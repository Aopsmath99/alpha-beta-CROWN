#!/usr/bin/env python
"""Alpha-beta-CROWN benchmark runner with structured metrics collection.

Runs the verifier on one or more configurations, collects detailed metrics
(verification time, alpha-CROWN optimization time, iteration convergence,
final bound tightness, verified/falsified/timeout counts, BaB subproblems),
and saves results to JSON with a summary table on stdout.

By default, each instance is run as a separate subprocess so the OS reclaims
all memory between instances (prevents the monotonic RAM growth that occurs
when CPython processes many heavy instances in a single process).

Requires the ABCROWN_METRICS_PATH instrumentation in auto_LiRPA/metrics.py.

Usage examples:

  # Single configuration run (each instance isolated in its own process)
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/baseline/

  # Limit to 10 instances for quick testing
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/quick/ --num-instances 10

  # Sweep over a single parameter
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/lr_sweep/ \\
      --sweep "solver.alpha-crown.lr_alpha=0.01,0.05,0.1,0.2,0.5"

  # Grid sweep over multiple parameters
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/grid/ --num-instances 10 \\
      --sweep "solver.alpha-crown.lr_alpha=0.01,0.1,0.5" \\
      --sweep "solver.alpha-crown.iteration=20,50,100"

  # Old behaviour: all instances in one process (faster startup, but leaks RAM)
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/baseline/ --batch-mode
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print('ERROR: PyYAML is required. Install with: pip install pyyaml')
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class SimplePbar:
    """Minimal progress bar fallback when tqdm is not available."""

    def __init__(self, total=None, desc='', **kwargs):
        self.total = total
        self.desc = desc
        self.n = 0
        self._start = time.time()

    def update(self, n=1):
        self.n += n
        elapsed = time.time() - self._start
        rate = self.n / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.n) / rate if rate > 0 else 0
        print(
            f'\r  {self.desc} {self.n}/{self.total} '
            f'[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]',
            end='', flush=True,
        )

    def set_postfix_str(self, s):
        pass

    def close(self):
        print()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_pbar(total, desc, **kwargs):
    """Create a progress bar (tqdm if available, otherwise SimplePbar)."""
    if HAS_TQDM:
        return tqdm(
            total=total, desc=desc, unit='inst',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                       '[{elapsed}<{remaining}, {rate_fmt}]',
            **kwargs,
        )
    return SimplePbar(total=total, desc=desc)


def parse_sweep(sweep_str):
    """Parse a sweep specification string.

    Format: 'dotted.key.path=val1,val2,val3'
    Values are auto-typed as int, float, bool, or string.

    Returns:
        (key_path: str, values: list)
    """
    if '=' not in sweep_str:
        raise ValueError(
            f'Sweep must be in format KEY=V1,V2,...  Got: {sweep_str}'
        )
    key, values_str = sweep_str.split('=', 1)
    values = []
    for v in values_str.split(','):
        v = v.strip()
        if not v:
            continue
        try:
            values.append(int(v))
        except ValueError:
            try:
                values.append(float(v))
            except ValueError:
                if v.lower() == 'true':
                    values.append(True)
                elif v.lower() == 'false':
                    values.append(False)
                else:
                    values.append(v)
    if not values:
        raise ValueError(f'No values found in sweep: {sweep_str}')
    return key, values


def set_nested(d, key_path, value):
    """Set a value in a nested dict using a dotted key path."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def get_nested(d, key_path, default=None):
    """Get a value from a nested dict using a dotted key path."""
    keys = key_path.split('.')
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


# ---------------------------------------------------------------------------
#  Per-instance subprocess isolation (default mode)
# ---------------------------------------------------------------------------

def run_instance_isolated(config, instance_idx, metrics_path, log_path,
                          python_exe):
    """Run a single verification instance in its own subprocess.

    Creates a temporary config with data.start=instance_idx and
    data.end=instance_idx+1, so the subprocess handles exactly one instance
    and then exits (releasing all RAM back to the OS).

    Returns:
        (returncode, instance_metrics_dict_or_None, status_line_or_None)
    """
    instance_config = copy.deepcopy(config)
    set_nested(instance_config, 'data.start', instance_idx)
    set_nested(instance_config, 'data.end', instance_idx + 1)

    config_path = metrics_path.parent / f'instance_{instance_idx}_config.yaml'
    inst_metrics = metrics_path.parent / f'instance_{instance_idx}_metrics.json'

    with open(config_path, 'w') as f:
        yaml.dump(instance_config, f, default_flow_style=False)

    if inst_metrics.exists():
        inst_metrics.unlink()

    env = os.environ.copy()
    env['ABCROWN_METRICS_PATH'] = str(inst_metrics)
    env['PYTHONUNBUFFERED'] = '1'

    cmd = [python_exe, '-u', 'abcrown.py', '--config', str(config_path)]

    status_line = None
    with open(log_path, 'w') as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            log_f.write(line)
            if line.strip().startswith('Result:'):
                status_line = line.strip()
        process.wait()

    metrics = None
    if inst_metrics.exists():
        try:
            with open(inst_metrics, 'r') as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Clean up the per-instance config (the metrics file is kept)
    try:
        config_path.unlink()
    except OSError:
        pass

    return process.returncode, metrics, status_line


def merge_instance_metrics(all_instance_metrics):
    """Merge per-instance metrics dicts into one combined metrics dict."""
    merged_instance_results = []
    merged_optimization_calls = []

    for m in all_instance_metrics:
        if m is None:
            continue
        merged_instance_results.extend(m.get('instance_results', []))
        merged_optimization_calls.extend(m.get('optimization_calls', []))

    if not merged_instance_results:
        return None

    times = [r['total_time_seconds'] for r in merged_instance_results]
    verified = [
        r for r in merged_instance_results
        if 'safe' in r['status'] and 'unsafe' not in r['status']
    ]
    falsified = [
        r for r in merged_instance_results
        if 'unsafe' in r['status']
    ]
    timeout = [
        r for r in merged_instance_results
        if 'unknown' in r['status'] or 'timeout' in r['status']
    ]

    alpha_calls = [
        c for c in merged_optimization_calls if c['type'] == 'alpha-crown'
    ]
    beta_calls = [
        c for c in merged_optimization_calls if c['type'] == 'beta-crown'
    ]

    summary = {
        'total_instances': len(merged_instance_results),
        'verified_count': len(verified),
        'falsified_count': len(falsified),
        'timeout_count': len(timeout),
        'verified_rate': (
            len(verified) / len(merged_instance_results)
            if merged_instance_results else 0
        ),
        'mean_time_all': sum(times) / len(times) if times else 0,
        'mean_time_verified': (
            sum(r['total_time_seconds'] for r in verified) / len(verified)
            if verified else 0
        ),
        'mean_time_falsified': (
            sum(r['total_time_seconds'] for r in falsified) / len(falsified)
            if falsified else 0
        ),
        'total_bab_domains': sum(
            r['bab_domains_visited'] for r in merged_instance_results
        ),
    }

    if alpha_calls:
        summary['alpha_crown_total_time'] = sum(
            c['time_seconds'] for c in alpha_calls
        )
        summary['alpha_crown_mean_iterations'] = (
            sum(c['iterations_completed'] for c in alpha_calls)
            / len(alpha_calls)
        )
        summary['alpha_crown_call_count'] = len(alpha_calls)

    if beta_calls:
        summary['beta_crown_total_time'] = sum(
            c['time_seconds'] for c in beta_calls
        )
        summary['beta_crown_call_count'] = len(beta_calls)
        summary['beta_crown_mean_iterations'] = (
            sum(c['iterations_completed'] for c in beta_calls)
            / len(beta_calls)
        )

    return {
        'summary': summary,
        'instance_results': merged_instance_results,
        'optimization_calls': merged_optimization_calls,
    }


def run_config_per_instance(config, data_start, data_end, config_dir,
                            python_exe, pbar):
    """Run all instances for one config, each in its own subprocess."""
    metrics_path = config_dir / 'metrics.json'
    all_instance_metrics = []
    error_count = 0

    for i in range(data_start, data_end):
        log_path = config_dir / f'instance_{i}.log'

        returncode, metrics, status_line = run_instance_isolated(
            config, i, metrics_path, log_path, python_exe,
        )

        if returncode != 0:
            error_count += 1

        all_instance_metrics.append(metrics)

        short = ''
        if status_line:
            short = status_line.replace('Result: ', '')
            if len(short) > 30:
                short = short[:27] + '...'
        pbar.update(1)
        pbar.set_postfix_str(short)

    merged = merge_instance_metrics(all_instance_metrics)
    if merged:
        with open(metrics_path, 'w') as f:
            json.dump(merged, f, indent=2)

    return merged, error_count


# ---------------------------------------------------------------------------
#  Batch mode (old behaviour: all instances in one subprocess)
# ---------------------------------------------------------------------------

def run_single_config_batch(config_path, metrics_path, python_exe,
                            on_instance=None):
    """Run the verifier on a single configuration (all instances at once).

    This is the legacy mode: one subprocess handles all instances.  Faster to
    start up (model loaded once), but CPython never returns freed RAM to the
    OS so memory grows monotonically across instances.
    """
    env = os.environ.copy()
    env['ABCROWN_METRICS_PATH'] = str(metrics_path)
    env['PYTHONUNBUFFERED'] = '1'

    cmd = [python_exe, '-u', 'abcrown.py', '--config', str(config_path)]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    stdout_lines = []
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        stdout_lines.append(line)
        if line.strip().startswith('Result:') and on_instance:
            on_instance(line.strip())

    process.wait()
    stdout_text = ''.join(stdout_lines)

    metrics = None
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'  WARNING: Could not read metrics file: {e}')

    return process.returncode, metrics, stdout_text


# ---------------------------------------------------------------------------
#  Config variant builder
# ---------------------------------------------------------------------------

def build_config_variants(base_config, sweep_specs, num_instances=None):
    """Build all configuration variants from base config and sweep specs."""
    if sweep_specs:
        sweep_keys = [s[0] for s in sweep_specs]
        sweep_values = [s[1] for s in sweep_specs]
        configs = []
        for combo in itertools.product(*sweep_values):
            config = copy.deepcopy(base_config)
            desc_parts = []
            for key, val in zip(sweep_keys, combo):
                set_nested(config, key, val)
                short_key = key.split('.')[-1]
                desc_parts.append(f'{short_key}={val}')
            configs.append((config, ', '.join(desc_parts)))
    else:
        configs = [(copy.deepcopy(base_config), 'baseline')]

    if num_instances is not None:
        for config, _ in configs:
            start = get_nested(config, 'data.start', 0)
            set_nested(config, 'data.end', start + num_instances)

    return configs


# ---------------------------------------------------------------------------
#  Summary display
# ---------------------------------------------------------------------------

def print_comparison_table(results):
    """Print a comparison table to stdout."""
    if not results:
        return

    cols = [
        ('Config', 20),
        ('Verified', 8),
        ('Falsified', 9),
        ('Timeout', 7),
        ('Mean Time', 10),
        ('a-CROWN Time', 12),
        ('a-CROWN Iters', 13),
        ('BaB Domains', 11),
    ]

    max_desc = max(len(r['description']) for r in results)
    cols[0] = ('Config', max(cols[0][1], max_desc + 2))

    header = ' | '.join(name.center(width) for name, width in cols)
    separator = '-+-'.join('-' * width for _, width in cols)

    print()
    print('Results Comparison')
    print('=' * len(header))
    print(header)
    print(separator)

    for r in results:
        s = r.get('summary', {})
        row_data = [
            r['description'],
            str(s.get('verified_count', '-')),
            str(s.get('falsified_count', '-')),
            str(s.get('timeout_count', '-')),
            f'{s.get("mean_time_all", 0):.2f}s',
            f'{s.get("alpha_crown_total_time", 0):.2f}s',
            f'{s.get("alpha_crown_mean_iterations", 0):.1f}',
            str(s.get('total_bab_domains', '-')),
        ]
        row = ' | '.join(
            val.center(width) for val, (_, width) in zip(row_data, cols)
        )
        print(row)

    print('=' * len(header))


def print_single_summary(result):
    """Print a detailed summary for a single configuration run."""
    s = result.get('summary', {})
    if not s:
        return

    print()
    print('  Verification Results')
    print('  ' + '-' * 40)
    total = s.get('total_instances', 0)
    print(f'  Total instances:       {total}')
    print(f'  Verified (safe):       {s.get("verified_count", 0)}')
    print(f'  Falsified (unsafe):    {s.get("falsified_count", 0)}')
    print(f'  Timeout (unknown):     {s.get("timeout_count", 0)}')
    if total > 0:
        print(f'  Verified rate:         {s.get("verified_rate", 0):.1%}')
    print()
    print('  Timing')
    print('  ' + '-' * 40)
    print(f'  Mean time (all):       {s.get("mean_time_all", 0):.2f}s')
    print(f'  Mean time (verified):  {s.get("mean_time_verified", 0):.2f}s')
    if s.get('alpha_crown_total_time') is not None:
        print(f'  Alpha-CROWN total:     '
              f'{s.get("alpha_crown_total_time", 0):.2f}s')
        print(f'  Alpha-CROWN avg iters: '
              f'{s.get("alpha_crown_mean_iterations", 0):.1f}')
    if s.get('beta_crown_total_time') is not None:
        print(f'  Beta-CROWN total:      '
              f'{s.get("beta_crown_total_time", 0):.2f}s')
        print(f'  Beta-CROWN calls:      '
              f'{s.get("beta_crown_call_count", 0)}')
    print()
    print('  Branch and Bound')
    print('  ' + '-' * 40)
    print(f'  Total domains visited: {s.get("total_bab_domains", 0)}')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Alpha-beta-CROWN Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to the base YAML configuration file.',
    )
    parser.add_argument(
        '--output', required=True,
        help='Directory to save results (metrics JSON, logs, summary).',
    )
    parser.add_argument(
        '--num-instances', type=int, default=None,
        help='Number of instances to verify (overrides data.end in config).',
    )
    parser.add_argument(
        '--sweep', action='append', default=[],
        help='Parameter sweep: KEY=V1,V2,V3. Repeatable for grid search. '
             'Example: --sweep "solver.alpha-crown.lr_alpha=0.01,0.1,0.5"',
    )
    parser.add_argument(
        '--python', default=sys.executable,
        help='Python interpreter to use (default: current interpreter).',
    )
    parser.add_argument(
        '--batch-mode', action='store_true',
        help='Run all instances in one subprocess (faster startup but '
             'monotonically increasing RAM usage). Default: each instance '
             'runs in its own subprocess for memory isolation.',
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f'ERROR: Config file not found: {config_path}')
        sys.exit(1)

    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    sweep_specs = []
    for sweep_str in args.sweep:
        try:
            sweep_specs.append(parse_sweep(sweep_str))
        except ValueError as e:
            print(f'ERROR: {e}')
            sys.exit(1)

    configs = build_config_variants(
        base_config, sweep_specs, args.num_instances,
    )

    sample_config = configs[0][0]
    data_start = get_nested(sample_config, 'data.start', 0)
    data_end = get_nested(sample_config, 'data.end', 100)
    instances_per_config = data_end - data_start

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_label = 'batch (shared process)' if args.batch_mode else 'isolated (per-instance process)'
    print()
    print('=' * 60)
    print('  Alpha-beta-CROWN Benchmark')
    print('=' * 60)
    print(f'  Base config:      {args.config}')
    if sweep_specs:
        for key, vals in sweep_specs:
            print(f'  Sweep:            {key} = {vals}')
    print(f'  Configurations:   {len(configs)}')
    print(f'  Instances/config: {instances_per_config}')
    print(f'  Instance mode:    {mode_label}')
    print(f'  Output directory: {output_dir}')
    print('=' * 60)
    print()

    all_results = []
    overall_start = time.time()

    for config_idx, (config, desc) in enumerate(configs):
        print(f'[{config_idx + 1}/{len(configs)}] {desc}')

        config_dir = output_dir / f'config_{config_idx}'
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path_run = config_dir / 'config.yaml'
        metrics_path = config_dir / 'metrics.json'
        with open(config_path_run, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        if metrics_path.exists():
            metrics_path.unlink()

        pbar = make_pbar(total=instances_per_config, desc='Verifying')

        if args.batch_mode:
            # Legacy mode: one subprocess for all instances.
            last_status = ''

            def on_instance(status_line):
                nonlocal last_status
                last_status = status_line
                pbar.update(1)
                short = status_line.replace('Result: ', '')
                if len(short) > 30:
                    short = short[:27] + '...'
                pbar.set_postfix_str(short)

            returncode, metrics, stdout = run_single_config_batch(
                config_path_run, metrics_path, args.python,
                on_instance=on_instance,
            )
            pbar.close()

            with open(config_dir / 'stdout.log', 'w') as f:
                f.write(stdout)

            if returncode != 0:
                print(f'  WARNING: Verifier exited with code {returncode}')
                print(f'  Check {config_dir / "stdout.log"} for details.')
        else:
            # Default mode: one subprocess per instance.
            cfg_data_start = get_nested(config, 'data.start', 0)
            cfg_data_end = get_nested(config, 'data.end', 100)

            metrics, error_count = run_config_per_instance(
                config, cfg_data_start, cfg_data_end, config_dir,
                args.python, pbar,
            )
            pbar.close()

            if error_count > 0:
                print(f'  WARNING: {error_count} instance(s) exited with '
                      f'non-zero code. Check instance logs in {config_dir}/')

        if metrics:
            result = {
                'config_idx': config_idx,
                'description': desc,
                'config': {k: str(v) for k, v in zip(
                    [s[0] for s in sweep_specs],
                    [get_nested(config, s[0]) for s in sweep_specs],
                )} if sweep_specs else {},
                'summary': metrics.get('summary', {}),
                'instance_results': metrics.get('instance_results', []),
                'optimization_calls_count': len(
                    metrics.get('optimization_calls', [])
                ),
            }
            all_results.append(result)

            s = result['summary']
            print(
                f'  => Verified: {s.get("verified_count", "?")}  '
                f'Falsified: {s.get("falsified_count", "?")}  '
                f'Timeout: {s.get("timeout_count", "?")}  '
                f'Mean time: {s.get("mean_time_all", 0):.2f}s'
            )
        else:
            print('  => ERROR: No metrics collected.')
            all_results.append({
                'config_idx': config_idx,
                'description': desc,
                'summary': {},
                'error': True,
            })

        print()

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - overall_start
    print('-' * 60)
    print(f'Benchmark completed in {total_time:.1f}s')
    print(f'Results saved to: {output_dir}')
    print(f'Summary JSON:     {summary_path}')

    if len(all_results) > 1:
        print_comparison_table(all_results)
    elif len(all_results) == 1:
        print_single_summary(all_results[0])

    print()

    print('Per-configuration files:')
    for i, r in enumerate(all_results):
        d = output_dir / f'config_{i}'
        print(f'  [{i}] {r["description"]}')
        print(f'      Config:  {d / "config.yaml"}')
        print(f'      Metrics: {d / "metrics.json"}')
        if args.batch_mode:
            print(f'      Log:     {d / "stdout.log"}')
        else:
            print(f'      Logs:    {d / "instance_*.log"}')

    return 0 if all(not r.get('error') for r in all_results) else 1


if __name__ == '__main__':
    sys.exit(main())
