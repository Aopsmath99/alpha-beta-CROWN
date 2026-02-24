#!/usr/bin/env python
"""Alpha-beta-CROWN benchmark runner with structured metrics collection.

Runs the verifier on one or more configurations, collects detailed metrics
(verification time, alpha-CROWN optimization time, iteration convergence,
final bound tightness, verified/falsified/timeout counts, BaB subproblems),
and saves results to JSON with a summary table on stdout.

Requires the ABCROWN_METRICS_PATH instrumentation in auto_LiRPA/metrics.py.

Usage examples:

  # Single configuration run
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/baseline/

  # Limit to 10 instances for quick testing
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/quick/ --num-instances 100

  # Sweep over a single parameter
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/lr_sweep/ \\
      --sweep "solver.alpha-crown.lr_alpha=0.01,0.05,0.1,0.2,0.5"

  # Grid sweep over multiple parameters
  python benchmark.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml \\
      --output results/grid/ --num-instances 10 \\
      --sweep "solver.alpha-crown.lr_alpha=0.01,0.1,0.5" \\
      --sweep "solver.alpha-crown.iteration=20,50,100"
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
        # Try int, then float, then bool, then keep as string
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
    """Set a value in a nested dict using a dotted key path.

    Example: set_nested(d, 'solver.alpha-crown.lr_alpha', 0.1)
    sets d['solver']['alpha-crown']['lr_alpha'] = 0.1
    """
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


def run_single_config(config_path, metrics_path, python_exe, on_instance=None):
    """Run the verifier on a single configuration.

    Args:
        config_path: path to the YAML config file.
        metrics_path: path where the MetricsCollector will dump JSON.
        python_exe: path to the Python interpreter.
        on_instance: callback(status_line) called when an instance completes.

    Returns:
        (returncode, metrics_dict_or_None, stdout_text)
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
        # Detect per-instance completion by looking for "Result:" lines
        if line.strip().startswith('Result:') and on_instance:
            on_instance(line.strip())

    process.wait()
    stdout_text = ''.join(stdout_lines)

    # Read the metrics JSON dumped by the instrumented code
    metrics = None
    metrics_path = Path(metrics_path)
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'  WARNING: Could not read metrics file: {e}')

    return process.returncode, metrics, stdout_text


def build_config_variants(base_config, sweep_specs, num_instances=None):
    """Build all configuration variants from base config and sweep specs.

    Args:
        base_config: dict loaded from the base YAML file.
        sweep_specs: list of (key_path, [values]) from --sweep args.
        num_instances: if set, override data.end.

    Returns:
        list of (config_dict, description_string) tuples.
    """
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

    # Override instance count if requested
    if num_instances is not None:
        for config, _ in configs:
            start = get_nested(config, 'data.start', 0)
            set_nested(config, 'data.end', start + num_instances)

    return configs


def print_comparison_table(results):
    """Print a comparison table to stdout."""
    if not results:
        return

    # Define columns
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

    # Adjust first column width to fit descriptions
    max_desc = max(len(r['description']) for r in results)
    cols[0] = ('Config', max(cols[0][1], max_desc + 2))

    # Print header
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
    args = parser.parse_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f'ERROR: Config file not found: {config_path}')
        sys.exit(1)

    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Parse sweep specifications
    sweep_specs = []
    for sweep_str in args.sweep:
        try:
            sweep_specs.append(parse_sweep(sweep_str))
        except ValueError as e:
            print(f'ERROR: {e}')
            sys.exit(1)

    # Build all configuration variants
    configs = build_config_variants(
        base_config, sweep_specs, args.num_instances,
    )

    # Determine instance count for progress tracking
    sample_config = configs[0][0]
    data_start = get_nested(sample_config, 'data.start', 0)
    data_end = get_nested(sample_config, 'data.end', 100)
    instances_per_config = data_end - data_start

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print banner
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
    print(f'  Output directory: {output_dir}')
    print('=' * 60)
    print()

    # Run each configuration
    all_results = []
    overall_start = time.time()

    for config_idx, (config, desc) in enumerate(configs):
        print(f'[{config_idx + 1}/{len(configs)}] {desc}')

        # Create per-config output directory
        config_dir = output_dir / f'config_{config_idx}'
        config_dir.mkdir(parents=True, exist_ok=True)

        # Write the config for this run
        config_path_run = config_dir / 'config.yaml'
        metrics_path = config_dir / 'metrics.json'
        with open(config_path_run, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Clean up old metrics file if it exists
        if metrics_path.exists():
            metrics_path.unlink()

        # Run with progress bar
        pbar = make_pbar(total=instances_per_config, desc='Verifying')
        last_status = ''

        def on_instance(status_line):
            nonlocal last_status
            last_status = status_line
            pbar.update(1)
            # Show the most recent result in the progress bar
            short = status_line.replace('Result: ', '')
            if len(short) > 30:
                short = short[:27] + '...'
            pbar.set_postfix_str(short)

        returncode, metrics, stdout = run_single_config(
            config_path_run, metrics_path, args.python,
            on_instance=on_instance,
        )
        pbar.close()

        # Save stdout log
        with open(config_dir / 'stdout.log', 'w') as f:
            f.write(stdout)

        if returncode != 0:
            print(f'  WARNING: Verifier exited with code {returncode}')
            print(f'  Check {config_dir / "stdout.log"} for details.')

        # Collect results
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

            # Quick inline summary
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

    # Save combined results (without the full optimization_calls to keep
    # the summary file manageable; those are in per-config metrics.json)
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
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

    # Per-config files listing
    print('Per-configuration files:')
    for i, r in enumerate(all_results):
        d = output_dir / f'config_{i}'
        print(f'  [{i}] {r["description"]}')
        print(f'      Config:  {d / "config.yaml"}')
        print(f'      Metrics: {d / "metrics.json"}')
        print(f'      Log:     {d / "stdout.log"}')

    return 0 if all(not r.get('error') for r in all_results) else 1


if __name__ == '__main__':
    sys.exit(main())
