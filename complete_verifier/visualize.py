#!/usr/bin/env python
"""Visualization tools for alpha-beta-CROWN verification results.

Creates four types of visualizations:

1. **Image Grid** (``results``): Shows test images organized by
   verification status (verified safe, falsified, timeout) with
   colour-coded borders and classification info.

2. **PGD Comparison** (``pgd``): For each image, shows the original
   (clean) image side-by-side with the PGD-attacked adversarial image
   and a magnified perturbation map.  Useful for demonstrating what
   an epsilon-ball attack looks like in practice.

3. **ReLU Relaxation Diagram** (``relaxation``): Shows how different
   alpha initializations affect the linear relaxation of a ReLU neuron.

4. **Convergence Curves** (``convergence``): Plots optimization loss
   and bound history across different configurations.

Usage:

  # Visualize verification results (status grid)
  python visualize.py results \\
      --metrics results/baseline/config_0/metrics.json \\
      --dataset CIFAR --output figures/

  # Show PGD adversarial effect for all images
  python visualize.py pgd \\
      --metrics results/baseline/config_0/metrics.json \\
      --dataset CIFAR \\
      --model-path models/cifar10_resnet/resnet2b.pth \\
      --epsilon 0.00784 --steps 50 \\
      --output figures/

  # Show PGD adversarial effect only for falsified images
  python visualize.py pgd \\
      --metrics results/baseline/config_0/metrics.json \\
      --dataset CIFAR --model-path models/resnet2b.pth \\
      --falsified-only --output figures/

  # Draw the ReLU relaxation diagram (no data needed)
  python visualize.py relaxation --output figures/

  # Draw convergence curves from one or more experiments
  python visualize.py convergence \\
      --metrics results/baseline/config_0/metrics.json \\
      --output figures/

Requires: matplotlib, torchvision (both usually already installed).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving to files
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── CIFAR-10 / MNIST metadata ──────────────────────────────────────

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2471, 0.2435, 0.2616]

MNIST_CLASSES = [str(i) for i in range(10)]
MNIST_MEAN = [0.1307]
MNIST_STD = [0.3081]


# ── Helper Functions ────────────────────────────────────────────────

def load_dataset(dataset_name, data_dir='./data', num_images=100, start=0):
    """Load test images from CIFAR-10 or MNIST."""
    if not HAS_TORCH:
        raise ImportError('PyTorch and torchvision are required. '
                          'Install with: pip install torch torchvision')

    if dataset_name.upper() in ('CIFAR', 'CIFAR10', 'CIFAR-10'):
        transform = transforms.ToTensor()
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)
        classes = CIFAR10_CLASSES
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name.upper() == 'MNIST':
        transform = transforms.ToTensor()
        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform)
        classes = MNIST_CLASSES
        mean, std = MNIST_MEAN, MNIST_STD
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}. Use CIFAR or MNIST.')

    end = min(start + num_images, len(testset))
    images = []
    labels = []
    for i in range(start, end):
        img, label = testset[i]
        images.append(img)
        labels.append(label)

    return images, labels, classes, mean, std


def unnormalize(img_tensor, mean, std):
    """Convert a normalized tensor back to displayable [0,1] range."""
    img = img_tensor.clone()
    if img.dim() == 3:
        for c in range(img.shape[0]):
            if c < len(mean):
                img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1)


def tensor_to_display(img_tensor):
    """Convert a CHW tensor to HWC numpy for matplotlib."""
    img = img_tensor.numpy()
    if img.shape[0] == 1:
        return img.squeeze(0)  # Grayscale
    return np.transpose(img, (1, 2, 0))  # CHW -> HWC


def load_model(model_path, device='cpu', model_name=None):
    """Load a model architecture + weights from a .pth file.

    Uses model_defs.py (the same module the verifier uses) to instantiate
    the architecture by calling the function whose name matches either
    *model_name* or the stem of *model_path* (e.g. ``resnet2b.pth`` ->
    ``resnet2b()``).

    Falls back to ``torch.load`` if model_defs lookup fails.
    """
    if model_name is None:
        model_name = Path(model_path).stem  # e.g. 'resnet2b'

    # Make sure the complete_verifier directory is on sys.path so
    # 'import model_defs' works regardless of cwd.
    verifier_dir = str(Path(__file__).resolve().parent)
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)

    model = None

    # Strategy 1: look up model_name in model_defs (same as load_model.py)
    try:
        import model_defs
        if hasattr(model_defs, model_name):
            model = getattr(model_defs, model_name)()
        else:
            # Try common naming variations
            for variant in [model_name.replace('-', '_'),
                            model_name.lower(),
                            model_name.upper()]:
                if hasattr(model_defs, variant):
                    model = getattr(model_defs, variant)()
                    break
    except ImportError:
        pass

    if model is not None:
        # Load state dict into the architecture
        sd = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        if isinstance(sd, list):
            sd = sd[0]
        if isinstance(sd, dict):
            model.load_state_dict(sd)
    else:
        # Strategy 2: torch.load the whole model (architecture + weights)
        model = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(model, dict):
            raise RuntimeError(
                f'Could not load model from {model_path}. '
                f'Model definition "{model_name}" was not found in '
                f'model_defs.py, and the .pth file contains only a '
                f'state_dict (not a full model).'
            )

    model.eval()
    model.to(device)
    return model


def classify_images(images, model_path, device='cpu'):
    """Run forward pass to get predictions and confidences."""
    if not HAS_TORCH:
        raise ImportError('PyTorch is required.')

    model = load_model(model_path, device)

    predictions = []
    confidences = []
    with torch.no_grad():
        for img in images:
            output = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0, pred].item()
            predictions.append(pred)
            confidences.append(conf)

    return predictions, confidences


# ── PGD Attack ──────────────────────────────────────────────────────

class NormalizedModel(torch.nn.Module):
    """Wrap a model so it normalizes [0,1] inputs before forwarding."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer(
            'mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer(
            'std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def pgd_attack(model, images, labels, epsilon, step_size=None,
               num_steps=50, device='cpu'):
    """Run a PGD attack on a batch of images.

    Args:
        model: a model that accepts [0,1] range images.
        images: list of CHW tensors in [0,1].
        labels: list of true label indices.
        epsilon: perturbation budget in [0,1] scale.
        step_size: PGD step size (default: epsilon/4).
        num_steps: number of PGD iterations.
        device: torch device.

    Returns:
        adv_images: list of adversarial CHW tensors in [0,1].
        adv_preds: list of adversarial predicted labels.
        adv_confs: list of confidence on true class after attack.
        succeeded: list of bools indicating if attack changed prediction.
    """
    if step_size is None:
        step_size = epsilon / 4.0

    model.eval()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    adv_images = []
    adv_preds = []
    adv_confs = []
    succeeded = []

    for img, label in zip(images, labels):
        x = img.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)

        # Random start within epsilon ball
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        delta = torch.clamp(x + delta, 0.0, 1.0) - x
        delta.requires_grad_(True)

        for _ in range(num_steps):
            x_adv = x + delta
            output = model(x_adv)
            loss = loss_fn(output, y)

            loss.backward()
            grad = delta.grad.detach()

            # Steepest ascent (maximize loss = fool the classifier)
            delta = delta.detach() + step_size * grad.sign()
            # Project back to epsilon ball
            delta = torch.clamp(delta, -epsilon, epsilon)
            # Project back to valid image range
            delta = torch.clamp(x + delta, 0.0, 1.0) - x
            delta.requires_grad_(True)

        with torch.no_grad():
            x_adv = (x + delta).clamp(0.0, 1.0)
            output = model(x_adv)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf_true = probs[0, label].item()

        adv_images.append(x_adv.squeeze(0).cpu())
        adv_preds.append(pred)
        adv_confs.append(conf_true)
        succeeded.append(pred != label)

    return adv_images, adv_preds, adv_confs, succeeded


def plot_pgd_comparison(images, adv_images, labels, classes, mean, std,
                        predictions=None, adv_preds=None, adv_confs=None,
                        succeeded=None, instance_results=None,
                        epsilon=None, output_path=None, title=None):
    """Create a comparison grid: Original | Adversarial | Perturbation.

    For each image, shows the clean version, the PGD-attacked version, and
    a magnified view of the perturbation.  Borders indicate verification
    status; red title highlights images where the attack succeeded.

    Args:
        images: list of clean CHW tensors in [0,1].
        adv_images: list of adversarial CHW tensors in [0,1].
        labels: list of true label indices.
        classes: list of class name strings.
        mean, std: normalization parameters (for display unnormalization).
        predictions: clean model predictions (optional).
        adv_preds: adversarial predictions.
        adv_confs: confidence on true class after attack.
        succeeded: list of bools (attack flipped prediction).
        instance_results: list of dicts from metrics.json (optional).
        epsilon: perturbation budget (for display).
        output_path: path to save figure (optional).
        title: figure title.
    """
    if not HAS_MPL:
        raise ImportError('matplotlib is required.')

    n = len(images)
    if n == 0:
        print('No images to display.')
        return

    # 3 columns per image: Original | Adversarial | Perturbation
    fig_width = 3 * 2.2
    fig_height = n * 2.4
    fig, axes = plt.subplots(n, 3, figsize=(fig_width, fig_height),
                             squeeze=False)

    col_headers = ['Original (clean)', 'Adversarial (PGD)', 'Perturbation (×10)']
    for j, header in enumerate(col_headers):
        axes[0, j].set_title(header, fontsize=10, fontweight='bold', pad=8)

    for i in range(n):
        # --- Original image ---
        ax_orig = axes[i, 0]
        img_disp = tensor_to_display(images[i].clamp(0, 1))
        if img_disp.ndim == 2:
            ax_orig.imshow(img_disp, cmap='gray', vmin=0, vmax=1)
        else:
            ax_orig.imshow(img_disp)

        true_name = classes[labels[i]]
        orig_label = f'True: {true_name}'
        if predictions is not None:
            pred_name = classes[predictions[i]]
            orig_label += f'\nPred: {pred_name}'
        ax_orig.set_ylabel(orig_label, fontsize=8, rotation=0, labelpad=60,
                           verticalalignment='center')

        # --- Adversarial image ---
        ax_adv = axes[i, 1]
        adv_disp = tensor_to_display(adv_images[i].clamp(0, 1))
        if adv_disp.ndim == 2:
            ax_adv.imshow(adv_disp, cmap='gray', vmin=0, vmax=1)
        else:
            ax_adv.imshow(adv_disp)

        adv_label = ''
        if adv_preds is not None:
            adv_name = classes[adv_preds[i]]
            adv_label = f'Pred: {adv_name}'
        if adv_confs is not None:
            adv_label += f'\nP(true)={adv_confs[i]:.2%}'
        if succeeded is not None and succeeded[i]:
            adv_label += '\nATTACK SUCCESS'
        ax_adv.set_xlabel(adv_label, fontsize=7,
                          color='#e74c3c' if (succeeded and succeeded[i])
                          else 'black')

        # --- Perturbation (magnified) ---
        ax_pert = axes[i, 2]
        perturbation = adv_images[i] - images[i]
        # Shift to [0,1] range for display: 0.5 = no change
        pert_vis = (perturbation * 10.0 + 0.5).clamp(0, 1)
        pert_disp = tensor_to_display(pert_vis)
        if pert_disp.ndim == 2:
            ax_pert.imshow(pert_disp, cmap='RdBu_r', vmin=0, vmax=1)
        else:
            ax_pert.imshow(pert_disp)

        l_inf = perturbation.abs().max().item()
        l2 = perturbation.norm(2).item()
        ax_pert.set_xlabel(f'L∞={l_inf:.5f}  L2={l2:.3f}', fontsize=7)

        # Set verification status border on all 3 columns
        for ax in [ax_orig, ax_adv, ax_pert]:
            ax.set_xticks([])
            ax.set_yticks([])
            if instance_results and i < len(instance_results):
                status = instance_results[i].get('status', '')
                border_color = get_status_color(status)
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)

    eps_str = f'  (ε={epsilon:.5f})' if epsilon else ''
    suptitle = title or f'PGD Adversarial Comparison{eps_str}'
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)

    if instance_results:
        legend_patches = [
            mpatches.Patch(color='#2ecc71', label='Verified (safe)'),
            mpatches.Patch(color='#e74c3c', label='Falsified (unsafe)'),
            mpatches.Patch(color='#f39c12', label='Timeout (unknown)'),
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=3, fontsize=9, frameon=True)

    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved PGD comparison to {output_path}')
    else:
        plt.show()
    plt.close(fig)


def load_metrics(metrics_path):
    """Load verification results from a metrics.json file."""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


# ── Visualization: Image Grid ──────────────────────────────────────

STATUS_COLORS = {
    'safe': '#2ecc71',            # green
    'safe-incomplete': '#2ecc71',
    'unsat': '#2ecc71',
    'unsafe': '#e74c3c',          # red
    'unsafe-pgd': '#e74c3c',
    'unsafe-bab': '#e74c3c',
    'sat': '#e74c3c',
    'unknown': '#f39c12',         # orange
    'timeout': '#f39c12',
}


def get_status_color(status):
    """Get display color for a verification status."""
    status_lower = status.lower()
    # Exact match first (handles 'unsafe-pgd', 'safe-incomplete', etc.)
    if status_lower in STATUS_COLORS:
        return STATUS_COLORS[status_lower]
    # Substring match, longest keys first so 'unsafe' beats 'safe'
    for key in sorted(STATUS_COLORS.keys(), key=len, reverse=True):
        if key in status_lower:
            return STATUS_COLORS[key]
    return '#95a5a6'  # gray fallback


def get_status_label(status):
    """Get short display label for a verification status."""
    s = status.lower()
    if 'safe' in s and 'unsafe' not in s:
        return 'VERIFIED'
    elif 'unsafe' in s or 'sat' == s:
        return 'FALSIFIED'
    elif 'unknown' in s or 'timeout' in s:
        return 'TIMEOUT'
    return status.upper()


def plot_image_grid(images, labels, classes, mean, std,
                    predictions=None, confidences=None,
                    instance_results=None, output_path=None,
                    max_images=50, title=None):
    """Create a grid of images annotated with classification and verification info.

    Args:
        images: list of CHW tensors.
        labels: list of true label indices.
        classes: list of class name strings.
        mean, std: normalization parameters.
        predictions: list of predicted label indices (optional).
        confidences: list of confidence values (optional).
        instance_results: list of dicts from metrics.json (optional).
        output_path: path to save figure (optional, shows if None).
        max_images: max images to show.
        title: figure title.
    """
    if not HAS_MPL:
        raise ImportError('matplotlib is required. Install: pip install matplotlib')

    n = min(len(images), max_images)
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    fig_width = cols * 2.0
    fig_height = rows * 2.4
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        # Unnormalize and display
        img_display = tensor_to_display(unnormalize(images[idx], mean, std))
        if img_display.ndim == 2:
            ax.imshow(img_display, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img_display)

        ax.set_xticks([])
        ax.set_yticks([])

        # Build annotation text
        true_name = classes[labels[idx]]
        line1 = f'True: {true_name}'

        if predictions is not None:
            pred_name = classes[predictions[idx]]
            correct = predictions[idx] == labels[idx]
            conf_str = f' ({confidences[idx]:.0%})' if confidences else ''
            line2 = f'Pred: {pred_name}{conf_str}'
            # Color title by correctness
            title_color = '#2ecc71' if correct else '#e74c3c'
        else:
            line2 = ''
            title_color = 'black'

        # Verification status border + label
        if instance_results and idx < len(instance_results):
            status = instance_results[idx].get('status', '')
            vtime = instance_results[idx].get('total_time_seconds', 0)
            border_color = get_status_color(status)
            status_label = get_status_label(status)
            line3 = f'{status_label} ({vtime:.1f}s)'

            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
        else:
            line3 = ''

        label_text = line1
        if line2:
            label_text += f'\n{line2}'
        if line3:
            label_text += f'\n{line3}'

        ax.set_title(label_text, fontsize=7, color=title_color, pad=2)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    # Legend
    if instance_results:
        legend_patches = [
            mpatches.Patch(color='#2ecc71', label='Verified (safe)'),
            mpatches.Patch(color='#e74c3c', label='Falsified (unsafe)'),
            mpatches.Patch(color='#f39c12', label='Timeout (unknown)'),
        ]
        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=3, fontsize=9, frameon=True)

    suptitle = title or 'Verification Results'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved image grid to {output_path}')
    else:
        plt.show()
    plt.close(fig)


# ── Visualization: ReLU Relaxation Diagram ─────────────────────────

def plot_relu_relaxation(output_path=None):
    """Draw the ReLU relaxation triangle showing different alpha values.

    This creates a publication-quality diagram that explains:
    - The ReLU function
    - The upper bound (fixed convex relaxation)
    - The lower bound with different alpha values (binary vs continuous)
    - The "gap" that alpha optimization minimizes
    """
    if not HAS_MPL:
        raise ImportError('matplotlib is required.')

    # Example neuron with pre-activation bounds l=-2, u=3
    l, u = -2.0, 3.0
    upper_k = u / (u - l)  # = 0.6

    x = np.linspace(l - 0.5, u + 0.5, 300)
    relu_y = np.maximum(x, 0)

    # Upper bound line: passes through (l, 0) and (u, u)
    upper_slope = u / (u - l)
    upper_intercept = -l * upper_slope
    upper_y = upper_slope * x + upper_intercept
    upper_y_clipped = np.clip(upper_y, None, None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax_idx, (ax, alpha_label, alpha_val, color) in enumerate(zip(
        axes,
        ['Binary Init (old): α=1',
         'Binary Init (old): α=0',
         f'Continuous Init (new): α={upper_k:.2f}'],
        [1.0, 0.0, upper_k],
        ['#3498db', '#3498db', '#e74c3c'],
    )):
        # ReLU
        ax.plot(x, relu_y, 'k-', linewidth=2.5, label='ReLU(x)', zorder=5)

        # Upper bound
        # Only draw in the [l, u] region
        mask = (x >= l) & (x <= u)
        ax.plot(x[mask], upper_y[mask], '--', color='#7f8c8d',
                linewidth=1.5, label=f'Upper bound (slope={upper_slope:.2f})')

        # Lower bound with given alpha
        lower_y = alpha_val * x
        ax.plot(x[mask], lower_y[mask], '-', color=color,
                linewidth=2.5, label=f'Lower bound (α={alpha_val:.2f})')

        # Fill the relaxation gap
        x_fill = x[mask]
        relu_fill = np.maximum(x_fill, 0)
        upper_fill = upper_slope * x_fill + upper_intercept
        lower_fill = alpha_val * x_fill

        # Gap between upper bound and ReLU
        ax.fill_between(x_fill, relu_fill, upper_fill,
                        alpha=0.15, color='#7f8c8d', label='Relaxation gap')
        # Gap between ReLU and lower bound
        ax.fill_between(x_fill, lower_fill, relu_fill,
                        alpha=0.15, color=color)

        # Mark the bounds
        ax.axvline(x=l, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axvline(x=u, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # Annotations
        ax.annotate(f'l={l}', xy=(l, -0.3), fontsize=9, ha='center',
                    color='gray')
        ax.annotate(f'u={u}', xy=(u, -0.3), fontsize=9, ha='center',
                    color='gray')

        # Compute and show gap area
        gap_area = np.trapz(upper_fill - lower_fill, x_fill)
        ax.text(0.95, 0.95, f'Gap area ≈ {gap_area:.2f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

        ax.set_title(alpha_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Pre-activation value (z)')
        if ax_idx == 0:
            ax.set_ylabel('Post-activation / bound')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(l - 0.5, u + 0.5)
        ax.set_ylim(-1.0, u + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        'ReLU Linear Relaxation: Effect of Alpha Initialization\n'
        f'(Example neuron with bounds [{l}, {u}], upper_k = u/(u−l) = {upper_k:.2f})',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved relaxation diagram to {output_path}')
    else:
        plt.show()
    plt.close(fig)


# ── Visualization: Convergence Curves ──────────────────────────────

def plot_convergence(metrics_paths, labels=None, output_path=None):
    """Plot optimization convergence curves from one or more metrics files.

    Compares loss curves and bound history across different configurations
    (e.g., different alpha initializations).

    Args:
        metrics_paths: list of paths to metrics.json files.
        labels: list of labels for each metrics file.
        output_path: path to save figure.
    """
    if not HAS_MPL:
        raise ImportError('matplotlib is required.')

    if labels is None:
        labels = [Path(p).parent.name for p in metrics_paths]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_paths)))

    for idx, (mpath, label) in enumerate(zip(metrics_paths, labels)):
        data = load_metrics(mpath)
        opt_calls = data.get('optimization_calls', [])

        # Find the first alpha-crown call (the initial incomplete verification)
        alpha_calls = [c for c in opt_calls if c['type'] == 'alpha-crown']
        if not alpha_calls:
            print(f'  No alpha-crown calls found in {mpath}')
            continue

        # Use the first alpha-crown call (largest, most representative)
        call = alpha_calls[0]

        loss_hist = call.get('loss_history', [])
        lower_hist = call.get('best_lower_history', [])
        iters = list(range(len(loss_hist)))

        color = colors[idx]

        if loss_hist:
            axes[0].plot(iters, loss_hist, color=color, label=label,
                         linewidth=1.5)
        if lower_hist:
            valid_lower = [v for v in lower_hist if v is not None]
            valid_iters = [i for i, v in enumerate(lower_hist) if v is not None]
            if valid_lower:
                axes[1].plot(valid_iters, valid_lower, color=color,
                             label=label, linewidth=1.5)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Optimization Loss (first α-CROWN call)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Best Lower Bound (sum)')
    axes[1].set_title('Lower Bound Convergence (first α-CROWN call)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Alpha-CROWN Convergence Comparison', fontsize=13,
                 fontweight='bold')
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved convergence plot to {output_path}')
    else:
        plt.show()
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────

def cmd_results(args):
    """Subcommand: visualize verification results on images."""
    if not HAS_TORCH or not HAS_MPL:
        print('ERROR: This command requires torch, torchvision, and matplotlib.')
        sys.exit(1)

    # Load metrics
    data = load_metrics(args.metrics)
    instance_results = data.get('instance_results', [])
    summary = data.get('summary', {})

    if not instance_results:
        print('No instance results found in metrics file.')
        sys.exit(1)

    # Determine index range from results
    indices = [r['index'] for r in instance_results]
    start = min(indices)
    num = max(indices) - start + 1

    print(f'Loading {args.dataset} dataset ({num} images starting at {start})...')
    images, labels, classes, mean, std = load_dataset(
        args.dataset, num_images=num, start=start,
    )

    # Try to get model predictions
    predictions = confidences = None
    if args.model_path and Path(args.model_path).exists():
        try:
            print(f'Running classification with {args.model_path}...')
            predictions, confidences = classify_images(
                images, args.model_path,
            )
        except Exception as e:
            print(f'Could not load model for classification: {e}')
            print('Continuing without predictions.')

    # Sort results to match image order
    result_map = {r['index']: r for r in instance_results}
    sorted_results = [result_map.get(start + i, {}) for i in range(len(images))]

    # Title with summary
    title = (
        f'{args.dataset} Verification Results  |  '
        f'Verified: {summary.get("verified_count", "?")}  '
        f'Falsified: {summary.get("falsified_count", "?")}  '
        f'Timeout: {summary.get("timeout_count", "?")}'
    )

    output_path = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'verification_results.png')

    plot_image_grid(
        images, labels, classes, mean, std,
        predictions=predictions,
        confidences=confidences,
        instance_results=sorted_results,
        output_path=output_path,
        max_images=args.max_images,
        title=title,
    )

    # Also generate per-category grids
    if args.output:
        for category, cat_label in [
            ('safe', 'Verified Safe'),
            ('unsafe', 'Falsified'),
            ('unknown', 'Timeout'),
        ]:
            cat_indices = [
                i for i, r in enumerate(sorted_results)
                if category in r.get('status', '').lower()
                and ('unsafe' not in r.get('status', '').lower()
                     if category == 'safe' else True)
            ]
            if cat_indices:
                cat_images = [images[i] for i in cat_indices]
                cat_labels = [labels[i] for i in cat_indices]
                cat_preds = ([predictions[i] for i in cat_indices]
                             if predictions else None)
                cat_confs = ([confidences[i] for i in cat_indices]
                             if confidences else None)
                cat_results = [sorted_results[i] for i in cat_indices]

                plot_image_grid(
                    cat_images, cat_labels, classes, mean, std,
                    predictions=cat_preds,
                    confidences=cat_confs,
                    instance_results=cat_results,
                    output_path=str(output_dir / f'{category}_images.png'),
                    max_images=args.max_images,
                    title=f'{cat_label} Images ({len(cat_indices)} total)',
                )


def cmd_relaxation(args):
    """Subcommand: draw the ReLU relaxation diagram."""
    if not HAS_MPL:
        print('ERROR: matplotlib is required. Install: pip install matplotlib')
        sys.exit(1)

    output_path = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'relu_relaxation.png')

    plot_relu_relaxation(output_path=output_path)


def cmd_pgd(args):
    """Subcommand: visualize PGD adversarial perturbations."""
    if not HAS_TORCH or not HAS_MPL:
        print('ERROR: This command requires torch, torchvision, and matplotlib.')
        sys.exit(1)

    if not args.model_path or not Path(args.model_path).exists():
        print('ERROR: --model-path is required and must point to an existing .pth file.')
        sys.exit(1)

    # Load metrics to know which instances to show and their statuses
    data = load_metrics(args.metrics)
    instance_results = data.get('instance_results', [])
    if not instance_results:
        print('No instance results in metrics file.')
        sys.exit(1)

    indices = [r['index'] for r in instance_results]
    start = min(indices)
    num = max(indices) - start + 1

    # Optionally filter to only falsified instances
    if args.falsified_only:
        keep = [
            i for i, r in enumerate(instance_results)
            if 'unsafe' in r.get('status', '').lower()
        ]
        if not keep:
            print('No falsified instances found.')
            sys.exit(0)
    else:
        keep = list(range(len(instance_results)))

    # Limit the number of images shown
    keep = keep[:args.max_images]

    print(f'Loading {args.dataset} dataset ({num} images starting at {start})...')
    all_images, all_labels, classes, mean, std = load_dataset(
        args.dataset, num_images=num, start=start,
    )

    # Select the images we want to display
    result_map = {r['index']: r for r in instance_results}
    sorted_results = [result_map.get(start + i, {}) for i in range(len(all_images))]

    # Map keep indices (in instance_results) to image indices
    selected_img_indices = []
    selected_results = []
    for k in keep:
        r = instance_results[k]
        img_idx = r['index'] - start
        if 0 <= img_idx < len(all_images):
            selected_img_indices.append(img_idx)
            selected_results.append(r)

    images = [all_images[i] for i in selected_img_indices]
    labels_sel = [all_labels[i] for i in selected_img_indices]

    if not images:
        print('No matching images found.')
        sys.exit(0)

    # Load model
    print(f'Loading model from {args.model_path}...')
    device = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
    raw_model = load_model(args.model_path, device,
                           model_name=args.model_name)

    # Decide whether to wrap with normalization
    # If --no-normalize is set, pass images raw (model has built-in norm)
    if args.normalize:
        model = NormalizedModel(raw_model, mean, std).to(device)
    else:
        model = raw_model

    # Get clean predictions
    print('Getting clean predictions...')
    clean_preds = []
    with torch.no_grad():
        for img in images:
            output = model(img.unsqueeze(0).to(device))
            clean_preds.append(output.argmax(dim=1).item())

    # Run PGD attack
    epsilon = args.epsilon
    print(f'Running PGD attack (ε={epsilon:.5f}, steps={args.steps}, '
          f'step_size={args.step_size or epsilon / 4:.5f}) on {len(images)} images...')

    adv_images, adv_preds, adv_confs, succeeded = pgd_attack(
        model, images, labels_sel,
        epsilon=epsilon,
        step_size=args.step_size,
        num_steps=args.steps,
        device=device,
    )

    n_success = sum(succeeded)
    print(f'PGD attack succeeded on {n_success}/{len(images)} images.')

    # Generate comparison figure
    output_path = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = '_falsified' if args.falsified_only else ''
        output_path = str(output_dir / f'pgd_comparison{suffix}.png')

    title = f'{args.dataset} PGD Comparison  (ε={epsilon:.5f}, {args.steps} steps)'
    if args.falsified_only:
        title += '  [falsified only]'

    plot_pgd_comparison(
        images, adv_images, labels_sel, classes, mean, std,
        predictions=clean_preds,
        adv_preds=adv_preds,
        adv_confs=adv_confs,
        succeeded=succeeded,
        instance_results=selected_results,
        epsilon=epsilon,
        output_path=output_path,
        title=title,
    )


def cmd_convergence(args):
    """Subcommand: plot convergence curves."""
    if not HAS_MPL:
        print('ERROR: matplotlib is required. Install: pip install matplotlib')
        sys.exit(1)

    output_path = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'convergence.png')

    metrics_paths = args.metrics
    labels = args.labels if args.labels else None

    plot_convergence(metrics_paths, labels=labels, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Visualization tools for alpha-beta-CROWN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Visualization type')

    # results subcommand
    p_results = subparsers.add_parser(
        'results', help='Visualize verification results on images',
    )
    p_results.add_argument('--metrics', required=True,
                           help='Path to metrics.json from benchmark run')
    p_results.add_argument('--dataset', required=True,
                           help='Dataset name: CIFAR or MNIST')
    p_results.add_argument('--model-path', default=None,
                           help='Path to model .pth for classification overlay')
    p_results.add_argument('--output', default=None,
                           help='Directory to save figures')
    p_results.add_argument('--max-images', type=int, default=50,
                           help='Maximum images per grid (default: 50)')
    p_results.set_defaults(func=cmd_results)

    # relaxation subcommand
    p_relax = subparsers.add_parser(
        'relaxation', help='Draw ReLU relaxation diagram',
    )
    p_relax.add_argument('--output', default=None,
                         help='Directory to save figures')
    p_relax.set_defaults(func=cmd_relaxation)

    # pgd subcommand
    p_pgd = subparsers.add_parser(
        'pgd', help='Visualize PGD adversarial perturbations',
    )
    p_pgd.add_argument('--metrics', required=True,
                       help='Path to metrics.json from benchmark run')
    p_pgd.add_argument('--dataset', required=True,
                       help='Dataset name: CIFAR or MNIST')
    p_pgd.add_argument('--model-path', required=True,
                       help='Path to model .pth file (required for PGD)')
    p_pgd.add_argument('--model-name', default=None,
                       help='Model architecture function name in model_defs.py '
                            '(default: inferred from .pth filename, e.g. '
                            'resnet2b.pth -> resnet2b)')
    p_pgd.add_argument('--epsilon', type=float, default=2.0 / 255,
                       help='Perturbation budget in [0,1] scale '
                            '(default: 2/255 ≈ 0.00784)')
    p_pgd.add_argument('--steps', type=int, default=50,
                       help='Number of PGD iterations (default: 50)')
    p_pgd.add_argument('--step-size', type=float, default=None,
                       help='PGD step size (default: epsilon/4)')
    p_pgd.add_argument('--falsified-only', action='store_true',
                       help='Only show images that were falsified (unsafe)')
    p_pgd.add_argument('--normalize', action='store_true',
                       help='Wrap model with normalization layer '
                            '(use if model expects raw [0,1] to be normalized)')
    p_pgd.add_argument('--output', default=None,
                       help='Directory to save figures')
    p_pgd.add_argument('--max-images', type=int, default=20,
                       help='Maximum images to show (default: 20)')
    p_pgd.set_defaults(func=cmd_pgd)

    # convergence subcommand
    p_conv = subparsers.add_parser(
        'convergence', help='Plot convergence curves',
    )
    p_conv.add_argument('--metrics', nargs='+', required=True,
                        help='One or more metrics.json paths to compare')
    p_conv.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each metrics file')
    p_conv.add_argument('--output', default=None,
                        help='Directory to save figures')
    p_conv.set_defaults(func=cmd_convergence)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
