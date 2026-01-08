"""
Visualizations for symmetry decomposition and eigenspace analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable

from ..core.decomposition import decompose, compute_coherence
from ..core.involutions import antipodal


def plot_decomposition(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray],
    figsize: Tuple[int, int] = (12, 4),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize x = x₊ + x₋ decomposition.

    Shows original vector and its symmetric/antisymmetric components.

    Args:
        x: Input vector
        sigma: Involution operator
        figsize: Figure size
        title: Custom title (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> from ..core.involutions import reverse
        >>> x = np.array([1, 2, 3, 2, 1])  # Palindrome
        >>> fig = plot_decomposition(x, reverse)
    """
    x_plus, x_minus = decompose(x, sigma)
    alpha = compute_coherence(x, sigma)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    indices = np.arange(len(x))

    # Original
    ax1.stem(indices, x, basefmt=' ')
    ax1.set_title('Original x', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.8)

    # Symmetric component
    ax2.stem(indices, x_plus, basefmt=' ', linefmt='g-', markerfmt='go')
    ax2.set_title(f'Symmetric x₊ (V₊)\n||x₊||²/||x||² = {alpha:.3f}',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.8)

    # Antisymmetric component
    ax3.stem(indices, x_minus, basefmt=' ', linefmt='r-', markerfmt='ro')
    ax3.set_title(f'Antisymmetric x₋ (V₋)\n||x₋||²/||x||² = {1-alpha:.3f}',
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', linewidth=0.8)

    if title is None:
        title = f'Eigenspace Decomposition: x = x₊ + x₋'

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_eigenspace_energy(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray],
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Visualize energy distribution between V₊ and V₋.

    Shows pie chart and bar chart of eigenspace energies.

    Args:
        x: Input vector
        sigma: Involution operator
        figsize: Figure size

    Returns:
        matplotlib Figure

    Example:
        >>> x = np.random.randn(100)
        >>> from ..core.involutions import antipodal
        >>> fig = plot_eigenspace_energy(x, antipodal)
    """
    x_plus, x_minus = decompose(x, sigma)

    energy_plus = np.linalg.norm(x_plus) ** 2
    energy_minus = np.linalg.norm(x_minus) ** 2
    total_energy = energy_plus + energy_minus

    alpha = energy_plus / total_energy if total_energy > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Pie chart
    colors = ['#2ecc71', '#e74c3c']  # Green, Red
    ax1.pie(
        [energy_plus, energy_minus],
        labels=['V₊ (symmetric)', 'V₋ (antisymmetric)'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    ax1.set_title(f'Energy Distribution\nCoherence α = {alpha:.3f}',
                  fontsize=12, fontweight='bold')

    # Bar chart
    components = ['V₊', 'V₋', 'Total']
    energies = [energy_plus, energy_minus, total_energy]
    bars = ax2.bar(components, energies, color=['green', 'red', 'blue'], alpha=0.7)

    ax2.set_ylabel('Energy (||·||²)', fontsize=11)
    ax2.set_title('Eigenspace Energies', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=10
        )

    plt.suptitle(
        f'Eigenspace Energy Analysis\n'
        f'||x₊||² = {energy_plus:.2f}, ||x₋||² = {energy_minus:.2f}',
        fontsize=13,
        fontweight='bold',
        y=1.00
    )
    plt.tight_layout()

    return fig


def plot_coherence_histogram(
    data_samples: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray],
    n_bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    show_critical: bool = True,
    K_lift: float = 1.0,
) -> plt.Figure:
    """
    Plot histogram of coherence values across multiple samples.

    Useful for understanding coherence distribution in datasets.

    Args:
        data_samples: Array of shape (num_samples, n)
        sigma: Involution operator
        n_bins: Number of histogram bins
        figsize: Figure size
        title: Custom title
        show_critical: Whether to show α_crit line
        K_lift: Orientation cost for critical threshold

    Returns:
        matplotlib Figure

    Example:
        >>> samples = np.random.randn(1000, 64)
        >>> from ..core.involutions import antipodal
        >>> fig = plot_coherence_histogram(samples, antipodal)
    """
    # Compute coherence for all samples
    alphas = [compute_coherence(x, sigma) for x in data_samples]
    alphas = np.array(alphas)

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    counts, bins, patches = ax.hist(
        alphas, bins=n_bins, density=True,
        alpha=0.7, color='skyblue', edgecolor='black'
    )

    # Statistics
    mean_alpha = np.mean(alphas)
    median_alpha = np.median(alphas)
    std_alpha = np.std(alphas)

    # Mark statistics
    ax.axvline(mean_alpha, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_alpha:.3f}')
    ax.axvline(median_alpha, color='green', linestyle='--', linewidth=2,
               label=f'Median = {median_alpha:.3f}')

    # Show critical threshold if requested
    if show_critical and data_samples.shape[1] > 0:
        n = data_samples.shape[1]
        alpha_crit = (n + K_lift) / (2 * n)

        ax.axvline(alpha_crit, color='purple', linestyle=':', linewidth=2.5,
                   label=f'α_crit = {alpha_crit:.3f}')

        # Shade decision regions
        ax.axvspan(0, alpha_crit, alpha=0.1, color='red', label='Ignore region')
        ax.axvspan(alpha_crit, 1.0, alpha=0.1, color='green', label='Exploit region')

        # Count samples in each region
        num_exploit = np.sum(alphas > alpha_crit)
        num_ignore = np.sum(alphas <= alpha_crit)
        pct_exploit = 100 * num_exploit / len(alphas)

        textstr = (
            f'Samples above α_crit: {num_exploit} ({pct_exploit:.1f}%)\n'
            f'Samples below α_crit: {num_ignore} ({100-pct_exploit:.1f}%)'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(
            0.98, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=props
        )

    ax.set_xlabel('Coherence α', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

    if title is None:
        title = (
            f'Coherence Distribution ({len(alphas)} samples)\n'
            f'μ = {mean_alpha:.3f}, σ = {std_alpha:.3f}'
        )

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_symmetry_test(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray],
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Visualize how well x satisfies σ(x) ≈ ±x.

    Plots x vs σ(x) to show symmetry relationship.

    Args:
        x: Input vector
        sigma: Involution operator
        figsize: Figure size

    Returns:
        matplotlib Figure

    Example:
        >>> x = np.array([1, 2, 3, 2, 1])
        >>> from ..core.involutions import reverse
        >>> fig = plot_symmetry_test(x, reverse)  # Should show perfect symmetry
    """
    sigma_x = sigma(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    indices = np.arange(len(x))

    # Overlay x and σ(x)
    ax1.plot(indices, x, 'b-o', label='x', linewidth=2, markersize=6)
    ax1.plot(indices, sigma_x, 'r--s', label='σ(x)', linewidth=2, markersize=6)
    ax1.plot(indices, -sigma_x, 'g:^', label='-σ(x)', linewidth=2, markersize=6, alpha=0.7)

    ax1.set_xlabel('Index', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Involution Action', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.8)

    # Scatter: x vs σ(x)
    ax2.scatter(x, sigma_x, alpha=0.6, s=50, c='blue', label='(x_i, σ(x)_i)')

    # Reference lines
    x_range = np.array([x.min(), x.max()])
    ax2.plot(x_range, x_range, 'g--', linewidth=2, alpha=0.7, label='y = x (symmetric)')
    ax2.plot(x_range, -x_range, 'r--', linewidth=2, alpha=0.7, label='y = -x (antisymmetric)')

    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('σ(x)', fontsize=11)
    ax2.set_title('Symmetry Relationship', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Compute distances
    dist_symmetric = np.linalg.norm(x - sigma_x)
    dist_antisymmetric = np.linalg.norm(x + sigma_x)

    plt.suptitle(
        f'Symmetry Test\n'
        f'||x - σ(x)|| = {dist_symmetric:.3f}, ||x + σ(x)|| = {dist_antisymmetric:.3f}',
        fontsize=13,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()

    return fig
