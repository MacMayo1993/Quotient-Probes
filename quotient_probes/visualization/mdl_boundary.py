"""
Interactive visualizations of MDL decision boundary.

Key plots:
1. ΔL(α) vs coherence for varying dimensions
2. α_crit vs dimension showing asymptotic behavior
3. Interactive sliders for exploring parameter space
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from ..core.mdl_decision import (
    batch_evaluate_boundary,
    critical_coherence,
    description_length_difference,
)


def plot_decision_boundary(
    n_values: List[int] = [64, 128, 256, 512],
    K_lift: float = 1.0,
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    num_points: int = 200,
    figsize: Tuple[int, int] = (10, 6),
    show_examples: bool = True,
) -> plt.Figure:
    """
    Plot ΔL(α) decision boundary for multiple dimensions.

    Shows how the critical threshold α_crit shifts with dimension n.

    Args:
        n_values: List of dimensions to plot
        K_lift: Orientation cost
        alpha_range: Range of coherence values
        num_points: Resolution of curves
        figsize: Figure size
        show_examples: Mark the worked examples from paper (n=64, n=256)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_decision_boundary([64, 256], K_lift=1.0)
        >>> plt.savefig('decision_boundary.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)

    # Plot boundary for each dimension
    for n in n_values:
        delta_Ls = batch_evaluate_boundary(alphas, n, K_lift)
        alpha_crit = critical_coherence(n, K_lift)

        ax.plot(alphas, delta_Ls, label=f"n={n} (α_crit={alpha_crit:.4f})", linewidth=2)

        # Mark critical point
        ax.plot(alpha_crit, 0, "o", markersize=8, color=ax.lines[-1].get_color())

    # Zero line (decision boundary)
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Decision boundary",
    )

    # Shade regions
    ax.axvspan(0, 0.5, alpha=0.1, color="red", label="Ignore (α < 0.5)")
    ax.axvspan(
        0.5, 1.0, alpha=0.1, color="green", label="Potentially exploit (α > 0.5)"
    )

    # Annotations
    if show_examples and 64 in n_values:
        alpha_64 = critical_coherence(64, K_lift)
        ax.annotate(
            f"n=64: α_crit={alpha_64:.4f}",
            xy=(alpha_64, 0),
            xytext=(alpha_64 + 0.1, -20),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
            fontsize=10,
            color="blue",
        )

    if show_examples and 256 in n_values:
        alpha_256 = critical_coherence(256, K_lift)
        ax.annotate(
            f"n=256: α_crit={alpha_256:.4f}",
            xy=(alpha_256, 0),
            xytext=(alpha_256 + 0.1, 20),
            arrowprops=dict(arrowstyle="->", color="purple", lw=1.5),
            fontsize=10,
            color="purple",
        )

    ax.set_xlabel("Coherence α", fontsize=12)
    ax.set_ylabel("ΔL (bits)", fontsize=12)
    ax.set_title(
        f"MDL Decision Boundary (K_lift = {K_lift})\n" f"ΔL(α) = n(2α - 1) - K_lift",
        fontsize=14,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box explaining decision rule
    textstr = (
        "Decision Rule:\n"
        "  ΔL < 0: Exploit symmetry\n"
        "  ΔL > 0: Ignore symmetry\n"
        f"  α_crit = (n + {K_lift})/(2n)"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    return fig


def plot_interactive_boundary(
    n_init: int = 128,
    K_lift_init: float = 1.0,
    n_range: Tuple[int, int] = (10, 1000),
    K_lift_range: Tuple[float, float] = (0.0, 5.0),
) -> Tuple[plt.Figure, dict]:
    """
    Create interactive plot with sliders for n and K_lift.

    Allows real-time exploration of how dimension and orientation cost
    affect the decision boundary.

    Args:
        n_init: Initial dimension
        K_lift_init: Initial orientation cost
        n_range: Range for dimension slider
        K_lift_range: Range for K_lift slider

    Returns:
        Tuple of (Figure, dict of widgets)

    Example:
        >>> fig, widgets = plot_interactive_boundary()
        >>> plt.show()  # Interact with sliders
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    # Initial plot
    alphas = np.linspace(0, 1, 300)

    def update_plot(n, K_lift):
        ax.clear()

        delta_Ls = batch_evaluate_boundary(alphas, n, K_lift)
        alpha_crit = critical_coherence(n, K_lift)

        # Plot curve
        ax.plot(alphas, delta_Ls, "b-", linewidth=2, label="ΔL(α)")

        # Decision boundary
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

        # Critical point
        ax.plot(alpha_crit, 0, "ro", markersize=10, label=f"α_crit = {alpha_crit:.4f}")

        # Shade regions
        ax.axvspan(0, alpha_crit, alpha=0.15, color="red", label="Ignore")
        ax.axvspan(alpha_crit, 1.0, alpha=0.15, color="green", label="Exploit")

        ax.set_xlabel("Coherence α", fontsize=12)
        ax.set_ylabel("ΔL (bits)", fontsize=12)
        ax.set_title(
            f"MDL Decision Boundary\n"
            f"n={n}, K_lift={K_lift:.2f}, α_crit={alpha_crit:.4f}",
            fontsize=13,
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        # Auto-scale y to show interesting region
        y_max = max(50, n * 0.5)
        ax.set_ylim(-y_max, y_max)

        fig.canvas.draw_idle()

    # Create sliders
    ax_n = plt.axes((0.15, 0.10, 0.7, 0.03))
    ax_K = plt.axes((0.15, 0.05, 0.7, 0.03))

    slider_n = Slider(
        ax_n, "Dimension n", n_range[0], n_range[1], valinit=n_init, valstep=1
    )

    slider_K = Slider(
        ax_K,
        "Orientation Cost K_lift",
        K_lift_range[0],
        K_lift_range[1],
        valinit=K_lift_init,
        valstep=0.1,
    )

    def on_update(val):
        n = int(slider_n.val)
        K_lift = slider_K.val
        update_plot(n, K_lift)

    slider_n.on_changed(on_update)
    slider_K.on_changed(on_update)

    # Initial plot
    update_plot(n_init, K_lift_init)

    widgets = {
        "slider_n": slider_n,
        "slider_K": slider_K,
    }

    return fig, widgets


def plot_dimension_sweep(
    n_max: int = 1000,
    K_lift_values: List[float] = [0.0, 0.5, 1.0, 2.0, 5.0],
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot how α_crit varies with dimension for different K_lift values.

    Shows asymptotic behavior: α_crit → 1/2 as n → ∞

    Args:
        n_max: Maximum dimension
        K_lift_values: List of orientation costs to plot
        figsize: Figure size

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_dimension_sweep(n_max=500, K_lift_values=[0, 1, 2])
        >>> plt.savefig('dimension_sweep.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_values = np.logspace(1, np.log10(n_max), 200).astype(int)
    n_values = np.unique(n_values)  # Remove duplicates

    for K_lift in K_lift_values:
        alpha_crits = [(n + K_lift) / (2 * n) for n in n_values]

        label = f"K_lift = {K_lift}"
        if K_lift == 0:
            label += " (no orientation cost)"
        elif K_lift == 1.0:
            label += " (Bernoulli)"

        ax.plot(n_values, alpha_crits, linewidth=2, label=label)

    # Asymptotic limit
    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Asymptotic limit (n→∞)",
    )

    ax.set_xlabel("Dimension n", fontsize=12)
    ax.set_ylabel("Critical Coherence α_crit", fontsize=12)
    ax.set_title(
        "How Dimension Affects Decision Threshold\n"
        "α_crit = (n + K_lift)/(2n) → 1/2 as n → ∞",
        fontsize=14,
    )
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0.49, max(alpha_crits) + 0.05)

    # Annotate key dimensions
    for n_mark in [64, 256, 1024]:
        if n_mark <= n_max:
            ax.axvline(n_mark, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax.text(
                n_mark,
                0.495,
                f"n={n_mark}",
                rotation=90,
                verticalalignment="bottom",
                fontsize=8,
                alpha=0.7,
            )

    plt.tight_layout()
    return fig


def plot_worked_examples(
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Recreate the worked examples from Section 3 of the paper.

    Shows decision regions for n=64 and n=256 with K_lift=1.

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    alphas = np.linspace(0, 1, 300)

    # Example 1: n=64
    n1 = 64
    K_lift = 1.0
    delta_L1 = batch_evaluate_boundary(alphas, n1, K_lift)
    alpha_crit1 = critical_coherence(n1, K_lift)

    ax1.plot(alphas, delta_L1, "b-", linewidth=2.5)
    ax1.axhline(0, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.plot(alpha_crit1, 0, "ro", markersize=12, label=f"α_crit = {alpha_crit1:.4f}")
    ax1.axvspan(0, alpha_crit1, alpha=0.2, color="red", label="Ignore")
    ax1.axvspan(alpha_crit1, 1.0, alpha=0.2, color="green", label="Exploit")

    ax1.set_xlabel("Coherence α", fontsize=11)
    ax1.set_ylabel("ΔL (bits)", fontsize=11)
    ax1.set_title(f"Example 1: n={n1}, K_lift={K_lift}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Example 2: n=256
    n2 = 256
    delta_L2 = batch_evaluate_boundary(alphas, n2, K_lift)
    alpha_crit2 = critical_coherence(n2, K_lift)

    ax2.plot(alphas, delta_L2, "purple", linewidth=2.5)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.plot(alpha_crit2, 0, "ro", markersize=12, label=f"α_crit = {alpha_crit2:.4f}")
    ax2.axvspan(0, alpha_crit2, alpha=0.2, color="red", label="Ignore")
    ax2.axvspan(alpha_crit2, 1.0, alpha=0.2, color="green", label="Exploit")

    ax2.set_xlabel("Coherence α", fontsize=11)
    ax2.set_ylabel("ΔL (bits)", fontsize=11)
    ax2.set_title(f"Example 2: n={n2}, K_lift={K_lift}", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Worked Examples from Paper\n" "ΔL(α) = n(2α - 1) - K_lift",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    return fig
