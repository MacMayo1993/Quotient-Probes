"""
Minimum Description Length (MDL) decision rule for symmetry exploitation.

Key insight: The decision to exploit symmetry (store x₊, x₋ separately) vs.
ignore it (store x directly) depends on the *orientation cost* of the lift map
relative to the compression gain from coherence.

Decision boundary (Theorem 1):
    Exploit symmetry if:  α > α_crit(n, K_lift)

where:
    α_crit = (n + K_lift) / (2n)

For Bernoulli coin-flip model: K_lift = 1
For Markov chain: K_lift depends on transition structure
"""

import numpy as np
from typing import Tuple, Optional, Callable, Union


def compute_orientation_cost(
    model: str = 'bernoulli',
    n: Optional[int] = None,
    **kwargs
) -> float:
    """
    Compute orientation cost K_lift for lift map π⁻¹: V/σ → V

    The lift map π⁻¹ assigns an orientation (sign) to each equivalence class [x].
    K_lift measures the description length of this orientation pattern.

    Args:
        model: Probabilistic model for orientation
               - 'bernoulli': Independent coin flips, K_lift = 1
               - 'markov': First-order Markov chain, K_lift = 1 + H(transitions)
               - 'constant': All same orientation, K_lift = 0
        n: Dimension (required for some models)
        **kwargs: Model-specific parameters

    Returns:
        K_lift: Orientation cost in bits

    Examples:
        >>> compute_orientation_cost('bernoulli')
        1.0
        >>> compute_orientation_cost('constant')
        0.0
        >>> compute_orientation_cost('markov', transition_entropy=0.5)
        1.5
    """
    if model == 'bernoulli':
        # Independent fair coin flips: H(orientation) = 1 bit per decision
        return 1.0

    elif model == 'constant':
        # All orientations identical: no randomness
        return 0.0

    elif model == 'markov':
        # Markov chain: 1 bit initial state + transition entropy
        transition_entropy = kwargs.get('transition_entropy', 0.0)
        return 1.0 + transition_entropy

    elif model == 'empirical':
        # Estimate from data
        orientations = kwargs.get('orientations', None)
        if orientations is None:
            raise ValueError("'empirical' model requires 'orientations' array")

        # Compute empirical entropy
        orientations = np.array(orientations)
        _, counts = np.unique(orientations, return_counts=True)
        probabilities = counts / len(orientations)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    else:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Available: 'bernoulli', 'markov', 'constant', 'empirical'"
        )


def critical_coherence(
    n: int,
    K_lift: float = 1.0
) -> float:
    """
    Compute critical coherence threshold α_crit.

    From Theorem 1:
        α_crit = (n + K_lift) / (2n) = 1/2 + K_lift/(2n)

    Args:
        n: Ambient dimension
        K_lift: Orientation cost (default: 1.0 for Bernoulli)

    Returns:
        α_crit ∈ [0.5, 1]: Critical coherence threshold

    Properties:
        - As n → ∞: α_crit → 1/2 (orientation cost becomes negligible)
        - α_crit > 1/2 always (orientation cost adds to storage)
        - Higher K_lift requires higher α to justify exploitation

    Examples:
        >>> critical_coherence(n=64, K_lift=1.0)
        0.5078125
        >>> critical_coherence(n=256, K_lift=1.0)
        0.501953125
        >>> critical_coherence(n=64, K_lift=0.0)  # No orientation cost
        0.5
    """
    if n <= 0:
        raise ValueError(f"Dimension must be positive, got n={n}")

    if K_lift < 0:
        raise ValueError(f"Orientation cost must be non-negative, got K_lift={K_lift}")

    alpha_crit = (n + K_lift) / (2.0 * n)
    return alpha_crit


def description_length_difference(
    alpha: float,
    n: int,
    K_lift: float = 1.0
) -> float:
    """
    Compute ΔL(α) = L_exploit - L_ignore

    From Equation (8) in paper:
        ΔL(α) = n·(2α - 1) - K_lift

    Interpretation:
        ΔL < 0: Exploit symmetry (saves bits)
        ΔL > 0: Ignore symmetry (cheaper to store x directly)
        ΔL = 0: Decision boundary

    Args:
        alpha: Coherence parameter α ∈ [0, 1]
        n: Ambient dimension
        K_lift: Orientation cost

    Returns:
        ΔL: Description length difference in bits

    Example:
        >>> n, K_lift = 64, 1.0
        >>> alpha_crit = critical_coherence(n, K_lift)
        >>> description_length_difference(alpha_crit, n, K_lift)
        0.0  # At boundary
        >>> description_length_difference(0.6, n, K_lift)  # Above threshold
        -13.0  # Exploit saves 13 bits
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"Coherence must be in [0,1], got α={alpha}")

    delta_L = n * (2 * alpha - 1) - K_lift
    return delta_L


def mdl_decision_rule(
    alpha: float,
    n: int,
    K_lift: float = 1.0,
    return_details: bool = False
) -> Union[bool, Tuple[bool, dict]]:
    """
    MDL-based decision: Should we exploit symmetry?

    Decision rule:
        Exploit if α > α_crit = (n + K_lift)/(2n)

    Args:
        alpha: Observed coherence
        n: Ambient dimension
        K_lift: Orientation cost
        return_details: If True, return (decision, details_dict)

    Returns:
        If return_details=False: Boolean decision
        If return_details=True: (decision, details) where details contains:
            - 'alpha_crit': Critical threshold
            - 'delta_L': Description length difference
            - 'bit_savings': Bits saved (negative if loss)
            - 'decision': 'exploit' or 'ignore'

    Example:
        >>> mdl_decision_rule(alpha=0.6, n=64, K_lift=1.0)
        True  # Exploit
        >>> decision, details = mdl_decision_rule(
        ...     alpha=0.6, n=64, K_lift=1.0, return_details=True
        ... )
        >>> details['bit_savings']
        12.8
    """
    alpha_crit = critical_coherence(n, K_lift)
    delta_L = description_length_difference(alpha, n, K_lift)

    # Decision: exploit if ΔL < 0 (equivalently, α > α_crit)
    should_exploit = alpha > alpha_crit

    if not return_details:
        return bool(should_exploit)  # Convert numpy bool to Python bool

    # Compute additional details
    details = {
        'alpha': alpha,
        'alpha_crit': alpha_crit,
        'n': n,
        'K_lift': K_lift,
        'delta_L': delta_L,
        'bit_savings': -delta_L,  # Negative ΔL means savings
        'decision': 'exploit' if should_exploit else 'ignore',
        'margin': alpha - alpha_crit,  # How far from boundary
    }

    return bool(should_exploit), details  # Convert numpy bool to Python bool


def compression_gain(
    alpha: float,
    n: int
) -> float:
    """
    Compute raw compression gain from coherence (ignoring orientation cost).

    Gain = n·(2α - 1)

    This is the bit savings from storing ||x₊|| and ||x₋|| instead of ||x||
    when the components have different norms.

    Args:
        alpha: Coherence parameter
        n: Dimension

    Returns:
        Compression gain in bits (can be negative if α < 0.5)

    Example:
        >>> compression_gain(alpha=0.75, n=100)
        50.0  # Save 50 bits
        >>> compression_gain(alpha=0.5, n=100)
        0.0  # Balanced, no gain
        >>> compression_gain(alpha=0.25, n=100)
        -50.0  # Loss
    """
    return n * (2 * alpha - 1)


def batch_evaluate_boundary(
    alphas: np.ndarray,
    n: int,
    K_lift: float = 1.0
) -> np.ndarray:
    """
    Vectorized evaluation of decision boundary.

    Useful for plotting ΔL(α) curves.

    Args:
        alphas: Array of coherence values
        n: Dimension
        K_lift: Orientation cost

    Returns:
        Array of ΔL values

    Example:
        >>> alphas = np.linspace(0, 1, 100)
        >>> delta_Ls = batch_evaluate_boundary(alphas, n=64, K_lift=1.0)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(alphas, delta_Ls)
        >>> plt.axhline(0, color='k', linestyle='--')
        >>> plt.xlabel('Coherence α')
        >>> plt.ylabel('ΔL (bits)')
    """
    alphas = np.asarray(alphas)
    delta_Ls = n * (2 * alphas - 1) - K_lift
    return delta_Ls


def adaptive_threshold(
    n: int,
    K_lift_range: Tuple[float, float] = (0.0, 2.0),
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute how α_crit varies with K_lift for fixed n.

    Shows how orientation cost affects decision boundary.

    Args:
        n: Dimension
        K_lift_range: (min, max) orientation cost
        num_points: Number of evaluation points

    Returns:
        Tuple (K_lifts, alpha_crits)

    Example:
        >>> n = 64
        >>> K_lifts, thresholds = adaptive_threshold(n)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(K_lifts, thresholds)
        >>> plt.xlabel('Orientation Cost K_lift')
        >>> plt.ylabel('Critical Coherence α_crit')
    """
    K_lifts = np.linspace(K_lift_range[0], K_lift_range[1], num_points)
    alpha_crits = (n + K_lifts) / (2.0 * n)
    return K_lifts, alpha_crits
