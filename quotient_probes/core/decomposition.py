"""
Symmetry decomposition into eigenspaces V₊ ⊕ V₋.

For an involution σ: V → V with σ² = I, any vector x decomposes as:
    x = x₊ + x₋

where:
    x₊ = (x + σ(x))/2  ∈ V₊  (symmetric component, eigenvalue +1)
    x₋ = (x - σ(x))/2  ∈ V₋  (antisymmetric component, eigenvalue -1)

The coherence parameter α = ||x₊||²/||x||² measures symmetry strength.
"""

import numpy as np
from typing import Tuple, Callable, Optional


def decompose_symmetric(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Extract symmetric component: x₊ = (x + σ(x))/2

    Args:
        x: Input vector or batch (shape: (..., n))
        sigma: Involution operator

    Returns:
        Symmetric component x₊ ∈ V₊

    Example:
        >>> from .involutions import antipodal
        >>> x = np.array([3, 1])  # Arbitrary vector
        >>> x_plus = decompose_symmetric(x, antipodal)
        >>> x_plus
        array([0., 0.])  # Antipodal has no symmetric component for generic x
    """
    return 0.5 * (x + sigma(x))


def decompose_antisymmetric(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Extract antisymmetric component: x₋ = (x - σ(x))/2

    Args:
        x: Input vector or batch (shape: (..., n))
        sigma: Involution operator

    Returns:
        Antisymmetric component x₋ ∈ V₋

    Example:
        >>> from .involutions import antipodal
        >>> x = np.array([3, 1])
        >>> x_minus = decompose_antisymmetric(x, antipodal)
        >>> x_minus
        array([3., 1.])  # For antipodal, x₋ = x for generic x
    """
    return 0.5 * (x - sigma(x))


def decompose(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full decomposition: x = x₊ + x₋

    Args:
        x: Input vector or batch (shape: (..., n))
        sigma: Involution operator

    Returns:
        Tuple (x₊, x₋) of symmetric and antisymmetric components

    Example:
        >>> from .involutions import reverse
        >>> x = np.array([1, 2, 3, 2, 1])  # Palindromic
        >>> x_plus, x_minus = decompose(x, reverse)
        >>> x_plus  # Should equal x (fully symmetric)
        array([1., 2., 3., 2., 1.])
        >>> np.allclose(x_minus, 0)  # No antisymmetric component
        True
    """
    sigma_x = sigma(x)
    x_plus = 0.5 * (x + sigma_x)
    x_minus = 0.5 * (x - sigma_x)
    return x_plus, x_minus


def compute_coherence(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray],
    epsilon: float = 1e-10
) -> float:
    """
    Compute coherence parameter: α = ||x₊||²/||x||²

    The coherence α ∈ [0, 1] measures the fraction of energy in the
    symmetric subspace V₊.

    Args:
        x: Input vector (shape: (n,))
        sigma: Involution operator
        epsilon: Small constant to avoid division by zero

    Returns:
        Coherence α ∈ [0, 1]

    Properties:
        - α = 1: Fully symmetric (x = x₊, x₋ = 0)
        - α = 0: Fully antisymmetric (x = x₋, x₊ = 0)
        - α = 0.5: Balanced (||x₊|| = ||x₋||)

    Example:
        >>> from .involutions import antipodal
        >>> x = np.array([1, 2, 3])
        >>> compute_coherence(x, antipodal)
        0.0  # Generic vector is purely antisymmetric under antipodal
    """
    x_plus = decompose_symmetric(x, sigma)

    norm_x = np.linalg.norm(x)
    norm_x_plus = np.linalg.norm(x_plus)

    if norm_x < epsilon:
        return 0.0  # Degenerate case

    alpha = (norm_x_plus ** 2) / (norm_x ** 2)
    return np.clip(alpha, 0.0, 1.0)  # Numerical safety


def compute_eigenspace_dimensions(
    sigma: Callable[[np.ndarray], np.ndarray],
    n: int,
    num_samples: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[int, int]:
    """
    Estimate dimensions of V₊ and V₋ via random sampling.

    For structured involutions, the eigenspace dimensions may not be n/2 each.

    Args:
        sigma: Involution operator
        n: Ambient dimension
        num_samples: Number of random vectors to test
        tolerance: Threshold for considering component as zero

    Returns:
        Tuple (dim_V_plus, dim_V_minus)

    Note:
        This is an empirical estimate. For exact dimensions, use algebraic methods.

    Example:
        >>> from .involutions import antipodal
        >>> dim_plus, dim_minus = compute_eigenspace_dimensions(antipodal, 100)
        >>> dim_plus, dim_minus
        (0, 100)  # Antipodal has empty V₊ for generic vectors
    """
    # Generate random vectors
    rng = np.random.RandomState(42)
    samples = rng.randn(num_samples, n)

    # Compute average ranks
    ranks_plus = []
    ranks_minus = []

    for x in samples:
        x_plus = decompose_symmetric(x, sigma)
        x_minus = decompose_antisymmetric(x, sigma)

        # Check if components are non-zero
        if np.linalg.norm(x_plus) > tolerance:
            ranks_plus.append(1)
        if np.linalg.norm(x_minus) > tolerance:
            ranks_minus.append(1)

    # Estimate as average presence
    dim_plus = int(np.mean(ranks_plus) * n) if ranks_plus else 0
    dim_minus = int(np.mean(ranks_minus) * n) if ranks_minus else 0

    return dim_plus, dim_minus


def verify_decomposition(
    x: np.ndarray,
    x_plus: np.ndarray,
    x_minus: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """
    Verify that x = x₊ + x₋ and components are orthogonal.

    Args:
        x: Original vector
        x_plus: Symmetric component
        x_minus: Antisymmetric component
        tolerance: Numerical tolerance

    Returns:
        True if decomposition is valid

    Example:
        >>> from .involutions import antipodal
        >>> x = np.array([1, 2, 3])
        >>> x_plus, x_minus = decompose(x, antipodal)
        >>> verify_decomposition(x, x_plus, x_minus)
        True
    """
    # Check x = x₊ + x₋
    reconstruction = x_plus + x_minus
    if not np.allclose(x, reconstruction, atol=tolerance):
        return False

    # Check orthogonality: ⟨x₊, x₋⟩ = 0
    inner_product = np.dot(x_plus, x_minus)
    if not np.abs(inner_product) < tolerance:
        return False

    return True
