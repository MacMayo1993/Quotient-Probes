"""
Involution operators for symmetry analysis.

An involution σ is a self-inverse linear operator: σ² = I
Common examples include antipodal reflection, time reversal, and spatial reflection.
"""

import numpy as np
from typing import Callable


def antipodal(x: np.ndarray) -> np.ndarray:
    """
    Antipodal involution: σ(x) = -x

    Most fundamental involution, maps vectors to their negation.
    Eigenspaces: V₊ (even subspace) and V₋ (odd subspace)

    Args:
        x: Input vector or batch of vectors (shape: (..., n))

    Returns:
        Reflected vector(s)

    Example:
        >>> x = np.array([1, 2, 3])
        >>> antipodal(x)
        array([-1, -2, -3])
    """
    return -x


def reverse(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Time-reversal involution: σ(x) = x[::-1]

    Reverses the order of elements along specified axis.
    Useful for time series and sequential data.

    Args:
        x: Input array
        axis: Axis along which to reverse (default: last axis)

    Returns:
        Reversed array

    Example:
        >>> x = np.array([1, 2, 3, 4])
        >>> reverse(x)
        array([4, 3, 2, 1])
    """
    return np.flip(x, axis=axis)


def reflection(x: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Hyperplane reflection: σ(x) = x - 2⟨x,n⟩n

    Reflects vector across hyperplane orthogonal to normal vector.

    Args:
        x: Input vector or batch of vectors (shape: (..., n))
        normal: Normal vector defining reflection plane (shape: (n,))
                Must be unit norm: ||n|| = 1

    Returns:
        Reflected vector(s)

    Example:
        >>> x = np.array([1, 0])
        >>> normal = np.array([1, 0])  # Reflect across y-axis
        >>> reflection(x, normal)
        array([-1,  0])
    """
    # Ensure normal is unit norm
    normal = normal / np.linalg.norm(normal)

    # Compute projection coefficient
    projection = np.dot(x, normal)

    # Reflect: x - 2⟨x,n⟩n
    return x - 2 * projection[..., np.newaxis] * normal


def custom_involution(sigma_func: Callable[[np.ndarray], np.ndarray]) -> Callable:
    """
    Wrapper to validate custom involution operators.

    Checks that σ² = I (within numerical tolerance).

    Args:
        sigma_func: Proposed involution function

    Returns:
        Validated involution function

    Raises:
        ValueError: If σ² ≠ I

    Example:
        >>> @custom_involution
        ... def my_sigma(x):
        ...     return -x  # Antipodal
        >>> my_sigma(np.array([1, 2, 3]))
        array([-1, -2, -3])
    """
    def validated_sigma(x: np.ndarray) -> np.ndarray:
        result = sigma_func(x)

        # Check σ² = I on a test input
        if x.size > 0:
            test_result = sigma_func(result)
            if not np.allclose(test_result, x, rtol=1e-10):
                raise ValueError(
                    f"Operator is not an involution: σ²(x) ≠ x\n"
                    f"Max difference: {np.max(np.abs(test_result - x))}"
                )

        return result

    return validated_sigma


def get_involution(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get built-in involution by name.

    Args:
        name: One of 'antipodal', 'reverse', or custom function

    Returns:
        Involution function

    Example:
        >>> sigma = get_involution('antipodal')
        >>> sigma(np.array([1, 2, 3]))
        array([-1, -2, -3])
    """
    involutions = {
        'antipodal': antipodal,
        'reverse': reverse,
        'time_reversal': reverse,
    }

    if name not in involutions:
        raise ValueError(
            f"Unknown involution '{name}'. "
            f"Available: {list(involutions.keys())}"
        )

    return involutions[name]
