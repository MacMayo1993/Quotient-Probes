"""
SymmetryProbe: Main API for quotient probe analysis.

Provides unified interface for:
1. Detecting symmetry (computing α)
2. Making MDL-based decisions
3. Decomposing into eigenspaces
4. Estimating orientation costs
"""

import numpy as np
from typing import Tuple, Callable, Optional, Union, Dict, Any

from .involutions import get_involution, antipodal
from .decomposition import (
    decompose,
    decompose_symmetric,
    decompose_antisymmetric,
    compute_coherence,
    verify_decomposition,
)
from .mdl_decision import (
    mdl_decision_rule,
    compute_orientation_cost,
    critical_coherence,
    description_length_difference,
)


class SymmetryProbe:
    """
    Analyzes symmetry structure and makes MDL-based exploitation decisions.

    Usage:
        >>> probe = SymmetryProbe(data, involution='antipodal')
        >>> alpha, gain, should_exploit = probe.analyze()
        >>> if should_exploit:
        ...     x_plus, x_minus = probe.decompose()

    Attributes:
        data: Input vector (n,) or batch (m, n)
        sigma: Involution operator
        n: Ambient dimension
        alpha: Coherence parameter (computed on analyze())
        K_lift: Orientation cost (estimated or provided)
    """

    def __init__(
        self,
        data: np.ndarray,
        involution: Union[str, Callable] = 'antipodal',
        K_lift: Optional[float] = None,
        orientation_model: str = 'bernoulli',
    ):
        """
        Initialize symmetry probe.

        Args:
            data: Input vector (n,) or batch (m, n)
            involution: Involution operator name or function
                       Options: 'antipodal', 'reverse', or custom callable
            K_lift: Orientation cost (if None, estimated from orientation_model)
            orientation_model: Model for estimating K_lift if not provided
                              Options: 'bernoulli', 'markov', 'constant'

        Example:
            >>> x = np.random.randn(100)
            >>> probe = SymmetryProbe(x, involution='antipodal')
            >>> probe.n
            100
        """
        self.data = np.asarray(data)

        # Handle batch vs single vector
        if self.data.ndim == 1:
            self.n = self.data.shape[0]
            self.is_batch = False
        elif self.data.ndim == 2:
            self.n = self.data.shape[1]
            self.is_batch = True
        else:
            raise ValueError(
                f"Data must be 1D (vector) or 2D (batch), got shape {self.data.shape}"
            )

        # Set up involution
        if isinstance(involution, str):
            self.sigma = get_involution(involution)
            self.involution_name = involution
        elif callable(involution):
            self.sigma = involution
            self.involution_name = 'custom'
        else:
            raise ValueError(
                f"involution must be string name or callable, got {type(involution)}"
            )

        # Set up orientation cost
        self.orientation_model = orientation_model
        if K_lift is None:
            self.K_lift = compute_orientation_cost(
                model=orientation_model,
                n=self.n
            )
        else:
            self.K_lift = K_lift

        # Computed during analysis
        self.alpha: Optional[float] = None
        self.x_plus: Optional[np.ndarray] = None
        self.x_minus: Optional[np.ndarray] = None
        self._analysis_results: Optional[Dict[str, Any]] = None

    def analyze(
        self,
        return_components: bool = False
    ) -> Union[Tuple[float, float, bool], Tuple[float, float, bool, np.ndarray, np.ndarray]]:
        """
        Analyze symmetry and make MDL decision.

        Computes:
        1. Coherence α = ||x₊||²/||x||²
        2. Description length difference ΔL
        3. Decision: exploit if α > α_crit

        Args:
            return_components: If True, also return (x₊, x₋)

        Returns:
            (alpha, bit_savings, should_exploit)
            or (alpha, bit_savings, should_exploit, x_plus, x_minus)

        Example:
            >>> x = np.array([1, 2, 3, 2, 1])  # Palindrome
            >>> from .involutions import reverse
            >>> probe = SymmetryProbe(x, involution='reverse')
            >>> alpha, savings, decision = probe.analyze()
            >>> alpha
            1.0  # Fully symmetric
            >>> decision
            True  # Exploit
        """
        if self.is_batch:
            raise NotImplementedError(
                "Batch analysis not yet implemented. "
                "Analyze individual vectors for now."
            )

        # Decompose
        self.x_plus, self.x_minus = decompose(self.data, self.sigma)

        # Compute coherence
        self.alpha = compute_coherence(self.data, self.sigma)

        # Make MDL decision
        should_exploit, details = mdl_decision_rule(
            self.alpha,
            self.n,
            self.K_lift,
            return_details=True
        )

        self._analysis_results = details
        bit_savings = details['bit_savings']

        if return_components:
            return self.alpha, bit_savings, should_exploit, self.x_plus, self.x_minus
        else:
            return self.alpha, bit_savings, should_exploit

    def decompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get eigenspace decomposition: x = x₊ + x₋

        Returns:
            Tuple (x_plus, x_minus)

        Note:
            Call analyze() first, or this will compute decomposition fresh.

        Example:
            >>> probe = SymmetryProbe(np.array([1, -1]))
            >>> x_plus, x_minus = probe.decompose()
            >>> np.allclose(probe.data, x_plus + x_minus)
            True
        """
        if self.x_plus is None or self.x_minus is None:
            self.x_plus, self.x_minus = decompose(self.data, self.sigma)

        return self.x_plus, self.x_minus

    def get_symmetric_component(self) -> np.ndarray:
        """Get symmetric component x₊ ∈ V₊"""
        if self.x_plus is None:
            self.x_plus = decompose_symmetric(self.data, self.sigma)
        return self.x_plus

    def get_antisymmetric_component(self) -> np.ndarray:
        """Get antisymmetric component x₋ ∈ V₋"""
        if self.x_minus is None:
            self.x_minus = decompose_antisymmetric(self.data, self.sigma)
        return self.x_minus

    def get_coherence(self) -> float:
        """
        Get coherence parameter α = ||x₊||²/||x||²

        Returns:
            α ∈ [0, 1]
        """
        if self.alpha is None:
            self.alpha = compute_coherence(self.data, self.sigma)
        return self.alpha

    def get_critical_coherence(self) -> float:
        """
        Get critical coherence threshold for this dimension and orientation cost.

        Returns:
            α_crit = 1/2 + K_lift/(2(n-1))
        """
        return critical_coherence(self.n, self.K_lift)

    def get_decision_details(self) -> Dict[str, Any]:
        """
        Get detailed analysis results.

        Returns:
            Dictionary with keys:
                - alpha: Coherence
                - alpha_crit: Critical threshold
                - delta_L: Description length difference
                - bit_savings: Bits saved by exploiting
                - decision: 'exploit' or 'ignore'
                - margin: α - α_crit

        Note:
            Must call analyze() first.
        """
        if self._analysis_results is None:
            raise RuntimeError("Must call analyze() before getting details")

        return self._analysis_results

    def verify(self, tolerance: float = 1e-10) -> bool:
        """
        Verify decomposition is valid: x = x₊ + x₋

        Args:
            tolerance: Numerical tolerance

        Returns:
            True if decomposition is valid

        Example:
            >>> probe = SymmetryProbe(np.random.randn(50))
            >>> probe.analyze()
            >>> probe.verify()
            True
        """
        if self.x_plus is None or self.x_minus is None:
            self.decompose()

        return verify_decomposition(
            self.data,
            self.x_plus,
            self.x_minus,
            tolerance=tolerance
        )

    def summary(self) -> str:
        """
        Generate human-readable summary of analysis.

        Returns:
            Formatted summary string

        Example:
            >>> probe = SymmetryProbe(np.random.randn(100))
            >>> probe.analyze()
            >>> print(probe.summary())
            Symmetry Probe Analysis
            =======================
            Involution: antipodal
            Dimension: 100
            Coherence α: 0.023
            Critical α: 0.505
            Decision: IGNORE (α < α_crit)
            Bit savings: -48.2 (loss)
        """
        if self._analysis_results is None:
            return "No analysis performed. Call analyze() first."

        details = self._analysis_results

        summary = [
            "Symmetry Probe Analysis",
            "=" * 50,
            f"Involution: {self.involution_name}",
            f"Dimension: {self.n}",
            f"Orientation model: {self.orientation_model} (K_lift = {self.K_lift:.2f})",
            "",
            f"Coherence α: {details['alpha']:.3f}",
            f"Critical α: {details['alpha_crit']:.3f}",
            f"Margin: {details['margin']:+.3f}",
            "",
            f"Description length ΔL: {details['delta_L']:+.2f} bits",
            f"Bit savings: {details['bit_savings']:+.2f}",
            "",
            f"Decision: {details['decision'].upper()}",
        ]

        if details['decision'] == 'exploit':
            summary.append(f"✓ Exploit symmetry (saves {details['bit_savings']:.1f} bits)")
        else:
            summary.append(f"✗ Ignore symmetry (costs {-details['bit_savings']:.1f} extra bits)")

        return "\n".join(summary)

    def __repr__(self) -> str:
        """String representation"""
        alpha_str = f", α={self.alpha:.3f}" if self.alpha is not None else ""
        return (
            f"SymmetryProbe(n={self.n}, "
            f"involution='{self.involution_name}', "
            f"K_lift={self.K_lift:.2f}"
            f"{alpha_str})"
        )
