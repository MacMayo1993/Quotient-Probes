"""
Tests for SymmetryProbe class and core functionality.
"""

import pytest
import numpy as np
from quotient_probes import SymmetryProbe
from quotient_probes.core.involutions import antipodal, reverse
from quotient_probes.core.decomposition import decompose, compute_coherence


class TestSymmetryProbe:
    """Test SymmetryProbe class"""

    def test_initialization(self):
        """Test probe initialization"""
        x = np.random.randn(50)
        probe = SymmetryProbe(x, involution='antipodal')

        assert probe.n == 50
        assert probe.involution_name == 'antipodal'
        assert probe.K_lift == 1.0  # Default Bernoulli

    def test_analyze(self):
        """Test analysis returns correct types"""
        x = np.random.randn(100)
        probe = SymmetryProbe(x)

        alpha, savings, decision = probe.analyze()

        assert isinstance(alpha, float)
        assert 0 <= alpha <= 1
        assert isinstance(savings, float)
        assert isinstance(decision, (bool, np.bool_))

    def test_palindrome_high_coherence(self):
        """Palindromic signal should have high coherence under time-reversal"""
        # Perfect palindrome
        half = np.array([1, 2, 3, 4, 5])
        x = np.concatenate([half, half[::-1]])

        probe = SymmetryProbe(x, involution='reverse')
        alpha = probe.get_coherence()

        # Should be very close to 1.0 (perfect symmetry)
        assert alpha > 0.99, f"Expected α ≈ 1.0 for palindrome, got {alpha}"

    def test_antisymmetric_low_coherence(self):
        """Antisymmetric signal should have low coherence"""
        # Create antisymmetric signal: x = -x[::-1]
        half = np.array([1, 2, 3, 4, 5])
        x = np.concatenate([half, -half[::-1]])

        probe = SymmetryProbe(x, involution='reverse')
        alpha = probe.get_coherence()

        # Should be very close to 0.0 (purely antisymmetric)
        assert alpha < 0.01, f"Expected α ≈ 0.0 for antisymmetric, got {alpha}"

    def test_decomposition_reconstruction(self):
        """Test that x = x₊ + x₋"""
        x = np.random.randn(64)
        probe = SymmetryProbe(x, involution='antipodal')

        x_plus, x_minus = probe.decompose()
        reconstruction = x_plus + x_minus

        np.testing.assert_allclose(x, reconstruction, rtol=1e-10)

    def test_decomposition_orthogonality(self):
        """Test that x₊ ⊥ x₋"""
        x = np.random.randn(128)
        probe = SymmetryProbe(x, involution='reverse')

        x_plus, x_minus = probe.decompose()
        inner_product = np.dot(x_plus, x_minus)

        assert abs(inner_product) < 1e-10, f"Expected orthogonality, got ⟨x₊,x₋⟩ = {inner_product}"

    def test_verify_decomposition(self):
        """Test decomposition verification"""
        x = np.random.randn(50)
        probe = SymmetryProbe(x)
        probe.analyze()

        assert probe.verify() is True

    def test_critical_coherence_formula(self):
        """Test α_crit = 1/2 + K_lift/(2(n-1))"""
        n = 64
        K_lift = 1.0

        probe = SymmetryProbe(np.zeros(n), K_lift=K_lift)
        alpha_crit = probe.get_critical_coherence()

        expected = 0.5 + (K_lift / (2 * (n - 1)))
        assert abs(alpha_crit - expected) < 1e-10

    def test_worked_example_n64(self):
        """Test worked example from paper: n=64, K_lift=1"""
        n = 64
        K_lift = 1.0

        probe = SymmetryProbe(np.zeros(n), K_lift=K_lift)
        alpha_crit = probe.get_critical_coherence()

        expected_crit = 0.5 + (K_lift / (2 * (n - 1)))
        assert abs(alpha_crit - expected_crit) < 1e-10, \
            f"Expected α_crit = {expected_crit:.6f}, got {alpha_crit:.6f}"

    def test_decision_boundary(self):
        """Test decision changes at α_crit"""
        from quotient_probes.core.mdl_decision import mdl_decision_rule

        n = 100
        K_lift = 1.0
        alpha_crit = 0.5 + (K_lift / (2 * (n - 1)))

        # Just below threshold
        assert mdl_decision_rule(alpha_crit - 0.001, n, K_lift) is False

        # Just above threshold
        assert mdl_decision_rule(alpha_crit + 0.001, n, K_lift) is True

        # Exactly at threshold (should not exploit due to ΔL = 0)
        assert mdl_decision_rule(alpha_crit, n, K_lift) is False


class TestInvolutions:
    """Test involution operators"""

    def test_antipodal_is_involution(self):
        """Test σ² = I for antipodal"""
        x = np.random.randn(50)
        sigma_x = antipodal(x)
        sigma_sigma_x = antipodal(sigma_x)

        np.testing.assert_allclose(x, sigma_sigma_x)

    def test_reverse_is_involution(self):
        """Test σ² = I for reverse"""
        x = np.random.randn(50)
        sigma_x = reverse(x)
        sigma_sigma_x = reverse(sigma_x)

        np.testing.assert_allclose(x, sigma_sigma_x)

    def test_antipodal_eigenvalues(self):
        """Test antipodal has eigenvalues ±1"""
        x = np.array([1, 2, 3])
        assert np.allclose(antipodal(x), -x)


class TestMDLDecision:
    """Test MDL decision rule"""

    def test_description_length_formula(self):
        """Test ΔL(α) = n(2α - 1) - K_lift"""
        from quotient_probes.core.mdl_decision import description_length_difference

        n = 64
        alpha = 0.6
        K_lift = 1.0

        delta_L = description_length_difference(alpha, n, K_lift)
        expected = n * (2 * alpha - 1) - K_lift

        assert abs(delta_L - expected) < 1e-10

    def test_orientation_cost_bernoulli(self):
        """Test K_lift = 1 for Bernoulli"""
        from quotient_probes.core.mdl_decision import compute_orientation_cost

        K_lift = compute_orientation_cost('bernoulli')
        assert K_lift == 1.0

    def test_orientation_cost_constant(self):
        """Test K_lift = 0 for constant orientation"""
        from quotient_probes.core.mdl_decision import compute_orientation_cost

        K_lift = compute_orientation_cost('constant')
        assert K_lift == 0.0

    def test_asymptotic_behavior(self):
        """Test α_crit → 0.5 as n → ∞"""
        from quotient_probes.core.mdl_decision import critical_coherence

        K_lift = 1.0

        # For large n, α_crit should approach 0.5
        alpha_crit_large = critical_coherence(n=10000, K_lift=K_lift)
        assert abs(alpha_crit_large - 0.5) < 0.001

        # For small n, α_crit should be noticeably above 0.5
        alpha_crit_small = critical_coherence(n=10, K_lift=K_lift)
        assert alpha_crit_small >= 0.55  # For n=10, K_lift=1.0: (10+1)/(2*10) = 0.55 exactly


class TestDecomposition:
    """Test eigenspace decomposition"""

    def test_energy_conservation(self):
        """Test ||x||² = ||x₊||² + ||x₋||²"""
        x = np.random.randn(100)
        x_plus, x_minus = decompose(x, antipodal)

        energy_x = np.linalg.norm(x) ** 2
        energy_plus = np.linalg.norm(x_plus) ** 2
        energy_minus = np.linalg.norm(x_minus) ** 2

        np.testing.assert_allclose(energy_x, energy_plus + energy_minus, rtol=1e-10)

    def test_coherence_bounds(self):
        """Test α ∈ [0, 1]"""
        for _ in range(10):
            x = np.random.randn(50)
            alpha = compute_coherence(x, antipodal)
            assert 0 <= alpha <= 1

    def test_coherence_interpretation(self):
        """Test α = ||x₊||²/||x||²"""
        x = np.random.randn(64)
        x_plus, x_minus = decompose(x, reverse)

        alpha_direct = np.linalg.norm(x_plus) ** 2 / np.linalg.norm(x) ** 2
        alpha_computed = compute_coherence(x, reverse)

        np.testing.assert_allclose(alpha_direct, alpha_computed, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
