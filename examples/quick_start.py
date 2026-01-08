"""
Quotient Probes: Quick Start Example

Demonstrates the core API for symmetry analysis and MDL-based decision making.
"""

import numpy as np
from quotient_probes import SymmetryProbe
from quotient_probes.core.involutions import antipodal, reverse


def example_1_basic_usage():
    """Basic usage: Analyze a vector for antipodal symmetry"""
    print("=" * 60)
    print("Example 1: Basic Usage - Antipodal Symmetry")
    print("=" * 60)

    # Create a random vector
    np.random.seed(42)
    x = np.random.randn(100)

    # Create probe
    probe = SymmetryProbe(x, involution='antipodal')

    # Analyze
    alpha, bit_savings, should_exploit = probe.analyze()

    print(f"\nVector dimension: {probe.n}")
    print(f"Coherence α: {alpha:.4f}")
    print(f"Critical threshold α_crit: {probe.get_critical_coherence():.4f}")
    print(f"Bit savings: {bit_savings:+.2f} bits")
    print(f"Decision: {'EXPLOIT' if should_exploit else 'IGNORE'}")

    print("\nFull summary:")
    print(probe.summary())


def example_2_palindrome():
    """Palindromic signal with time-reversal symmetry"""
    print("\n" + "=" * 60)
    print("Example 2: Palindrome - Perfect Symmetry")
    print("=" * 60)

    # Create perfectly palindromic signal
    half = np.array([1, 2, 3, 4, 5])
    x = np.concatenate([half, half[::-1]])

    print(f"\nSignal: {x}")
    print(f"Reversed: {x[::-1]}")

    # Analyze with time-reversal involution
    probe = SymmetryProbe(x, involution='reverse')
    alpha, bit_savings, should_exploit = probe.analyze()

    print(f"\nCoherence α: {alpha:.4f} (1.0 = perfect symmetry)")
    print(f"Decision: {'EXPLOIT' if should_exploit else 'IGNORE'}")

    # Show decomposition
    x_plus, x_minus = probe.decompose()
    print(f"\nSymmetric component x₊: {x_plus}")
    print(f"Antisymmetric component x₋: {x_minus}")
    print(f"||x₋||: {np.linalg.norm(x_minus):.6f} (should be ≈0)")


def example_3_worked_example():
    """Recreate worked example from paper (Section 3)"""
    print("\n" + "=" * 60)
    print("Example 3: Worked Example from Paper (n=64, K_lift=1)")
    print("=" * 60)

    # Parameters from paper
    n = 64
    K_lift = 1.0

    # Create vector with α = 0.6 (above threshold)
    # We want ||x₊||²/||x||² = 0.6
    # So ||x₊||/||x|| = sqrt(0.6) ≈ 0.775
    # And ||x₋||/||x|| = sqrt(0.4) ≈ 0.632

    target_alpha = 0.6
    target_norm = 10.0

    # For antipodal involution, x₊ = 0 for generic x
    # So we use time-reversal to get controllable α

    # Create symmetric and antisymmetric parts with desired norms
    np.random.seed(123)
    x_plus = np.random.randn(n)
    x_plus = (x_plus + x_plus[::-1]) / 2  # Make symmetric

    x_minus = np.random.randn(n)
    x_minus = (x_minus - x_minus[::-1]) / 2  # Make antisymmetric

    # Scale to achieve target α
    norm_plus_desired = target_norm * np.sqrt(target_alpha)
    norm_minus_desired = target_norm * np.sqrt(1 - target_alpha)

    x_plus = x_plus / np.linalg.norm(x_plus) * norm_plus_desired
    x_minus = x_minus / np.linalg.norm(x_minus) * norm_minus_desired

    x = x_plus + x_minus

    # Analyze
    probe = SymmetryProbe(x, involution='reverse', K_lift=K_lift)
    alpha, bit_savings, should_exploit = probe.analyze()

    print(f"\nDimension n: {n}")
    print(f"Orientation cost K_lift: {K_lift}")
    print(f"Critical coherence α_crit: {probe.get_critical_coherence():.4f}")
    print(f"Measured coherence α: {alpha:.4f}")
    print(f"Target coherence: {target_alpha:.4f}")
    print(f"\nDescription length difference ΔL: {-bit_savings:+.2f} bits")
    print(f"Bit savings: {bit_savings:+.2f} bits")
    print(f"\nDecision: {'EXPLOIT ✓' if should_exploit else 'IGNORE ✗'}")

    if should_exploit:
        print(f"\n✓ Since α = {alpha:.3f} > α_crit = {probe.get_critical_coherence():.3f},")
        print(f"  we save {bit_savings:.1f} bits by exploiting symmetry!")


def example_4_dimension_sweep():
    """Show how decision changes with dimension"""
    print("\n" + "=" * 60)
    print("Example 4: How Dimension Affects Decision")
    print("=" * 60)

    # Fixed coherence
    target_alpha = 0.52
    K_lift = 1.0

    print(f"\nFixed coherence α = {target_alpha}")
    print(f"Orientation cost K_lift = {K_lift}")
    print(f"\nCritical thresholds:")
    print(f"{'n':>6} | {'α_crit':>8} | {'Decision':>10} | {'ΔL (bits)':>12}")
    print("-" * 45)

    for n in [16, 32, 64, 128, 256, 512, 1024]:
        alpha_crit = (n + K_lift) / (2 * n)
        delta_L = n * (2 * target_alpha - 1) - K_lift

        decision = "EXPLOIT" if target_alpha > alpha_crit else "IGNORE"

        print(f"{n:>6} | {alpha_crit:>8.5f} | {decision:>10} | {delta_L:>+12.2f}")

    print(f"\nAs n → ∞: α_crit → 0.5")
    print(f"With α = {target_alpha}, we eventually EXPLOIT for large enough n")


def example_5_compare_involutions():
    """Compare different involution operators"""
    print("\n" + "=" * 60)
    print("Example 5: Comparing Different Involutions")
    print("=" * 60)

    # Create a specific signal
    np.random.seed(100)
    x = np.sin(np.linspace(0, 2*np.pi, 64)) + 0.1 * np.random.randn(64)

    print(f"Analyzing sinusoidal signal (n={len(x)}):\n")

    # Test multiple involutions
    involutions = [
        ('antipodal', 'σ(x) = -x'),
        ('reverse', 'σ(x) = x[::-1]'),
    ]

    results = []
    for inv_name, description in involutions:
        probe = SymmetryProbe(x, involution=inv_name)
        alpha, savings, decision = probe.analyze()

        results.append({
            'involution': inv_name,
            'description': description,
            'alpha': alpha,
            'savings': savings,
            'decision': decision
        })

        print(f"{inv_name.upper()}: {description}")
        print(f"  α = {alpha:.4f}")
        print(f"  Bit savings: {savings:+.2f}")
        print(f"  Decision: {decision}")
        print()

    print("Different involutions reveal different symmetry structures!")


if __name__ == "__main__":
    print("QUOTIENT PROBES - Quick Start Examples")
    print("=" * 60)
    print("Orientation Cost in Symmetry-Adapted Hilbert Spaces\n")

    example_1_basic_usage()
    example_2_palindrome()
    example_3_worked_example()
    example_4_dimension_sweep()
    example_5_compare_involutions()

    print("\n" + "=" * 60)
    print("Examples complete! See notebooks/ for interactive tutorials.")
    print("=" * 60)
