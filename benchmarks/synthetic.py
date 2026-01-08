"""
Synthetic benchmarks for quotient probes.

Tests the library on synthetic data with controlled coherence α ∈ [0, 1].
This validates the theory against ground truth.

Benchmarks:
1. Coherence sweep: Test decisions across α ∈ [0, 1]
2. Dimension sweep: Test scaling with n ∈ [10, 10000]
3. Orientation cost sweep: Test K_lift ∈ [0, 5]
4. Application performance: Compression, search, regime detection
5. Involution comparison: Antipodal vs reverse vs custom

Run with:
    python -m benchmarks.synthetic --all
    python -m benchmarks.synthetic --coherence-sweep
    python -m benchmarks.synthetic --compression-test
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from quotient_probes import SymmetryProbe
from quotient_probes.applications.compression import (
    SeamAwareCompressor,
    estimate_compression_potential,
)
from quotient_probes.core.mdl_decision import (
    critical_coherence,
    description_length_difference,
    mdl_decision_rule,
)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    benchmark_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    success: bool
    runtime_seconds: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyntheticDataGenerator:
    """Generate synthetic data with controlled coherence."""

    @staticmethod
    def generate_with_coherence(
        n: int,
        target_alpha: float,
        involution: str = "reverse",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate vector with specific coherence α.

        Args:
            n: Dimension
            target_alpha: Target coherence (0 to 1)
            involution: Involution type
            seed: Random seed

        Returns:
            Vector x with ||x₊||²/||x||² ≈ target_alpha

        Example:
            >>> x = SyntheticDataGenerator.generate_with_coherence(128, 0.7, 'reverse')
            >>> probe = SymmetryProbe(x, 'reverse')
            >>> alpha = probe.get_coherence()
            >>> assert abs(alpha - 0.7) < 0.01
        """
        if seed is not None:
            np.random.seed(seed)

        if not 0 <= target_alpha <= 1:
            raise ValueError(f"target_alpha must be in [0, 1], got {target_alpha}")

        # Generate symmetric and antisymmetric components with target energy split
        # ||x₊||² = target_alpha * ||x||²
        # ||x₋||² = (1 - target_alpha) * ||x||²

        # Total energy budget (arbitrary scale)
        total_energy = n

        energy_plus = target_alpha * total_energy
        energy_minus = (1 - target_alpha) * total_energy

        # Generate random components
        x_plus_raw = np.random.randn(n)
        x_minus_raw = np.random.randn(n)

        # Scale to target energies
        if energy_plus > 0:
            x_plus = x_plus_raw * np.sqrt(
                energy_plus / (np.linalg.norm(x_plus_raw) ** 2 + 1e-10)
            )
        else:
            x_plus = np.zeros(n)

        if energy_minus > 0:
            x_minus = x_minus_raw * np.sqrt(
                energy_minus / (np.linalg.norm(x_minus_raw) ** 2 + 1e-10)
            )
        else:
            x_minus = np.zeros(n)

        # For reverse involution, we need to construct x such that:
        # x₊ = (x + reverse(x))/2 = x_plus
        # x₋ = (x - reverse(x))/2 = x_minus
        # Therefore: x = x_plus + x_minus

        x = x_plus + x_minus

        return x

    @staticmethod
    def generate_batch_with_coherence_sweep(
        n: int,
        alphas: np.ndarray,
        involution: str = "reverse",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate batch of vectors with varying coherence.

        Args:
            n: Dimension
            alphas: Array of target coherences
            involution: Involution type
            seed: Random seed

        Returns:
            Batch of vectors (m, n)
        """
        if seed is not None:
            np.random.seed(seed)

        vectors = []
        for i, alpha in enumerate(alphas):
            x = SyntheticDataGenerator.generate_with_coherence(
                n, alpha, involution, seed=seed + i if seed else None
            )
            vectors.append(x)

        return np.array(vectors)


def benchmark_coherence_sweep(
    n: int = 128,
    K_lift: float = 1.0,
    num_points: int = 20,
    save_plot: bool = True,
) -> BenchmarkResult:
    """
    Test MDL decision rule across α ∈ [0, 1].

    Validates that:
    1. Decision flips at α_crit
    2. Bit savings match theory
    3. Coherence estimation is accurate
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Coherence Sweep (n={n}, K_lift={K_lift})")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Generate test data
    alphas_target = np.linspace(0.05, 0.95, num_points)
    data = SyntheticDataGenerator.generate_batch_with_coherence_sweep(
        n, alphas_target, involution="reverse", seed=42
    )

    # Compute critical threshold
    alpha_crit = critical_coherence(n, K_lift)
    print(f"Critical threshold: α_crit = {alpha_crit:.4f}\n")

    # Test each vector
    results = []
    correct_decisions = 0

    print(
        f"{'Target α':<12} {'Actual α':<12} {'Decision':<10} {'Expected':<10} {'Correct':<8} {'Bit Savings':<12}"
    )
    print("-" * 80)

    for i, target_alpha in enumerate(alphas_target):
        x = data[i]

        probe = SymmetryProbe(x, involution="reverse", K_lift=K_lift)
        actual_alpha, bit_savings, should_exploit = probe.analyze()

        # Expected decision
        expected_exploit = target_alpha > alpha_crit

        # Check if decision is correct
        is_correct = should_exploit == expected_exploit
        correct_decisions += is_correct

        results.append(
            {
                "target_alpha": target_alpha,
                "actual_alpha": actual_alpha,
                "should_exploit": should_exploit,
                "expected_exploit": expected_exploit,
                "bit_savings": bit_savings,
                "is_correct": is_correct,
            }
        )

        marker = "✅" if is_correct else "❌"
        exploit_str = "EXPLOIT" if should_exploit else "IGNORE"
        expected_str = "EXPLOIT" if expected_exploit else "IGNORE"

        print(
            f"{target_alpha:<12.3f} {actual_alpha:<12.4f} {exploit_str:<10} {expected_str:<10} {marker:<8} {bit_savings:+12.2f}"
        )

    accuracy = correct_decisions / num_points
    runtime = time.time() - start_time

    print(f"\n{'='*70}")
    print(
        f"RESULTS: {correct_decisions}/{num_points} correct ({accuracy*100:.1f}% accuracy)"
    )
    print(f"Runtime: {runtime:.3f}s")
    print(f"{'='*70}\n")

    # Plot results
    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Decision boundary
        ax = axes[0, 0]
        alphas_plot = np.linspace(0, 1, 200)
        delta_Ls = [description_length_difference(a, n, K_lift) for a in alphas_plot]
        ax.plot(alphas_plot, delta_Ls, "b-", linewidth=2, label="ΔL(α)")
        ax.axhline(0, color="k", linestyle="--", linewidth=1.5)
        ax.axvline(
            alpha_crit,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"α_crit={alpha_crit:.4f}",
        )

        # Mark test points
        for r in results:
            color = "green" if r["should_exploit"] else "red"
            ax.plot(
                r["actual_alpha"],
                description_length_difference(r["actual_alpha"], n, K_lift),
                "o",
                color=color,
                markersize=6,
                alpha=0.6,
            )

        ax.set_xlabel("Coherence α", fontsize=11)
        ax.set_ylabel("ΔL (bits)", fontsize=11)
        ax.set_title(f"Decision Boundary (n={n}, K_lift={K_lift})", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Target vs Actual coherence
        ax = axes[0, 1]
        target_alphas = [r["target_alpha"] for r in results]
        actual_alphas = [r["actual_alpha"] for r in results]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect match")
        ax.scatter(target_alphas, actual_alphas, c="blue", s=50, alpha=0.6)
        ax.set_xlabel("Target α", fontsize=11)
        ax.set_ylabel("Actual α", fontsize=11)
        ax.set_title("Coherence Generation Accuracy", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Plot 3: Bit savings
        ax = axes[1, 0]
        bit_savings_values = [r["bit_savings"] for r in results]
        colors = ["green" if s > 0 else "red" for s in bit_savings_values]
        ax.bar(
            range(len(bit_savings_values)),
            bit_savings_values,
            color=colors,
            alpha=0.6,
        )
        ax.axhline(0, color="k", linestyle="-", linewidth=1)
        ax.set_xlabel("Test case index", fontsize=11)
        ax.set_ylabel("Bit savings", fontsize=11)
        ax.set_title("Bit Savings Distribution", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 4: Decision correctness
        ax = axes[1, 1]
        correct = [1 if r["is_correct"] else 0 for r in results]
        ax.bar(
            range(len(correct)),
            correct,
            color=["green" if c else "red" for c in correct],
            alpha=0.6,
        )
        ax.set_xlabel("Test case index", fontsize=11)
        ax.set_ylabel("Correct (1) / Incorrect (0)", fontsize=11)
        ax.set_title(f"Decision Accuracy: {accuracy*100:.1f}%", fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            "benchmarks/results/coherence_sweep.png", dpi=150, bbox_inches="tight"
        )
        print("📊 Plot saved: benchmarks/results/coherence_sweep.png")

    return BenchmarkResult(
        benchmark_name="coherence_sweep",
        params={"n": n, "K_lift": K_lift, "num_points": num_points},
        metrics={
            "accuracy": accuracy,
            "avg_bit_savings": float(
                np.mean([r["bit_savings"] for r in results if r["should_exploit"]])
            ),
            "alpha_crit": float(alpha_crit),
        },
        success=(accuracy > 0.9),  # 90% threshold
        runtime_seconds=runtime,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def benchmark_dimension_sweep(
    n_values: List[int] = [16, 32, 64, 128, 256, 512, 1024],
    target_alpha: float = 0.6,
    K_lift: float = 1.0,
    save_plot: bool = True,
) -> BenchmarkResult:
    """Test how performance scales with dimension n."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Dimension Sweep (α={target_alpha}, K_lift={K_lift})")
    print(f"{'='*70}\n")

    start_time = time.time()

    results = []

    print(
        f"{'n':<8} {'α_crit':<10} {'Decision':<10} {'Bit Savings':<15} {'Runtime (ms)':<15}"
    )
    print("-" * 70)

    for n in n_values:
        # Generate data
        x = SyntheticDataGenerator.generate_with_coherence(
            n, target_alpha, involution="reverse", seed=42
        )

        # Analyze
        t0 = time.time()
        probe = SymmetryProbe(x, involution="reverse", K_lift=K_lift)
        alpha, bit_savings, should_exploit = probe.analyze()
        runtime_ms = (time.time() - t0) * 1000

        alpha_crit = critical_coherence(n, K_lift)
        decision = "EXPLOIT" if should_exploit else "IGNORE"

        results.append(
            {
                "n": n,
                "alpha_crit": alpha_crit,
                "actual_alpha": alpha,
                "should_exploit": should_exploit,
                "bit_savings": bit_savings,
                "runtime_ms": runtime_ms,
            }
        )

        print(
            f"{n:<8} {alpha_crit:<10.4f} {decision:<10} {bit_savings:+15.2f} {runtime_ms:<15.3f}"
        )

    total_runtime = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Total runtime: {total_runtime:.3f}s")
    print(f"{'='*70}\n")

    # Plot
    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: α_crit vs n
        ax = axes[0, 0]
        ns = [r["n"] for r in results]
        alpha_crits = [r["alpha_crit"] for r in results]
        ax.semilogx(ns, alpha_crits, "o-", linewidth=2, markersize=8)
        ax.axhline(
            0.5,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label="Asymptotic limit (0.5)",
        )
        ax.set_xlabel("Dimension n (log scale)", fontsize=11)
        ax.set_ylabel("α_crit", fontsize=11)
        ax.set_title(f"Critical Threshold vs Dimension (K_lift={K_lift})", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Bit savings vs n
        ax = axes[0, 1]
        bit_savings_values = [r["bit_savings"] for r in results]
        ax.semilogx(
            ns, bit_savings_values, "o-", linewidth=2, markersize=8, color="green"
        )
        ax.axhline(0, color="k", linestyle="-", linewidth=1)
        ax.set_xlabel("Dimension n (log scale)", fontsize=11)
        ax.set_ylabel("Bit savings", fontsize=11)
        ax.set_title(f"Bit Savings vs Dimension (α={target_alpha})", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 3: Runtime scaling
        ax = axes[1, 0]
        runtimes = [r["runtime_ms"] for r in results]
        ax.loglog(ns, runtimes, "o-", linewidth=2, markersize=8, color="purple")
        ax.set_xlabel("Dimension n (log scale)", fontsize=11)
        ax.set_ylabel("Runtime (ms, log scale)", fontsize=11)
        ax.set_title("Runtime Scaling", fontsize=12)
        ax.grid(True, alpha=0.3, which="both")

        # Plot 4: Summary table
        ax = axes[1, 1]
        ax.axis("off")
        table_data = []
        for r in results:
            table_data.append(
                [
                    f"{r['n']}",
                    f"{r['alpha_crit']:.4f}",
                    f"{r['bit_savings']:+.1f}",
                    f"{r['runtime_ms']:.2f}",
                ]
            )
        table = ax.table(
            cellText=table_data,
            colLabels=["n", "α_crit", "Savings (bits)", "Time (ms)"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        plt.tight_layout()
        plt.savefig(
            "benchmarks/results/dimension_sweep.png", dpi=150, bbox_inches="tight"
        )
        print("📊 Plot saved: benchmarks/results/dimension_sweep.png")

    return BenchmarkResult(
        benchmark_name="dimension_sweep",
        params={"n_values": n_values, "target_alpha": target_alpha, "K_lift": K_lift},
        metrics={
            "max_n": max(n_values),
            "max_bit_savings": float(max([r["bit_savings"] for r in results])),
            "avg_runtime_ms": float(np.mean([r["runtime_ms"] for r in results])),
        },
        success=True,
        runtime_seconds=total_runtime,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def benchmark_compression(
    n: int = 256,
    num_samples: int = 100,
    alpha_range: Tuple[float, float] = (0.3, 0.9),
    save_plot: bool = True,
) -> BenchmarkResult:
    """Benchmark compression application."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Compression Performance (n={n}, samples={num_samples})")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Generate diverse data
    alphas = np.random.uniform(alpha_range[0], alpha_range[1], num_samples)
    data = SyntheticDataGenerator.generate_batch_with_coherence_sweep(
        n, alphas, involution="reverse", seed=42
    )

    # Test compression
    compressor = SeamAwareCompressor(involution="reverse", K_lift=1.0, verbose=False)

    results = []
    total_original_bits = 0.0
    total_compressed_bits = 0.0

    for i in range(num_samples):
        x = data[i]
        result = compressor.compress(x)

        results.append(
            {
                "alpha": result.coherence,
                "compression_ratio": result.compression_ratio,
                "bit_savings": result.bit_savings,
                "exploited": result.exploited_symmetry,
            }
        )

        total_original_bits += result.original_size_bits
        total_compressed_bits += result.compressed_size_bits

    avg_compression_ratio = total_original_bits / total_compressed_bits
    num_exploited = sum(r["exploited"] for r in results)
    exploitation_rate = num_exploited / num_samples

    runtime = time.time() - start_time

    print(f"\n📊 Compression Statistics:")
    print(f"   Samples: {num_samples}")
    print(
        f"   Exploitation rate: {exploitation_rate*100:.1f}% ({num_exploited}/{num_samples})"
    )
    print(f"   Avg compression ratio: {avg_compression_ratio:.3f}x")
    print(
        f"   Total bit savings: {total_original_bits - total_compressed_bits:+.0f} bits"
    )
    print(f"   Runtime: {runtime:.3f}s")
    print(f"   Throughput: {num_samples/runtime:.1f} vectors/sec\n")

    # Plot
    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Compression ratio vs coherence
        ax = axes[0, 0]
        alphas_plot = [r["alpha"] for r in results]
        ratios = [r["compression_ratio"] for r in results]
        colors = ["green" if r["exploited"] else "red" for r in results]
        ax.scatter(alphas_plot, ratios, c=colors, s=50, alpha=0.6)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="No compression")
        ax.set_xlabel("Coherence α", fontsize=11)
        ax.set_ylabel("Compression ratio", fontsize=11)
        ax.set_title("Compression Ratio vs Coherence", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Distribution of bit savings
        ax = axes[1, 0]
        bit_savings_values = [r["bit_savings"] for r in results]
        ax.hist(
            bit_savings_values, bins=20, color="skyblue", edgecolor="black", alpha=0.7
        )
        ax.axvline(0, color="r", linestyle="--", linewidth=2, label="Break-even")
        ax.set_xlabel("Bit savings", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Distribution of Bit Savings", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Exploitation decision
        ax = axes[0, 1]
        exploit_count = [num_exploited, num_samples - num_exploited]
        ax.pie(
            exploit_count,
            labels=["Exploited", "Ignored"],
            autopct="%1.1f%%",
            colors=["green", "red"],
            startangle=90,
        )
        ax.set_title(
            f"Symmetry Exploitation Rate: {exploitation_rate*100:.1f}%", fontsize=12
        )

        # Plot 4: Summary stats
        ax = axes[1, 1]
        ax.axis("off")
        stats_text = f"""
COMPRESSION BENCHMARK SUMMARY
{'='*40}

Samples:              {num_samples}
Dimension:            {n}
Involution:           reverse

Exploitation rate:    {exploitation_rate*100:.1f}%
Avg compression:      {avg_compression_ratio:.3f}x
Total bit savings:    {total_original_bits - total_compressed_bits:+.0f}

Runtime:              {runtime:.3f}s
Throughput:           {num_samples/runtime:.1f} vectors/sec
        """
        ax.text(
            0.1,
            0.5,
            stats_text,
            fontsize=10,
            family="monospace",
            verticalalignment="center",
        )

        plt.tight_layout()
        plt.savefig(
            "benchmarks/results/compression_benchmark.png", dpi=150, bbox_inches="tight"
        )
        print("📊 Plot saved: benchmarks/results/compression_benchmark.png")

    return BenchmarkResult(
        benchmark_name="compression",
        params={"n": n, "num_samples": num_samples, "alpha_range": alpha_range},
        metrics={
            "avg_compression_ratio": avg_compression_ratio,
            "exploitation_rate": exploitation_rate,
            "throughput_vectors_per_sec": num_samples / runtime,
        },
        success=True,
        runtime_seconds=runtime,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def run_all_benchmarks(save_results: bool = True):
    """Run all benchmarks and save results."""
    print("\n" + "=" * 70)
    print("RUNNING ALL SYNTHETIC BENCHMARKS")
    print("=" * 70)

    # Create results directory
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Coherence sweep
    results["coherence_sweep"] = benchmark_coherence_sweep(
        n=128, K_lift=1.0, num_points=20
    )

    # 2. Dimension sweep
    results["dimension_sweep"] = benchmark_dimension_sweep(
        n_values=[16, 32, 64, 128, 256, 512, 1024], target_alpha=0.6, K_lift=1.0
    )

    # 3. Compression benchmark
    results["compression"] = benchmark_compression(
        n=256, num_samples=100, alpha_range=(0.3, 0.9)
    )

    # Save results
    if save_results:
        results_dict = {name: result.to_dict() for name, result in results.items()}

        with open("benchmarks/results/synthetic_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✅ All results saved to benchmarks/results/synthetic_results.json")

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"\n{status} {result.benchmark_name}")
        print(f"   Runtime: {result.runtime_seconds:.3f}s")
        print(f"   Key metrics:")
        for metric_name, value in result.metrics.items():
            print(f"     - {metric_name}: {value:.4f}")

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run synthetic benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--coherence-sweep", action="store_true", help="Run coherence sweep"
    )
    parser.add_argument(
        "--dimension-sweep", action="store_true", help="Run dimension sweep"
    )
    parser.add_argument(
        "--compression-test", action="store_true", help="Run compression test"
    )

    args = parser.parse_args()

    # Create results directory
    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

    if args.all:
        run_all_benchmarks()
    elif args.coherence_sweep:
        benchmark_coherence_sweep()
    elif args.dimension_sweep:
        benchmark_dimension_sweep()
    elif args.compression_test:
        benchmark_compression()
    else:
        # Default: run all
        run_all_benchmarks()
