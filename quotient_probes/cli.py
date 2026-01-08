"""
Command-line interface for quotient-probes.

Usage:
    quotient-probe analyze data.npy --involution=reverse
    quotient-probe compress data.npy --output=compressed.npz
    quotient-probe benchmark --synthetic --n=256
"""

import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

from .applications.compression import (
    SeamAwareCompressor,
    estimate_compression_potential,
)
from .core.symmetry_probe import SymmetryProbe
from .visualization.mdl_boundary import plot_decision_boundary, plot_worked_examples


@click.group()
@click.version_option()
def main():
    """
    Quotient Probes: Orientation Cost in Symmetry-Adapted Hilbert Spaces.

    Analyze symmetry structure and make MDL-based exploitation decisions.
    """
    pass


@main.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--involution",
    "-i",
    default="antipodal",
    type=click.Choice(["antipodal", "reverse", "time_reversal"]),
    help="Involution operator",
)
@click.option(
    "--k-lift",
    "-k",
    type=float,
    default=None,
    help="Orientation cost (default: auto-detect from model)",
)
@click.option(
    "--orientation-model",
    "-m",
    default="bernoulli",
    type=click.Choice(["bernoulli", "markov", "constant"]),
    help="Model for orientation cost",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--plot", "-p", is_flag=True, help="Generate visualization")
def analyze(data_file, involution, k_lift, orientation_model, verbose, plot):
    """
    Analyze symmetry structure of data file.

    DATA_FILE: Path to numpy array (.npy, .npz, or .txt)

    Example:
        quotient-probe analyze data.npy --involution=reverse --verbose
    """
    click.echo(f"📊 Analyzing {data_file}...")

    # Load data
    try:
        data = load_data(data_file)
        click.echo(f"✅ Loaded data with shape: {data.shape}")
    except Exception as e:
        click.echo(f"❌ Error loading data: {e}", err=True)
        sys.exit(1)

    # Handle batch vs single vector
    if data.ndim > 1:
        if data.shape[0] > 1:
            click.echo(f"📦 Batch mode: analyzing {data.shape[0]} vectors")
            analyze_batch(data, involution, k_lift, orientation_model, verbose)
            return
        else:
            data = data[0]

    # Create probe
    probe = SymmetryProbe(
        data, involution=involution, K_lift=k_lift, orientation_model=orientation_model
    )

    # Analyze
    alpha, bit_savings, should_exploit = probe.analyze()

    # Print summary
    if verbose:
        click.echo("\n" + probe.summary())
    else:
        click.echo(f"\n{'='*60}")
        click.echo(f"Coherence α: {alpha:.4f}")
        click.echo(f"Decision: {'✅ EXPLOIT' if should_exploit else '❌ IGNORE'}")
        click.echo(f"Bit savings: {bit_savings:+.1f} bits")
        click.echo(f"{'='*60}\n")

    # Plot if requested
    if plot:
        try:
            import matplotlib.pyplot as plt

            from .visualization.mdl_boundary import plot_decision_boundary

            details = probe.get_decision_details()
            n = probe.n
            K_lift = probe.K_lift

            fig = plot_decision_boundary(
                n_values=[n], K_lift=K_lift, show_examples=False
            )

            # Mark observed point
            ax = fig.axes[0]
            delta_L = details["delta_L"]
            ax.plot(
                alpha,
                delta_L,
                "ro",
                markersize=15,
                label=f"Your data (α={alpha:.3f})",
                zorder=10,
            )
            ax.legend()

            output_file = Path(data_file).stem + "_analysis.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            click.echo(f"📊 Plot saved: {output_file}")

        except ImportError:
            click.echo("⚠️  Install matplotlib for plotting: pip install matplotlib")


def analyze_batch(data, involution, k_lift, orientation_model, verbose):
    """Analyze batch of vectors."""
    results = []

    for i, vec in enumerate(data):
        probe = SymmetryProbe(
            vec,
            involution=involution,
            K_lift=k_lift,
            orientation_model=orientation_model,
        )
        alpha, bit_savings, should_exploit = probe.analyze()
        results.append(
            {
                "alpha": alpha,
                "should_exploit": should_exploit,
                "bit_savings": bit_savings,
            }
        )

    # Aggregate statistics
    alphas = [r["alpha"] for r in results]
    exploit_count = sum(r["should_exploit"] for r in results)
    total_savings = sum(r["bit_savings"] for r in results)

    click.echo(f"\n{'='*60}")
    click.echo(f"BATCH ANALYSIS SUMMARY")
    click.echo(f"{'='*60}")
    click.echo(f"Vectors analyzed:     {len(results)}")
    click.echo(
        f"Exploitation rate:    {exploit_count/len(results)*100:.1f}% ({exploit_count}/{len(results)})"
    )
    click.echo(f"Avg coherence:        {np.mean(alphas):.4f} ± {np.std(alphas):.4f}")
    click.echo(f"Total bit savings:    {total_savings:+.1f} bits")
    click.echo(f"{'='*60}\n")


@main.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for compressed data"
)
@click.option(
    "--involution",
    "-i",
    default="antipodal",
    type=click.Choice(["antipodal", "reverse", "time_reversal"]),
    help="Involution operator",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def compress(data_file, output, involution, verbose):
    """
    Compress data using MDL-driven symmetry exploitation.

    Example:
        quotient-probe compress data.npy --output=compressed.npz
    """
    click.echo(f"🗜️  Compressing {data_file}...")

    # Load data
    try:
        data = load_data(data_file)
        click.echo(f"✅ Loaded data with shape: {data.shape}")
    except Exception as e:
        click.echo(f"❌ Error loading data: {e}", err=True)
        sys.exit(1)

    # Compress
    compressor = SeamAwareCompressor(involution=involution, verbose=verbose)
    result = compressor.compress(data)

    click.echo(f"\n{'='*60}")
    click.echo(f"COMPRESSION RESULTS")
    click.echo(f"{'='*60}")
    click.echo(f"Original size:        {result.original_size_bits:.0f} bits")
    click.echo(f"Compressed size:      {result.compressed_size_bits:.0f} bits")
    click.echo(f"Compression ratio:    {result.compression_ratio:.3f}x")
    click.echo(
        f"Exploited symmetry:   {'✅ Yes' if result.exploited_symmetry else '❌ No'}"
    )
    click.echo(f"Coherence:            {result.coherence:.4f}")
    click.echo(f"Bit savings:          {result.bit_savings:+.1f} bits")
    click.echo(f"{'='*60}\n")

    # Save if output specified
    if output:
        try:
            np.savez_compressed(
                output, compressed_data=result.compressed_data, metadata=result.metadata
            )
            click.echo(f"💾 Compressed data saved: {output}")
        except Exception as e:
            click.echo(f"❌ Error saving: {e}", err=True)


@main.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--involution",
    "-i",
    default="antipodal",
    type=click.Choice(["antipodal", "reverse"]),
    help="Involution operator",
)
@click.option("--k-lift", "-k", type=float, default=1.0, help="Orientation cost")
def potential(data_file, involution, k_lift):
    """
    Estimate compression potential without actual compression.

    Fast pre-analysis to determine if symmetry exploitation is worthwhile.

    Example:
        quotient-probe potential data.npy --involution=reverse
    """
    # Load data
    data = load_data(data_file)

    # Estimate
    analysis = estimate_compression_potential(data, involution, k_lift)

    click.echo(f"\n{'='*60}")
    click.echo(f"COMPRESSION POTENTIAL ANALYSIS")
    click.echo(f"{'='*60}")
    click.echo(
        f"Should exploit:       {'✅ Yes' if analysis['should_exploit'] else '❌ No'}"
    )
    click.echo(f"Coherence:            {analysis['coherence']:.4f}")
    click.echo(f"Critical threshold:   {analysis['alpha_crit']:.4f}")
    click.echo(f"Estimated savings:    {analysis['estimated_savings_bits']:+.1f} bits")
    click.echo(f"Est. compression:     {analysis['estimated_compression_ratio']:.3f}x")
    click.echo(f"Margin:               {analysis['margin']:+.4f}")
    click.echo(f"{'='*60}\n")


@main.command()
@click.option("--synthetic", is_flag=True, help="Run synthetic benchmarks")
@click.option("--n", type=int, default=128, help="Dimension for synthetic data")
@click.option("--num-points", type=int, default=20, help="Number of test points")
def benchmark(synthetic, n, num_points):
    """
    Run benchmarks to validate the library.

    Example:
        quotient-probe benchmark --synthetic --n=256
    """
    if synthetic:
        click.echo("🧪 Running synthetic benchmarks...")

        try:
            from benchmarks.synthetic import benchmark_coherence_sweep

            result = benchmark_coherence_sweep(
                n=n, num_points=num_points, save_plot=False
            )

            if result.success:
                click.echo(f"\n✅ Benchmark PASSED")
            else:
                click.echo(f"\n❌ Benchmark FAILED")

            click.echo(f"\nKey metrics:")
            for metric, value in result.metrics.items():
                click.echo(f"  {metric}: {value:.4f}")

        except ImportError as e:
            click.echo(f"❌ Error importing benchmark module: {e}", err=True)
            click.echo("Make sure you're in the repository root directory.")
            sys.exit(1)
    else:
        click.echo("Specify --synthetic to run synthetic benchmarks")


@main.command()
@click.option("--n", type=int, default=64, help="Dimension")
@click.option("--k-lift", type=float, default=1.0, help="Orientation cost")
@click.option("--output", "-o", type=click.Path(), help="Output file")
def visualize(n, k_lift, output):
    """
    Generate MDL decision boundary visualization.

    Example:
        quotient-probe visualize --n=64 --k-lift=1.0 --output=boundary.png
    """
    try:
        import matplotlib.pyplot as plt

        from .visualization.mdl_boundary import plot_decision_boundary

        click.echo(f"📊 Generating visualization (n={n}, K_lift={k_lift})...")

        fig = plot_decision_boundary(n_values=[n], K_lift=k_lift, show_examples=False)

        if output:
            plt.savefig(output, dpi=150, bbox_inches="tight")
            click.echo(f"✅ Saved: {output}")
        else:
            plt.show()

    except ImportError:
        click.echo("❌ Install matplotlib: pip install matplotlib")
        sys.exit(1)


def load_data(file_path: str) -> np.ndarray:
    """Load data from file."""
    path = Path(file_path)

    if path.suffix == ".npy":
        return np.load(file_path)
    elif path.suffix == ".npz":
        data = np.load(file_path)
        # Try common array names
        for key in ["data", "arr_0", "x"]:
            if key in data:
                return data[key]
        # Return first array
        return data[list(data.keys())[0]]
    elif path.suffix in [".txt", ".csv"]:
        return np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


if __name__ == "__main__":
    main()
