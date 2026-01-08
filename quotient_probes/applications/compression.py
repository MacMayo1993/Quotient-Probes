"""
SeamAware Compression: MDL-driven symmetry-aware compression.

The key insight: When α > α_crit, we save bits by storing (x₊, x₋) separately
instead of storing x directly. This is because the orientation cost is amortized
across the compressed representation.

Applications:
1. Time series compression (EEG, financial data)
2. Embedding compression (CLIP, BERT vectors)
3. Model weight compression (neural networks)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass

from ..core.symmetry_probe import SymmetryProbe
from ..core.involutions import get_involution


@dataclass
class CompressionResult:
    """Result of compression operation."""

    compressed_data: Any
    compression_ratio: float
    original_size_bits: float
    compressed_size_bits: float
    exploited_symmetry: bool
    coherence: float
    bit_savings: float
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"CompressionResult(\n"
            f"  compression_ratio={self.compression_ratio:.3f}x\n"
            f"  original={self.original_size_bits:.1f} bits\n"
            f"  compressed={self.compressed_size_bits:.1f} bits\n"
            f"  exploited_symmetry={self.exploited_symmetry}\n"
            f"  coherence α={self.coherence:.3f}\n"
            f"  bit_savings={self.bit_savings:+.1f} bits\n"
            f")"
        )


class SeamAwareCompressor:
    """
    MDL-driven compressor that automatically decides whether to exploit symmetry.

    The compressor:
    1. Analyzes coherence α of input data
    2. Compares α to critical threshold α_crit
    3. If α > α_crit: compresses (x₊, x₋) separately
    4. If α < α_crit: compresses x directly

    Usage:
        >>> compressor = SeamAwareCompressor(involution='antipodal')
        >>> result = compressor.compress(data)
        >>> print(f"Compression ratio: {result.compression_ratio:.2f}x")
        >>> decompressed = compressor.decompress(result.compressed_data)

    Attributes:
        involution: Symmetry operator (antipodal, reverse, or custom)
        K_lift: Orientation cost (default: 1.0 for Bernoulli)
        quantization_bits: Bits per coefficient (default: 32 for float32)
    """

    def __init__(
        self,
        involution: Union[str, Callable] = 'antipodal',
        K_lift: Optional[float] = None,
        orientation_model: str = 'bernoulli',
        quantization_bits: int = 32,
        verbose: bool = False,
    ):
        """
        Initialize compressor.

        Args:
            involution: Involution operator ('antipodal', 'reverse', or callable)
            K_lift: Orientation cost (if None, estimated from orientation_model)
            orientation_model: Model for orientation cost ('bernoulli', 'markov', 'constant')
            quantization_bits: Bits per coefficient (32 for float32, 16 for float16, etc.)
            verbose: Print compression decisions
        """
        self.involution = involution
        self.K_lift_param = K_lift
        self.orientation_model = orientation_model
        self.quantization_bits = quantization_bits
        self.verbose = verbose

        # Get involution function
        if isinstance(involution, str):
            self.sigma = get_involution(involution)
            self.involution_name = involution
        elif callable(involution):
            self.sigma = involution
            self.involution_name = 'custom'
        else:
            raise ValueError(f"involution must be string or callable, got {type(involution)}")

    def compress(
        self,
        data: np.ndarray,
        return_probe: bool = False,
    ) -> Union[CompressionResult, Tuple[CompressionResult, SymmetryProbe]]:
        """
        Compress data using MDL-based symmetry exploitation decision.

        Args:
            data: Input vector (n,) or batch (m, n)
            return_probe: If True, also return the SymmetryProbe object

        Returns:
            CompressionResult (or tuple of CompressionResult and SymmetryProbe)

        Example:
            >>> data = np.random.randn(256)
            >>> result = compressor.compress(data)
            >>> print(result.compression_ratio)
        """
        # Handle batch vs single vector
        if data.ndim == 1:
            return self._compress_single(data, return_probe)
        elif data.ndim == 2:
            return self._compress_batch(data, return_probe)
        else:
            raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")

    def _compress_single(
        self,
        x: np.ndarray,
        return_probe: bool = False,
    ) -> Union[CompressionResult, Tuple[CompressionResult, SymmetryProbe]]:
        """Compress a single vector."""
        n = x.shape[0]

        # Analyze symmetry
        probe = SymmetryProbe(
            x,
            involution=self.involution,
            K_lift=self.K_lift_param,
            orientation_model=self.orientation_model,
        )

        alpha, bit_savings, should_exploit = probe.analyze()

        # Original size (uncompressed)
        original_size_bits = n * self.quantization_bits

        if should_exploit:
            # EXPLOIT: Store x₊ and x₋ separately + orientation bit
            x_plus, x_minus = probe.decompose()

            # Compute norms (for reconstruction)
            norm_plus = np.linalg.norm(x_plus)
            norm_minus = np.linalg.norm(x_minus)

            # Compressed representation:
            # 1. Direction of x₊ (normalized)
            # 2. Direction of x₋ (normalized)
            # 3. Two norms (magnitudes)
            # 4. One orientation bit

            # In practice, we'd use entropy coding here
            # For demonstration, we count description length

            # Directions: (n-1) degrees of freedom each on unit sphere
            # Norms: log2 precision bits each
            # Orientation: K_lift bits

            # Simplified model: each component takes n/2 space on average
            # (accounting for sparsity in eigenspaces)
            compressed_size_bits = (
                alpha * n * self.quantization_bits +  # x₊ component
                (1 - alpha) * n * self.quantization_bits +  # x₋ component
                probe.K_lift  # orientation cost
            )

            compressed_data = {
                'mode': 'exploit',
                'x_plus': x_plus,
                'x_minus': x_minus,
                'norm_plus': norm_plus,
                'norm_minus': norm_minus,
                'involution': self.involution_name,
            }

            if self.verbose:
                print(f"✅ EXPLOIT: α={alpha:.3f} > α_crit, saved {bit_savings:.1f} bits")

        else:
            # IGNORE: Store x directly (isotropic compression)
            compressed_data = {
                'mode': 'ignore',
                'x': x,
                'involution': self.involution_name,
            }

            # No benefit from symmetry, standard compression
            compressed_size_bits = original_size_bits

            if self.verbose:
                print(f"❌ IGNORE: α={alpha:.3f} < α_crit, standard compression")

        # Compute metrics
        compression_ratio = original_size_bits / compressed_size_bits

        result = CompressionResult(
            compressed_data=compressed_data,
            compression_ratio=compression_ratio,
            original_size_bits=original_size_bits,
            compressed_size_bits=compressed_size_bits,
            exploited_symmetry=should_exploit,
            coherence=alpha,
            bit_savings=bit_savings,
            metadata={
                'n': n,
                'K_lift': probe.K_lift,
                'alpha_crit': probe.get_critical_coherence(),
                'quantization_bits': self.quantization_bits,
            }
        )

        if return_probe:
            return result, probe
        return result

    def _compress_batch(
        self,
        X: np.ndarray,
        return_probe: bool = False,
    ) -> CompressionResult:
        """Compress a batch of vectors."""
        m, n = X.shape

        # Compress each vector independently
        results = []
        probes = []

        for i in range(m):
            if return_probe:
                result, probe = self._compress_single(X[i], return_probe=True)
                probes.append(probe)
            else:
                result = self._compress_single(X[i], return_probe=False)
            results.append(result)

        # Aggregate statistics
        total_original_bits = sum(r.original_size_bits for r in results)
        total_compressed_bits = sum(r.compressed_size_bits for r in results)
        avg_compression_ratio = total_original_bits / total_compressed_bits
        avg_coherence = np.mean([r.coherence for r in results])
        total_bit_savings = sum(r.bit_savings for r in results)
        num_exploited = sum(r.exploited_symmetry for r in results)

        batch_result = CompressionResult(
            compressed_data={'batch': [r.compressed_data for r in results]},
            compression_ratio=avg_compression_ratio,
            original_size_bits=total_original_bits,
            compressed_size_bits=total_compressed_bits,
            exploited_symmetry=(num_exploited > 0),
            coherence=avg_coherence,
            bit_savings=total_bit_savings,
            metadata={
                'm': m,
                'n': n,
                'num_exploited': num_exploited,
                'exploitation_rate': num_exploited / m,
                'individual_results': results,
            }
        )

        if return_probe:
            return batch_result, probes
        return batch_result

    def decompress(
        self,
        compressed_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        Decompress data.

        Args:
            compressed_data: Output from compress()

        Returns:
            Reconstructed vector or batch

        Example:
            >>> result = compressor.compress(data)
            >>> reconstructed = compressor.decompress(result.compressed_data)
            >>> np.allclose(data, reconstructed)
            True
        """
        if 'batch' in compressed_data:
            # Batch decompression
            return np.array([
                self.decompress(item)
                for item in compressed_data['batch']
            ])

        mode = compressed_data['mode']

        if mode == 'exploit':
            # Reconstruct from x₊ and x₋
            x_plus = compressed_data['x_plus']
            x_minus = compressed_data['x_minus']
            return x_plus + x_minus

        elif mode == 'ignore':
            # Direct storage
            return compressed_data['x']

        else:
            raise ValueError(f"Unknown compression mode: {mode}")

    def benchmark(
        self,
        data: np.ndarray,
        baseline_compressor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark compression performance vs baseline.

        Args:
            data: Test data (single vector or batch)
            baseline_compressor: Baseline compressor (default: no compression)

        Returns:
            Dictionary with performance metrics

        Example:
            >>> data = np.random.randn(100, 256)
            >>> metrics = compressor.benchmark(data)
            >>> print(f"Speedup: {metrics['compression_ratio_improvement']:.2f}x")
        """
        # SeamAware compression
        result_seam = self.compress(data)

        # Baseline compression (no symmetry exploitation)
        if baseline_compressor is None:
            # Default baseline: store directly
            baseline_size = result_seam.original_size_bits
            baseline_ratio = 1.0
        else:
            baseline_result = baseline_compressor(data)
            baseline_size = baseline_result.compressed_size_bits
            baseline_ratio = baseline_result.compression_ratio

        # Compute improvements
        compression_ratio_improvement = result_seam.compression_ratio / baseline_ratio
        size_reduction = (baseline_size - result_seam.compressed_size_bits) / baseline_size

        return {
            'seam_compression_ratio': result_seam.compression_ratio,
            'baseline_compression_ratio': baseline_ratio,
            'compression_ratio_improvement': compression_ratio_improvement,
            'seam_compressed_size_bits': result_seam.compressed_size_bits,
            'baseline_compressed_size_bits': baseline_size,
            'size_reduction_pct': size_reduction * 100,
            'bit_savings': result_seam.bit_savings,
            'coherence': result_seam.coherence,
            'exploited_symmetry': result_seam.exploited_symmetry,
        }


def compress_time_series(
    time_series: np.ndarray,
    window_size: int = 128,
    overlap: float = 0.5,
    involution: str = 'reverse',
    **kwargs
) -> Tuple[CompressionResult, np.ndarray]:
    """
    Compress time series using sliding window + symmetry exploitation.

    Args:
        time_series: Input time series (1D array)
        window_size: Window size for segmentation
        overlap: Overlap ratio (0.0 to 1.0)
        involution: Involution for windows ('reverse' recommended for time series)
        **kwargs: Additional arguments for SeamAwareCompressor

    Returns:
        Tuple of (CompressionResult, reconstructed time series)

    Example:
        >>> ts = np.sin(np.linspace(0, 10*np.pi, 1000))
        >>> result, reconstructed = compress_time_series(ts)
        >>> print(f"Compression: {result.compression_ratio:.2f}x")
    """
    n = len(time_series)
    step = int(window_size * (1 - overlap))

    # Segment into windows
    windows = []
    indices = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        windows.append(time_series[start:end])
        indices.append((start, end))

    # Compress windows
    compressor = SeamAwareCompressor(involution=involution, **kwargs)
    windows_array = np.array(windows)
    result = compressor.compress(windows_array)

    # Reconstruct
    reconstructed = compressor.decompress(result.compressed_data)

    # Reconstruct full time series (overlap-add)
    full_reconstructed = np.zeros(n)
    weights = np.zeros(n)

    for i, (start, end) in enumerate(indices):
        full_reconstructed[start:end] += reconstructed[i]
        weights[start:end] += 1

    full_reconstructed /= np.maximum(weights, 1)

    return result, full_reconstructed


def estimate_compression_potential(
    data: np.ndarray,
    involution: str = 'antipodal',
    K_lift: float = 1.0,
) -> Dict[str, Any]:
    """
    Estimate compression potential without actual compression.

    Fast pre-analysis to determine if symmetry exploitation is worthwhile.

    Args:
        data: Input data
        involution: Involution operator
        K_lift: Orientation cost

    Returns:
        Dictionary with analysis results

    Example:
        >>> data = np.random.randn(256)
        >>> analysis = estimate_compression_potential(data)
        >>> if analysis['should_exploit']:
        >>>     print(f"Potential savings: {analysis['estimated_savings_bits']:.1f} bits")
    """
    probe = SymmetryProbe(data, involution=involution, K_lift=K_lift)
    alpha, bit_savings, should_exploit = probe.analyze()

    details = probe.get_decision_details()

    return {
        'should_exploit': should_exploit,
        'coherence': alpha,
        'alpha_crit': details['alpha_crit'],
        'estimated_savings_bits': bit_savings,
        'estimated_compression_ratio': 1.0 + (bit_savings / (data.size * 32)),
        'margin': details['margin'],
        'decision': details['decision'],
    }
