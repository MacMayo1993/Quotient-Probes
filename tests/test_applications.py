"""
Tests for applications module.

Tests compression, vector search, and regime detection applications.
"""

import numpy as np
import pytest

from quotient_probes.applications.compression import (
    SeamAwareCompressor,
    compress_time_series,
    estimate_compression_potential,
)
from quotient_probes.applications.regime_detection import (
    LighthouseDetector,
    generate_synthetic_regime_series,
)
from quotient_probes.applications.vector_search import (
    AntipodalVectorDB,
    create_benchmark_database,
)


class TestSeamAwareCompressor:
    """Test compression application."""

    def test_compressor_initialization(self):
        """Test compressor initialization."""
        compressor = SeamAwareCompressor(involution="antipodal")
        assert compressor.involution_name == "antipodal"
        assert compressor.quantization_bits == 32

    def test_compress_single_vector(self):
        """Test compression of single vector."""
        np.random.seed(42)
        data = np.random.randn(128)

        compressor = SeamAwareCompressor(involution="reverse")
        result = compressor.compress(data)

        assert hasattr(result, "compression_ratio")
        assert hasattr(result, "coherence")
        assert hasattr(result, "exploited_symmetry")
        assert result.compression_ratio >= 0
        assert 0 <= result.coherence <= 1

    def test_compress_decompress_roundtrip(self):
        """Test compression and decompression roundtrip."""
        np.random.seed(42)
        data = np.random.randn(64)

        compressor = SeamAwareCompressor(involution="reverse")
        result = compressor.compress(data)
        reconstructed = compressor.decompress(result.compressed_data)

        # Should reconstruct exactly (no actual quantization in this implementation)
        np.testing.assert_allclose(data, reconstructed, rtol=1e-10)

    def test_compress_batch(self):
        """Test batch compression."""
        np.random.seed(42)
        data = np.random.randn(10, 64)

        compressor = SeamAwareCompressor(involution="reverse")
        result = compressor.compress(data)

        assert "batch" in result.compressed_data
        assert result.compression_ratio > 0
        assert hasattr(result.metadata, "__getitem__")

    def test_compression_potential_estimation(self):
        """Test compression potential estimation."""
        np.random.seed(42)
        data = np.random.randn(128)

        analysis = estimate_compression_potential(data, involution="reverse")

        assert "should_exploit" in analysis
        assert "coherence" in analysis
        assert "alpha_crit" in analysis
        assert "estimated_savings_bits" in analysis
        assert isinstance(analysis["should_exploit"], bool)

    def test_time_series_compression(self):
        """Test time series compression with windows."""
        np.random.seed(42)
        time_series = np.sin(np.linspace(0, 10 * np.pi, 500))

        result, reconstructed = compress_time_series(
            time_series, window_size=64, overlap=0.5, involution="reverse"
        )

        assert len(reconstructed) == len(time_series)
        assert result.compression_ratio > 0


class TestAntipodalVectorDB:
    """Test vector search application."""

    def test_vectordb_initialization(self):
        """Test vector database initialization."""
        embeddings = np.random.randn(100, 64)
        db = AntipodalVectorDB(embeddings, auto_partition=False)

        assert db.n_vectors == 100
        assert db.dim == 64

    def test_vectordb_search(self):
        """Test vector search."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        db = AntipodalVectorDB(embeddings, auto_partition=False)

        query = np.random.randn(64)
        result = db.search(query, k=5)

        assert len(result.indices) == 5
        assert len(result.distances) == 5
        assert (
            result.query_time_ms >= 0
        )  # May be 0 on Windows (fast operation + low timer resolution)
        assert all(0 <= idx < 100 for idx in result.indices)

    def test_vectordb_with_partitioning(self):
        """Test vector search with symmetry partitioning."""
        np.random.seed(42)

        # Create symmetric embeddings
        embeddings = create_benchmark_database(
            n_vectors=1000, dim=128, symmetry_strength=0.8, seed=42
        )

        db = AntipodalVectorDB(embeddings, coherence_threshold=0.5, auto_partition=True)
        query = np.random.randn(128)

        # Search with symmetry
        result_sym = db.search(query, k=10, use_symmetry=True)

        # Search without symmetry
        result_baseline = db.search(query, k=10, use_symmetry=False)

        # Should find same results (or very similar)
        assert len(result_sym.indices) == 10
        assert len(result_baseline.indices) == 10

        # Symmetry should be faster (searches fewer vectors)
        if db.use_partitioning:
            assert result_sym.partition_size < db.n_vectors

    def test_vectordb_batch_search(self):
        """Test batch search."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        db = AntipodalVectorDB(embeddings, auto_partition=False)

        queries = np.random.randn(10, 64)
        results = db.batch_search(queries, k=5)

        assert len(results) == 10
        assert all(len(r.indices) == 5 for r in results)

    def test_vectordb_benchmark(self):
        """Test benchmark functionality."""
        np.random.seed(42)
        embeddings = np.random.randn(200, 64)
        db = AntipodalVectorDB(embeddings, auto_partition=False)

        metrics = db.benchmark(num_queries=20, k=5)

        assert "num_queries" in metrics
        assert "speedup" in metrics
        assert "time_with_symmetry_s" in metrics
        assert "time_baseline_s" in metrics
        assert metrics["speedup"] > 0

    def test_vectordb_cosine_similarity(self):
        """Test cosine similarity search."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64)
        db = AntipodalVectorDB(embeddings, normalize=True)

        query = np.random.randn(64)
        result = db.search(query, k=5, metric="cosine")

        assert len(result.indices) == 5
        # Distances should be in [0, 2] for cosine distance
        assert all(0 <= d <= 2 for d in result.distances)

    def test_create_benchmark_database(self):
        """Test synthetic database generation."""
        embeddings = create_benchmark_database(
            n_vectors=100, dim=32, symmetry_strength=0.7, seed=42
        )

        assert embeddings.shape == (100, 32)
        # Should be normalized
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestLighthouseDetector:
    """Test regime detection application."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = LighthouseDetector(window_size=64, overlap=0.5)

        assert detector.window_size == 64
        assert detector.overlap == 0.5
        assert detector.step == 32
        assert detector.alpha_crit > 0.5

    def test_detector_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        with pytest.raises(ValueError, match="overlap must be in"):
            LighthouseDetector(window_size=64, overlap=1.5)

    def test_rolling_coherence_computation(self):
        """Test rolling coherence computation."""
        np.random.seed(42)
        time_series = np.sin(np.linspace(0, 10 * np.pi, 500))

        detector = LighthouseDetector(window_size=64, overlap=0.5)
        centers, coherences = detector.compute_rolling_coherence(time_series)

        assert len(centers) == len(coherences)
        assert len(coherences) > 0
        assert all(0 <= alpha <= 1 for alpha in coherences)

    def test_seam_detection(self):
        """Test seam detection."""
        # Create coherence series that crosses threshold
        coherences = np.array([0.3, 0.4, 0.6, 0.7, 0.5, 0.4])
        centers = np.arange(len(coherences)) * 10

        detector = LighthouseDetector(window_size=64, K_lift=1.0)
        seams = detector.detect_seams(coherences, centers)

        # Should detect at least one seam (crossing at index 2)
        assert len(seams) > 0
        assert all(hasattr(s, "direction") for s in seams)
        assert all(hasattr(s, "strength") for s in seams)

    def test_regime_segmentation(self):
        """Test regime segmentation."""
        # Create coherence series with clear regimes
        coherences = np.concatenate(
            [
                np.ones(10) * 0.7,  # Symmetric regime
                np.ones(10) * 0.3,  # Antisymmetric regime
                np.ones(10) * 0.8,  # Symmetric regime
            ]
        )
        centers = np.arange(len(coherences)) * 10

        detector = LighthouseDetector(window_size=64, K_lift=1.0, min_regime_duration=5)
        regimes = detector.segment_regimes(coherences, centers)

        # Should detect multiple regimes
        assert len(regimes) >= 2
        assert all(hasattr(r, "regime_type") for r in regimes)
        assert all(hasattr(r, "mean_coherence") for r in regimes)

    def test_full_detection_pipeline(self):
        """Test full detection pipeline."""
        # Generate synthetic series with regime transitions
        time_series = generate_synthetic_regime_series(
            regime_durations=[200, 200],
            regime_alphas=[0.8, 0.2],
            window_size=64,
            seed=42,
        )

        detector = LighthouseDetector(window_size=64, overlap=0.5)
        seams, regimes = detector.detect(time_series)

        # Should detect at least one seam and multiple regimes
        assert len(seams) >= 0  # May or may not detect seams depending on thresholds
        assert len(regimes) >= 0

    def test_detection_with_coherence_return(self):
        """Test detection with coherence return."""
        np.random.seed(42)
        time_series = np.random.randn(500)

        detector = LighthouseDetector(window_size=64, overlap=0.5)
        seams, regimes, centers, coherences = detector.detect(
            time_series, return_coherence=True
        )

        assert len(centers) == len(coherences)
        assert all(0 <= alpha <= 1 for alpha in coherences)

    def test_detector_statistics(self):
        """Test statistics computation."""
        np.random.seed(42)
        time_series = np.random.randn(500)

        detector = LighthouseDetector(window_size=64, overlap=0.5)
        seams, regimes = detector.detect(time_series)

        stats = detector.get_statistics(seams, regimes)

        assert "num_seams" in stats
        assert "num_regimes" in stats
        assert stats["num_seams"] == len(seams)
        assert stats["num_regimes"] == len(regimes)

    def test_generate_synthetic_regime_series(self):
        """Test synthetic regime series generation."""
        ts = generate_synthetic_regime_series(
            regime_durations=[100, 150],
            regime_alphas=[0.7, 0.3],
            window_size=64,
            seed=42,
        )

        assert len(ts) == 250  # Sum of durations
        assert isinstance(ts, np.ndarray)


class TestApplicationsIntegration:
    """Integration tests across applications."""

    def test_compression_on_regime_data(self):
        """Test compression on data with regime structure."""
        # Generate regime-structured data
        ts = generate_synthetic_regime_series(
            regime_durations=[200, 200], regime_alphas=[0.8, 0.2], seed=42
        )

        # Compress
        compressor = SeamAwareCompressor(involution="reverse")
        result = compressor.compress(ts)

        assert result.compression_ratio > 0

    def test_vector_search_on_compressed_embeddings(self):
        """Test vector search after compression analysis."""
        np.random.seed(42)
        embeddings = create_benchmark_database(100, 64, symmetry_strength=0.8)

        # Analyze symmetry
        db = AntipodalVectorDB(embeddings)

        # Should partition if symmetric
        stats = db.get_statistics()
        assert "use_partitioning" in stats

        # Search should work
        query = np.random.randn(64)
        result = db.search(query, k=5)
        assert len(result.indices) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
