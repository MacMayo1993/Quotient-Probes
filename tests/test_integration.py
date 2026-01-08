"""
Integration tests for end-to-end workflows.

Tests complete workflows combining multiple modules.
"""

import numpy as np
import pytest

from quotient_probes import SymmetryProbe
from quotient_probes.applications.compression import SeamAwareCompressor
from quotient_probes.applications.regime_detection import LighthouseDetector
from quotient_probes.applications.vector_search import AntipodalVectorDB
from quotient_probes.core.mdl_decision import critical_coherence, mdl_decision_rule


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_full_symmetry_analysis_workflow(self):
        """Test complete symmetry analysis workflow."""
        np.random.seed(42)

        # 1. Generate data
        n = 128
        data = np.random.randn(n)

        # 2. Create probe and analyze
        probe = SymmetryProbe(data, involution="reverse", K_lift=1.0)
        alpha, bit_savings, should_exploit = probe.analyze()

        # 3. Verify analysis results
        assert 0 <= alpha <= 1
        assert isinstance(should_exploit, bool)

        # 4. Get detailed results
        details = probe.get_decision_details()
        assert "alpha_crit" in details
        assert "delta_L" in details

        # 5. Decompose if should exploit
        if should_exploit:
            x_plus, x_minus = probe.decompose()
            assert np.allclose(data, x_plus + x_minus)

        # 6. Verify decomposition
        assert probe.verify()

    def test_compression_workflow(self):
        """Test compression workflow from start to finish."""
        np.random.seed(42)

        # 1. Generate time series
        t = np.linspace(0, 4 * np.pi, 256)
        time_series = np.sin(t) + 0.5 * np.cos(3 * t)

        # 2. Analyze symmetry
        probe = SymmetryProbe(time_series, involution="reverse")
        alpha = probe.get_coherence()

        # 3. Compress
        compressor = SeamAwareCompressor(involution="reverse")
        result = compressor.compress(time_series)

        # 4. Verify compression metadata
        assert result.coherence == pytest.approx(alpha, abs=1e-5)
        assert result.compression_ratio > 0

        # 5. Decompress
        reconstructed = compressor.decompress(result.compressed_data)

        # 6. Verify reconstruction
        np.testing.assert_allclose(time_series, reconstructed, rtol=1e-10)

    def test_vector_search_workflow(self):
        """Test vector search workflow."""
        np.random.seed(42)

        # 1. Generate embedding database
        n_vectors = 500
        dim = 128
        embeddings = np.random.randn(n_vectors, dim)

        # 2. Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 3. Create database
        db = AntipodalVectorDB(embeddings, coherence_threshold=0.6, auto_partition=True)

        # 4. Generate query
        query = np.random.randn(dim)
        query = query / np.linalg.norm(query)

        # 5. Search with symmetry
        result_sym = db.search(query, k=10, metric="cosine", use_symmetry=True)

        # 6. Search without symmetry (baseline)
        result_baseline = db.search(query, k=10, metric="cosine", use_symmetry=False)

        # 7. Verify results
        assert len(result_sym.indices) == 10
        assert len(result_baseline.indices) == 10

        # 8. If partitioning enabled, should be faster
        if db.use_partitioning:
            assert result_sym.speedup >= 1.0

    def test_regime_detection_workflow(self):
        """Test regime detection workflow."""
        np.random.seed(42)

        # 1. Generate synthetic data with regime changes
        # Symmetric regime
        symmetric_segment = np.tile(np.array([1, 2, 3, 2, 1]), 40)  # 200 samples

        # Antisymmetric regime
        anti_segment = np.tile(np.array([1, 2, 0, -2, -1]), 40)  # 200 samples

        # Combine
        time_series = np.concatenate([symmetric_segment, anti_segment])

        # 2. Create detector
        detector = LighthouseDetector(
            window_size=64, overlap=0.5, involution="reverse", K_lift=1.0
        )

        # 3. Detect regimes
        seams, regimes, centers, coherences = detector.detect(
            time_series, return_coherence=True
        )

        # 4. Verify detection
        assert len(centers) == len(coherences)
        assert all(0 <= alpha <= 1 for alpha in coherences)

        # 5. Get statistics
        stats = detector.get_statistics(seams, regimes)
        assert "num_seams" in stats
        assert "num_regimes" in stats

    def test_mdl_decision_consistency(self):
        """Test consistency of MDL decisions across modules."""
        np.random.seed(42)

        n = 128
        K_lift = 1.0

        # Generate data with known coherence
        x = np.random.randn(n)

        # 1. Compute coherence using SymmetryProbe
        probe = SymmetryProbe(x, involution="reverse", K_lift=K_lift)
        alpha_probe = probe.get_coherence()
        should_exploit_probe = probe.analyze()[2]

        # 2. Compute decision using mdl_decision_rule directly
        should_exploit_mdl = mdl_decision_rule(alpha_probe, n, K_lift)

        # 3. Check critical coherence
        alpha_crit = critical_coherence(n, K_lift)

        # 4. All should agree
        assert should_exploit_probe == should_exploit_mdl
        assert should_exploit_mdl == (alpha_probe > alpha_crit)

    def test_batch_processing_workflow(self):
        """Test batch processing across multiple vectors."""
        np.random.seed(42)

        # 1. Generate batch of vectors
        batch_size = 20
        n = 64
        data_batch = np.random.randn(batch_size, n)

        # 2. Compress batch
        compressor = SeamAwareCompressor(involution="reverse")
        result = compressor.compress(data_batch)

        # 3. Verify batch metadata
        assert "batch" in result.compressed_data
        assert result.metadata["m"] == batch_size
        assert result.metadata["n"] == n

        # 4. Decompress batch
        reconstructed_batch = compressor.decompress(result.compressed_data)

        # 5. Verify reconstruction
        assert reconstructed_batch.shape == data_batch.shape
        np.testing.assert_allclose(data_batch, reconstructed_batch, rtol=1e-10)

        # 6. Check individual results
        individual_results = result.metadata["individual_results"]
        assert len(individual_results) == batch_size

    def test_cross_involution_consistency(self):
        """Test that results are consistent across involutions."""
        np.random.seed(42)

        n = 64
        x = np.random.randn(n)

        involutions = ["antipodal", "reverse"]
        results = {}

        for inv in involutions:
            probe = SymmetryProbe(x, involution=inv, K_lift=1.0)
            alpha, savings, decision = probe.analyze()

            results[inv] = {
                "alpha": alpha,
                "decision": decision,
                "savings": savings,
            }

        # Each involution should give valid results
        for inv, res in results.items():
            assert 0 <= res["alpha"] <= 1
            assert isinstance(res["decision"], bool)

    def test_performance_comparison_workflow(self):
        """Test performance comparison between methods."""
        np.random.seed(42)

        # Generate test data
        embeddings = np.random.randn(1000, 128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create database
        db = AntipodalVectorDB(embeddings, auto_partition=True)

        # Benchmark
        metrics = db.benchmark(num_queries=50, k=10)

        # Verify metrics
        assert metrics["speedup"] > 0
        assert metrics["time_baseline_s"] > 0
        assert metrics["time_with_symmetry_s"] > 0
        assert metrics["throughput_sym_qps"] > 0
        assert metrics["throughput_baseline_qps"] > 0

    def test_error_handling_workflow(self):
        """Test error handling in workflows."""

        # Test dimension mismatch
        with pytest.raises(ValueError):
            probe = SymmetryProbe(np.random.randn(10, 10, 10))  # 3D array

        # Test invalid involution
        with pytest.raises(ValueError):
            SymmetryProbe(np.random.randn(64), involution="invalid")

        # Test invalid overlap in detector
        with pytest.raises(ValueError):
            LighthouseDetector(window_size=64, overlap=1.5)

        # Test query dimension mismatch in vector search
        embeddings = np.random.randn(100, 64)
        db = AntipodalVectorDB(embeddings)
        with pytest.raises(ValueError):
            db.search(np.random.randn(128), k=5)  # Wrong dimension


class TestReproducibility:
    """Test reproducibility of results."""

    def test_deterministic_compression(self):
        """Test that compression is deterministic."""
        np.random.seed(42)
        data = np.random.randn(128)

        compressor = SeamAwareCompressor(involution="reverse")

        # Compress twice
        result1 = compressor.compress(data)
        result2 = compressor.compress(data)

        # Should get same results
        assert result1.coherence == result2.coherence
        assert result1.compression_ratio == result2.compression_ratio
        assert result1.exploited_symmetry == result2.exploited_symmetry

    def test_deterministic_regime_detection(self):
        """Test that regime detection is deterministic."""
        np.random.seed(42)
        time_series = np.random.randn(500)

        detector = LighthouseDetector(window_size=64, overlap=0.5)

        # Detect twice
        seams1, regimes1 = detector.detect(time_series)
        seams2, regimes2 = detector.detect(time_series)

        # Should get same results
        assert len(seams1) == len(seams2)
        assert len(regimes1) == len(regimes2)

    def test_seed_reproducibility(self):
        """Test that setting seed gives reproducible results."""

        # Run 1
        np.random.seed(42)
        data1 = np.random.randn(128)
        probe1 = SymmetryProbe(data1, involution="reverse")
        alpha1 = probe1.get_coherence()

        # Run 2 with same seed
        np.random.seed(42)
        data2 = np.random.randn(128)
        probe2 = SymmetryProbe(data2, involution="reverse")
        alpha2 = probe2.get_coherence()

        # Should be identical
        np.testing.assert_array_equal(data1, data2)
        assert alpha1 == alpha2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
