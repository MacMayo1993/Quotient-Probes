"""
Tests for CLI --json output functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from quotient_probes.cli import main


@pytest.fixture
def sample_data_file():
    """Create a temporary .npy file with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        # Create a vector with known coherence properties
        data = np.random.randn(64)
        np.save(f.name, data)
        yield f.name
        # Cleanup
        Path(f.name).unlink()


@pytest.fixture
def batch_data_file():
    """Create a temporary .npy file with batch data."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        # Create batch of vectors
        data = np.random.randn(5, 64)
        np.save(f.name, data)
        yield f.name
        # Cleanup
        Path(f.name).unlink()


class TestCLIJsonOutput:
    """Test suite for CLI --json flag functionality."""

    def test_analyze_single_vector_json_output(self, sample_data_file):
        """Test that --json flag produces valid JSON for single vector."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", sample_data_file, "--json"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Parse JSON output
        output = json.loads(result.output)

        # Verify required fields
        assert "file" in output
        assert "coherence" in output
        assert "should_exploit" in output
        assert "bit_savings" in output
        assert "dimension" in output
        assert "K_lift" in output
        assert "alpha_critical" in output
        assert "delta_L" in output

        # Verify types
        assert isinstance(output["coherence"], float)
        assert isinstance(output["should_exploit"], bool)
        assert isinstance(output["bit_savings"], float)
        assert isinstance(output["dimension"], int)
        assert isinstance(output["K_lift"], float)

        # Verify coherence is in valid range [0, 1]
        assert 0 <= output["coherence"] <= 1

    def test_analyze_batch_json_output(self, batch_data_file):
        """Test that --json flag produces valid JSON for batch data."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", batch_data_file, "--json"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Parse JSON output
        output = json.loads(result.output)

        # Verify batch structure
        assert "batch_size" in output
        assert "summary" in output
        assert "results" in output

        # Verify summary fields
        summary = output["summary"]
        assert "vectors_analyzed" in summary
        assert "exploitation_count" in summary
        assert "exploitation_rate" in summary
        assert "avg_coherence" in summary
        assert "std_coherence" in summary
        assert "total_bit_savings" in summary

        # Verify results array
        assert isinstance(output["results"], list)
        assert len(output["results"]) == output["batch_size"]

        # Verify each result has required fields
        for result_item in output["results"]:
            assert "index" in result_item
            assert "alpha" in result_item
            assert "should_exploit" in result_item
            assert "bit_savings" in result_item
            assert "dimension" in result_item

    def test_json_output_with_involution_option(self, sample_data_file):
        """Test JSON output with different involution types."""
        runner = CliRunner()

        # Test with reverse involution
        result = runner.invoke(
            main, ["analyze", sample_data_file, "--json", "--involution=reverse"]
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["involution"] == "reverse"

        # Test with antipodal involution
        result = runner.invoke(
            main, ["analyze", sample_data_file, "--json", "--involution=antipodal"]
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["involution"] == "antipodal"

    def test_json_output_with_orientation_model(self, sample_data_file):
        """Test JSON output with different orientation models."""
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                "analyze",
                sample_data_file,
                "--json",
                "--orientation-model=bernoulli",
            ],
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["orientation_model"] == "bernoulli"

    def test_json_output_with_custom_k_lift(self, sample_data_file):
        """Test JSON output with custom K_lift value."""
        runner = CliRunner()

        result = runner.invoke(
            main, ["analyze", sample_data_file, "--json", "--k-lift=0.5"]
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["K_lift"] == 0.5

    def test_json_no_emoji_or_formatting(self, sample_data_file):
        """Test that JSON output contains no emoji or formatting characters."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", sample_data_file, "--json"])

        assert result.exit_code == 0

        # Verify output is pure JSON (no emoji, no formatting chars)
        assert "📊" not in result.output
        assert "✅" not in result.output
        assert "❌" not in result.output
        assert "=" * 60 not in result.output

        # Verify it's valid JSON
        output = json.loads(result.output)
        assert output is not None

    def test_json_output_parseable_by_jq(self, sample_data_file):
        """Test that JSON output structure is compatible with jq queries."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", sample_data_file, "--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)

        # Test common jq-style access patterns
        assert output["coherence"] is not None  # .coherence
        assert output["should_exploit"] is not None  # .should_exploit
        assert isinstance(output["bit_savings"], (int, float))  # .bit_savings

    def test_batch_json_exploitation_rate(self, batch_data_file):
        """Test that batch JSON correctly calculates exploitation rate."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", batch_data_file, "--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)

        # Manually calculate exploitation rate
        exploit_count = sum(r["should_exploit"] for r in output["results"])
        expected_rate = exploit_count / output["batch_size"]

        assert abs(output["summary"]["exploitation_rate"] - expected_rate) < 1e-10

    def test_json_coherence_matches_decision(self, sample_data_file):
        """Test that coherence and decision are consistent in JSON output."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", sample_data_file, "--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)

        # Verify decision logic consistency
        alpha = output["coherence"]
        alpha_crit = output["alpha_critical"]
        should_exploit = output["should_exploit"]

        if alpha > alpha_crit:
            assert should_exploit is True
        else:
            assert should_exploit is False


class TestCLIJsonEdgeCases:
    """Test edge cases for JSON output."""

    def test_json_with_perfect_symmetry(self):
        """Test JSON output with perfectly symmetric data."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            # Create perfectly palindromic sequence
            data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
            np.save(f.name, data)

            runner = CliRunner()
            result = runner.invoke(
                main, ["analyze", f.name, "--json", "--involution=reverse"]
            )

            assert result.exit_code == 0
            output = json.loads(result.output)

            # Perfect symmetry should have coherence close to 1
            assert output["coherence"] > 0.9
            assert output["should_exploit"] is True

            Path(f.name).unlink()

    def test_json_with_zero_symmetry(self):
        """Test JSON output with completely antisymmetric data."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            # Create data with minimal symmetry
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            np.save(f.name, data)

            runner = CliRunner()
            result = runner.invoke(
                main, ["analyze", f.name, "--json", "--involution=reverse"]
            )

            assert result.exit_code == 0
            output = json.loads(result.output)

            # Low symmetry data
            assert output["coherence"] < 0.6

            Path(f.name).unlink()

    def test_json_output_is_deterministic(self, sample_data_file):
        """Test that running CLI multiple times produces identical JSON."""
        runner = CliRunner()

        # Run twice with same parameters
        result1 = runner.invoke(main, ["analyze", sample_data_file, "--json"])
        result2 = runner.invoke(main, ["analyze", sample_data_file, "--json"])

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Parse and compare (accounting for floating point)
        output1 = json.loads(result1.output)
        output2 = json.loads(result2.output)

        assert abs(output1["coherence"] - output2["coherence"]) < 1e-10
        assert output1["should_exploit"] == output2["should_exploit"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
