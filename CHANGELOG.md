# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Additional notebooks (Bernoulli vs Markov, Real World Case Studies)
- More comprehensive benchmarks on real datasets
- Performance optimization with numba
- Neural architecture application (seam-gated layers)

## [0.1.0] - 2026-01-08

### Added

#### Core Library
- Initial release of quotient-probes library
- `SymmetryProbe` class for analyzing involution symmetries
- MDL-based decision rules for symmetry exploitation
- Decomposition into V₊ ⊕ V₋ eigenspaces
- Support for antipodal, reverse, and reflection involutions
- Coherence parameter α computation
- Critical coherence threshold α_crit calculation
- Orientation cost models (Bernoulli, Markov, constant)

#### Applications
- **SeamAwareCompressor**: MDL-driven compression with automatic symmetry exploitation
  - Single vector and batch compression
  - Time series compression with sliding windows
  - Compression potential estimation
  - Full compression/decompression pipeline
- **AntipodalVectorDB**: Symmetry-partitioned vector search
  - 2x speedup on symmetric embeddings
  - Automatic partitioning based on coherence
  - Support for cosine, Euclidean, and dot product metrics
  - Batch search and benchmarking
- **LighthouseDetector**: Regime detection using coherence as a lighthouse signal
  - Rolling coherence computation
  - Seam detection (regime transitions)
  - Regime segmentation
  - Visualization with timeline plots

#### Interactive Notebooks
- **01_symmetry_decomposition.ipynb**: Interactive decomposition visualization
  - Antipodal, reverse, and reflection examples
  - Real-time signal explorer with sliders
  - Energy distribution analysis
  - Orthogonality and reconstruction verification
- **02_mdl_decision_rule.ipynb**: MDL boundary explorer
  - Interactive sliders for n, K_lift, α
  - "Should I exploit symmetry?" calculator widget
  - Worked examples (n=64, n=256) from paper
  - Dimension sweep and asymptotic behavior
  - Real data testing with various coherence levels

#### CLI Tool
- Command-line interface with Click
- Commands: `analyze`, `compress`, `potential`, `benchmark`, `visualize`
- Support for .npy, .npz, .txt, .csv file formats
- Batch processing capability
- Verbose output mode

#### Benchmarks
- **synthetic.py**: Comprehensive validation suite
  - Coherence sweep across α ∈ [0, 1]
  - Dimension sweep across n ∈ [16, 1024]
  - Compression performance benchmarking
  - Automatic plot generation and JSON export
  - SyntheticDataGenerator with controlled coherence

#### CI/CD
- GitHub Actions workflow with 8 jobs:
  - Multi-OS testing (Ubuntu, macOS, Windows)
  - Multi-Python version (3.8, 3.9, 3.10, 3.11)
  - Code quality checks (black, flake8, mypy, isort)
  - Coverage reporting (Codecov)
  - Benchmark execution
  - Notebook validation
  - Security scanning (safety, bandit)
  - Package building and validation

#### Tests
- Core functionality tests (`test_symmetry_probe.py`)
- Application tests (`test_applications.py`)
  - Compression tests
  - Vector search tests
  - Regime detection tests
- Integration tests (`test_integration.py`)
  - End-to-end workflows
  - Cross-module consistency
  - Reproducibility tests

#### Documentation
- Comprehensive README with examples
- CONTRIBUTING.md with development guidelines
- NumPy-style docstrings throughout
- Type hints for all public APIs
- Example code in all docstrings

#### Visualization
- MDL decision boundary plots
- Dimension sweep visualizations
- Worked examples from paper
- Interactive matplotlib widgets
- Coherence timeline plots
- Regime detection visualizations

### Dependencies
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- click >= 8.0.0

### Optional Dependencies
- **interactive**: ipywidgets, plotly, jupyter
- **dev**: pytest, pytest-cov, black, flake8, mypy, isort, sphinx
- **performance**: numba
- **data**: h5py, pandas

### Performance
- Coherence computation: O(n) time, O(n) space
- MDL decision: O(1) given coherence
- Decomposition: O(n) time, O(n) space
- Vector search with partitioning: ~2x speedup
- Compression: Adaptive based on coherence

### Breaking Changes
None (initial release)

### Deprecated
None

### Removed
None

### Fixed
None (initial release)

### Security
- Input validation on all public APIs
- Numerical stability with epsilon parameters
- Type checking with mypy
- Security scanning in CI pipeline

---

## Release Notes

### v0.1.0 - "Lighthouse Launch" 🚢

The inaugural release of Quotient Probes! This release provides:

**For Researchers:**
- Complete implementation of MDL-based symmetry exploitation
- Rigorous validation via synthetic benchmarks
- Interactive notebooks for exploring the theory
- Reference implementations for paper reproduction

**For Practitioners:**
- Production-ready compression application
- Fast vector search with symmetry partitioning
- Regime detection for time series
- CLI tool for command-line usage

**For Contributors:**
- Comprehensive test suite (90%+ coverage goal)
- CI/CD pipeline for quality assurance
- Detailed contribution guidelines
- Type-safe codebase

**Highlights:**
- 🎮 Interactive Notebook 02: The "killer demo" with live sliders
- 🗜️ SeamAware compression: Automatic MDL-driven decisions
- 🔍 Antipodal vector search: 2x speedup on symmetric embeddings
- 🌊 Lighthouse detector: Find regime transitions in time series
- 🧪 Synthetic benchmarks: Validate theory on controlled data
- 🚀 CLI tool: `quotient-probe analyze data.npy --verbose`

---

## Version History

- **v0.1.0** (2026-01-08): Initial release

---

## Upgrade Guide

### From pre-release to v0.1.0

This is the initial release, no migration needed.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## Links

- [PyPI](https://pypi.org/project/quotient-probes/) (Coming soon)
- [Documentation](https://github.com/MacMayo1993/Quotient-Probes) (GitHub README)
- [Issues](https://github.com/MacMayo1993/Quotient-Probes/issues)
- [Changelog](https://github.com/MacMayo1993/Quotient-Probes/blob/main/CHANGELOG.md)

---

[Unreleased]: https://github.com/MacMayo1993/Quotient-Probes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MacMayo1993/Quotient-Probes/releases/tag/v0.1.0
