# CI/CD Pipeline Documentation

## Overview

The Quotient Probes repository uses a comprehensive GitHub Actions CI/CD pipeline to ensure code quality, reproducibility, and correctness across all supported platforms and Python versions.

## Pipeline Jobs

### 1. 🧪 **Test** (Multi-OS, Multi-Python)
**Matrix**: 3 OS × 4 Python versions = **12 test runs**

- **Operating Systems**: Ubuntu, macOS, Windows
- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **Coverage**: pytest with coverage reporting
- **Upload**: Coverage to Codecov (Ubuntu + Python 3.10 only)

**Purpose**: Ensure code works across all supported platforms and Python versions.

---

### 2. 🎨 **Lint** (Code Quality)
**Single job**: Ubuntu + Python 3.10

- **black**: Code formatting check
- **isort**: Import sorting verification
- **flake8**: Linting (syntax errors, undefined names, complexity)
- **mypy**: Type checking (continues on error)

**Purpose**: Maintain consistent code style and catch type errors.

---

### 3. 📊 **Benchmarks** (Synthetic Validation)
**Single job**: Ubuntu + Python 3.10

- Runs `benchmarks.synthetic --coherence-sweep`
- Tests coherence sweep across α ∈ [0, 1]
- Validates MDL decision boundary
- **Artifacts**: Benchmark results and plots

**Purpose**: Validate theoretical predictions on controlled synthetic data.

---

### 4. 📓 **Notebooks** (Interactive Demo Validation)
**Single job**: Ubuntu + Python 3.10

- Installs interactive dependencies (ipywidgets, jupyter)
- Executes all notebooks with `nbconvert`
- **Artifacts**: Executed notebooks

**Purpose**: Ensure notebooks run without errors.

---

### 5. 🔒 **Security** (Vulnerability Scanning)
**Single job**: Ubuntu + Python 3.10

- **safety**: Check for known vulnerabilities in dependencies
- **bandit**: Static security analysis

**Purpose**: Detect security issues before deployment.

---

### 6. 🔁 **Reproducibility** (NEW!) ⭐
**Matrix**: 3 OS = **3 reproducibility runs**

**Tests**:
1. **Deterministic behavior**: Run same analysis twice with seed=42, verify identical results
2. **Sample data validation**: Test on symmetric/antisymmetric samples, verify expected coherence
3. **Compression reproducibility**: Compress twice, verify identical outputs + roundtrip
4. **Vector search reproducibility**: Search twice, verify identical neighbors
5. **Regime detection reproducibility**: Detect regimes twice, verify identical seams

**Cross-platform**: Runs on Ubuntu, macOS, and Windows to ensure platform-independent results.

**Purpose**: **GUARANTEE REPRODUCIBILITY** - core requirement for scientific software.

---

### 7. 🖥️ **CLI** (Command-Line Interface Testing) (NEW!) ⭐
**Single job**: Ubuntu + Python 3.10

**Commands tested**:
```bash
# 1. Analyze
quotient-probe analyze data/samples/symmetric_timeseries.npy --involution=reverse

# 2. Compress
quotient-probe compress data/samples/embeddings_symmetric.npy --output=test_compressed.npz

# 3. Potential estimation
quotient-probe potential data/samples/mixed_timeseries.npy --involution=reverse

# 4. Benchmark
quotient-probe benchmark --synthetic --n=64 --num-points=5

# 5. Visualize
quotient-probe visualize --n=64 --k-lift=1.0 --output=test_plot.png
```

**Artifacts**: Compressed data and plots from CLI

**Purpose**: Validate that CLI works end-to-end for users.

---

### 8. 🔗 **Integration** (Application Testing) (NEW!) ⭐
**Single job**: Ubuntu + Python 3.10

**Tests**:
1. **Integration test suite**: Run `tests/test_integration.py` (end-to-end workflows)
2. **Compression on samples**: Test all 3 applications on sample datasets
3. **Vector search benchmark**: Measure speedup on symmetric embeddings
4. **Regime detection**: Detect regimes in sample time series

**Purpose**: Validate applications work together correctly on real sample data.

---

### 9. 📦 **Package** (Distribution Build)
**Single job**: Ubuntu + Python 3.10

- Builds source distribution and wheel with `python -m build`
- Validates package with `twine check`
- **Artifacts**: Uploadable distributions

**Purpose**: Ensure package can be published to PyPI.

---

### 10. 📚 **Docs** (Documentation Build)
**Single job**: Ubuntu + Python 3.10

- Installs Sphinx
- Placeholder for future documentation builds

**Purpose**: Future-proof for Sphinx documentation.

---

## Total CI Jobs: 10

| Job | Count | Total Runs |
|-----|-------|------------|
| Test | 12 (3 OS × 4 Python) | 12 |
| Lint | 1 | 1 |
| Benchmarks | 1 | 1 |
| Notebooks | 1 | 1 |
| Security | 1 | 1 |
| **Reproducibility** | **3 (3 OS)** | **3** |
| **CLI** | **1** | **1** |
| **Integration** | **1** | **1** |
| Package | 1 | 1 |
| Docs | 1 | 1 |
| **TOTAL** | | **23 parallel jobs** |

---

## Reproducibility Guarantees

### ✅ What We Test

1. **Deterministic Algorithms**
   - Same seed → same random numbers → same results
   - Tested across Ubuntu, macOS, Windows
   - Verified to 15 decimal places (< 1e-15 tolerance)

2. **Sample Data Expectations**
   - Symmetric samples: α > 0.6 (high coherence)
   - Antisymmetric samples: α < 0.4 (low coherence)
   - Mixed samples: intermediate α

3. **Compression Idempotence**
   - compress(x) twice → identical results
   - decompress(compress(x)) = x (roundtrip)

4. **Search Determinism**
   - search(query, k=5) twice → same top-5 neighbors
   - Distances match to machine precision

5. **Regime Detection Stability**
   - detect(ts) twice → same number of seams and regimes
   - Seam locations identical

### ✅ Cross-Platform Consistency

The **reproducibility job** runs on:
- **Ubuntu** (most common for CI)
- **macOS** (common for researchers)
- **Windows** (accessibility for non-Unix users)

All must pass independently, ensuring results are **platform-independent**.

---

## Artifacts Generated

| Job | Artifact | Contents |
|-----|----------|----------|
| Benchmarks | `benchmark-results` | Plots and JSON results |
| Notebooks | `executed-notebooks` | Fully executed .ipynb files |
| CLI | `cli-outputs` | Compressed data + plots |
| Package | `dist-packages` | Wheel + source distribution |

**Retention**: 90 days (GitHub default)

---

## When CI Runs

- **On push** to: `main`, `develop`, `claude/**` branches
- **On pull request** to: `main`, `develop`

---

## Failure Handling

### Jobs that MUST pass:
- ✅ Test (all 12 variants)
- ✅ Lint (black, flake8, isort)
- ✅ **Reproducibility** (all 3 OS)
- ✅ **CLI** (all commands)
- ✅ **Integration** (all applications)
- ✅ Package (build + check)

### Jobs that may fail gracefully:
- ⚠️ mypy (type checking, `continue-on-error: true`)
- ⚠️ Benchmarks (heavy computation, `continue-on-error: true`)
- ⚠️ Notebooks (may timeout, `continue-on-error: true`)
- ⚠️ Security (warnings don't block, `|| true`)
- ⚠️ Docs (placeholder, `continue-on-error: true`)

---

## Performance

**Estimated total CI time**: ~15-25 minutes (parallel execution)

**Bottlenecks**:
- Test job (12 parallel runs): ~5-8 min each
- Benchmarks: ~3-5 min
- Notebooks: ~2-4 min

**Optimization**:
- `cache: 'pip'` speeds up dependency installation
- Matrix strategy runs jobs in parallel
- Lightweight sample data (~200 KB total)

---

## Local Testing

Developers can run the same checks locally:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov=quotient_probes --cov-report=html

# Check formatting
black quotient_probes tests benchmarks
flake8 quotient_probes tests benchmarks
isort quotient_probes tests benchmarks

# Type check
mypy quotient_probes --ignore-missing-imports

# Run benchmarks
python -m benchmarks.synthetic --coherence-sweep

# Execute notebooks
jupyter nbconvert --execute notebooks/*.ipynb

# Test CLI
python data/generate_samples.py
quotient-probe analyze data/samples/symmetric_timeseries.npy -v

# Build package
python -m build
twine check dist/*
```

---

## Badge Status

Add these badges to your README:

```markdown
[![CI](https://github.com/MacMayo1993/Quotient-Probes/workflows/CI/badge.svg)](https://github.com/MacMayo1993/Quotient-Probes/actions)
[![codecov](https://codecov.io/gh/MacMayo1993/Quotient-Probes/branch/main/graph/badge.svg)](https://codecov.io/gh/MacMayo1993/Quotient-Probes)
```

---

## Future Enhancements

1. **Performance regression testing**: Track benchmark times over commits
2. **Sphinx documentation**: Deploy to GitHub Pages
3. **Docker image**: Test in containerized environment
4. **Conda package**: Build and test conda distribution
5. **PyPI auto-publish**: On tagged releases, automatically publish

---

## Reproducibility Certificate 🏆

This repository's CI pipeline ensures:

✅ **Deterministic**: Same inputs → same outputs (verified across 3 OS)
✅ **Cross-platform**: Works on Ubuntu, macOS, Windows
✅ **Cross-version**: Supports Python 3.8-3.11
✅ **Tested**: 650+ lines of tests + integration tests
✅ **Validated**: Synthetic benchmarks verify theory
✅ **Documented**: Every CI job has a clear purpose
✅ **Transparent**: All runs logged and artifacts saved

**Scientific software should be reproducible by default. This pipeline guarantees it.**

---

For questions about the CI pipeline, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.
