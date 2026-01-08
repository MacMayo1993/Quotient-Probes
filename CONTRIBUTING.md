# Contributing to Quotient Probes

Thank you for your interest in contributing to Quotient Probes! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept criticism gracefully
- Prioritize the community's best interests

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of symmetry analysis and MDL principles (helpful but not required)

### Setting Up Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Quotient-Probes.git
   cd Quotient-Probes
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev,interactive]"
   ```

4. **Verify installation:**
   ```bash
   pytest tests/
   ```

---

## Development Workflow

### Branch Naming

Use descriptive branch names with prefixes:

- `feature/`: New features (`feature/add-fft-involution`)
- `bugfix/`: Bug fixes (`bugfix/fix-coherence-calculation`)
- `docs/`: Documentation (`docs/update-readme`)
- `test/`: Test additions (`test/add-compression-tests`)
- `refactor/`: Code refactoring (`refactor/simplify-mdl-logic`)

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(compression): add LZ77 backend for SeamAwareCompressor

Implement LZ77 compression as an optional backend for the compressor.
This provides better compression ratios for highly redundant data.

Closes #42
```

```
fix(mdl): correct coherence computation for edge case

Fixed division by zero when input vector has zero norm.
Added epsilon parameter for numerical stability.
```

---

## Code Style

### Python Style Guide

We follow **PEP 8** with some modifications:

#### Formatting

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Single quotes for strings, double quotes for docstrings
- **Imports**: Use absolute imports, group standard/third-party/local

#### Tools

Run these before committing:

```bash
# Format code
black quotient_probes tests benchmarks

# Sort imports
isort quotient_probes tests benchmarks

# Lint
flake8 quotient_probes tests benchmarks

# Type check
mypy quotient_probes --ignore-missing-imports
```

#### Type Hints

Use type hints for all public functions:

```python
def compute_coherence(
    x: np.ndarray,
    sigma: Callable[[np.ndarray], np.ndarray],
    epsilon: float = 1e-10
) -> float:
    """Compute coherence parameter."""
    ...
```

### Docstring Style

Use **NumPy-style docstrings**:

```python
def mdl_decision_rule(
    alpha: float,
    n: int,
    K_lift: float = 1.0,
    return_details: bool = False
) -> Union[bool, Tuple[bool, dict]]:
    """
    MDL-based decision: Should we exploit symmetry?

    Decision rule:
        Exploit if α > α_crit = (n + K_lift)/(2n)

    Parameters
    ----------
    alpha : float
        Observed coherence (0 to 1)
    n : int
        Ambient dimension
    K_lift : float, optional
        Orientation cost (default: 1.0)
    return_details : bool, optional
        If True, return (decision, details_dict)

    Returns
    -------
    bool or tuple
        Boolean decision or (decision, details) if return_details=True

    Examples
    --------
    >>> mdl_decision_rule(alpha=0.6, n=64, K_lift=1.0)
    True

    Notes
    -----
    Based on Theorem 1 from the paper.

    See Also
    --------
    critical_coherence : Compute α_crit
    description_length_difference : Compute ΔL
    """
    ...
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quotient_probes --cov-report=term --cov-report=html

# Run specific test file
pytest tests/test_symmetry_probe.py

# Run specific test
pytest tests/test_symmetry_probe.py::TestSymmetryProbe::test_decomposition
```

### Writing Tests

- Place tests in `tests/` directory
- Use `test_` prefix for test files and functions
- Group related tests in classes
- Use descriptive test names

**Example:**

```python
import pytest
import numpy as np
from quotient_probes import SymmetryProbe

class TestSymmetryProbe:
    """Tests for SymmetryProbe class."""

    def test_coherence_bounds(self):
        """Test that coherence is always in [0, 1]."""
        np.random.seed(42)
        for _ in range(100):
            x = np.random.randn(64)
            probe = SymmetryProbe(x)
            alpha = probe.get_coherence()
            assert 0 <= alpha <= 1

    def test_decomposition_reconstruction(self):
        """Test that x = x₊ + x₋."""
        x = np.random.randn(128)
        probe = SymmetryProbe(x, involution='reverse')
        x_plus, x_minus = probe.decompose()
        np.testing.assert_allclose(x, x_plus + x_minus)
```

### Test Coverage

- Aim for **90%+ coverage**
- Cover edge cases:
  - Empty inputs
  - Single-element arrays
  - Very large dimensions
  - Zero vectors
  - Extreme coherence values (0, 1)

---

## Documentation

### Code Documentation

- **Public APIs**: Must have docstrings
- **Private functions**: Docstrings recommended
- **Complex algorithms**: Add inline comments explaining the logic

### Notebook Documentation

When adding notebooks:

- Start with clear learning objectives
- Include interactive elements (sliders, widgets)
- Add worked examples
- Provide references to paper sections

### README Updates

Update the README when:

- Adding new features
- Changing public API
- Adding dependencies
- Updating installation instructions

---

## Pull Request Process

### Before Submitting

1. **Run tests:**
   ```bash
   pytest --cov=quotient_probes
   ```

2. **Run linters:**
   ```bash
   black quotient_probes tests
   flake8 quotient_probes tests
   mypy quotient_probes
   ```

3. **Update CHANGELOG.md** with your changes

4. **Update documentation** if you changed public APIs

### PR Template

```markdown
## Description

Brief description of changes.

## Motivation

Why is this change needed?

## Changes Made

- Added X
- Fixed Y
- Refactored Z

## Testing

- [ ] Added tests for new functionality
- [ ] All tests passing
- [ ] Coverage maintained or improved

## Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow convention
- [ ] No breaking changes (or clearly documented)

## Related Issues

Closes #XX
```

### Review Process

1. **Automated checks**: CI must pass (tests, linting, coverage)
2. **Code review**: At least one maintainer approval required
3. **Documentation review**: Ensure docs are clear and complete
4. **Performance check**: No significant performance regressions

### Merge Requirements

- ✅ All CI checks passing
- ✅ Approved by maintainer
- ✅ Up to date with main branch
- ✅ No merge conflicts
- ✅ CHANGELOG updated

---

## Development Tips

### Debugging

Use pytest fixtures for common test data:

```python
@pytest.fixture
def random_signal():
    """Generate random time series."""
    np.random.seed(42)
    return np.random.randn(256)
```

### Benchmarking

Add benchmarks for performance-critical code:

```python
def test_compression_performance(benchmark):
    """Benchmark compression speed."""
    data = np.random.randn(1000, 512)
    compressor = SeamAwareCompressor()

    result = benchmark(compressor.compress, data)
    assert result.compression_ratio > 1.0
```

### Profiling

Use cProfile for performance analysis:

```bash
python -m cProfile -o profile.stats benchmarks/synthetic.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/MacMayo1993/Quotient-Probes/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MacMayo1993/Quotient-Probes/discussions)
- **Email**: (Add maintainer email)

---

## Recognition

Contributors will be:

- Listed in `AUTHORS.md`
- Mentioned in release notes
- Credited in academic citations (for significant contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Quotient Probes!** 🎉
