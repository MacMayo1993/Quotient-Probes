"""
Generate sample datasets for testing and reproducibility.

Run this script to create sample data files in data/samples/
"""

import numpy as np
from pathlib import Path

# Create samples directory
samples_dir = Path(__file__).parent / 'samples'
samples_dir.mkdir(exist_ok=True)

print("Generating sample datasets...")

# 1. Symmetric time series (palindromic)
print("  - symmetric_timeseries.npy")
t = np.linspace(0, 2*np.pi, 128)
symmetric_signal = np.sin(t) + 0.5*np.cos(2*t)
# Make it symmetric by averaging with reverse
symmetric_signal = (symmetric_signal + symmetric_signal[::-1]) / 2
np.save(samples_dir / 'symmetric_timeseries.npy', symmetric_signal)

# 2. Antisymmetric time series
print("  - antisymmetric_timeseries.npy")
antisymmetric_signal = (symmetric_signal - symmetric_signal[::-1]) / 2
np.save(samples_dir / 'antisymmetric_timeseries.npy', antisymmetric_signal)

# 3. Mixed coherence time series
print("  - mixed_timeseries.npy")
mixed_signal = 0.6 * symmetric_signal + 0.4 * np.random.randn(128)
np.save(samples_dir / 'mixed_timeseries.npy', mixed_signal)

# 4. Embedding vectors (simulated CLIP-like)
print("  - embeddings_symmetric.npy")
n_vectors = 100
dim = 256
embeddings = np.random.randn(n_vectors, dim)
# Add antipodal symmetry
for i in range(0, n_vectors, 2):
    if i + 1 < n_vectors:
        embeddings[i+1] = -embeddings[i] + np.random.randn(dim) * 0.1
# Normalize
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
np.save(samples_dir / 'embeddings_symmetric.npy', embeddings)

# 5. Random embeddings (no symmetry)
print("  - embeddings_random.npy")
random_embeddings = np.random.randn(100, 256)
random_embeddings = random_embeddings / np.linalg.norm(random_embeddings, axis=1, keepdims=True)
np.save(samples_dir / 'embeddings_random.npy', random_embeddings)

# 6. Regime-structured time series
print("  - regime_timeseries.npy")
# Create 3 regimes with different symmetry properties
regime1 = np.tile([1, 2, 3, 2, 1], 40)  # Symmetric (200 samples)
regime2 = np.tile([1, 2, 0, -2, -1], 40)  # Antisymmetric (200 samples)
regime3 = np.tile([1, 2, 3, 2, 1], 40)  # Symmetric again (200 samples)
regime_series = np.concatenate([regime1, regime2, regime3])
np.save(samples_dir / 'regime_timeseries.npy', regime_series)

# 7. High-coherence vector
print("  - high_coherence_vector.npy")
high_coh = np.random.randn(128)
high_coh = (high_coh + high_coh[::-1]) / 2 + np.random.randn(128) * 0.1
np.save(samples_dir / 'high_coherence_vector.npy', high_coh)

# 8. Low-coherence vector
print("  - low_coherence_vector.npy")
low_coh = np.random.randn(128)
np.save(samples_dir / 'low_coherence_vector.npy', low_coh)

# Create README
readme_content = """# Sample Datasets

This directory contains sample datasets for testing and demonstrations.

## Files

### Time Series

- **symmetric_timeseries.npy**: Palindromic signal (α ≈ 1.0)
  - Shape: (128,)
  - Use case: Test high-coherence detection

- **antisymmetric_timeseries.npy**: Anti-palindromic signal (α ≈ 0.0)
  - Shape: (128,)
  - Use case: Test low-coherence detection

- **mixed_timeseries.npy**: Mixed symmetry (α ≈ 0.5-0.7)
  - Shape: (128,)
  - Use case: Test decision boundary

- **regime_timeseries.npy**: Multiple regime transitions
  - Shape: (600,)
  - Regimes: Symmetric → Antisymmetric → Symmetric
  - Use case: Test lighthouse detector

### Embeddings

- **embeddings_symmetric.npy**: Antipodal-symmetric vectors
  - Shape: (100, 256)
  - ~50% of vectors are near-opposite pairs
  - Use case: Test vector search speedup

- **embeddings_random.npy**: Random vectors (no symmetry)
  - Shape: (100, 256)
  - Use case: Baseline comparison

### Single Vectors

- **high_coherence_vector.npy**: High α vector (α > 0.7)
  - Shape: (128,)
  - Use case: Test exploitation decision

- **low_coherence_vector.npy**: Low α vector (α < 0.4)
  - Shape: (128,)
  - Use case: Test ignore decision

## Usage

### Python API

```python
import numpy as np
from quotient_probes import SymmetryProbe

# Load sample
data = np.load('data/samples/symmetric_timeseries.npy')

# Analyze
probe = SymmetryProbe(data, involution='reverse')
alpha, savings, decision = probe.analyze()
print(f"Coherence: {alpha:.3f}, Decision: {'EXPLOIT' if decision else 'IGNORE'}")
```

### CLI

```bash
# Analyze sample
quotient-probe analyze data/samples/high_coherence_vector.npy --verbose

# Compress sample
quotient-probe compress data/samples/embeddings_symmetric.npy

# Detect regimes
python -c "
import numpy as np
from quotient_probes.applications.regime_detection import LighthouseDetector

data = np.load('data/samples/regime_timeseries.npy')
detector = LighthouseDetector()
seams, regimes = detector.detect(data)
print(f'Found {len(seams)} seams and {len(regimes)} regimes')
"
```

## Regenerating Samples

To regenerate all samples:

```bash
python data/generate_samples.py
```

This will overwrite existing files.

## License

These sample datasets are provided for testing and educational purposes under the MIT License.
"""

with open(samples_dir / 'README.md', 'w') as f:
    f.write(readme_content)

print("\n✅ Sample datasets generated successfully!")
print(f"📁 Location: {samples_dir}")
print("\nGenerated files:")
for file in sorted(samples_dir.glob('*.npy')):
    size_kb = file.stat().st_size / 1024
    print(f"  - {file.name} ({size_kb:.1f} KB)")
