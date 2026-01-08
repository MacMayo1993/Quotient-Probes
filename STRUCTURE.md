# Quotient-Probes Repository Structure

## Overview
Reference implementation and demonstrations for "Quotient Probes: Orientation Cost in Symmetry-Adapted Hilbert Spaces"

## Directory Layout

```
quotient-probes/
├── quotient_probes/              # Core Python package
│   ├── __init__.py
│   ├── core/                     # Core algorithms
│   │   ├── __init__.py
│   │   ├── symmetry_probe.py    # Main SymmetryProbe class
│   │   ├── mdl_decision.py      # MDL decision rule implementation
│   │   ├── decomposition.py     # V₊ ⊕ V₋ decomposition utilities
│   │   └── involutions.py       # Involution operators (antipodal, etc.)
│   ├── visualization/            # Interactive visualizers
│   │   ├── __init__.py
│   │   ├── mdl_boundary.py      # ΔL(α) vs K_lift plotter
│   │   └── symmetry_plots.py    # Symmetry decomposition visualizations
│   ├── benchmarks/               # Benchmark suite
│   │   ├── __init__.py
│   │   ├── synthetic.py         # Synthetic signals with controlled α
│   │   ├── eeg.py               # EEG data handlers
│   │   ├── financial.py         # Financial time series
│   │   └── audio.py             # Audio waveform processing
│   └── applications/             # Application prototypes
│       ├── __init__.py
│       ├── compression.py       # SeamAware compression demo
│       ├── vector_search.py     # Antipodal database
│       ├── regime_detection.py  # Lighthouse-style seam finder
│       └── neural_layers.py     # Seam-gated layer implementation
│
├── notebooks/                    # Pedagogical Jupyter notebooks
│   ├── 01_symmetry_decomposition.ipynb
│   ├── 02_mdl_decision_rule.ipynb
│   ├── 03_bernoulli_vs_markov.ipynb
│   └── 04_real_world_case_studies.ipynb
│
├── examples/                     # Standalone example scripts
│   ├── quick_start.py
│   ├── synthetic_demo.py
│   ├── eeg_analysis.py
│   └── financial_regimes.py
│
├── tests/                        # Test suite
│   ├── test_symmetry_probe.py
│   ├── test_mdl_decision.py
│   ├── test_decomposition.py
│   └── test_benchmarks.py
│
├── data/                         # Sample datasets
│   ├── synthetic/
│   ├── eeg_samples/
│   └── financial_samples/
│
├── docs/                         # Documentation
│   ├── theory_overview.md       # Theory summary linking to paper
│   ├── api_reference.md         # API documentation
│   ├── tutorials/
│   └── applications.md          # Application guide
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── README.md                     # Main documentation
├── LICENSE
└── .gitignore
```

## Implementation Priority

### Phase 1: Core Foundation (Immediate)
1. Core package structure
2. SymmetryProbe API implementation
3. MDL decision rule module
4. Basic tests

### Phase 2: Visualization & Pedagogy (Week 1)
1. Interactive MDL boundary visualizer
2. First two notebooks (01, 02)
3. Synthetic benchmark suite

### Phase 3: Applications (Week 2)
1. Compression demo
2. Vector search demo
3. Remaining notebooks (03, 04)

### Phase 4: Advanced Demos (Week 3)
1. Regime detection
2. Neural architecture demo
3. Real-world benchmarks (EEG, financial)

### Phase 5: Polish & Documentation (Week 4)
1. Comprehensive README
2. API documentation
3. Tutorial guides
4. arXiv linkage

## Key Design Principles

1. **Clean API**: Simple, intuitive interface matching the paper's mathematical clarity
2. **Pedagogical First**: Every component should teach the core concepts
3. **Reproducible**: All results and figures should be regeneratable
4. **Minimal Dependencies**: Only essential packages (numpy, scipy, matplotlib)
5. **Interactive**: Visualizations should be explorable and intuitive
