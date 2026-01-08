"""
Setup script for quotient-probes package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Quotient Probes: Orientation Cost in Symmetry-Adapted Hilbert Spaces"

setup(
    name="quotient-probes",
    version="0.1.0",
    author="Mac Mayo",
    author_email="",
    description="MDL-based decision rules for exploiting involution symmetries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MacMayo1993/Quotient-Probes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "interactive": [
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "performance": [
            "numba>=0.54.0",
        ],
        "data": [
            "h5py>=3.0.0",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quotient-probe=quotient_probes.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "symmetry",
        "mdl",
        "minimum description length",
        "involution",
        "quotient space",
        "compression",
        "information theory",
    ],
)
