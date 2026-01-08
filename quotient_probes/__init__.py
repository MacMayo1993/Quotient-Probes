"""
Quotient Probes: Orientation Cost in Symmetry-Adapted Hilbert Spaces

Reference implementation for the paper demonstrating MDL-based decision rules
for exploiting involution symmetries in high-dimensional data.
"""

__version__ = "0.1.0"
__author__ = "Mac Mayo"

from .core.decomposition import decompose_antisymmetric, decompose_symmetric
from .core.mdl_decision import compute_orientation_cost, mdl_decision_rule
from .core.symmetry_probe import SymmetryProbe

__all__ = [
    "SymmetryProbe",
    "mdl_decision_rule",
    "compute_orientation_cost",
    "decompose_symmetric",
    "decompose_antisymmetric",
]
