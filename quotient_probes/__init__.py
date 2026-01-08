"""
Quotient Probes: Orientation Cost in Symmetry-Adapted Hilbert Spaces

Reference implementation for the paper demonstrating MDL-based decision rules
for exploiting involution symmetries in high-dimensional data.
"""

__version__ = "0.1.0"
__author__ = "Mac Mayo"

from .core.symmetry_probe import SymmetryProbe
from .core.mdl_decision import mdl_decision_rule, compute_orientation_cost
from .core.decomposition import decompose_symmetric, decompose_antisymmetric

__all__ = [
    "SymmetryProbe",
    "mdl_decision_rule",
    "compute_orientation_cost",
    "decompose_symmetric",
    "decompose_antisymmetric",
]
