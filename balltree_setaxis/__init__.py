"""
Ball tree with global PCA alignment for rotation-equivariant construction.

Like balltree by Maksim Zhdanov, but each batch is PCA-aligned before building
the tree, so the result is rotation equivariant (except in degenerate cases:
colinear points, symmetric configurations, or splits exactly on a point/line).
Cython + OpenMP for fast parallel construction.
"""

__version__ = "0.1.0"

try:
    from .balltree import (
        build_balltree,
        partition_balltree,
        build_balltree_with_rotations,
        build_balltree_with_offsets,
        partition_balltree_with_offsets,
        build_balltree_with_idx,
        partition_balltree_with_idx,
        generate_rotation_matrix,
    )
except ImportError as e:
    raise ImportError(
        "Could not import compiled Cython modules. "
        "Please ensure the package was properly installed with: "
        "pip install balltree-setaxis"
    ) from e

__all__ = [
    "build_balltree",
    "partition_balltree", 
    "build_balltree_with_rotations",
    "build_balltree_with_offsets",
    "partition_balltree_with_offsets",
    "build_balltree_with_idx",
    "partition_balltree_with_idx",
    "generate_rotation_matrix",
]