# balltree_setaxis

Like [balltree](https://github.com/maxxxzdn/balltree) by Maksim Zhdanov (maxxxzdn), but with **global PCA alignment first**, so construction is **rotation equivariant**—the same tree structure (up to permutation) is produced regardless of how the point cloud is rotated. Equivariance holds in typical cases; degenerate cases (e.g. colinear points, symmetric configurations, or splits exactly on a hyperplane through the data) can break uniqueness.

Fast, parallel ball tree implementation with PyTorch integration, built with Cython and OpenMP.

## Features

- **Rotation equivariant**: PCA-align each batch before building the tree so the result is invariant to global rotation (modulo degeneracies).
- **High performance**: C++ with OpenMP parallelization.
- **PyTorch integration**: Works with PyTorch tensors.
- **Flexible API**: Batch builds, offsets, rotations, and level partitioning.

## Installation

### From Git

```bash
pip install "git+https://github.com/CoenvdE/balltree_setaxis.git"
```

### From source

```bash
git clone https://github.com/CoenvdE/balltree_setaxis.git
cd balltree_setaxis
pip install -e .
```

### Requirements

- Python >= 3.8
- NumPy, PyTorch
- Cython (for building from source)

### System dependencies

**macOS**

```bash
brew install libomp
```

**Linux**  
OpenMP is usually provided by GCC (`libgomp`). If needed: `sudo apt install libomp-dev` (or equivalent).

### OpenMP and PyTorch

If OpenMP from this package and PyTorch conflict (e.g. crashes or wrong behaviour when using both), force the process to use PyTorch’s OpenMP. In a virtualenv on macOS:

```bash
export TORCH_OMP_DIR="$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib"
export DYLD_LIBRARY_PATH="$TORCH_OMP_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Use your venv’s Python version in place of `3.11` if it differs. You can add these to your shell profile or run them before starting Python.

## Quick start

```python
import torch
import balltree_setaxis

# Data: (num_samples, num_features)
data = torch.randn(1000, 3)
batch_idx = torch.zeros(1000, dtype=torch.long)  # single batch

# Build ball trees with PCA alignment (rotation equivariant)
tree_indices, tree_mask, rotation_matrices = balltree_setaxis.build_balltree_with_offsets(
    data.numpy(), 
    torch.tensor([0, 1000]).numpy()  # batch boundaries
)

# Or use batch indices
tree_indices, tree_mask, rotation_matrices = balltree_setaxis.build_balltree(data.numpy(), batch_idx.numpy())

# With rotations for cross-ball interactions
tree_idx, tree_mask, rot_indices = balltree_setaxis.build_balltree_with_rotations(
    data.numpy(), batch_idx.numpy(),
    strides=[2, 2], ball_sizes=[64, 32, 16], angle=45.0
)
```

## API overview

- **`build_balltree_with_offsets(data, batch_offsets)`** — Build trees for contiguous batches with PCA alignment. Returns `(tree_indices, tree_mask, rotation_matrices)`.
- **`partition_balltree_with_offsets(data, batch_offsets, target_level)`** — Partition to a given level.
- **`build_balltree(data, batch_idx)`** / **`partition_balltree(...)`** — Same idea with batch indices.
- **`build_balltree_with_rotations(data, batch_idx, strides, ball_sizes, angle)`** — Build with rotational variants.

See docstrings in `balltree_setaxis` for full parameters and return values.

## License

MIT — see [LICENSE](LICENSE).


