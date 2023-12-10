from typing import TypeVar

from jax import numpy as jnp, tree_map, tree_util

T = TypeVar('T')

__all__ = [
    'add_chunk_dim',
    'remove_chunk_dim'
]


def remove_chunk_dim(py_tree: T) -> T:
    """
    Remove the chunk dimension from a pytree

    Args:
        py_tree: pytree to remove chunk dimension from

    Returns:
        pytree with chunk dimension removed
    """
    leaves = tree_util.tree_leaves(py_tree)

    # Check consistency
    for leaf in leaves:
        if len(leaf.shape) < 1:
            raise ValueError(f"Expected all leaves to have at least one dimension, got {leaf.shape}")
        if leaf.shape[0] != leaves[0].shape[0]:
            raise ValueError(
                f"Expected all leaves to have the same batch dimension, got {leaf.shape[0]} != {leaves[0].shape[0]}"
            )

    def _remove_chunk_dim(a):
        shape = list(a.shape)
        if len(shape) == 1:
            return a[0]
        shape = [shape[0] * shape[1]] + shape[2:]
        return jnp.reshape(a, shape)

    return tree_map(_remove_chunk_dim, py_tree)


def add_chunk_dim(py_tree: T, chunk_size: int) -> T:
    """
    Add a chunk dimension to a pytree

    Args:
        py_tree: pytree to add chunk dimension to
        chunk_size: size of chunk dimension

    Returns:
        pytree with chunk dimension added
    """

    def _add_chunk_dim(a):
        shape = list(a.shape)
        shape = [chunk_size, shape[0] // chunk_size] + shape[1:]
        return jnp.reshape(a, shape)

    return tree_map(_add_chunk_dim, py_tree)
