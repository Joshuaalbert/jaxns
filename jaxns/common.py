from typing import TypeVar

from jax import numpy as jnp, tree_map

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

    def _remove_chunk_dim(a):
        shape = list(a.shape)
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
