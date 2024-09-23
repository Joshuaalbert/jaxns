import dataclasses
import warnings
from typing import TypeVar

import jax
from jax import numpy as jnp

from jaxns.internals.logging import get_grandparent_info

# if not jax.config.threefry_partitionable:
#     warnings.warn("JAX threefry partitionable is not enabled. Setting it now. Check for errors.")
#     jax.config.update('jax_threefry_partitionable', True)

if not jax.config.read('jax_enable_x64'):
    warnings.warn("JAX x64 is not enabled. Setting it now. Check for errors.")
    jax.config.update('jax_enable_x64', True)

# Create a float scalar to lock in dtype choices.
if jnp.array(1., jnp.float64).dtype != jnp.float64:
    raise RuntimeError("Failed to set float64 as default dtype.")

T = TypeVar("T")


def _cast_floating_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, float):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.floating):
                warnings.warn(f"Expected float type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_complex_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, complex):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.complexfloating):
                warnings.warn(f"Expected complex type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_integer_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, int):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.integer):
                warnings.warn(f"Expected integer type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


def _cast_bool_to(tree: T, dtype: jnp.dtype, quiet: bool) -> T:
    def conditional_cast(x):
        if isinstance(x, bool):
            return jnp.asarray(x, dtype=dtype)
        try:
            if not quiet and not jnp.issubdtype(x.dtype, jnp.bool_):
                warnings.warn(f"Expected bool type, got {x.dtype}, {get_grandparent_info()}.")
            return x.astype(dtype)
        except AttributeError:
            warnings.warn(f"Failed to cast {x} to {dtype}.")
            return x

    return jax.tree.map(conditional_cast, tree)


X = TypeVar("X")


@dataclasses.dataclass(frozen=True)
class Policy:
    """Encapsulates casting for inputs, outputs and parameters."""
    measure_dtype: jnp.dtype = jnp.float64
    index_dtype: jnp.dtype = jnp.int64
    count_dtype: jnp.dtype = jnp.int64

    def cast_to_index(self, x: X, quiet: bool = False) -> X:
        """Converts index values to the index dtype."""
        return _cast_integer_to(x, self.index_dtype, quiet=quiet)

    def cast_to_measure(self, x: X, quiet: bool = False) -> X:
        """Converts measure values to the measure dtype."""
        return _cast_floating_to(x, self.measure_dtype, quiet=quiet)

    def cast_to_count(self, x: X, quiet: bool = False) -> X:
        """Converts count values to the count dtype."""
        return _cast_integer_to(x, self.count_dtype, quiet=quiet)


mp_policy = Policy()

float_type = jnp.result_type(float)
int_type = jnp.result_type(int)
complex_type = jnp.result_type(complex)
