import jax.lax
import jax.tree
from jax import numpy as jnp

from jaxns.nested_samplers.types import IntArray


def scan_or_while_loop(scan_fn, carry_init, xs, length: IntArray | None = None, unroll: int = 1) -> tuple:
    # if length is None or static use scan, other wise use while_loop
    if length is None or isinstance(length, int):
        return jax.lax.scan(scan_fn, carry_init, xs, length=length, unroll=unroll)

    def cond_fn(carry):
        i, _, _ = carry
        return i < length

    def body_fn(carry):
        i, carry_inner, ys = carry
        x = jax.tree.map(lambda x: x[i], xs)
        carry_inner, y = scan_fn(carry_inner, x)
        ys = jax.tree.map(lambda y, y_new: y.at[i, ...].set(y_new), ys, y)
        return (i + 1, carry_inner, ys)

    # aeval to build ys structure
    max_length = jax.tree.leaves(xs)[0].shape[0]
    carry_struct, ys_struct = jax.eval_shape(scan_fn, carry_init, jax.tree.map(lambda x: x[0], xs))
    ys_init = jax.tree.map(lambda y: jnp.zeros((max_length,) + y.shape, dtype=y.dtype), ys_struct)
    carry = (0, carry_init, ys_init)
    _, carry_inner, ys = jax.lax.while_loop(cond_fn, body_fn, carry)
    return carry_inner, ys


def test_scan_or_while_loop():
    def scan_fn(carry, x):
        return carry + x, carry + x

    carry_init = 0
    xs = jnp.arange(10)

    # Test with static length
    _, ys_scan = scan_or_while_loop(scan_fn, carry_init, xs, length=10)
    assert jnp.all(ys_scan == jnp.cumsum(xs))

    # Test with dynamic length
    length = jnp.array(10)
    _, ys_while = scan_or_while_loop(scan_fn, carry_init, xs, length=length)
    assert jnp.all(ys_while == jnp.cumsum(xs))

    # Test with dynamic length
    length = jnp.array(7)
    _, ys_while = scan_or_while_loop(scan_fn, carry_init, xs, length=length)
    assert jnp.all(ys_while == jnp.cumsum(xs).at[7:].set(0))
