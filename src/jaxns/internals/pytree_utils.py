import jax
import jax.numpy as jnp


def tree_dot(x, y):
    dots = jax.tree.leaves(jax.tree.map(jnp.vdot, x, y))
    return sum(dots[1:], start=dots[0])


def tree_norm(x):
    norm2 = tree_dot(x, x)
    if jnp.issubdtype(norm2.dtype, jnp.complexfloating):
        return jnp.sqrt(norm2.real)
    return jnp.sqrt(norm2)


def tree_mul(x, y):
    return jax.tree.map(jax.lax.mul, x, y)


def tree_sub(x, y):
    return jax.tree.map(jax.lax.sub, x, y)


def tree_div(x, y):
    return jax.tree.map(jax.lax.div, x, y)
