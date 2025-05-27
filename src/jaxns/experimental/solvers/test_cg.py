from jax import numpy as jnp

from dsa2000_cal.solvers.cg import cg_solve


def test_cg():
    A = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    b = jnp.array([1.0, 1.0])
    x0 = jnp.array([0.0, 0.0])

    def A_op(x):
        return jnp.dot(A, x)

    x, diag = cg_solve(A_op, b, x0)
    assert jnp.allclose(x, b)
    assert diag.iterations == 1
