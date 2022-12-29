from jax import numpy as jnp, random

from jaxns.nested_sampling.internals.linalg import msqrt, cholesky_update


def test_msqrt():
    for i in range(10):
        A = random.normal(random.PRNGKey(i), shape=(30, 30))
        B = A @ A.T
        L = msqrt(B)
        assert jnp.allclose(B, L @ L.T, atol=2e-5)


def test_cholesky_update():
    from jax import random
    for i in range(10):
        A = random.normal(random.PRNGKey(i), shape=(3, 3))
        A = A @ A.T
        x = jnp.array([1., 0., -2.])
        # with disable_jit():
        #     print(cholesky_update(jnp.linalg.cholesky(A), x))
        #     print(jnp.linalg.cholesky(A + x[:,None]*x[None,:]))
        assert jnp.allclose(cholesky_update(jnp.linalg.cholesky(A), x),
                            jnp.linalg.cholesky(A + x[:, None] * x[None, :]), atol=5e-5)
