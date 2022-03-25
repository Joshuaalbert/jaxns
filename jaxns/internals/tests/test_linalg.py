from jax import numpy as jnp, random

from jaxns.internals.linalg import msqrt, cholesky_update, rank_one_update_matrix_inv


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


def test_inverse_update():
    A = random.normal(random.PRNGKey(2), shape=(3, 3))
    A = A @ A.T
    u = random.normal(random.PRNGKey(7), shape=(3,))
    v = random.normal(random.PRNGKey(6), shape=(3,))
    B = u[:, None] * v
    Ainv = jnp.linalg.inv(A)
    detAinv = jnp.linalg.det(Ainv)
    C1, logdetC1 = rank_one_update_matrix_inv(Ainv, jnp.log(detAinv), u, v, add=True)
    assert jnp.isclose(jnp.linalg.inv(A + B), C1).all()
    assert jnp.isclose(jnp.log(jnp.linalg.det(jnp.linalg.inv(A + B))), logdetC1)
    C2, logdetC2 = rank_one_update_matrix_inv(Ainv, jnp.log(detAinv), u, v, add=False)
    print(jnp.log(jnp.linalg.det(jnp.linalg.inv(A - B))), logdetC2)
    assert jnp.isclose(jnp.linalg.inv(A - B), C2).all()
    assert jnp.isclose(jnp.log(jnp.linalg.det(jnp.linalg.inv(A - B))), logdetC2)