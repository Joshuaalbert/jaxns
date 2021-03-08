from jax import numpy as jnp, random

from jaxns.likelihood_samplers.ellipsoid_utils import rank_one_update_matrix_inv, log_ellipsoid_volume


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


def test_log_ellipsoid_volume():
    radii = jnp.ones(2)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(jnp.pi))
    radii = jnp.ones(3)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(4. * jnp.pi / 3.))

