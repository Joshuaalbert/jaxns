import numpy as np
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, random, tree_map, disable_jit, vmap

import jaxns.internals.maps
from jaxns.framework.bases import PriorModelGen
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns.nested_sampler.standard_static import draw_uniform_samples
from jaxns.internals.random import random_ortho_matrix
from jaxns.samplers.multi_ellipsoid.multi_ellipsoid_utils import log_ellipsoid_volume, ellipsoid_clustering, \
    bounding_ellipsoid, covariance_to_rotational, ellipsoid_params, point_in_ellipsoid, plot_ellipses, \
    EllipsoidParams, maha_ellipsoid, circle_to_ellipsoid, ellipsoid_to_circle
from jaxns.internals.types import float_type, Sample

tfpd = tfp.distributions


def test_ellipsoid_clustering():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=0, scale=2))

        return x, y

    def log_likelihood(x, y):
        return jnp.log(jnp.exp(-0.5 * ((x - 0.5) / 0.1) ** 2 - 0.5 * ((y - 0.5) / 0.1) ** 2) + jnp.exp(
            -0.5 * ((x - 1.5) / 0.1) ** 2 - 0.5 * ((y - 1.5) / 0.1) ** 2))

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    n = 1000
    live_points = draw_uniform_samples(random.PRNGKey(43),
                                       num_live_points=n,
                                       model=model)
    keep = live_points.log_L > log_likelihood(1.1, 1.1)
    reservoir: Sample = tree_map(lambda x: x[keep], live_points)
    plt.scatter(reservoir.U_sample[:, 0], reservoir.U_sample[:, 1])
    with disable_jit():
        state = ellipsoid_clustering(random.PRNGKey(42), points=reservoir.U_sample,
                                     log_VS=jnp.asarray(0., float_type),
                                     max_num_ellipsoids=10)
        plot_ellipses(params=state.params)
    # plt.show()
    plt.close('all')


def test_log_ellipsoid_volume():
    radii = jnp.ones(2)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(jnp.pi))
    radii = jnp.ones(3)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(4. * jnp.pi / 3.))


def test_bounding_ellipsoid():
    n = 1_000_000
    mean = jnp.asarray([0., 0.])
    cov = jnp.asarray([[1., 0.4], [0.4, 1.]])
    X = random.multivariate_normal(random.PRNGKey(42), mean=mean,
                                   cov=cov, shape=(n,))
    mask = jnp.ones(n, jnp.bool_)
    mu, Sigma = bounding_ellipsoid(points=X, mask=mask)
    assert jnp.allclose(mu, mean, atol=1e-2)
    assert jnp.allclose(Sigma, cov, atol=1e-2)


def test_covariance_to_rotational():
    n = 5
    random_rotation = random_ortho_matrix(random.PRNGKey(0), n=n, special_orthogonal=True)
    random_radii = random.uniform(random.PRNGKey(1), shape=(n,))

    J = random_rotation @ jnp.diag(1 / random_radii)
    cov_J = jnp.linalg.inv(J @ J.T)
    cov = random_rotation @ jnp.diag(random_radii ** 2) @ random_rotation.T

    np.testing.assert_allclose(cov, cov_J, atol=1e-6)

    radii, rotation = covariance_to_rotational(cov)

    _cov = rotation @ jnp.diag(radii ** 2) @ rotation.T

    np.testing.assert_allclose(cov, _cov, atol=1e-6)


def test_ellipsoid_params():
    n = 1000

    N = 2
    random_rotation = random_ortho_matrix(random.PRNGKey(0), n=N, special_orthogonal=True)
    random_radii = random.uniform(random.PRNGKey(1), shape=(N,))
    cov = random_rotation @ jnp.diag(random_radii ** 2) @ random_rotation.T

    X = random.multivariate_normal(random.PRNGKey(42),
                                   mean=jnp.zeros(N),
                                   cov=cov,
                                   shape=(n,))

    mu, radii, rotation = ellipsoid_params(points=X, mask=jnp.ones(n, jnp.bool_))
    inside = vmap(lambda x: point_in_ellipsoid(x, mu, radii, rotation))(X)
    plt.scatter(X[:, 0], X[:, 1], c=inside)
    plot_ellipses(tree_map(lambda x: x[None], EllipsoidParams(mu, radii, rotation)))

    assert np.all(inside)

    rho_max = jnp.max(vmap(lambda x: maha_ellipsoid(x, mu, radii, rotation))(X))
    assert jnp.isclose(rho_max, 1.)

    points = jnp.asarray([[0., 1.], [0., -1.], [1.5, 0.], [-1.5, 0.]])
    mu, radii, rotation = ellipsoid_params(points=points, mask=jnp.ones(4, jnp.bool_))
    # print(mu, radii, rotation)
    mu_true = jnp.zeros(2)
    radii_true = jnp.asarray([1.5, 1.])
    rotation_true = jnp.eye(2)
    assert jnp.allclose(mu, mu_true)
    assert jnp.allclose(radii, radii_true)
    assert jnp.allclose(rotation, rotation_true)


def test_ellipsoid_transforms():
    n = 1000

    N = 2
    random_rotation = random_ortho_matrix(random.PRNGKey(0), n=N, special_orthogonal=True)
    random_radii = random.uniform(random.PRNGKey(1), shape=(N,))
    mu = jnp.zeros(N)
    cov = random_rotation @ jnp.diag(random_radii ** 2) @ random_rotation.T

    X = random.multivariate_normal(random.PRNGKey(42),
                                   mean=jnp.zeros(N),
                                   cov=cov,
                                   shape=(n,))
    X_out = vmap(lambda x: circle_to_ellipsoid(ellipsoid_to_circle(x, mu, random_radii, random_rotation),
                                               mu, random_radii, random_rotation))(X)

    np.testing.assert_allclose(X_out, X, atol=1e-6)
