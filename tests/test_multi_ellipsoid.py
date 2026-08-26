import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
from jax import random, vmap
from jaxctx.priors.prior import Prior

from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.multi_ellipsoid_utils import (
    EllipsoidParams,
    bounding_ellipsoid,
    circle_to_ellipsoid,
    covariance_to_rotational,
    ellipsoid_clustering,
    ellipsoid_params,
    ellipsoid_to_circle,
    empty_sampler_data,
    log_ellipsoid_volume,
    maha_ellipsoid,
    plot_ellipses,
    point_in_ellipsoid,
    update_sampler_data,
)
from jaxns.random_utils import random_ortho_matrix

plt.switch_backend("Agg")

tfpd = tfp.distributions


def test_ellipsoid_clustering():
    def prior_model():
        x = Prior(tfpd.Uniform(low=0, high=2), name='x').realise()
        y = Prior(tfpd.Normal(loc=0, scale=2), name='y').realise()
        return log_likelihood(x, y)

    def log_likelihood(x, y):
        return jnp.log(jnp.exp(-0.5 * ((x - 0.5) / 0.1) ** 2 - 0.5 * ((y - 0.5) / 0.1) ** 2) + jnp.exp(
            -0.5 * ((x - 1.5) / 0.1) ** 2 - 0.5 * ((y - 1.5) / 0.1) ** 2))

    model = Model(prior_model=prior_model)

    n = 1000
    keys = random.split(random.PRNGKey(42), n)
    U_samples = vmap(model.sample_U)(keys)
    log_L = vmap(model.log_likelihood)(U_samples)
    threshold = jnp.percentile(log_L, 75)
    keep = log_L > threshold
    points = jnp.stack(jax.tree.leaves(U_samples), axis=-1).astype(mp_policy.measure_dtype)
    reservoir = points[keep]
    plt.scatter(reservoir[:, 0], reservoir[:, 1])
    state = ellipsoid_clustering(
        random.PRNGKey(42),
        points=reservoir,
        log_VS=jnp.asarray(0., mp_policy.measure_dtype),
        max_num_ellipsoids=10
    )
    plot_ellipses(params=state.params, show=False)
    plt.close('all')


def test_log_ellipsoid_volume():
    radii = jnp.ones(2)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(jnp.pi))
    radii = jnp.ones(3)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(4. * jnp.pi / 3.))


def test_bounding_ellipsoid():
    n = 200_000
    mean = jnp.asarray([0., 0.])
    cov = jnp.asarray([[1., 0.4], [0.4, 1.]])
    X = random.multivariate_normal(random.PRNGKey(42), mean=mean,
                                   cov=cov, shape=(n,))
    mask = jnp.ones(n, jnp.bool_)
    mu, Sigma = bounding_ellipsoid(points=X, mask=mask)
    assert jnp.allclose(mu, mean, atol=2e-2)
    assert jnp.allclose(Sigma, cov, atol=2e-2)


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
    random_radii = random.uniform(random.PRNGKey(1), shape=(N,), dtype=mp_policy.measure_dtype)
    cov = random_rotation @ jnp.diag(random_radii ** 2) @ random_rotation.T

    X = random.multivariate_normal(
        random.PRNGKey(42),
        mean=jnp.zeros(N),
        cov=cov,
        shape=(n,),
        dtype=mp_policy.measure_dtype
    )

    mu, radii, rotation = ellipsoid_params(points=X, mask=jnp.ones(n, jnp.bool_))
    inside = vmap(lambda x: point_in_ellipsoid(x, mu, radii, rotation))(X)
    plt.scatter(X[:, 0], X[:, 1], c=inside)
    plot_ellipses(jax.tree.map(lambda x: x[None], EllipsoidParams(mu, radii, rotation)), show=False)

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


def test_sampler_data_warm_update_retains_geometry_after_failed_fit():
    first = random.normal(random.PRNGKey(20), shape=(16, 2)) * 0.05 + 0.2
    second = random.normal(random.PRNGKey(21), shape=(16, 2)) * 0.08 + 0.8
    points = jnp.concatenate([first, second], axis=0)
    log_L = -jnp.sum(jnp.square(points - 0.5), axis=1)
    mask = jnp.ones((32,), jnp.bool_)
    log_weights = jnp.full((32,), -jnp.log(32.0))
    initial = empty_sampler_data(num_components=2, dimension=2)

    fitted = jax.jit(
        update_sampler_data,
        static_argnames=(
            "n_iters",
            "min_effective_samples",
            "regularisation",
        ),
    )(
        random.PRNGKey(22),
        initial,
        points,
        log_L,
        log_weights,
        mask,
        jnp.asarray(32),
        n_iters=5,
        min_effective_samples=8,
        regularisation=1e-6,
    )
    assert jnp.all(fitted.valid)
    assert int(fitted.num_updates) == 1
    assert jnp.all(jnp.isfinite(fitted.log_volumes))

    # An exactly collapsed population contains no full-dimensional component.
    # The attempted warm update must retain every previously valid parameter
    # rather than installing ridge-created but scientifically false geometry.
    collapsed = jnp.full_like(points, 0.5)
    retained = update_sampler_data(
        random.PRNGKey(23),
        fitted,
        collapsed,
        jnp.zeros_like(log_L),
        log_weights,
        mask,
        jnp.asarray(64),
        n_iters=3,
        min_effective_samples=8,
        regularisation=1e-6,
    )
    np.testing.assert_array_equal(retained.centres, fitted.centres)
    np.testing.assert_array_equal(retained.radii, fitted.radii)
    np.testing.assert_array_equal(retained.rotations, fitted.rotations)
    assert int(retained.num_updates) == 1
    assert int(retained.num_attempted) == 64
