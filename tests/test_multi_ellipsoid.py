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
    _accumulate_mixture_moments,
    accumulate_mixture_moments_reference,
    bounding_ellipsoid,
    circle_to_ellipsoid,
    covariance_to_rotational,
    ellipsoid_clustering,
    ellipsoid_params,
    ellipsoid_to_circle,
    empty_sampler_data,
    initialise_streaming_sampler_data,
    log_ellipsoid_volume,
    maha_ellipsoid,
    plot_ellipses,
    point_in_ellipsoid,
    stream_sampler_data,
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


def test_streaming_moments_match_reference_and_do_not_forget_old_rows():
    data = empty_sampler_data(
        num_components=2,
        dimension=2,
        streaming=True,
    )
    points = jnp.asarray([
        [-1.1, -0.9],
        [-0.9, -1.1],
        [0.9, 1.1],
        [1.1, 0.9],
    ])
    responsibilities = jnp.asarray([
        [0.9, 0.8, 0.2, 0.1],
        [0.1, 0.2, 0.8, 0.9],
    ])
    mask = jnp.asarray([True, True, True, False])
    observed = jax.jit(_accumulate_mixture_moments)(
        data.moments,
        points,
        responsibilities,
        mask,
    )
    reference = accumulate_mixture_moments_reference(
        data.moments,
        np.asarray(points),
        np.asarray(responsibilities),
        np.asarray(mask),
    )
    for actual, expected in zip(
        jax.tree.leaves(observed),
        jax.tree.leaves(reference),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-15)
    # Responsibilities sum to one per valid row, so cumulative component
    # mass records three observations exactly; no decay erased older rows.
    np.testing.assert_allclose(jnp.sum(observed.mass), 3.0, rtol=1e-15)


def test_streaming_update_discovers_higher_contour_and_survives_collapse():
    first = random.normal(random.PRNGKey(30), shape=(16, 2)) * 0.05 + 0.2
    second = random.normal(random.PRNGKey(31), shape=(16, 2)) * 0.05 + 0.8
    points = jnp.concatenate([first, second], axis=0)
    log_L = -jnp.sum(jnp.square(points - 0.5), axis=1)
    mask = jnp.ones((32,), jnp.bool_)
    initial = empty_sampler_data(
        num_components=2,
        dimension=2,
        streaming=True,
    )
    fitted = jax.jit(
        initialise_streaming_sampler_data,
        static_argnames=(
            "n_iters",
            "min_effective_samples",
            "regularisation",
        ),
    )(
        random.PRNGKey(32),
        initial,
        points,
        log_L,
        mask,
        jnp.asarray(32),
        n_iters=5,
        min_effective_samples=8,
        regularisation=1e-6,
    )
    assert fitted.moments is not None
    assert jnp.all(fitted.valid)
    np.testing.assert_allclose(
        jnp.sum(fitted.moments.mass),
        32.0,
        rtol=1e-14,
    )

    new_points = random.normal(
        random.PRNGKey(33),
        shape=(8, 2),
    ) * 0.02 + 0.8
    new_log_L = jnp.arange(8, dtype=mp_policy.measure_dtype) + 10.0
    updated = jax.jit(
        stream_sampler_data,
        static_argnames=("regularisation",),
    )(
        fitted,
        new_points,
        new_log_L,
        jnp.ones((8,), jnp.bool_),
        jnp.asarray(40),
        regularisation=1e-6,
    )
    np.testing.assert_allclose(
        jnp.sum(updated.moments.mass),
        40.0,
        rtol=1e-14,
    )
    assert jnp.max(updated.log_L_max) == 17.0
    assert int(updated.num_samples) == 40
    assert int(updated.num_updates) == 2

    # A degenerate new batch must not replace full-dimensional direction
    # geometry with a singular component. Persistent statistics may record
    # the observation, while the last healthy geometry remains usable.
    collapsed = stream_sampler_data(
        updated,
        jnp.full((8, 2), 0.8),
        jnp.full((8,), 18.0),
        jnp.ones((8,), jnp.bool_),
        jnp.asarray(48),
        regularisation=1e-6,
    )
    assert jnp.any(collapsed.valid)
    assert jnp.all(jnp.isfinite(collapsed.radii[collapsed.valid]))
    assert jnp.max(collapsed.log_L_max) == 18.0


def test_streaming_soft_responsibilities_can_discover_a_later_mode():
    initial_points = (
        random.normal(random.PRNGKey(40), shape=(32, 2)) * 0.08 - 1.0
    )
    data = initialise_streaming_sampler_data(
        random.PRNGKey(41),
        empty_sampler_data(2, 2, streaming=True),
        initial_points,
        -jnp.sum(jnp.square(initial_points), axis=1),
        jnp.ones((32,), jnp.bool_),
        jnp.asarray(32),
        n_iters=5,
        min_effective_samples=8,
        regularisation=1e-6,
    )
    assert jnp.all(data.valid)

    # The second mode did not exist during initial training. Repeated new
    # observations must be able to pull one existing component across without
    # deleting the mass already accumulated by the first mode.
    for step in range(8):
        points = (
            random.normal(
                random.PRNGKey(42 + step),
                shape=(16, 2),
            ) * 0.08 + 1.0
        )
        data = stream_sampler_data(
            data,
            points,
            jnp.full((16,), float(step + 1)),
            jnp.ones((16,), jnp.bool_),
            jnp.asarray(32 + 16 * (step + 1)),
            regularisation=1e-6,
        )
    assert jnp.max(data.mixture.centres[:, 0]) > 0.5
    np.testing.assert_allclose(
        jnp.sum(data.moments.mass),
        160.0,
        rtol=1e-14,
    )


def test_streaming_singular_initial_population_retries_without_pseudo_data():
    points = jnp.full((16, 2), 0.5)
    initial = empty_sampler_data(2, 2, streaming=True)
    attempted = initialise_streaming_sampler_data(
        random.PRNGKey(51),
        initial,
        points,
        jnp.zeros((16,)),
        jnp.ones((16,), jnp.bool_),
        jnp.asarray(16),
        n_iters=3,
        min_effective_samples=8,
        regularisation=1e-6,
    )
    assert not jnp.any(attempted.valid)
    assert int(attempted.num_attempted) == 16
    assert int(attempted.num_updates) == 0
    np.testing.assert_array_equal(
        attempted.moments.mass,
        initial.moments.mass,
    )
