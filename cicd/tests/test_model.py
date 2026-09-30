import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jaxctx import scope
from tensorflow_probability.substrates import jax as tfp

from jaxns.model import Model
from jaxns.priors import Prior

tfpd = tfp.distributions


@pytest.mark.parametrize("dimension", [1, 3])
def test_prior_transform_preserves_hierarchical_measure_and_dimension(dimension):
    def prior_model():
        location = Prior(
            tfpd.Uniform(jnp.zeros(dimension), jnp.ones(dimension)),
            name="location",
        ).realise()
        value = Prior(
            tfpd.Normal(jnp.sum(location), 0.5), name="value",
        ).realise()
        return -jnp.square(value)

    model = Model(prior_model)
    single = model.sample_U(jax.random.PRNGKey(303))
    assert model.U_ndims() == dimension + 1
    assert sum(leaf.size for leaf in jax.tree.leaves(single)) == dimension + 1
    samples = jax.jit(jax.vmap(model.sample_U))(
        jax.random.split(jax.random.PRNGKey(304), 8192),
    )
    transformed = jax.jit(jax.vmap(model.transform_to_X))(samples)
    locations = np.asarray(transformed["location"])
    # Testing the conditional residual catches a lost dependency even when
    # each named variable's unconditional mean happens to remain correct.
    residual = (np.asarray(transformed["value"]) - locations.sum(axis=1)) / 0.5
    np.testing.assert_allclose(locations.mean(axis=0), 0.5, atol=0.02)
    np.testing.assert_allclose(locations.var(axis=0), 1 / 12, atol=0.006)
    np.testing.assert_allclose(residual.mean(), 0.0, atol=0.06)
    np.testing.assert_allclose(residual.var(), 1.0, atol=0.08)
    assert np.all((locations >= 0.0) & (locations < 1.0))
    np.testing.assert_allclose(
        np.mean((locations - 0.5) * residual[:, None], axis=0),
        0.0, atol=0.02,
    )


def test_nan_likelihood_is_zero_but_sanity_check_reports_raw_output() -> None:
    def prior_model(log_likelihood):
        Prior(tfpd.Uniform(0.0, 1.0), name="value").realise()
        return jnp.asarray(log_likelihood)

    model = Model(prior_model)
    sample = model.sample_U(jax.random.PRNGKey(0), args=(0.0,))
    # Even after a successful check, every later evaluation must apply the
    # policy. The diagnostic must still see NaNs before that conversion.
    model.sanity_check(jax.random.PRNGKey(1), args=(0.0,), num_samples=4)
    values = jnp.asarray([jnp.nan, -jnp.inf, -2.0, 0.0, jnp.inf])
    expected = jnp.asarray([-jnp.inf, -jnp.inf, -2.0, 0.0, jnp.inf])
    for evaluate in (model.log_likelihood, model.log_joint):
        actual = jax.jit(jax.vmap(
            lambda value, evaluate=evaluate: evaluate(sample, args=(value,)),
        ))(values)
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="log_likelihood: nan"):
        model.sanity_check(jax.random.PRNGKey(1), args=(jnp.nan,), num_samples=4)


def test_sanity_check_rejects_invalid_model_outputs() -> None:
    """Invalid model outputs fail before they can look scientifically valid."""

    def model_with_likelihood(log_likelihood):
        def prior_model():
            Prior(
                tfpd.Uniform(low=0.0, high=1.0),
                name="value",
            ).realise()
            return jnp.asarray(log_likelihood)

        return Model(prior_model=prior_model)

    for invalid_likelihood in (jnp.nan, jnp.inf):
        with pytest.raises(ValueError, match="invalid prior sample"):
            model_with_likelihood(invalid_likelihood).sanity_check(
                jax.random.PRNGKey(0),
                num_samples=4,
            )

    def invalid_prior_model():
        Prior(
            tfpd.Normal(loc=jnp.nan, scale=1.0),
            name="value",
        ).realise()
        return jnp.asarray(0.0)

    with pytest.raises(ValueError, match="invalid prior sample"):
        Model(prior_model=invalid_prior_model).sanity_check(
            jax.random.PRNGKey(1),
            num_samples=4,
        )

    # A negative-infinite log likelihood is valid zero likelihood and is not
    # conflated with NaN or a divergent positive likelihood.
    model_with_likelihood(-jnp.inf).sanity_check(
        jax.random.PRNGKey(2),
        num_samples=4,
    )

    with pytest.raises(ValueError, match="num_samples must be positive"):
        model_with_likelihood(0.0).sanity_check(
            jax.random.PRNGKey(3),
            num_samples=0,
        )

    with pytest.raises(ValueError, match="scalar log likelihood"):
        model_with_likelihood(jnp.zeros((1,))).sanity_check(
            jax.random.PRNGKey(4),
            num_samples=4,
        )


def test_init_params_forwards_model_and_explicit_args() -> None:
    """Model data and parameters remain explicit rather than closure-bound."""

    def prior_model(observations):
        location = Prior(
            tfpd.Normal(loc=0.0, scale=1.0),
            name="location",
        ).realise()
        uncertainty = Prior(
            tfpd.Exponential(rate=1.0),
            name="uncertainty",
        ).parameter()
        likelihood = tfpd.Normal(location, uncertainty)
        return jnp.sum(likelihood.log_prob(observations))

    model = Model(prior_model=prior_model)
    args = (jnp.asarray([-0.1, 0.2]),)
    params = model.init_params(
        key=jax.random.PRNGKey(0),
        args=args,
    )
    model.sanity_check(
        key=jax.random.PRNGKey(1),
        args=args,
        params=params,
        num_samples=4,
    )
    sample = model.sample_U(
        key=jax.random.PRNGKey(2),
        args=args,
        params=params,
    )
    log_likelihood = model.log_likelihood(
        sample,
        args=args,
        params=params,
    )
    transformed = model.transform_to_X(
        sample,
        args=args,
        params=params,
    )
    log_prior = model.log_prior(
        sample,
        args=args,
        params=params,
    )
    log_joint = model.log_joint(
        sample,
        args=args,
        params=params,
    )

    assert "uncertainty" in params
    assert "location" in transformed
    assert model.U_ndims(args=args, params=params) > 0
    assert jnp.all(jnp.isfinite(jnp.asarray([
        log_likelihood,
        log_prior,
        log_joint,
    ])))
    assert jnp.allclose(log_joint, log_prior + log_likelihood)


def test_periodic_coordinates_expand_in_sampler_order() -> None:
    """Whole-prior declarations become scalar flags without changing U."""

    def prior_model():
        with scope("calibration"):
            angles = Prior(
                tfpd.Uniform(
                    low=jnp.zeros((2,)),
                    high=jnp.ones((2,)),
                ),
                name="angles",
            ).realise(periodic=True)
            radius = Prior(
                tfpd.Uniform(low=0.0, high=1.0),
                name="radius",
            ).realise()
        return -jnp.sum(jnp.square(angles)) - jnp.square(radius)

    model = Model(prior_model=prior_model)
    sample = model.sample_U(jax.random.PRNGKey(2))

    assert model._periodic_coordinates() == (True, True, False)
    assert jax.tree.structure(sample) == jax.tree.structure(
        model.sample_U(jax.random.PRNGKey(3))
    )


def test_periodic_coordinates_report_all_false_topology() -> None:
    """Ordinary models retain an aligned all-false topology."""

    def prior_model():
        value = Prior(
            tfpd.Uniform(low=0.0, high=1.0),
            name="value",
        ).realise()
        return -jnp.square(value)

    assert Model(prior_model=prior_model)._periodic_coordinates() == (False,)
