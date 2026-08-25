"""Focused validation contracts for public model evaluation."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model

tfpd = tfp.distributions


def test_unit_cube_transform_matches_dependent_prior_likelihood_and_dimension():
    """The U representation preserves a dependent declared prior contract."""
    conditional_scale = 0.2

    def likelihood(x, y):
        return -0.5 * jnp.square(x + 2.0 * y)

    def prior_model():
        x = Prior(
            tfpd.Uniform(low=0.0, high=1.0),
            name="x",
        ).realise()
        y = Prior(
            tfpd.Normal(loc=x, scale=conditional_scale),
            name="y",
        ).realise()
        return likelihood(x, y)

    model = Model(prior_model=prior_model)
    sample = model.sample_U(jax.random.PRNGKey(0))
    consumed_coordinates = sum(
        np.size(np.asarray(leaf)) for leaf in jax.tree.leaves(sample)
    )
    assert model.U_ndims() == consumed_coordinates == 2

    num_samples = 4096
    U_samples = jax.vmap(model.sample_U)(
        jax.random.split(jax.random.PRNGKey(1), num_samples)
    )
    X_samples = jax.vmap(model.transform_to_X)(U_samples)
    x = np.asarray(X_samples.get_dotted("x"))
    y = np.asarray(X_samples.get_dotted("y"))
    residual = y - x

    # Test the joint law through its independent base coordinate and the
    # conditional residual, which directly exposes a lost dependency.
    assert abs(np.mean(x) - 0.5) < 0.02
    assert abs(np.var(x) - (1.0 / 12.0)) < 0.01
    assert abs(np.mean(residual)) < 0.015
    assert abs(np.std(residual) - conditional_scale) < 0.015
    assert np.corrcoef(x, y)[0, 1] > 0.7

    unit_likelihood = jax.vmap(model.log_likelihood)(U_samples)
    transformed_likelihood = likelihood(
        X_samples.get_dotted("x"),
        X_samples.get_dotted("y"),
    )
    np.testing.assert_allclose(unit_likelihood, transformed_likelihood)


def test_model_sanity_check_accepts_zero_likelihood_prior_draws():
    """Negative infinity is valid log zero, unlike NaN or positive infinity."""
    def prior_model():
        Prior(
            tfpd.Uniform(low=0.0, high=1.0),
            name="x",
        ).realise()
        return jnp.asarray(-jnp.inf)

    model = Model(prior_model=prior_model)
    model.sanity_check(jax.random.PRNGKey(2), num_samples=2)


@pytest.mark.parametrize("invalid_log_likelihood", [jnp.nan, jnp.inf])
def test_initialise_rejects_nonfinite_model_outputs_before_returning_state(
        invalid_log_likelihood,
):
    def prior_model():
        Prior(
            tfpd.Uniform(low=0.0, high=1.0),
            name="x",
        ).realise()
        return jnp.asarray(invalid_log_likelihood)

    sampler = NestedSampler(
        model=Model(prior_model=prior_model),
        root_allocation_degree=1,
        shell_size=1,
        max_samples=2,
        initial_capacity=2,
    )

    with pytest.raises(
        ValueError,
        match="log-likelihood must return one finite scalar",
    ):
        sampler.initialise(jax.random.PRNGKey(7))
