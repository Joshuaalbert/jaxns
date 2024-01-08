import pytest
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from jaxns.framework.model import Model
from jaxns.framework.prior import Prior

tfpd = tfp.distributions


def test_gh95():
    def prior_model():
        x = yield Prior(dist_or_value=jnp.asarray(0.))
        return x

    def log_likelihood(x):
        return jnp.sum(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.U_ndims == 0


def test_parametrised_singular():
    def prior_model():
        x = yield Prior(dist_or_value=jnp.asarray(0.), name='x').parametrised()
        return x

    def log_likelihood(x):
        return jnp.sum(x)

    with pytest.raises(ValueError, match="Cannot parametrise a prior without distribution."):
        _ = Model(prior_model=prior_model, log_likelihood=log_likelihood)


def test_parametrised():
    def prior_model():
        x = yield Prior(tfpd.Uniform(),name='x').parametrised()
        return x

    def log_likelihood(x):
        return jnp.sum(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.num_params == 1
    assert model.U_ndims == 0

    log_prob_joint = model.log_prob_joint(model.U_placeholder, allow_nan=True)
    assert log_prob_joint.shape == ()
    assert log_prob_joint == 0.5, "Didn't init at median of uniform"
