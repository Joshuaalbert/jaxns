from jax import numpy as jnp

from jaxns.framework.prior import Prior
from jaxns.framework.model import Model


def test_gh95():
    def prior_model():
        x = yield Prior(dist_or_value=jnp.asarray(0.))
        return x

    def log_likelihood(x):
        return jnp.sum(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.U_ndims == 0
