import jax
import jax.random
import numpy as np

from jaxns import Prior, Model
from jaxns.framework.jaxify import jaxify_likelihood
from jaxns.framework.tests.test_model import tfpd


def test_jaxify_likelihood():
    def log_likelihood(x, y):
        return np.sum(x, axis=-1) + np.sum(y, axis=-1)

    wrapped_ll = jaxify_likelihood(log_likelihood)
    np.testing.assert_allclose(wrapped_ll(np.array([1, 2]), np.array([3, 4])), 10)

    vmaped_wrapped_ll = jax.vmap(jaxify_likelihood(log_likelihood, vectorised=True))

    np.testing.assert_allclose(vmaped_wrapped_ll(np.array([[1, 2], [2, 2]]), np.array([[3, 4], [4, 4]])),
                               np.array([10, 12]))


def test_jaxify():
    def prior_model():
        x = yield Prior(tfpd.Uniform(), name='x').parametrised()
        return x

    @jaxify_likelihood
    def log_likelihood(x):
        return x

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    model.sanity_check(key=jax.random.PRNGKey(0), S=10)
    assert model.U_ndims == 0
    assert model.num_params == 1
