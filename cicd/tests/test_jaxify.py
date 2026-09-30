import jax
import jax.random
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from jaxns.jaxify import jaxify_likelihood

tfpd = tfp.distributions


def test_jaxify_likelihood():
    def log_likelihood(x, y):
        return np.sum(x, axis=-1) + np.sum(y, axis=-1)

    wrapped_ll = jaxify_likelihood(log_likelihood)
    np.testing.assert_allclose(wrapped_ll(np.array([1, 2]), np.array([3, 4])), 10)

    vmaped_wrapped_ll = jax.vmap(jaxify_likelihood(log_likelihood, vectorised=True))

    np.testing.assert_allclose(
        vmaped_wrapped_ll(np.array([[1, 2], [2, 2]]), np.array([[3, 4], [4, 4]])),
        np.array([10, 12])
    )
