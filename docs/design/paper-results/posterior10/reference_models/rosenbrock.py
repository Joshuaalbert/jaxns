import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Model, Prior

tfpd = tfp.distributions


def build_rosenbrock_model(ndim: int) -> Model:
    """
    Builds the Rosenbrock model.

    Args:
        ndim: Number of input dimensions the function should take.

    Returns:
        model: The Rosenbrock model.
    """

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-5 * jnp.ones(ndim), high=5 * jnp.ones(ndim)), name='z')
        return z

    def log_likelihood(z):
        y = 0.
        for i in range(ndim - 1):
            y += (100. * jnp.power(z[i + 1] - jnp.power(z[i], 2), 2) + jnp.power(1 - z[i], 2))
        return -y

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    return model
