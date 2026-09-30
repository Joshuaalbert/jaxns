import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Model, Prior

tfpd = tfp.distributions


def build_rastrigin_model(ndim: int) -> Model:
    """
    Builds the Rastrigin model.

    Args:
        ndim:  The number of dimensions of the rastrigin function.

    Returns:
        model: The rastrigin model.
    """

    def prior_model():
        x_min = -5.12
        x_max = 5.12
        z = yield Prior(tfpd.Uniform(low=x_min * jnp.ones(ndim), high=x_max * jnp.ones(ndim)), name='z')
        return z

    def log_likelihood(z):
        a = jnp.asarray(10.)
        y = a * ndim
        for i in range(ndim):
            y += jnp.power(z[i], 2) - a * jnp.cos(2 * jnp.pi * z[i])
        return -y

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    return model
