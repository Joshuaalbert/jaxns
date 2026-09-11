import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Model, Prior

tfpd = tfp.distributions


def build_eggbox_model(ndim: int) -> Model:
    """
    Builds the eggbox model.

    Args:
        ndim:  The number of dimensions of the eggbox function.

    Returns:
        model: The eggbox model.
    """

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=jnp.zeros(ndim), high=10. * jnp.pi * jnp.ones(ndim)), name='z')
        return z

    def log_likelihood(z):
        y = 1
        for i in range(ndim):
            y *= jnp.cos(z[i] / 2)
        y = jnp.power(2. + y, 5)
        return y

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    return model
