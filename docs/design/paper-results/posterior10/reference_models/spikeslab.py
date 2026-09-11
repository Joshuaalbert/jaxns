import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import Model, Prior

tfpd = tfp.distributions


def build_spikeslab_model(ndim: int) -> Model:
    """
    Builds the SpokeSlab model.

    Args:
        ndim: Number of input dimensions the function should take.

    Returns:
        model: The SpokeSlab model.
    """

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-4. * jnp.ones(ndim), high=8. * jnp.ones(ndim)), name='z')
        return z

    def log_likelihood(z):
        mean_1 = jnp.array([6., 6.])
        mean_2 = jnp.array([2.5, 2.5])
        for i in range(ndim - 2):
            mean_1 = jnp.append(mean_1, 0.)
            mean_2 = jnp.append(mean_2, 0.)
        cov_1 = 0.08 * jnp.eye(ndim)
        cov_2 = 0.8 * jnp.eye(ndim)
        gauss_1 = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_1, covariance_matrix=cov_1).log_prob(z)
        gauss_2 = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_2, covariance_matrix=cov_2).log_prob(z)
        y = jnp.logaddexp(gauss_1, gauss_2)
        return y

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    return model
