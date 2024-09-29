from time import monotonic_ns
from typing import Dict

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jaxns import Model, Prior, NestedSampler

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


def all_models() -> Dict[str, Model]:
    """
    Return all the models

    Returns:
        A dictionary of models
    """
    return dict(
        eggbox=build_eggbox_model(ndim=10),
        rastrigin=build_rastrigin_model(ndim=10),
        rosenbrock=build_rosenbrock_model(ndim=10),
        spikeslab=build_spikeslab_model(ndim=10)
    )


class Timer:
    def __enter__(self):
        self.t0 = monotonic_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time to execute: {(monotonic_ns() - self.t0) / 1e9} seconds.")


def main():
    for model_name, model in all_models().items():
        print(f"Testing model {model_name}")
        model.sanity_check(jax.random.PRNGKey(0), 1000)
        ns = NestedSampler(
            model=model,
            difficult_model=True,
            parameter_estimation=True
        )
        ns_jit = jax.jit(lambda key: ns(key))
        ns_compiled = ns_jit.lower(jax.random.PRNGKey(42)).compile()
        with Timer():
            termination_reason, state = ns_compiled(jax.random.PRNGKey(42))
            termination_reason.block_until_ready()
        results = ns.to_results(termination_reason=termination_reason, state=state)
        ns.plot_diagnostics(results, save_name=f"{model_name}_diagnostics.png")
        ns.plot_cornerplot(results, save_name=f"{model_name}_cornerplot.png")
        ns.summary(results, f_obj=f"{model_name}_summary.txt")


if __name__ == '__main__':
    main()
