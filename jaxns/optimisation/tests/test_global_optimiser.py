from jax import numpy as jnp, random

from jaxns import GlobalOptimiser
from jaxns.prior_transforms import PriorChain, UniformPrior


def test_nested_sampling_max_likelihood():
    def log_likelihood(x):
        return -0.5*jnp.sum(x**4 - 16 * x**2 + 5 * x)

    def test_example(ndim):
        with PriorChain() as prior_chain:
            UniformPrior('x', -5.*jnp.ones(ndim), 5.*jnp.ones(ndim))

        go = GlobalOptimiser(log_likelihood, prior_chain)
        results = go(key=random.PRNGKey(42), termination_frac_likelihood_improvement=1e-3, termination_patience=3)
        assert results.termination_reason == 1
        lower_bound = 39.16616*ndim
        upper_bound = 39.16617*ndim
        x_max = -2.903534
        print(ndim,jnp.abs(results.log_L_max-0.5*(lower_bound+upper_bound)))
        print(ndim,jnp.abs(results.sample_L_max['x']-x_max))
        assert jnp.isclose(results.log_L_max,
                           0.5 * (lower_bound + upper_bound),
                           atol=2. * (upper_bound - lower_bound))
        assert jnp.allclose(results.sample_L_max['x'], x_max, atol=2e-1*ndim/8)
        print(results)


    test_example(21)