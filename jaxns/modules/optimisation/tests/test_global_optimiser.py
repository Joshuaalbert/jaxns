from jax import numpy as jnp, random

from jaxns import GlobalOptimiser
from jaxns.prior_transforms import PriorChain, UniformPrior


def test_nested_sampling_max_likelihood():
    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    def log_likelihood(x):
        return -0.5 * jnp.sum(x ** 4 - 16 * x ** 2 + 5 * x)

    def test_example(ndim):
        with PriorChain() as prior_chain:
            UniformPrior('x', -5. * jnp.ones(ndim), 5. * jnp.ones(ndim))

        go = GlobalOptimiser(log_likelihood, prior_chain, num_parallel_samplers=2,
                             sampler_kwargs=dict(gradient_boost=True))
        results = go(key=random.PRNGKey(42), num_live_points=prior_chain.U_ndims * 30)
        go.summary(results)
        assert (results.termination_reason == 1) or (results.termination_reason == 33)
        lower_bound = 39.16616 * ndim
        upper_bound = 39.16617 * ndim
        x_max = -2.903534
        print(ndim, jnp.abs(results.log_L_max - 0.5 * (lower_bound + upper_bound)))
        print(ndim, jnp.abs(results.sample_L_max['x'] - x_max))
        assert jnp.isclose(results.log_L_max,
                           0.5 * (lower_bound + upper_bound),
                           atol=2. * (upper_bound - lower_bound))
        assert jnp.allclose(results.sample_L_max['x'], x_max, atol=2e-1 * ndim / 8)

    test_example(20)
