# from jax.config import config
#
# config.update("jax_enable_x64", True)

from jax import numpy as jnp, random, vmap
from jaxns.internals.maps import prepare_func_args
from jaxns.modules.bayesian_optimisation.utils import latin_hypercube
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.modules.bayesian_optimisation.bayesian_optimiser import BayesianOptimiser




def test_bayesian_optimisation():
    num_steps = 10

    @prepare_func_args
    def objective(x):
        return -0.5 * jnp.sum(x ** 4 - 16 * x ** 2 + 5 * x)

    def test_example(ndim):
        with PriorChain() as search_prior_chain:
            UniformPrior('x', -5. * jnp.ones(ndim), 5. * jnp.ones(ndim))

        bo = BayesianOptimiser(search_prior_chain)
        U, X = bo.initialise_experiment(ndim*2)
        Y = list(map(lambda x: objective(**x), X))

        for u,x,y in zip(U, X, Y):
            bo.add_result(u,x,y)

        print(bo)

        for i in range(num_steps):
            u, x = bo.choose_next_sample_location()
            y = objective(**x)
            bo.add_result(u,x,y)
            print(u, x, y)
            print(bo)

        lower_bound = 39.16616 * ndim
        upper_bound = 39.16617 * ndim
        x_max = -2.903534
        # print(ndim, jnp.abs(results.log_L_max - 0.5 * (lower_bound + upper_bound)))
        # print(ndim, jnp.abs(results.sample_L_max['x'] - x_max))
        # assert jnp.isclose(results.log_L_max,
        #                    0.5 * (lower_bound + upper_bound),
        #                    atol=2. * (upper_bound - lower_bound))
        # assert jnp.allclose(results.sample_L_max['x'], x_max, atol=2e-1 * ndim / 8)
        # print(results)

    test_example(2)


def test_latin_hyper_cube():
    num_samples = 50
    ndim = 2
    samples = latin_hypercube(random.PRNGKey(442525), num_samples, ndim, 0.)
    s = jnp.sort(samples, axis=0) * num_samples
    assert jnp.all(s < jnp.arange(1, num_samples + 1)[:, None])
    assert jnp.all(s > jnp.arange(0, num_samples)[:, None])

    num_samples = 50
    ndim = 2
    samples = latin_hypercube(random.PRNGKey(442525), num_samples, ndim, 1.)
    s = jnp.sort(samples, axis=0) * num_samples
    assert jnp.all(s < jnp.arange(1, num_samples + 1)[:, None])
    assert jnp.all(s > jnp.arange(0, num_samples)[:, None])