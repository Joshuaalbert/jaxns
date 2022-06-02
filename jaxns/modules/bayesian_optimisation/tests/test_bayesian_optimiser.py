# from jax.config import config
#
# config.update("jax_enable_x64", True)

from jax import numpy as jnp, random
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

        lower_bound = 39.16616 * ndim
        upper_bound = 39.16617 * ndim
        print(f"Optimal value in ({lower_bound}, {upper_bound}).")

        x_max = -2.903534

        print(f"Global optimum at {jnp.ones(ndim) * x_max}")

        with PriorChain() as search_prior_chain:
            UniformPrior('x', -5. * jnp.ones(ndim), 5. * jnp.ones(ndim))

        bo = BayesianOptimiser(search_prior_chain, key=random.PRNGKey(6465249))
        observations = bo.initialise_experiment(ndim * 2)
        for obs in observations:
            obs.set_response(objective(**obs.sample_point))
            bo.add_result(obs)

        bo.save_state("save_file.npz")
        bo.load_state("save_file.npz")

        print(bo)
        for i in range(num_steps):
            obs = bo.choose_next_sample_location()
            obs.set_response(objective(**obs.sample_point))
            bo.add_result(obs)
            print(obs)
            print(bo)

        bo.plot_progress()

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