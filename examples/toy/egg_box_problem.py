from jaxns.nested_sampler.nested_sampling import NestedSampler, save_state, load_state
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.utils import summary
from jax import random, jit, vmap
from jax import numpy as jnp
import pylab as plt
from jax import disable_jit


def main():
    def log_likelihood(theta, **kwargs):
        return 5. * (2. + jnp.prod(jnp.cos(0.5 * theta)))

    ndim = 2
    with PriorChain() as prior_chain:
        UniformPrior('theta', low=jnp.zeros(ndim), high=jnp.pi * 10. * jnp.ones(ndim))

    prior_chain.build()

    theta = vmap(lambda key: prior_chain(prior_chain.sample_U_flat(key)))(
        random.split(random.PRNGKey(0), 10000))
    lik = vmap(lambda theta: log_likelihood(**theta))(theta)
    sc = plt.scatter(theta['theta'][:, 0], theta['theta'][:, 1], c=lik)
    plt.colorbar(sc)
    plt.show()

    # static run
    ns = NestedSampler(log_likelihood, prior_chain,
                       dynamic=True,
                       max_samples=6e3,
                       max_num_live_points=prior_chain.U_ndims*40)

    # with disable_jit():
    results, state = jit(ns,static_argnames='return_state')(
            key=random.PRNGKey(42),
            terminate_evidence_uncert=1e-2,
            termination_ess=None,
            num_live_points=prior_chain.U_ndims*30,
            delta_num_live_points=prior_chain.U_ndims*20,
            terminate_max_num_threads=20,
            dynamic_kwargs=dict(f=0.9, G=0.),
            return_state=True)

    # # save_state(state,'state.npz')
    #
    # # state = load_state('state.npz')
    #
    # # dynamic run
    # ns = NestedSampler(log_likelihood, prior_chain,
    #                    dynamic=True,
    #                    max_samples=6e3,
    #                    max_num_live_points=prior_chain.U_ndims * 40)
    #
    # # with disable_jit():
    # results, state = jit(ns, static_argnames='return_state')(
    #     key=random.PRNGKey(42),
    #     terminate_evidence_uncert=1e-2,
    #     termination_ess=400.,
    #     num_live_points=prior_chain.U_ndims * 30,
    #     delta_num_live_points=prior_chain.U_ndims * 5,
    #     terminate_max_num_threads=3,
    #     dynamic_kwargs=dict(f=0.5, G=0.),
    #     return_state=True,
    #     refine_state=state)

    print(state)
    summary(results)
    print(results.thread_stats)
    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
