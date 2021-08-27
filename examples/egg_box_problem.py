from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.utils import summary
from jax import random, jit, vmap
from jax import numpy as jnp
import pylab as plt


def main():
    def log_likelihood(theta, **kwargs):
        return 5. * (2. + jnp.prod(jnp.cos(0.5 * theta)))
    ndim = 4
    prior_chain = PriorChain() \
        .push(UniformPrior('theta', low=jnp.zeros(ndim), high=jnp.pi * 10. * jnp.ones(ndim)))

    theta = vmap(lambda key: prior_chain(prior_chain.compactify_U(prior_chain.sample_U(key))))(
        random.split(random.PRNGKey(0), 10000))
    lik = vmap(lambda theta: log_likelihood(**theta))(theta)
    sc = plt.scatter(theta['theta'][:, 0], theta['theta'][:, 1], c=lik)
    plt.colorbar(sc)
    plt.show()

    ns = NestedSampler(log_likelihood, prior_chain, num_live_points=100*prior_chain.U_ndims)
    results = jit(ns)(key=random.PRNGKey(42))

    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
