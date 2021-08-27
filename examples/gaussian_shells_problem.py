from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.utils import summary
from jax import random, jit,vmap
from jax import numpy as jnp
import pylab as plt
from timeit import default_timer


def main():
    def log_likelihood(theta, **kwargs):
        def log_circ(theta, c, r, w):
            return -0.5*(jnp.linalg.norm(theta - c) - r)**2/w**2 - jnp.log(jnp.sqrt(2*jnp.pi*w**2))
        w1=w2=jnp.array(0.1)
        r1=r2=jnp.array(2.)
        c1 = jnp.array([0., -3.])
        c2 = jnp.array([0., 3.])
        return jnp.logaddexp(log_circ(theta, c1,r1,w1) , log_circ(theta,c2,r2,w2))


    prior_chain = PriorChain() \
        .push(UniformPrior('theta', low=-6.*jnp.ones(2), high=6.*jnp.ones(2)))

    theta = vmap(lambda key: prior_chain(prior_chain.compactify_U(prior_chain.sample_U(key))))(random.split(random.PRNGKey(0),10000))
    lik = vmap(lambda theta: log_likelihood(**theta))(theta)
    sc=plt.scatter(theta['theta'][:,0], theta['theta'][:,1],c=jnp.exp(lik))
    plt.colorbar(sc)
    plt.show()

    ns = NestedSampler(log_likelihood,
                       prior_chain)
    results = jit(ns)(key=random.PRNGKey(4525280))

    summary(results)

    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
