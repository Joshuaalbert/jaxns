from jaxns.gaussian_process.kernels import RBF, M12
from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
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

    theta = vmap(lambda key: prior_chain(random.uniform(key, (prior_chain.U_ndims,))))(random.split(random.PRNGKey(0),10000))
    lik = vmap(lambda theta: log_likelihood(**theta))(theta)
    sc=plt.scatter(theta['theta'][:,0], theta['theta'][:,1],c=jnp.exp(lik))
    plt.colorbar(sc)
    plt.show()

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice')

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      sampler_kwargs=dict(depth=7, num_slices=1))

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print("Efficiency", results.efficiency)
        print("Time to run (including compile)", default_timer() - t0)
        t0 = default_timer()
        results = run(random.PRNGKey(1))
        print(results.efficiency)
        print("Time to run (no compile)", default_timer() - t0)
        return results

    for n in [1000]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.ylabel('log Z')
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr


if __name__ == '__main__':
    main()
