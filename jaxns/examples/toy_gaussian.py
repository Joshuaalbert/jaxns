from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, MVNDiagPrior, MVNPrior, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics, plot_samples_development
from timeit import default_timer
from jax import random, jit, disable_jit, make_jaxpr
from jax import numpy as jnp
import pylab as plt


def main():


    ndims = 4
    sigma = 0.1

    def log_likelihood(theta, **kwargs):
        r2 = jnp.sum(theta**2)
        logL = -0.5*jnp.log(2.*jnp.pi*sigma**2)*ndims
        logL += -0.5*r2/sigma**2
        return logL


    prior_transform = PriorChain().push(UniformPrior('theta', -jnp.ones(ndims), jnp.ones(ndims)))
    ns = NestedSampler(log_likelihood, prior_transform, sampler_name='slice')

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                      sampler_kwargs=dict(depth=3))

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print(results.efficiency)
        print("Time to run including compile:", default_timer() - t0)
        print("Time efficiency normalised:", results.efficiency*(default_timer() - t0))
        t0 = default_timer()
        results = run(random.PRNGKey(1))
        print(results.efficiency)
        print("Time to run no compile:", default_timer() - t0)
        print("Time efficiency normalised:", results.efficiency * (default_timer() - t0))
        return results

    for n in [1000]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)

    plt.show()

    # plot_samples_development(results, save_name='./example.mp4')
    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    # import jax.profiler
    # server = jax.profiler.start_server(9999)
    # input("Ready? ")
    main()
