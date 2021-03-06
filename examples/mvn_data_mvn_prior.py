from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import PriorChain, MVNPrior, MVNPrior
from jaxns.utils import summary
from jaxns.plotting import plot_cornerplot, plot_diagnostics, plot_samples_development
from jax.scipy.linalg import solve_triangular
from jax import random, jit, disable_jit, make_jaxpr
from jax import numpy as jnp
import pylab as plt
from timeit import default_timer
import os


def main():
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={4}"

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 8
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu
    post_cov = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_cov

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    print("True logZ={}".format(true_logZ))
    print("True posterior m={}\nCov={}".format(post_mu, post_cov))

    def run_with_n(n):
        prior_transform = PriorChain().push(MVNPrior('x', prior_mu, prior_cov))

        def param_mean(x, **args):
            return x

        def param_covariance(x, **args):
            return jnp.outer(x, x)

        ns = NestedSampler(log_likelihood, prior_transform, sampler_name='slice',
                           num_live_points=n,
                           max_samples=1e6,
                           collect_samples=True,
                           num_parallel_samplers=2,
                           sampler_kwargs=dict(depth=2, num_slices=5 * ndims),
                           marginalised=dict(x_mean=param_mean,
                                             x_cov=param_covariance)
                           )

        @jit
        def run(key):
            return ns(key=key, termination_frac=0.001)

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print('efficiency', results.efficiency)
        print("time to run including compile", default_timer() - t0)
        t0 = default_timer()
        results = run(random.PRNGKey(747645))
        print('efficiency', results.efficiency, results.num_samples)
        print("time to run not including compile", default_timer() - t0)
        return results

    for n in [100]:
        results = run_with_n(n)
        # can always save results to play with later
        save_results(results, 'save.npz')
        # loads results that you may have saved
        results = load_results('save.npz')

        summary(results)

        print(results.marginalised['x_mean'],
              results.marginalised['x_cov'] - jnp.outer(results.marginalised['x_mean'], results.marginalised['x_mean']))
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)

    plt.hlines(true_logZ, 0, n)
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
