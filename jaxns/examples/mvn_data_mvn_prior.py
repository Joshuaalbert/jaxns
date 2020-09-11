from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, MVNDiagPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax.scipy.linalg import solve_triangular
from jax import random, jit, disable_jit, make_jaxpr
from jax import numpy as jnp
import pylab as plt


def main():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 2
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.ones(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu
    post_cov = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_cov

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    prior_transform = PriorChain().push(MVNDiagPrior('x', prior_mu, jnp.sqrt(jnp.diag(prior_cov))))
    # prior_transform = LaplacePrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    # prior_transform = UniformPrior(-20.*jnp.ones(ndims), 20.*jnp.ones(ndims))
    ns = NestedSampler(log_likelihood, prior_transform, sampler_name='ellipsoid')

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n * ndims,
                      max_samples=1e4,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=True)
        print(make_jaxpr(run)(random.PRNGKey(0)))
        results = run(random.PRNGKey(0))
        return results

    for n in [10]:
        with disable_jit():
            results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.hlines(true_logZ, 50, 200)
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)

    print("True logZ={}".format(true_logZ))
    print("True posterior m={}\nCov={}".format(post_mu, post_cov))


if __name__ == '__main__':
    import jax.profiler
    server = jax.profiler.start_server(9999)
    input("Ready? ")
    main()
