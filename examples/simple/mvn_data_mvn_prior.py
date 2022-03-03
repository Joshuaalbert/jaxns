from jaxns.nested_sampler.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import PriorChain, MVNPrior
from jaxns.utils import summary
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax.scipy.linalg import solve_triangular
from jax import random, jit
from jax import numpy as jnp
from timeit import default_timer

def main():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 3
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.99, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu
    post_cov = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_cov

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    print("True logZ={}".format(true_logZ))
    print("True post_mu={}".format(post_mu))

    with PriorChain() as prior_chain:
        MVNPrior('x', prior_mu, prior_cov)

    def param_mean(x, **args):
        return x

    def param_covariance(x, **args):
        return jnp.outer(x, x)

    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_kwargs=dict(midpoint_shrink=False, num_slices=prior_chain.U_ndims*10),
                       dynamic=False
                       )

    ns = jit(ns)
    results = ns(random.PRNGKey(4525325), terminate_max_num_threads=50, terminate_evidence_uncert=1e-2, termination_ess=None,dynamic_kwargs=dict(G=0.))
    # run once to make sure it compiles
    results.log_efficiency.block_until_ready()
    t0 = default_timer()
    # run again and time it
    results = ns(random.PRNGKey(4525325), terminate_max_num_threads=50,terminate_evidence_uncert=1e-2, termination_ess=None, dynamic_kwargs=dict(G=0.))
    results.log_efficiency.block_until_ready()
    run_time = default_timer() - t0
    print(f"Total run time: {run_time}")

    # # can always save results to play with later
    # save_results(results, 'save.npz')
    # # loads results that you may have saved
    # results = load_results('save.npz')

    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
