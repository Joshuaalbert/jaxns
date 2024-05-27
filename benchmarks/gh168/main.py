import time

import jax
import jax.numpy as jnp
import numpy as np
import pkg_resources
import tensorflow_probability.substrates.jax as tfp
from jax._src.scipy.linalg import solve_triangular

from jaxns import Model, Prior, DefaultNestedSampler

tfpd = tfp.distributions


def run_model(key):
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
            - 0.5 * dx @ dx

    ndims = 8
    prior_mu = 15 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.99, data_cov)

    log_Z_true = log_normal(data_mu, prior_mu, prior_cov + data_cov)
    # not super happy with this being 1.58 and being off by like 0.1. Probably related to the ESS.
    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    # print(f"True post mu:{post_mu}")
    # print(f"True log Z: {log_Z_true}")

    def prior_model():
        x = yield Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x')
        return x

    def log_likelihood(x):
        return tfpd.MultivariateNormalTriL(loc=data_mu, scale_tril=jnp.linalg.cholesky(data_cov)).log_prob(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    ns = DefaultNestedSampler(model=model, max_samples=100000, verbose=False)

    termination_reason, state = ns(key)
    results = ns.to_results(termination_reason=termination_reason, state=state, trim=False)
    return results.log_Z_mean - log_Z_true, results.log_Z_uncert


def main():
    jaxns_version = pkg_resources.get_distribution("jaxns").version
    m = 10
    run_model_aot = jax.jit(run_model).lower(jax.random.PRNGKey(0)).compile()
    dt = []

    errors = []
    uncerts = []

    for i in range(m):
        t0 = time.time()
        log_Z_error, log_Z_uncert = run_model_aot(jax.random.PRNGKey(i))
        log_Z_error.block_until_ready()
        t1 = time.time()
        dt.append(t1 - t0)
        errors.append(log_Z_error)
        uncerts.append(log_Z_uncert)
    total_time = sum(dt)
    best_3 = sum(sorted(dt)[:3]) / 3.
    # print(f"Errors: {errors}")
    # print(f"Uncerts: {uncerts}")
    print(f"JAXNS {jaxns_version}\n"
          f"\tMean error: {np.mean(errors)}\n"
          f"\tMean uncert: {np.mean(uncerts)}\n"
          f"Avg. time taken: {total_time / m:.5f} seconds.\n"
          f"The best 3 of {m} runs took {best_3:.5f} seconds.")

    with open('results', 'a') as fp:
        fp.write(f"{jaxns_version},{np.mean(errors)},{np.mean(uncerts)},{total_time / m},{best_3}\n")

# Before fix
# 2.5.0,2.851858615875244,0.3351728320121765,0.7272443532943725,0.7075355052947998

# After fix
# 2.4.12,0.42119064927101135,0.3309990465641022,1.001870584487915,0.9800511995951334
if __name__ == '__main__':
    main()
