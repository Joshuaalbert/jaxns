import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import tensorflow_probability.substrates.jax as tfp
from jax._src.scipy.linalg import solve_triangular

from jaxns import Model, Prior

try:
    from jaxns import NestedSampler
except ImportError:
    from jaxns import DefaultNestedSampler as NestedSampler

tfpd = tfp.distributions


def run_model(key, prior_mu, prior_cov, data_mu, data_cov):
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
            - 0.5 * dx @ dx

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    J = jnp.linalg.solve(data_cov + prior_cov, prior_cov)
    post_mu = prior_mu + J.T @ (data_mu - prior_mu)
    post_cov = prior_cov - J.T @ (prior_cov + data_cov) @ J

    print("True logZ={}".format(true_logZ))
    print("True post_mu={}".format(post_mu))
    print("True post_cov={}".format(post_cov))

    # KL posterior || prior
    dist_posterior = tfpd.MultivariateNormalFullCovariance(loc=post_mu, covariance_matrix=post_cov)
    dist_prior = tfpd.MultivariateNormalFullCovariance(loc=prior_mu, covariance_matrix=prior_cov)
    H_true = -tfp.distributions.kl_divergence(dist_posterior, dist_prior)
    print("True H={}".format(H_true))

    def prior_model():
        x = yield Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x')
        return x

    def log_likelihood(x):
        return tfpd.MultivariateNormalTriL(loc=data_mu, scale_tril=jnp.linalg.cholesky(data_cov)).log_prob(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    ns = NestedSampler(model=model, verbose=False)

    termination_reason, state = ns(key)
    results = ns.to_results(termination_reason=termination_reason, state=state, trim=False)

    error = results.H_mean - H_true
    log_Z_error = results.log_Z_mean - true_logZ
    return results.H_mean, H_true, error, log_Z_error


def get_data():
    ndims = 8
    prior_mu = 15 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.99, data_cov)
    return prior_mu, prior_cov, data_mu, data_cov


def main():
    jaxns_version = pkg_resources.get_distribution("jaxns").version
    m = 100

    data = get_data()
    run_model_aot = jax.jit(run_model).lower(jax.random.PRNGKey(0), *data).compile()

    dt = []
    H_errors = []
    log_z_errors = []

    for i in range(m):
        t0 = time.time()
        H, H_true, H_error, log_Z_error = jax.block_until_ready(run_model_aot(jax.random.PRNGKey(i), *data))
        print(f"i={i} H={H} H_true={H_true} H_error={H_error} log_Z_error={log_Z_error}")
        t1 = time.time()
        dt.append(t1 - t0)
        H_errors.append(H_error)
        log_z_errors.append(log_Z_error)
    total_time = sum(dt)
    best_3 = sum(sorted(dt)[:3]) / 3.
    # print(f"Errors: {errors}")
    # print(f"Uncerts: {uncerts}")
    print(f"JAXNS {jaxns_version}\n"
          f"\tMean error: {np.mean(H_errors)}\n"
          f"Avg. time taken: {total_time / m:.5f} seconds.\n"
          f"The best 3 of {m} runs took {best_3:.5f} seconds.")

    with open('results', 'a') as fp:
        # jaxns_version,mean_H_error,mean_logZ_error,avg_time,best_3
        fp.write(f"{jaxns_version},{np.mean(H_errors)},{np.mean(log_z_errors)},{total_time / m},{best_3}\n")

    plt.scatter(log_z_errors, H_errors)
    plt.title("H error vs log Z error")
    plt.ylabel("H error [nat]")
    plt.xlabel("log Z error [nat]")
    plt.savefig(f"H_error_logZ_error_v{jaxns_version}.png")
    plt.show()


if __name__ == '__main__':
    main()
