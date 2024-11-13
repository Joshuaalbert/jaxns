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


def build_run_model(num_slices, gradient_guided, ndims):
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

        # print("True logZ={}".format(true_logZ))
        # print("True post_mu={}".format(post_mu))
        # print("True post_cov={}".format(post_cov))

        # KL posterior || prior
        dist_posterior = tfpd.MultivariateNormalFullCovariance(loc=post_mu, covariance_matrix=post_cov)
        dist_prior = tfpd.MultivariateNormalFullCovariance(loc=prior_mu, covariance_matrix=prior_cov)
        H_true = -tfp.distributions.kl_divergence(dist_posterior, dist_prior)

        # print("True H={}".format(H_true))

        def prior_model():
            x = yield Prior(
                tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
                name='x')
            return x

        def log_likelihood(x):
            return tfpd.MultivariateNormalTriL(loc=data_mu, scale_tril=jnp.linalg.cholesky(data_cov)).log_prob(x)

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

        ns = NestedSampler(model=model, verbose=False, k=0, num_slices=num_slices, gradient_guided=gradient_guided)

        termination_reason, state = ns(key)
        results = ns.to_results(termination_reason=termination_reason, state=state, trim=False)

        error = results.H_mean - H_true
        log_Z_error = results.log_Z_mean - true_logZ
        return results.H_mean, H_true, error, log_Z_error

    return run_model


def get_data(ndims):
    prior_mu = 15 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.99, data_cov)
    return prior_mu, prior_cov, data_mu, data_cov


def main():
    jaxns_version = pkg_resources.get_distribution("jaxns").version
    m = 90
    d = 16

    data = get_data(d)

    # Row 1: Plot logZ error for gradient guided vs baseline for different s, with errorbars
    # Row 2: Plot H error for gradient guided vs baseline for different s, with errorbars
    # Row 3: Plot time taken for gradient guided vs baseline for different s, with errorbars

    s_array = [0.5, 1, 2, 3, 4, 5]

    run_model_baseline_aot_array = [
        jax.jit(build_run_model(num_slices=int(s * d), gradient_guided=False, ndims=d)).lower(jax.random.PRNGKey(0),
                                                                                              *data).compile() for
        s in
        s_array]
    run_model_gg_aot_array = [
        jax.jit(build_run_model(num_slices=int(s * d), gradient_guided=True, ndims=d)).lower(jax.random.PRNGKey(0),
                                                                                             *data).compile() for s
        in
        s_array]

    H_errors = np.zeros((len(s_array), m, 2))
    log_z_errors = np.zeros((len(s_array), m, 2))
    dt = np.zeros((len(s_array), m, 2))

    for s_idx in range(len(s_array)):
        s = s_array[s_idx]
        for i in range(m):
            key = jax.random.PRNGKey(i * 42)
            baseline_model = run_model_baseline_aot_array[s_idx]
            gg_model = run_model_gg_aot_array[s_idx]
            t0 = time.time()
            H, H_true, H_error, log_Z_error = jax.block_until_ready(baseline_model(key, *data))
            t1 = time.time()
            dt[s_idx, i, 0] = t1 - t0
            H_errors[s_idx, i, 0] = H_error
            log_z_errors[s_idx, i, 0] = log_Z_error
            print(f"Baseline: i={i} k=0 s={s} H={H} H_true={H_true} H_error={H_error} log_Z_error={log_Z_error}")
            t0 = time.time()
            H, H_true, H_error, log_Z_error = jax.block_until_ready(gg_model(key, *data))
            t1 = time.time()
            dt[s_idx, i, 1] = t1 - t0
            H_errors[s_idx, i, 1] = H_error
            log_z_errors[s_idx, i, 1] = log_Z_error
            print(f"GG: i={i} k=0 s={s} H={H} H_true={H_true} H_error={H_error} log_Z_error={log_Z_error}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    # Row 1
    H_error_mean = np.mean(H_errors, axis=1)  # [s, 2]
    H_error_std = np.std(H_errors, axis=1)  # [s, 2]
    axs[0].plot(s_array, H_error_mean[:, 0], label="Baseline", c='b')
    axs[0].plot(s_array, H_error_mean[:, 1], label="Gradient Guided", c='r')
    axs[0].fill_between(s_array, H_error_mean[:, 0] - H_error_std[:, 0], H_error_mean[:, 0] + H_error_std[:, 0],
                        color='b', alpha=0.2)
    axs[0].fill_between(s_array, H_error_mean[:, 1] - H_error_std[:, 1], H_error_mean[:, 1] + H_error_std[:, 1],
                        color='r', alpha=0.2)
    axs[0].set_ylabel("H error")
    axs[0].legend()

    # Row 2
    logZ_error_mean = np.mean(log_z_errors, axis=1)  # [s, 2]
    logZ_error_std = np.std(log_z_errors, axis=1)  # [s, 2]
    axs[1].plot(s_array, logZ_error_mean[:, 0], label="Baseline", c='b')
    axs[1].plot(s_array, logZ_error_mean[:, 1], label="Gradient Guided", c='r')
    axs[1].fill_between(s_array, logZ_error_mean[:, 0] - logZ_error_std[:, 0],
                        logZ_error_mean[:, 0] + logZ_error_std[:, 0], color='b', alpha=0.2)
    axs[1].fill_between(s_array, logZ_error_mean[:, 1] - logZ_error_std[:, 1],
                        logZ_error_mean[:, 1] + logZ_error_std[:, 1], color='r', alpha=0.2)
    axs[1].set_ylabel("logZ error")
    axs[1].legend()

    # Row 3
    dt_mean = np.mean(dt, axis=1)  # [s, 2]
    dt_std = np.std(dt, axis=1)  # [s, 2]
    axs[2].plot(s_array, dt_mean[:, 0], label="Baseline", c='b')
    axs[2].plot(s_array, dt_mean[:, 1], label="Gradient Guided", c='r')
    axs[2].fill_between(s_array, dt_mean[:, 0] - dt_std[:, 0], dt_mean[:, 0] + dt_std[:, 0], color='b', alpha=0.2)
    axs[2].fill_between(s_array, dt_mean[:, 1] - dt_std[:, 1], dt_mean[:, 1] + dt_std[:, 1], color='r', alpha=0.2)
    axs[2].set_ylabel("Time taken")
    axs[2].legend()
    axs[2].set_xlabel(r"s, slices per dim")

    axs[0].set_title(f"Gradient guided vs baseline, D={d}, v{jaxns_version}")

    plt.savefig(f"Gradient_guided_vs_baseline_D{d}_v{jaxns_version}.png")

    plt.show()


if __name__ == '__main__':
    main()
