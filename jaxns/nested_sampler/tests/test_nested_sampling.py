import numpy as np
from jax import numpy as jnp, random, jit
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln

from jaxns import NestedSampler, resample, summary, plot_diagnostics, analytic_log_evidence
from jaxns.nested_sampler.nested_sampling import compute_remaining_evidence
from jaxns.nested_sampler.utils import evidence_posterior_samples
from jaxns.prior_transforms import PriorChain, UniformPrior, MVNPrior, GammaPrior


def test_shrinkage():
    n = 2

    def log_likelihood(x):
        return jnp.log(1. - x ** n)

    def exact_X(L):
        return (1. - L) ** (1. / n)

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(log_likelihood, prior_chain)
    results = ns(key=random.PRNGKey(43))
    ns.summary(results)

    # print(list(results.log_X_mean[:results.total_num_samples]))
    # print(list(results.log_L_samples[:results.total_num_samples]))
    # print(list(jnp.log(exact_X(jnp.exp(results.log_L_samples[:results.total_num_samples])))))
    diff = results.log_X_mean[:results.total_num_samples] - jnp.log(
        exact_X(jnp.exp(results.log_L_samples[:results.total_num_samples])))
    diff = jnp.where(jnp.isfinite(diff), diff, jnp.nan)
    # print(jnp.nanstd(diff))
    assert jnp.nanstd(diff) < 0.09


def test_nested_sampling_basic():
    def log_likelihood(x):
        return - jnp.sum(x ** 2)

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(log_likelihood, prior_chain)
    results = ns(key=random.PRNGKey(42))
    plot_diagnostics(results)
    summary(results)
    ns.plot_cornerplot(results)

    log_Z_true = analytic_log_evidence(prior_chain, log_likelihood, S=200)

    log_Z_samples = evidence_posterior_samples(random.PRNGKey(42),
                                               results.num_live_points_per_sample[:results.total_num_samples],
                                               results.log_L_samples[:results.total_num_samples], S=1000)

    assert jnp.isclose(results.log_Z_mean, jnp.mean(log_Z_samples), atol=1e-3)
    assert jnp.isclose(results.log_Z_uncert, jnp.std(log_Z_samples), atol=1e-3)

    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=1.75 * results.log_Z_uncert)


def test_nested_sampling_plateau():
    def log_likelihood(x):
        return 0.

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(log_likelihood, prior_chain)

    results = ns(key=random.PRNGKey(43))
    plot_diagnostics(results)
    summary(results)
    ns.plot_cornerplot(results)

    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, 0., atol=1.75 * results.log_Z_uncert)


def test_nested_sampling_basic_parallel():
    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    def log_likelihood(x):
        return - jnp.sum(x ** 2)

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(log_likelihood, prior_chain, num_parallel_samplers=2)
    results = ns(key=random.PRNGKey(42))
    ns.plot_diagnostics(results)

    ns_serial = NestedSampler(log_likelihood, prior_chain)
    results_serial = ns_serial(key=random.PRNGKey(42))
    assert jnp.isclose(results_serial.log_Z_mean, results.log_Z_mean, atol=1e-3)


def test_nested_sampling_mvn_static():
    from jaxns import summary
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 4
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)
    # not super happy with this being 1.58 and being off by like 0.1. Probably related to the ESS.
    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    print(f"True post mu:{post_mu}")
    print(f"True log Z: {true_logZ}")

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    with PriorChain() as prior_chain:
        MVNPrior('x', prior_mu, prior_cov)

    ns = NestedSampler(log_likelihood, prior_chain)
    results = ns(key=random.PRNGKey(42))
    ns.summary(results)
    ns.plot_diagnostics(results)
    assert jnp.isclose(results.log_Z_mean, true_logZ, atol=1.75 * results.log_Z_uncert)


def test_nested_sampling_mvn_dynamic():
    # TODO: passing, but not with the correct results. Need to change the test.
    from jaxns import summary
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 4
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)
    # not super happy with this being 1.58 and being off by like 0.1. Probably related to the ESS.
    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    print(f"True post mu:{post_mu}")
    print(f"True log Z: {true_logZ}")

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    with PriorChain() as prior_chain:
        MVNPrior('x', prior_mu, prior_cov)

    ns = NestedSampler(log_likelihood, prior_chain, dynamic=True)
    results = ns(key=random.PRNGKey(42),
                 adaptive_evidence_patience=1,
                 G=0.,
                 termination_evidence_uncert=0.05,
                 termination_max_num_steps=100)
    ns.summary(results)
    ns.plot_diagnostics(results)
    assert jnp.isclose(results.log_Z_mean, true_logZ, atol=1.75 * results.log_Z_uncert)


def test_gh21():
    num_samples = 10
    true_k = 1.
    true_theta = 0.5

    _gamma = np.random.gamma(true_k, true_theta, size=num_samples)
    samples = jnp.asarray(np.random.poisson(_gamma, size=num_samples))

    prior_k = 5.
    prior_theta = 0.3

    true_post_k = prior_k + jnp.sum(samples)
    true_post_theta = prior_theta / (num_samples * prior_theta + 1.)

    def log_likelihood(gamma, **kwargs):
        """
        Poisson likelihood.
        """
        return jnp.sum(samples * jnp.log(gamma) - gamma - gammaln(samples + 1))

    with PriorChain() as prior_chain:
        gamma = GammaPrior('gamma', prior_k, prior_theta)

    ns = NestedSampler(loglikelihood=log_likelihood, prior_chain=prior_chain)
    results = jit(ns)(random.PRNGKey(32564))

    samples = resample(random.PRNGKey(43083245), results.samples, results.log_dp_mean, S=int(results.ESS))

    sample_mean = jnp.mean(samples['gamma'], axis=0)

    true_mean = true_post_k * true_post_theta

    assert jnp.allclose(sample_mean, true_mean, atol=0.05)


def test_compute_remaining_evidence():
    # [a,b,-inf], 2 -> [a+b, b, -inf]
    log_dZ_mean = jnp.asarray([0., 1., -jnp.inf])
    sample_idx = 2
    expect = jnp.asarray([jnp.logaddexp(0, 1), 1, -jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)

    # [-inf, -inf,-inf], 0 -> [-inf, -inf, -inf]
    log_dZ_mean = jnp.asarray([-jnp.inf, -jnp.inf - jnp.inf])
    sample_idx = 0
    expect = jnp.asarray([-jnp.inf, -jnp.inf - jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)
