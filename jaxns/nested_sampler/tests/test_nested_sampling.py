import numpy as np
from jax import numpy as jnp, random, jit
from jax._src.scipy.linalg import solve_triangular
from jax._src.scipy.special import gammaln

from jaxns import NestedSampler, resample, summary, plot_diagnostics
from jaxns.nested_sampler.nested_sampling import _get_static_goal, _get_likelihood_maximisation_goal, \
    compute_remaining_evidence
from jaxns.nested_sampler.utils import evidence_posterior_samples
from jaxns.prior_transforms import PriorChain, UniformPrior, MVNPrior, GammaPrior


def test_shrinkage():
    n = 1
    def log_likelihood(x):
        return jnp.log(1. - x**n)

    def exact_X(L):
        return (1. - L)**(1./n)

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(log_likelihood, prior_chain)
    results = ns(key=random.PRNGKey(43), adaptive_evidence_stopping_threshold=0.01)
    print(list(results.log_L_samples[:500]))
    print(list(results.log_X_mean[:500]))


    import pylab as plt
    plt.plot(results.log_L_samples, results.log_X_mean, label='predict')
    plt.plot(results.log_L_samples, jnp.log(exact_X(jnp.exp(results.log_L_samples))), label='exact')
    plt.legend()
    plt.show()

    plt.plot(results.log_L_samples,
             results.log_X_mean - jnp.log(exact_X(jnp.exp(results.log_L_samples))),
             label='diff')
    plt.legend()
    plt.show()

    x_exact = exact_X(jnp.exp(results.log_L_samples))
    X = jnp.concatenate([jnp.asarray([1.]), x_exact])
    n = X[1:]/(X[:-1] - X[1:])
    plt.plot(n[:results.total_num_samples], label='exact n')
    plt.plot(results.num_live_points_per_sample[:results.total_num_samples], label='predict n')
    plt.ylim(0,500)
    plt.legend()
    plt.show()

    plt.plot(n[:results.total_num_samples] - results.num_live_points_per_sample[:results.total_num_samples], label='diff n')
    plt.ylim(-500,500)
    plt.legend()
    plt.show()



def test_nested_sampling_basic():
    def log_likelihood(x):
        return - jnp.sum(x**2)

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(log_likelihood, prior_chain)
    results = ns(key=random.PRNGKey(43), num_live_points=None, termination_live_evidence_frac=1e-2)
    plot_diagnostics(results)
    summary(results)

    log_Z_samples = evidence_posterior_samples(random.PRNGKey(42),
                                               results.num_live_points_per_sample[:results.total_num_samples],
                                               results.log_L_samples[:results.total_num_samples], S=1000)
    assert jnp.isclose(results.log_Z_mean, jnp.mean(log_Z_samples), atol=1e-3)
    assert jnp.isclose(results.log_Z_uncert,jnp.std(log_Z_samples), atol=1e-3)

    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, -1./3., atol=1.75*results.log_Z_uncert)


def test_nested_sampling_basic_parallel():

    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    def log_likelihood(x):
        return - jnp.sum(x**2)

    with PriorChain() as prior_chain:
        UniformPrior('x', 0., 1.)

    ns = NestedSampler(log_likelihood, prior_chain, num_parallel_samplers=2)
    results = ns(key=random.PRNGKey(42))

    ns_serial = NestedSampler(log_likelihood, prior_chain)
    results_serial = ns_serial(key=random.PRNGKey(42))
    assert jnp.isclose(results_serial.log_Z_mean, results.log_Z_mean)


def test_nested_sampling_mvn():
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

    ns = NestedSampler(log_likelihood, prior_chain, sampler_kwargs=dict(min_num_slices=prior_chain.U_ndims*10,
                                                                        max_num_slices=prior_chain.U_ndims*20))
    results = jit(ns)(key=random.PRNGKey(42))
    summary(results)
    plot_diagnostics(results)
    assert jnp.isclose(results.log_Z_mean, true_logZ, atol=1.75 * results.log_Z_uncert)

    # evidence better with gradient boost
    ns = NestedSampler(log_likelihood, prior_chain, sampler_kwargs=dict(gradient_boost=True,
                                                                        min_num_slices=prior_chain.U_ndims * 10,
                                                                        max_num_slices=prior_chain.U_ndims * 20
                                                                        ))
    results = jit(ns)(key=random.PRNGKey(43))
    summary(results)
    plot_diagnostics(results)
    assert jnp.isclose(results.log_Z_mean,  true_logZ, atol=1.75 * results.log_Z_uncert)


def test_nested_sampling_dynamic():
    from jaxns.plotting import plot_diagnostics, plot_cornerplot
    from jaxns import evidence_posterior_samples
    from jaxns.nested_sampler.utils import summary
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 16
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    with PriorChain() as prior_chain:
        MVNPrior('x', prior_mu, prior_cov)

    ns = NestedSampler(log_likelihood, prior_chain, dynamic=True,
                       sampler_kwargs=dict(midpoint_shrink=False,
                                           gradient_boost=False,
                                           min_num_slices=prior_chain.U_ndims*15,
                                           max_num_slices=prior_chain.U_ndims*20))
    results = ns(key=random.PRNGKey(42),
                 dynamic_kwargs=dict(G=0.),
                 termination_evidence_uncert=1e-3,
                 termination_live_evidence_frac=1e-5,
                 termination_ess=None,
                 termination_max_num_steps=100)
    print(results)
    print(f"True posterior mean: {post_mu}")
    print(f"True log(Z): {true_logZ}")
    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)
    log_Z_samples = evidence_posterior_samples(random.PRNGKey(42), results.num_live_points_per_sample, results.log_L_samples,S=1000)
    import pylab as plt

    plt.hist(log_Z_samples, bins='auto')
    plt.xlim(-7., -5.)
    plt.show()
    assert jnp.isclose(results.log_Z_mean, true_logZ, atol= 1.75 * results.log_Z_uncert)


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


def test_static_goal():

    static_num_live_points = 2
    num_samples = 5
    num_live_points = jnp.asarray([0,0,0,0,0])
    expect = jnp.asarray([0,0,1,1,2])
    assert jnp.allclose(_get_static_goal(num_live_points, static_num_live_points, num_samples), expect)

    num_live_points = jnp.asarray([2, 0, 0, 0, 0])
    expect = jnp.asarray([1,1,2,2,3])
    assert jnp.allclose(_get_static_goal(num_live_points, static_num_live_points, num_samples), expect)

    num_live_points = jnp.asarray([3, 0, 0, 0, 0])
    expect = jnp.asarray([1, 1, 2, 2, 3])
    assert jnp.allclose(_get_static_goal(num_live_points, static_num_live_points, num_samples), expect)

    num_live_points = jnp.asarray([2, 2, 2, 2, 2])
    expect = jnp.asarray([4,4,4,4,4])
    assert jnp.allclose(_get_static_goal(num_live_points, static_num_live_points, num_samples), expect)


def test_get_likelihood_maximisation_goal():
    sample_idx = 3
    search_top_n = 3
    # search_top_n=2, log_L_samples=[a, b, c, 0, 0], sample_idx=3, contours=[l0,a,b,c,0], search from a (i=1) to b (i=2)
    contours = jnp.asarray([-jnp.inf, 1, 2, 3, jnp.inf, jnp.inf])
    expect = jnp.asarray([0,0,0,-jnp.inf, -jnp.inf, -jnp.inf])
    assert jnp.allclose(_get_likelihood_maximisation_goal(contours, sample_idx, search_top_n), expect)

    sample_idx = 3
    search_top_n = 2
    # search_top_n=2, log_L_samples=[a, b, c, 0, 0], sample_idx=3, contours=[l0,a,b,c,0], search from a (i=1) to b (i=2)
    contours = jnp.asarray([-jnp.inf, 1, 2, 3, jnp.inf, jnp.inf])
    expect = jnp.asarray([-jnp.inf, 0, 0, -jnp.inf, -jnp.inf, -jnp.inf])
    assert jnp.allclose(_get_likelihood_maximisation_goal(contours, sample_idx, search_top_n), expect)


def test_compute_remaining_evidence():
    # [a,b,-inf], 2 -> [a+b, b, -inf]
    log_dZ_mean = jnp.asarray([0., 1., -jnp.inf])
    sample_idx=2
    expect = jnp.asarray([jnp.logaddexp(0,1),1,-jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)

    # [-inf, -inf,-inf], 0 -> [-inf, -inf, -inf]
    log_dZ_mean = jnp.asarray([-jnp.inf, -jnp.inf -jnp.inf])
    sample_idx = 0
    expect = jnp.asarray([-jnp.inf, -jnp.inf - jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)