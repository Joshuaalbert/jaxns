from time import monotonic_ns

import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp
from jax._src.lax.control_flow import scan
from jax._src.scipy.linalg import solve_triangular

from jaxns.internals.log_semiring import LogSpace
from jaxns.model import Model
from jaxns.nested_sampler import ApproximateNestedSampler, ExactNestedSampler
from jaxns.prior import PriorModelGen, Prior
from jaxns.static_uniform import BadUniformSampler
from jaxns.types import TerminationCondition, float_type
from jaxns.utils import sample_evidence, bruteforce_evidence

tfpd = tfp.distributions


class Timer:
    def __enter__(self):
        self.t0 = monotonic_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time to execute: {(monotonic_ns() - self.t0) / 1e9} seconds.")


def test_approximate_nested_sampler():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=jnp.asarray([1., 1.])))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -jnp.sum(z ** 2)

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    approx_ns = ApproximateNestedSampler(model=model, num_live_points=50, num_parallel_samplers=1,
                                         max_samples=1000)
    # print(termination_reason)
    with Timer():
        termination_reason, state = approx_ns(random.PRNGKey(42),
                                              term_cond=TerminationCondition(live_evidence_frac=1e-4))
        results = approx_ns.to_results(state, termination_reason)
        results.log_Z_mean.block_until_ready()
    approx_ns.summary(approx_ns.to_results(state, termination_reason))

    with Timer():
        termination_reason, state = approx_ns(random.PRNGKey(42),
                                              term_cond=TerminationCondition(live_evidence_frac=1e-4))
        results = approx_ns.to_results(state, termination_reason)
        results.log_Z_mean.block_until_ready()


def test_exact_nested_sampler():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -z ** 2

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    model.sanity_check(random.PRNGKey(0), S=100)

    exact_ns = ExactNestedSampler(model=model, num_live_points=50, num_parallel_samplers=1, max_samples=1000,
                                  patience=1)
    # print(termination_reason)
    # print(state)
    with Timer():
        termination_reason, state = exact_ns(random.PRNGKey(42),
                                             term_cond=TerminationCondition(live_evidence_frac=1e-4))
        results = exact_ns.to_results(state, termination_reason)
        exact_ns.summary(results)
        exact_ns.plot_diagnostics(results)

    with Timer():
        termination_reason, state = exact_ns(random.PRNGKey(42),
                                             term_cond=TerminationCondition(live_evidence_frac=1e-4))
        results = exact_ns.to_results(state, termination_reason)


def test_nested_sampling_basic():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1), name='x')
        return x

    def log_likelihood(x):
        return - jnp.sum(x ** 2)

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    exact_ns = ExactNestedSampler(model=model, num_live_points=50, num_parallel_samplers=1,
                                  max_samples=1000)
    with Timer():
        termination_reason, state = exact_ns(random.PRNGKey(42),
                                             term_cond=TerminationCondition(live_evidence_frac=1e-4))
        results = exact_ns.to_results(state, termination_reason)

        # exact_ns.plot_diagnostics(results)
        exact_ns.summary(results)
        # exact_ns.plot_cornerplot(results)

    log_Z_true = bruteforce_evidence(model=model, S=200)

    log_Z_samples = sample_evidence(random.PRNGKey(42),
                                    results.num_live_points_per_sample,
                                    results.log_L_samples, S=1000)

    assert jnp.isclose(results.log_Z_mean, jnp.mean(log_Z_samples), atol=1e-3)
    assert jnp.isclose(results.log_Z_uncert, jnp.std(log_Z_samples), atol=1e-2)

    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=1.75 * results.log_Z_uncert)


#
# def test_nested_sampling_basic_parallel():
#     def prior_model() -> PriorModelGen:
#         x = yield Prior(tfpd.Uniform(low=0, high=1), name='x')
#         return x
#
#     def log_likelihood(x):
#         return - jnp.sum(x ** 2)
#
#     model = Model(prior_model=prior_model,
#                   log_likelihood=log_likelihood)
#     exact_ns = ExactNestedSampler(model=model, num_live_points=50, num_parallel_samplers=2,
#                                   max_samples=1000)
#
#     termination_reason, state = exact_ns(random.PRNGKey(42),
#                                          term_cond=TerminationCondition(live_evidence_frac=1e-4))
#     results = exact_ns.to_results(state, termination_reason)
#
#     exact_ns.plot_diagnostics(results)
#     exact_ns.summary(results)
#     exact_ns.plot_cornerplot(results)
#
#     log_Z_true = bruteforce_evidence(model=model, S=200)
#
#     log_Z_samples = sample_evidence(random.PRNGKey(42),
#                                                results.num_live_points_per_sample,
#                                                results.log_L_samples, S=1000)
#
#     assert jnp.isclose(results.log_Z_mean, jnp.mean(log_Z_samples), atol=1e-3)
#     assert jnp.isclose(results.log_Z_uncert, jnp.std(log_Z_samples), atol=1e-3)
#
#     assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
#     assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=1.75 * results.log_Z_uncert)


def test_nested_sampling_mvn_static():
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

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)
    # not super happy with this being 1.58 and being off by like 0.1. Probably related to the ESS.
    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    print(f"True post mu:{post_mu}")
    print(f"True log Z: {true_logZ}")

    def prior_model() -> PriorModelGen:
        x = yield Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x')
        return x

    def log_likelihood(x):
        return log_normal(x, data_mu, data_cov)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    # model.sanity_check(random.PRNGKey(42), S=100)
    exact_ns = ExactNestedSampler(model=model, num_live_points=100, num_parallel_samplers=1,
                                  max_samples=40000, patience=20)
    with Timer():
        termination_reason, state = exact_ns(random.PRNGKey(41),
                                             term_cond=TerminationCondition(live_evidence_frac=1e-5))
        results = exact_ns.to_results(state, termination_reason)
        exact_ns.summary(results)
    exact_ns.plot_diagnostics(results)
    actual_log_Z_mean = results.log_Z_mean
    expected_log_Z_mean = true_logZ
    tol = 1.75 * results.log_Z_uncert
    assert jnp.isclose(actual_log_Z_mean, expected_log_Z_mean, atol=tol)


# def test_multi_needs_fewer_slices():
#     def log_normal(x, mean, cov):
#         L = jnp.linalg.cholesky(cov)
#         dx = x - mean
#         dx = solve_triangular(L, dx, lower=True)
#         return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * dx @ dx
#
#     ndims = 4
#     prior_mu = 2 * jnp.ones(ndims)
#     prior_cov = jnp.diag(jnp.ones(ndims)) ** 2
#
#     data_mu = jnp.zeros(ndims)
#     data_cov = jnp.diag(jnp.ones(ndims)) ** 2
#     data_cov = jnp.where(data_cov == 0., 0.95, data_cov)
#
#     true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)
#     # not super happy with this being 1.58 and being off by like 0.1. Probably related to the ESS.
#     post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
#         prior_cov + data_cov) @ prior_mu
#
#     print(f"True post mu:{post_mu}")
#     print(f"True log Z: {true_logZ}")
#
#     def prior_model() -> PriorModelGen:
#         x = yield Prior(
#             tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
#             name='x')
#         return x
#
#     def log_likelihood(x):
#         return log_normal(x, data_mu, data_cov)
#
#     model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
#
#     model.sanity_check(random.PRNGKey(52), S=100)
#     exact_ns = ExactNestedSampler(model=model, num_live_points=200, num_parallel_samplers=1,
#                                   max_samples=1e4)
#     with Timer():
#         print("Using Multi Dimensional Slice Sampling")
#         termination_reason, state = exact_ns(random.PRNGKey(42),
#                                              term_cond=TerminationCondition(live_evidence_frac=1e-4))
#         results = exact_ns.to_results(state, termination_reason)
#         exact_ns.summary(results)
#         num_slices_multi = results.total_num_slices
#     assert jnp.isclose(results.log_Z_mean, true_logZ, atol=1.75 * results.log_Z_uncert)
#
#     slice_sampler = UniDimSliceSampler(model=model, midpoint_shrink=True)
#     exact_ns = ExactNestedSampler(model=model, num_live_points=200, num_parallel_samplers=1,
#                                   max_samples=1e4, refinement_slice_sampler=slice_sampler)
#     with Timer():
#         print("Using Uni Dimensional Slice Sampling")
#         termination_reason, state = exact_ns(random.PRNGKey(42),
#                                              term_cond=TerminationCondition(live_evidence_frac=1e-4))
#         results = exact_ns.to_results(state, termination_reason)
#         exact_ns.summary(results)
#         num_slices_uni = results.total_num_slices
#     assert jnp.isclose(results.log_Z_mean, true_logZ, atol=1.75 * results.log_Z_uncert)
#     # TODO(Joshuaalbert): broken test, probably means multidim slice sampling is not robust to structure.
#     # assert num_slices_multi < num_slices_uni


def test_shrinkage():
    n = 2

    # Prior is uniform in U[0,1]
    # Likelihood is 1 - x**n
    # Z = 1 - 1/n+1

    log_Z_true = jnp.log(1. - 1. / (n + 1))

    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1))
        return x

    def log_likelihood(x):
        return jnp.log(1. - x ** n)

    def exact_X(L):
        return (1. - L) ** (1. / n)

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    exact_ns = ExactNestedSampler(model=model, num_live_points=50, num_parallel_samplers=1,
                                  max_samples=1000)

    termination_reason, state = exact_ns(random.PRNGKey(42),
                                         term_cond=TerminationCondition(live_evidence_frac=1e-4))
    results = exact_ns.to_results(state, termination_reason)

    # print(list(results.log_X_mean[:results.total_num_samples]))
    # print(list(results.log_L_samples[:results.total_num_samples]))
    # print(list(jnp.log(exact_X(jnp.exp(results.log_L_samples[:results.total_num_samples])))))
    diff = results.log_X_mean - jnp.log(exact_X(jnp.exp(results.log_L_samples)))
    diff = jnp.where(jnp.isfinite(diff), diff, jnp.nan)
    # print(jnp.nanstd(diff))
    assert jnp.nanstd(diff) < 0.26
    assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=results.log_Z_uncert * 1.75)


def compute_shrinkage_stats(num_live_points):
    def _update_stats(state, num_live_points):
        X_mean = LogSpace(state['log_X_mean'])
        X2_mean = LogSpace(state['log_X2_mean'])

        T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
        T2_mean = LogSpace(- jnp.logaddexp((0.), jnp.log(2.) - jnp.log(num_live_points)))

        next_X_mean = X_mean * T_mean
        next_X2_mean = X2_mean * T2_mean

        return dict(log_X_mean=next_X_mean.log_abs_val, log_X2_mean=next_X2_mean.log_abs_val), dict(
            log_X_mean=next_X_mean.log_abs_val, log_X2_mean=next_X2_mean.log_abs_val)

    state = dict(
        log_X_mean=jnp.asarray(0., float_type),
        log_X2_mean=jnp.asarray(0., float_type),
    )
    _, stats = scan(_update_stats,
                    state,
                    num_live_points)
    log_X_uncert = (LogSpace(stats['log_X2_mean']) - LogSpace(stats['log_X_mean']).square()).sqrt().log_abs_val
    return stats['log_X_mean'], log_X_uncert


def test_adaptive_refinement():
    n = 2

    # Prior is uniform in U[0,1]
    # Likelihood is 1 - x**n
    # Z = 1 - 1/n+1

    log_Z_true = jnp.log(1. - 1. / (n + 1))
    print(f"True log(Z): {log_Z_true}")

    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1))
        return x

    def log_likelihood(x):
        return (LogSpace(0.) - LogSpace(n * jnp.log(x))).log_abs_val

    def exact_X(L):
        return (1. - L) ** (1. / n)

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    ns = ApproximateNestedSampler(
        model=model,
        num_live_points=50,
        num_parallel_samplers=1,
        max_samples=1e4,
        sampler_chain=[
            BadUniformSampler(mis_fraction=0., model=model)
        ]
    )

    termination_reason, state = ns(random.PRNGKey(42),
                                   term_cond=TerminationCondition(live_evidence_frac=1e-4))
    results = ns.to_results(state, termination_reason)
    ns.summary(results)
    ns.plot_diagnostics(results)

    # ensure there is no bug in control flow. X should be same to controlled evaluation
    log_X_mean, log_X_std = compute_shrinkage_stats(results.num_live_points_per_sample)
    assert jnp.allclose(results.log_X_mean, log_X_mean)

    # ensure the deviation from the exact shrinkage is correct
    X_exact = exact_X(jnp.exp(results.log_L_samples))

    rel_diff = jnp.abs(jnp.exp(log_X_mean) - X_exact) / jnp.exp(log_X_std)

    # ensure log_Z is close to truth
    assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=results.log_Z_uncert * 1.75)

    print("Relative shrinkage errors", jnp.percentile(rel_diff, jnp.asarray([50, 75, 90, 95])))
    assert jnp.all(jnp.percentile(rel_diff, jnp.asarray([50, 75, 90, 95])) < jnp.asarray([0.9, 1.1, 1.4, 1.5]))


def test_nested_sampling_plateau():
    def log_likelihood(x):
        return 0.

    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1))
        return x

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    exact_ns = ExactNestedSampler(
        model=model,
        num_live_points=50,
        max_samples=1000
    )

    termination_reason, state = exact_ns(
        random.PRNGKey(42),
        term_cond=TerminationCondition(live_evidence_frac=1e-4)
    )

    results = exact_ns.to_results(state, termination_reason)
    exact_ns.summary(results)
    exact_ns.plot_diagnostics(results)

    log_Z_samples = sample_evidence(random.PRNGKey(42),
                                    results.num_live_points_per_sample,
                                    results.log_L_samples, S=1000)
    plt.hist(log_Z_samples, bins='auto')
    plt.show()

    true_log_Z_uncert = jnp.std(log_Z_samples)
    print(f"true logZ uncert: {true_log_Z_uncert}")

    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, 0., atol=1.75 * true_log_Z_uncert)
