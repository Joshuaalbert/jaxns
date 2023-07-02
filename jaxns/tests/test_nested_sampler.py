from jax import random, numpy as jnp

from jaxns.statistics import compute_evidence_dual, compute_num_live_points_from_unit_threads, compute_evidence, \
    compute_shrinkage_stats
from jaxns.utils import sample_evidence


def test_nested_sampling_basic(basic_run_results):
    log_Z_true, state, results = basic_run_results
    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=1.75 * results.log_Z_uncert)


def test_sample_evidence(basic_run_results):
    log_Z_true, state, results = basic_run_results
    log_Z_samples = sample_evidence(random.PRNGKey(42),
                                    results.num_live_points_per_sample,
                                    results.log_L_samples, S=1000)
    assert jnp.isclose(results.log_Z_mean, jnp.mean(log_Z_samples), atol=1e-3)
    assert jnp.isclose(results.log_Z_uncert, jnp.std(log_Z_samples), atol=1e-2)


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

def test_compute_evidence_dual(basic_mvn_run_results):
    true_logZ, state, results = basic_mvn_run_results

    num_live_points = compute_num_live_points_from_unit_threads(
        log_L_samples=state.sample_collection.reservoir.log_L,
        log_L_constraints=state.sample_collection.reservoir.log_L_constraint,
        num_samples=state.sample_collection.sample_idx,
        sorted_collection=True
    )
    evidence_calculation_dual, stats = compute_evidence_dual(
        sample_collection=state.sample_collection,
        num_live_points=num_live_points
    )
    print('Dual', evidence_calculation_dual)
    evidence_calculation, stats = compute_evidence(
        sample_collection=state.sample_collection,
        num_live_points=num_live_points
    )
    print('Primal', evidence_calculation)

    assert jnp.isclose(evidence_calculation_dual.log_Z_mean, evidence_calculation.log_Z_mean, atol=0.05)


def test_nested_sampling_mvn_static(basic_mvn_run_results):
    true_logZ, state, results = basic_mvn_run_results
    actual_log_Z_mean = results.log_Z_mean
    expected_log_Z_mean = true_logZ
    tol = 1.75 * results.log_Z_uncert
    assert jnp.isclose(actual_log_Z_mean, expected_log_Z_mean, atol=tol)


def test_nested_sampling_mvn_static_multiellipsoid_sampler(multiellipsoidal_mvn_run_results):
    true_logZ, state, results = multiellipsoidal_mvn_run_results
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

def test_nested_sampling_plateau(plateau_run_results):
    log_Z_true, true_log_Z_uncert, state, results = plateau_run_results
    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, 0., atol=1.75 * true_log_Z_uncert)


def test_basic2_correctness(basic2_results):
    log_Z_true, state, results, X_exact = basic2_results
    assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=results.log_Z_uncert * 1.75)


def test_compute_shrinkage_stats(basic2_results):
    log_Z_true, state, results, X_exact = basic2_results

    # ensure there is no bug in control flow. X should be same to controlled evaluation
    log_X_mean, log_X_std = compute_shrinkage_stats(results.num_live_points_per_sample)
    assert jnp.allclose(results.log_X_mean, log_X_mean)


def test_shrinkage(basic2_results):
    log_Z_true, state, results, X_exact = basic2_results

    diff = results.log_X_mean - jnp.log(X_exact)
    diff = jnp.where(jnp.isfinite(diff), diff, jnp.nan)
    # print(jnp.nanstd(diff))
    assert jnp.nanstd(diff) < 0.26

    log_X_mean, log_X_std = compute_shrinkage_stats(results.num_live_points_per_sample)
    rel_diff = jnp.abs(jnp.exp(log_X_mean) - X_exact) / jnp.exp(log_X_std)
    print("Relative shrinkage errors", jnp.percentile(rel_diff, jnp.asarray([50, 75, 90, 95])))
    assert jnp.all(jnp.percentile(rel_diff, jnp.asarray([50, 75, 90, 95])) < jnp.asarray([0.9, 1.1, 1.4, 1.5]))
