import os

from jax import random, numpy as jnp

# Force 2 jax  hosts
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

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
    log_Z_mean = jnp.mean(log_Z_samples)
    log_Z_std = jnp.std(log_Z_samples)
    assert jnp.isclose(results.log_Z_mean, log_Z_mean, atol=1e-2)
    assert jnp.isclose(results.log_Z_uncert, log_Z_std, atol=1e-2)


def test_nested_sampling_mvn_static(basic_mvn_run_results):
    true_logZ, state, results = basic_mvn_run_results
    actual_log_Z_mean = results.log_Z_mean
    expected_log_Z_mean = true_logZ
    tol = 1.9 * results.log_Z_uncert
    assert jnp.isclose(actual_log_Z_mean, expected_log_Z_mean, atol=tol)


def test_nested_sampling_mvn_static_parallel(basic_mvn_run_results_parallel):
    true_logZ, state, results = basic_mvn_run_results_parallel
    actual_log_Z_mean = results.log_Z_mean
    expected_log_Z_mean = true_logZ
    tol = 1.9 * results.log_Z_uncert
    assert jnp.isclose(actual_log_Z_mean, expected_log_Z_mean, atol=tol)


def test_nested_sampling_mvn_static_multiellipsoid_sampler(multiellipsoidal_mvn_run_results):
    true_logZ, state, results = multiellipsoidal_mvn_run_results
    actual_log_Z_mean = results.log_Z_mean
    expected_log_Z_mean = true_logZ
    tol = 1.75 * results.log_Z_uncert
    assert jnp.isclose(actual_log_Z_mean, expected_log_Z_mean, atol=tol)


def test_nested_sampling_plateau(plateau_run_results):
    log_Z_true, true_log_Z_uncert, state, results = plateau_run_results
    assert jnp.bitwise_not(jnp.isnan(results.log_Z_mean))
    assert jnp.isclose(results.log_Z_mean, 0., atol=1.75 * true_log_Z_uncert)


def test_basic2_correctness(basic2_results):
    log_Z_true, state, results, X_exact = basic2_results
    assert jnp.isclose(results.log_Z_mean, log_Z_true, atol=results.log_Z_uncert * 1.75)
