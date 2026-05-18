import copy
import math

import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.v3_validation import deterministic_fixtures
from benchmarks.v3_validation.deterministic_fixtures import (
    analytic_evidence_fixtures,
    posterior_weighting_fixtures,
    plateau_equality_recovery_fixture,
    race_tree_accounting_fixture,
)
from benchmarks.v3_validation.schema_checks import (
    CALIBRATION_LOGZ_CONVENTION,
    CALIBRATION_SIGMA_CONVENTION,
    assert_plateau_equality_record,
    compute_calibration_summary,
)
from jaxns import v3_shrinkage
from jaxns.race_tree import BlockState, build_block_state
from jaxns.samples import PhantomSamples, Samples


def _benchmark_metadata(problem: str, seed: int = 0) -> dict:
    return {
        "method_setting": {
            "method": "phantom-conditioned",
            "allocation": "uniform",
        },
        "seed": seed,
        "problem": problem,
        "likelihood_evaluations": 128,
        "wall_clock_seconds": 0.01,
        "worker_count": 1,
        "commit": "test-fixture",
    }


def test_analytic_evidence_fixtures_have_exact_survival_and_logz_refs():
    for fixture in analytic_evidence_fixtures():
        likelihood = np.asarray(fixture.likelihood_levels, dtype=float)
        equality_mass = np.asarray(fixture.equality_masses, dtype=float)

        np.testing.assert_allclose(np.sum(equality_mass), 1.0)
        expected_survival = np.asarray([
            np.sum(equality_mass[idx + 1:])
            for idx in range(equality_mass.shape[0])
        ])
        expected_logZ = math.log(float(np.dot(likelihood, equality_mass)))

        np.testing.assert_allclose(
            np.asarray(fixture.strict_survival_after_block),
            expected_survival,
        )
        np.testing.assert_allclose(fixture.logZ_ref, expected_logZ)


def test_calibration_summary_declares_hat_logz_and_sigma_convention():
    log_Z_samples = np.asarray([math.log(2.0), math.log(3.0), math.log(5.0)])
    logZ_ref = math.log(3.0)

    summary = compute_calibration_summary(log_Z_samples, logZ_ref=logZ_ref)

    expected_hat_logZ = float(np.mean(log_Z_samples))
    expected_sigma = float(np.std(log_Z_samples, ddof=1))
    np.testing.assert_allclose(summary["hat_logZ"], expected_hat_logZ)
    np.testing.assert_allclose(summary["sigma_logZ"], expected_sigma)
    np.testing.assert_allclose(
        summary["z_logZ"],
        (expected_hat_logZ - logZ_ref) / expected_sigma,
    )
    assert summary["summary_convention"] == CALIBRATION_LOGZ_CONVENTION
    assert summary["sigma_convention"] == CALIBRATION_SIGMA_CONVENTION

    with pytest.raises(ValueError, match="At least two"):
        compute_calibration_summary([logZ_ref], logZ_ref=logZ_ref)


def test_plateau_fixture_records_equality_mass_recovery_separately():
    fixture = plateau_equality_recovery_fixture()
    assert not hasattr(fixture, "rho_g")
    np.testing.assert_allclose(
        np.asarray(fixture.R_cg),
        np.asarray(fixture.A_cg) - np.asarray(fixture.B_cg) - np.asarray(fixture.E_cg),
    )
    assert fixture.phantom_gate_active
    assert (
        abs(fixture.phantom_equality_mean - fixture.equality_mass_ref)
        < abs(fixture.no_phantom_equality_mean - fixture.equality_mass_ref)
    )

    summary = compute_calibration_summary(
        [
            fixture.logZ_ref - 0.03,
            fixture.logZ_ref + 0.01,
            fixture.logZ_ref + 0.02,
        ],
        logZ_ref=fixture.logZ_ref,
    )
    record = {
        "metric_family": "evidence_calibration",
        "metadata": _benchmark_metadata(fixture.name),
        "evidence_calibration": {
            **summary,
            "rmse_logZ": 0.021602468994692867,
            "reported_uncertainty_logZ": summary["sigma_logZ"],
            "empirical_uncertainty_logZ": summary["sigma_logZ"],
            "expectation_logZ": fixture.logZ_ref,
            "mc_shrinkage_logZ": summary["hat_logZ"],
            "kish_participating_cluster_counts": [
                fixture.kish_participating_cluster_count
            ],
            "phantom_gate_active": [fixture.phantom_gate_active],
            "phantom_A_g": [fixture.phantom_A],
            "phantom_B_g": [fixture.phantom_B],
            "phantom_E_g": [fixture.phantom_E],
            "phantom_R_g": [fixture.phantom_R],
            "C_min": fixture.C_min,
        },
        "plateau_equality": {
            "likelihood_level": fixture.likelihood_level,
            "equality_mass_ref": fixture.equality_mass_ref,
            "hat_equality_mass": fixture.phantom_equality_mean,
            "equality_mass_error": (
                fixture.phantom_equality_mean - fixture.equality_mass_ref
            ),
            "per_sample_equality_mass": (
                fixture.phantom_equality_mean / fixture.block_size
            ),
        },
    }

    assert_plateau_equality_record(record)

    bad_record = copy.deepcopy(record)
    del bad_record["plateau_equality"]["equality_mass_ref"]
    with pytest.raises(AssertionError, match="equality_mass_ref"):
        assert_plateau_equality_record(bad_record)


def test_gamma_weighted_phantom_fixture_forces_explicit_draw_surface():
    fixture_fn = getattr(
        deterministic_fixtures,
        "gamma_weighted_phantom_conditioning_fixture",
        None,
    )
    assert callable(fixture_fn), (
        "v3 validation fixtures must expose "
        "gamma_weighted_phantom_conditioning_fixture() with per-cluster "
        "A_cg/B_cg/E_cg/R_cg counts, explicit race gammas, explicit "
        "cluster weights, C_min, gate activity, and expected probabilities."
    )
    fixture = fixture_fn()
    assert fixture.C_min == 20
    np.testing.assert_allclose(
        np.asarray(fixture.R_cg),
        np.asarray(fixture.A_cg) - np.asarray(fixture.B_cg) - np.asarray(fixture.E_cg),
    )
    assert np.any(np.asarray(fixture.E_cg) > 0.0)
    assert np.any(np.asarray(fixture.R_cg) > 0.0)

    draw_fn = getattr(
        v3_shrinkage,
        "gamma_weighted_phantom_probabilities_from_draws",
        None,
    )
    assert callable(draw_fn), (
        "v3_shrinkage.gamma_weighted_phantom_probabilities_from_draws(...) "
        "is required so deterministic fixtures can validate explicit race "
        "gammas and shared v_c weights."
    )

    num_blocks = len(fixture.log_L_blocks)
    block_state = BlockState(
        log_L_blocks=jnp.asarray(fixture.log_L_blocks),
        block_first_idx=jnp.arange(num_blocks, dtype=jnp.int32),
        block_size=jnp.asarray(fixture.block_size, dtype=jnp.int32),
        incoming_K=jnp.asarray(fixture.incoming_K, dtype=jnp.int32),
        block_out_degree=jnp.zeros((num_blocks,), dtype=jnp.int32),
        valid=jnp.ones((num_blocks,), dtype=jnp.bool_),
        block_sample_indices=jnp.asarray(fixture.block_sample_indices),
    )
    draw = draw_fn(
        block_state=block_state,
        A_cg=jnp.asarray(fixture.A_cg),
        B_cg=jnp.asarray(fixture.B_cg),
        E_cg=jnp.asarray(fixture.E_cg),
        race_gamma_gt=jnp.asarray(fixture.race_gamma_gt),
        race_gamma_eq=jnp.asarray(fixture.race_gamma_eq),
        race_gamma_lt=jnp.asarray(fixture.race_gamma_lt),
        cluster_weights=jnp.asarray(fixture.cluster_weights),
        C_min=fixture.C_min,
    )

    np.testing.assert_allclose(
        np.asarray(draw.kish_participating_cluster_counts),
        np.asarray(fixture.kish_participating_cluster_counts),
    )
    np.testing.assert_array_equal(
        np.asarray(draw.phantom_gate_active, dtype=bool),
        np.asarray(fixture.phantom_gate_active, dtype=bool),
    )
    np.testing.assert_allclose(np.asarray(draw.p_gt), fixture.expected_p_gt)
    np.testing.assert_allclose(np.asarray(draw.p_eq), fixture.expected_p_eq)
    np.testing.assert_allclose(np.asarray(draw.p_lt), fixture.expected_p_lt)


def test_race_tree_accounting_fixture_records_block_assumptions():
    fixture = race_tree_accounting_fixture()
    num_samples = len(fixture.log_likelihoods)
    samples = Samples(
        log_L_constraints=jnp.asarray(fixture.log_L_constraints),
        log_likelihoods=jnp.asarray(fixture.log_likelihoods),
        U_samples=jnp.zeros((num_samples, 1)),
        out_degree=jnp.asarray(fixture.out_degree, dtype=jnp.int32),
        num_likelihood_evaluations=jnp.ones((num_samples,), dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((num_samples, 0, 1)),
            valid_mask=jnp.zeros((num_samples, 0), dtype=jnp.bool_),
            log_L=jnp.zeros((num_samples, 0)),
        ),
    )

    block_state = build_block_state(
        samples=samples,
        root_out_degree=jnp.asarray(fixture.root_out_degree, dtype=jnp.int32),
        num_samples=jnp.asarray(num_samples, dtype=jnp.int32),
        sample_indices=jnp.asarray(fixture.sample_indices, dtype=jnp.int32),
        validate=True,
    )

    np.testing.assert_allclose(
        np.asarray(block_state.log_L_blocks),
        np.asarray(fixture.expected_log_L_blocks),
    )
    np.testing.assert_array_equal(
        np.asarray(block_state.block_size),
        np.asarray(fixture.expected_block_size),
    )
    np.testing.assert_array_equal(
        np.asarray(block_state.incoming_K),
        np.asarray(fixture.expected_incoming_K),
    )
    np.testing.assert_array_equal(
        np.asarray(block_state.block_out_degree),
        np.asarray(fixture.expected_block_out_degree),
    )
    np.testing.assert_array_equal(
        np.asarray(block_state.block_sample_indices),
        np.asarray(fixture.expected_block_sample_indices),
    )
    np.testing.assert_array_equal(
        np.asarray(fixture.expected_incoming_K[:-1])
        - np.asarray(fixture.expected_block_size[:-1])
        + np.asarray(fixture.expected_block_out_degree[:-1]),
        np.asarray(fixture.expected_incoming_K[1:]),
    )


@pytest.mark.parametrize(
    "fixture",
    posterior_weighting_fixtures(),
    ids=lambda fixture: fixture.name,
)
def test_posterior_weighting_fixtures_cover_plateaus_and_non_plateaus(fixture):
    num_blocks = len(fixture.log_L_blocks)
    block_state = BlockState(
        log_L_blocks=jnp.asarray(fixture.log_L_blocks),
        block_first_idx=jnp.arange(num_blocks, dtype=jnp.int32),
        block_size=jnp.asarray(fixture.block_size, dtype=jnp.int32),
        incoming_K=jnp.asarray(fixture.incoming_K, dtype=jnp.int32),
        block_out_degree=jnp.zeros((num_blocks,), dtype=jnp.int32),
        valid=jnp.ones((num_blocks,), dtype=jnp.bool_),
        block_sample_indices=jnp.asarray(fixture.block_sample_indices),
    )
    concentrations = v3_shrinkage.DirichletConcentrations(
        alpha_gt=jnp.asarray(fixture.alpha_gt),
        alpha_eq=jnp.asarray(fixture.alpha_eq),
        alpha_lt=jnp.asarray(fixture.alpha_lt),
        epsilon=jnp.zeros((num_blocks,)),
    )

    log_weights = v3_shrinkage.expected_v3_log_posterior_weights(
        block_state,
        concentrations,
    )

    np.testing.assert_allclose(
        np.exp(np.asarray(log_weights)),
        np.asarray(fixture.expected_sample_weights),
        rtol=1e-12,
        atol=1e-12,
    )
