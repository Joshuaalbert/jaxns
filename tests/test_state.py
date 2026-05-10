import numpy as np
import pytest
from jax import numpy as jnp
from jax import random
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.race_tree import build_block_state
from jaxns.samples import PhantomSamples, Samples
from jaxns.state import State
from jaxns.stats_utils import linear_to_log_stats
from jaxns.v3_shrinkage import (
    classic_dirichlet_concentrations,
    expected_v3_evidence_summary,
)
from tests.distributed_support import make_toy_model


tfpd = tfp.distributions


def _make_basic_model() -> Model:
    def prior_model():
        x = Prior(tfpd.Uniform(low=0.0, high=1.0), name='x').realise()
        return -x ** 2

    return Model(prior_model=prior_model)


def _make_invalid_plateau_capacity_state() -> State:
    samples = Samples(
        log_L_constraints=jnp.array([-jnp.inf, -jnp.inf]),
        log_likelihoods=jnp.array([0.0, 0.0]),
        U_samples=jnp.array([[0.25], [0.75]]),
        out_degree=jnp.array([1, 0], dtype=jnp.int32),
        num_likelihood_evaluations=jnp.array([1, 1], dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((2, 0, 1)),
            valid_mask=jnp.zeros((2, 0), dtype=jnp.bool_),
            log_L=jnp.zeros((2, 0)),
        ),
    )
    return State(
        root_out_degree=jnp.asarray(1, dtype=jnp.int32),
        samples=samples,
        num_samples=jnp.asarray(2, dtype=jnp.int32),
        log_L_supremum=jnp.asarray(0.0),
        U_supremum=jnp.array([0.75]),
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
        model=_make_basic_model(),
    )


def _make_strict_contour_violation_state() -> State:
    samples = Samples(
        log_L_constraints=jnp.array([0.0]),
        log_likelihoods=jnp.array([0.0]),
        U_samples=jnp.array([[0.5]]),
        out_degree=jnp.array([0], dtype=jnp.int32),
        num_likelihood_evaluations=jnp.array([1], dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((1, 0, 1)),
            valid_mask=jnp.zeros((1, 0), dtype=jnp.bool_),
            log_L=jnp.zeros((1, 0)),
        ),
    )
    return State(
        root_out_degree=jnp.asarray(1, dtype=jnp.int32),
        samples=samples,
        num_samples=jnp.asarray(1, dtype=jnp.int32),
        log_L_supremum=jnp.asarray(0.0),
        U_supremum=jnp.array([0.5]),
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
        model=_make_basic_model(),
    )


def test_to_result_marks_no_phantoms_invalid():
    model = _make_basic_model()
    results = NestedSampler(model=model).run().to_result().trim()

    assert results.log_L_phantom.shape[1] == 0
    assert int(results.total_phantom_samples) == 0
    assert not np.any(np.asarray(results.valid_phantom))

    diagnostics = results.phantom_conditioning_diagnostics()
    assert hasattr(diagnostics, "kish_participating_cluster_counts")
    assert hasattr(diagnostics, "phantom_gate_active")
    np.testing.assert_allclose(
        np.asarray(diagnostics.kish_participating_cluster_counts),
        np.zeros_like(np.asarray(results.log_L_blocks), dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.phantom_gate_active, dtype=bool),
        np.zeros_like(np.asarray(results.log_L_blocks), dtype=bool),
    )

    evidence_samples = results.sample_mc_shrinkage(num_samples=16, C_min=20)
    np.testing.assert_allclose(
        np.asarray(evidence_samples.kish_participating_cluster_counts),
        np.zeros_like(np.asarray(evidence_samples.log_L_blocks), dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(evidence_samples.phantom_gate_active, dtype=bool),
        np.zeros_like(np.asarray(evidence_samples.log_L_blocks), dtype=bool),
    )
    for old_name in ("rho_values", "rho_fit", "rho_samples", "rho_eta_samples"):
        if hasattr(evidence_samples, old_name):
            assert getattr(evidence_samples, old_name) is None


def test_samples_resize_preserves_constraints_and_provenance_fields():
    samples = Samples(
        log_L_constraints=jnp.array([-jnp.inf, 0.0]),
        log_likelihoods=jnp.array([0.0, 1.0]),
        U_samples=jnp.array([[0.25], [0.75]]),
        out_degree=jnp.array([1, 0], dtype=jnp.int32),
        num_likelihood_evaluations=jnp.array([2, 3], dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.array([[[0.1]], [[0.2]]]),
            valid_mask=jnp.array([[True], [False]]),
            log_L=jnp.array([[0.5], [1.5]]),
        ),
    )

    resized = samples.resize(4)

    np.testing.assert_allclose(
        np.asarray(resized.log_L_constraints[:2]),
        np.array([-np.inf, 0.0]),
    )
    np.testing.assert_allclose(
        np.asarray(resized.log_likelihoods[:2]),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(resized.out_degree[:2]),
        np.array([1, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(resized.num_likelihood_evaluations[:2]),
        np.array([2, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(resized.phantom_samples.valid_mask[:2]),
        np.array([[True], [False]]),
    )
    np.testing.assert_allclose(
        np.asarray(resized.log_L_constraints[2:]),
        np.array([np.inf, np.inf]),
    )
    np.testing.assert_allclose(
        np.asarray(resized.log_likelihoods[2:]),
        np.array([np.inf, np.inf]),
    )
    np.testing.assert_array_equal(
        np.asarray(resized.out_degree[2:]),
        np.array([0, 0], dtype=np.int32),
    )


def test_state_consistency_rejects_strict_contour_violation():
    state = _make_strict_contour_violation_state()

    with pytest.raises(ValueError, match="Strict contour"):
        state.ensure_consistency()


def test_state_sample_logZ_rejects_strict_contour_equality():
    state = _make_strict_contour_violation_state()

    with pytest.raises(ValueError, match="Strict contour.*must be greater"):
        state.sample_logZ(random.PRNGKey(7), num_samples=2)


def test_state_to_result_rejects_strict_contour_equality():
    state = _make_strict_contour_violation_state()

    with pytest.raises(ValueError, match="Strict contour.*must be greater"):
        state.to_result()


def test_state_sample_logZ_uses_public_v3_path():
    samples = Samples(
        log_L_constraints=jnp.array([-jnp.inf, 0.0]),
        log_likelihoods=jnp.array([0.0, 1.0]),
        U_samples=jnp.array([[0.25], [0.75]]),
        out_degree=jnp.array([1, 0], dtype=jnp.int32),
        num_likelihood_evaluations=jnp.array([1, 1], dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((2, 0, 1)),
            valid_mask=jnp.zeros((2, 0), dtype=jnp.bool_),
            log_L=jnp.zeros((2, 0)),
        ),
    )
    state = State(
        root_out_degree=jnp.asarray(1, dtype=jnp.int32),
        samples=samples,
        num_samples=jnp.asarray(2, dtype=jnp.int32),
        log_L_supremum=jnp.asarray(1.0),
        U_supremum=jnp.array([0.75]),
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
        model=_make_basic_model(),
    )

    log_Z = state.sample_logZ(random.PRNGKey(3), num_samples=5)

    assert np.asarray(log_Z).shape == (5,)
    assert np.all(np.isfinite(np.asarray(log_Z)))


def test_state_sample_logZ_rejects_invalid_plateau_capacity():
    state = _make_invalid_plateau_capacity_state()

    with pytest.raises(ValueError, match="K_g|m_g|incoming|plateau"):
        state.sample_logZ(random.PRNGKey(5), num_samples=2)


def test_state_to_result_evidence_summary_uses_v3_block_model():
    samples = Samples(
        log_L_constraints=jnp.array([-jnp.inf, -jnp.inf, -jnp.inf]),
        log_likelihoods=jnp.array([0.0, 1.0, 1.0]),
        U_samples=jnp.array([0.20, 0.50, 0.80]),
        out_degree=jnp.array([1, 0, 0], dtype=jnp.int32),
        num_likelihood_evaluations=jnp.array([1, 1, 1], dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=jnp.zeros((3, 0)),
            valid_mask=jnp.zeros((3, 0), dtype=jnp.bool_),
            log_L=jnp.zeros((3, 0)),
        ),
    )
    state = State(
        root_out_degree=jnp.asarray(2, dtype=jnp.int32),
        samples=samples,
        num_samples=jnp.asarray(3, dtype=jnp.int32),
        log_L_supremum=jnp.asarray(1.0),
        U_supremum=jnp.asarray(0.50),
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
        model=make_toy_model(),
    )

    block_state = build_block_state(
        samples=samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        validate=True,
    )
    expected_summary = expected_v3_evidence_summary(
        block_state,
        classic_dirichlet_concentrations(block_state),
    )

    results = state.to_result().trim()

    np.testing.assert_allclose(
        np.asarray(results.log_Z_mean),
        np.asarray(expected_summary.log_Z_mean),
    )
    np.testing.assert_allclose(
        np.asarray(results.log_Z_uncert),
        np.asarray(expected_summary.log_Z_uncert),
    )
    np.testing.assert_array_equal(
        np.asarray(results.block_incoming_K)[np.isfinite(np.asarray(results.log_L_blocks))],
        np.array([2, 2], dtype=np.int32),
    )

    legacy_evidence, _ = state.evaluate_evidence()
    legacy_log_Z_mean, _ = linear_to_log_stats(
        legacy_evidence.Z_mean.log_abs_val,
        log_f2_mean=legacy_evidence.Z2_mean.log_abs_val,
    )
    assert not np.isclose(
        float(results.log_Z_mean),
        float(legacy_log_Z_mean),
    )


def test_state_to_result_preserves_phantom_provenance_for_kish_diagnostics():
    samples = Samples(
        log_L_constraints=jnp.asarray([-jnp.inf, -jnp.inf, 0.0]),
        log_likelihoods=jnp.asarray([0.0, 1.0, 2.0]),
        U_samples=jnp.asarray([0.20, 0.50, 0.80]),
        out_degree=jnp.asarray([1, 0, 0], dtype=jnp.int32),
        num_likelihood_evaluations=jnp.asarray([3, 3, 3], dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=None,
            valid_mask=jnp.asarray(
                [
                    [True, True],
                    [True, True],
                    [True, True],
                ],
                dtype=jnp.bool_,
            ),
            log_L=jnp.asarray(
                [
                    [-0.5, 1.0],
                    [0.5, 2.0],
                    [2.5, 1.5],
                ],
                dtype=jnp.float32,
            ),
        ),
    )
    state = State(
        root_out_degree=jnp.asarray(2, dtype=jnp.int32),
        samples=samples,
        num_samples=jnp.asarray(3, dtype=jnp.int32),
        log_L_supremum=jnp.asarray(2.0),
        U_supremum=jnp.asarray(0.80),
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
        model=make_toy_model(),
    )

    results = state.to_result().trim()
    diagnostics = results.phantom_conditioning_diagnostics(C_min=1)

    np.testing.assert_allclose(
        np.asarray(results.log_L_constraints),
        np.asarray([-np.inf, -np.inf, 0.0]),
    )
    np.testing.assert_allclose(
        np.asarray(results.log_L_phantom),
        np.asarray([[-0.5, 1.0], [0.5, 2.0], [2.5, 1.5]]),
    )
    np.testing.assert_array_equal(
        np.asarray(results.valid_phantom, dtype=bool),
        np.asarray([True, True, True]),
    )
    np.testing.assert_allclose(np.asarray(diagnostics.A_g), [4.0, 5.0, 3.0])
    np.testing.assert_allclose(np.asarray(diagnostics.B_g), [3.0, 3.0, 1.0])
    np.testing.assert_allclose(np.asarray(diagnostics.E_g), [0.0, 1.0, 1.0])
    np.testing.assert_allclose(np.asarray(diagnostics.R_g), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(diagnostics.kish_participating_cluster_counts),
        [2.0, 25.0 / 9.0, 9.0 / 5.0],
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.phantom_gate_active, dtype=bool),
        np.asarray([True, True, True]),
    )


def test_state_to_result_rejects_invalid_plateau_capacity():
    state = _make_invalid_plateau_capacity_state()

    with pytest.raises(ValueError, match="K_g|m_g|incoming|plateau"):
        state.to_result()
