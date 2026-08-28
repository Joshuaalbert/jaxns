import numpy as np
import pytest

import jaxns.shrinkage.reference as ref_phantom


def _require_callable(name: str):
    fn = getattr(ref_phantom, name, None)
    assert callable(fn), f"jaxns.shrinkage.reference.{name} is required by Ticket 0013."
    return fn


def _toy_inputs(num_phantom: int = 3):
    log_L_constraints = np.asarray([-np.inf, 0.0, 1.0], dtype=float)
    log_L_classic = np.asarray([0.0, 1.0, 2.0], dtype=float)
    K_classic = np.asarray([8, 7, 6], dtype=np.int32)
    valid_phantom = np.asarray([True, True, True], dtype=bool)
    log_L_phantom = np.asarray(
        [
            [0.0, 0.5, 2.5],
            [1.0, 1.5, 2.0],
            [2.0, 2.5, 3.0],
        ],
        dtype=float,
    )[:, :num_phantom]
    num_samples = np.int32(3)
    return (
        log_L_constraints,
        log_L_classic,
        K_classic,
        valid_phantom,
        log_L_phantom,
        num_samples,
    )


def test_reference_count_matrices_expose_parent_gated_R_and_aggregates():
    count_fn = _require_callable("compute_phantom_count_matrices")
    (
        log_L_constraints,
        log_L_classic,
        _,
        valid_phantom,
        log_L_phantom,
        num_samples,
    ) = _toy_inputs()
    log_L_blocks = log_L_classic.copy()

    counts = count_fn(
        log_L_blocks=log_L_blocks,
        block_valid_mask=np.ones_like(log_L_blocks, dtype=bool),
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=np.arange(log_L_classic.size) < int(num_samples),
    )

    for field_name in ("A_cg", "B_cg", "E_cg", "R_cg", "A_g", "B_g", "E_g", "R_g"):
        assert hasattr(counts, field_name), f"Reference count result missing {field_name}."
    np.testing.assert_allclose(counts.R_cg, counts.A_cg - counts.B_cg - counts.E_cg)
    np.testing.assert_allclose(counts.A_g, np.sum(counts.A_cg, axis=0))
    np.testing.assert_allclose(counts.B_g, np.sum(counts.B_cg, axis=0))
    np.testing.assert_allclose(counts.E_g, np.sum(counts.E_cg, axis=0))
    np.testing.assert_allclose(counts.R_g, np.sum(counts.R_cg, axis=0))


def test_reference_sample_mc_shrinkage_runs_with_kish_gate_diagnostics():
    (
        log_L_constraints,
        log_L_classic,
        K_classic,
        valid_phantom,
        log_L_phantom,
        num_samples,
    ) = _toy_inputs()

    out = ref_phantom.sample_mc_shrinkage(
        seed=321,
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
        num_Z_samples=20,
        C_min=2,
    )

    assert out.log_Z_samples.shape == (20,)
    assert np.all(np.isfinite(out.log_Z_samples))
    for field_name in (
            "kish_participating_cluster_counts",
            "phantom_gate_active",
            "phantom_A",
            "phantom_B",
            "phantom_E",
            "phantom_R",
    ):
        assert hasattr(out, field_name), f"EvidenceSamples missing {field_name}."
        values = np.asarray(getattr(out, field_name))
        assert values.shape == log_L_classic.shape
    np.testing.assert_allclose(out.phantom_R, out.phantom_A - out.phantom_B - out.phantom_E)


def test_reference_sample_mc_shrinkage_no_phantoms_uses_classic_gamma_path():
    (
        log_L_constraints,
        log_L_classic,
        K_classic,
        _,
        _,
        num_samples,
    ) = _toy_inputs(num_phantom=0)

    out = ref_phantom.sample_mc_shrinkage(
        seed=0,
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=np.zeros((3,), dtype=bool),
        log_L_phantom=np.zeros((3, 0), dtype=float),
        num_samples=num_samples,
        num_Z_samples=5,
        C_min=20,
    )

    assert out.log_Z_samples.shape == (5,)
    np.testing.assert_allclose(out.kish_participating_cluster_counts, np.zeros((3,)))
    np.testing.assert_array_equal(out.phantom_gate_active, np.zeros((3,), dtype=bool))
    for old_name in ("rho_samples", "rho_values", "rho_fit", "rho_eta_samples"):
        if hasattr(out, old_name):
            assert getattr(out, old_name) is None


def test_reference_phantom_gate_rejects_zero_participation_when_threshold_is_zero():
    gate_fn = _require_callable("compute_phantom_gate_active")

    gate = gate_fn(np.zeros((3, 2), dtype=float), C_min=0)

    np.testing.assert_array_equal(gate, np.asarray([False, False]))


def test_reference_gamma_sampler_rejects_malformed_count_matrices():
    sampler = _require_callable("sample_gamma_weighted_phantom_probabilities")
    block_state, _ = ref_phantom._block_state_from_arrays(
        log_L_classic=np.asarray([0.0], dtype=float),
        K_classic=np.asarray([5], dtype=np.int32),
        num_samples=np.int32(1),
    )

    with pytest.raises(ValueError, match="B_cg \\+ E_cg must be <= A_cg"):
        sampler(
            rng=np.random.default_rng(0),
            block_state=block_state,
            A_cg=np.asarray([[1.0]], dtype=float),
            B_cg=np.asarray([[1.0]], dtype=float),
            E_cg=np.asarray([[1.0]], dtype=float),
            num_samples=2,
            C_min=0,
        )


def test_reference_rejects_malformed_per_phantom_validity_mask():
    (
        log_L_constraints,
        log_L_classic,
        K_classic,
        _,
        log_L_phantom,
        num_samples,
    ) = _toy_inputs()

    with pytest.raises(ValueError, match="valid_phantom|per-cluster|one-dimensional"):
        ref_phantom.sample_mc_shrinkage(
            seed=0,
            log_L_constraints=log_L_constraints,
            log_L_classic=log_L_classic,
            K_classic=K_classic,
            valid_phantom=np.ones_like(log_L_phantom, dtype=bool),
            log_L_phantom=log_L_phantom,
            num_samples=num_samples,
            num_Z_samples=2,
            C_min=20,
        )
