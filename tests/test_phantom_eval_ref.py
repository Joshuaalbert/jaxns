import numpy as np

from jaxns.phantom_eval_ref import sample_mc_shrinkage


def _make_toy_inputs():
    log_L_classic = np.array([0.0, 1.0, -123.0], dtype=float)
    num_samples = np.int32(2)

    # Live counts before each classic discard.
    K_classic = np.array([20, 10, 1], dtype=int)

    # Constraints: sample0 from prior; sample1 constrained to > 0.
    log_L_constraints = np.array([-np.inf, 0.0, 0.0], dtype=float)

    # Only the second cluster has usable phantoms.
    valid_phantom = np.array([False, True, True], dtype=bool)

    # 11 phantoms for cluster 1:
    # 10 above b=1.0 (success), 1 between a=0 and b=1 (failure).
    num_ph = 11
    log_L_phantom = np.full((3, num_ph), 0.5, dtype=float)
    log_L_phantom[1, :10] = 2.0
    log_L_phantom[1, 10:] = 0.5

    return log_L_constraints, log_L_classic, K_classic, valid_phantom, log_L_phantom, num_samples


def test_mc_shrinkage_v2_basic():
    log_L_constraints, log_L_classic, K_classic, valid_phantom, log_L_phantom, num_samples = _make_toy_inputs()

    out = sample_mc_shrinkage(
        seed=123,
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
        num_Z_samples=10,
    )

    assert out.log_Z_samples.shape == (10,)
    assert out.log_dZ_mean.shape == log_L_classic.shape
    assert out.log_dZ_var.shape == log_L_classic.shape
    assert out.rho_samples.shape == (10,)
    assert out.eta_samples.shape == (10,)
    assert out.rho_eta_samples.shape == (10,)
    assert np.all(np.isfinite(out.log_Z_samples))
    assert np.all(out.eta_samples >= 0.0)
    assert np.all(out.eta_samples <= 1.0)
    np.testing.assert_allclose(out.rho_eta_samples, out.rho_samples * out.eta_samples)
    block_mask = np.isfinite(out.log_L_blocks)
    assert np.all(np.isfinite(out.log_dZ_mean[block_mask]))
    assert np.all(np.isfinite(out.log_dZ_var[block_mask]))
    assert np.all(np.isneginf(out.log_dZ_mean[~block_mask]))
    assert np.all(np.isneginf(out.log_dZ_var[~block_mask]))


def test_mc_shrinkage_v2_runs_and_shapes():
    log_L_constraints, log_L_classic, K_classic, valid_phantom, log_L_phantom, num_samples = _make_toy_inputs()

    out = sample_mc_shrinkage(
        seed=321,
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
        num_Z_samples=20,
    )

    assert out.log_Z_samples.shape == (20,)
    assert out.log_dZ_mean.shape == log_L_classic.shape
    assert out.log_dZ_var.shape == log_L_classic.shape
    assert out.rho_samples.shape == (20,)
    assert out.eta_samples.shape == (20,)
    assert out.rho_eta_samples.shape == (20,)
    assert np.all(np.isfinite(out.log_Z_samples))

    # v2 should have some variability across samples
    assert np.std(out.log_Z_samples) > 0.0


def test_mc_shrinkage_v2_falls_back_without_phantoms():
    # Same schedule, but no valid phantoms.
    log_L_classic = np.array([0.0, 1.0, 2.0], dtype=float)
    K_classic = np.array([20, 10, 5], dtype=int)
    log_L_constraints = np.array([-np.inf, 0.0, 1.0], dtype=float)
    valid_phantom = np.array([False, False, False], dtype=bool)
    log_L_phantom = np.full((3, 5), 0.5, dtype=float)
    num_samples = np.int32(3)

    out = sample_mc_shrinkage(
        seed=0,
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
        num_Z_samples=5,
    )
    assert out.log_Z_samples.shape == (5,)
    assert np.all(np.isfinite(out.log_Z_samples))
    assert np.all(out.eta_samples == 0.0)
    assert np.all(out.rho_eta_samples == 0.0)
    block_mask = np.isfinite(out.log_L_blocks)
    assert np.all(out.block_first_idx[block_mask] >= 0)
    assert np.all(out.block_first_idx[~block_mask] == -1)
