import numpy as np
import jax
from jax import numpy as jnp

import jaxns.phantom_eval as jax_phantom
import jaxns.phantom_eval_ref as ref_phantom


def _to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _make_scenario(seed, num_blocks, num_phantom, valid_mask, offset, pad_invalid=0):
    rng = np.random.default_rng(seed)
    increments = rng.uniform(0.2, 0.8, size=num_blocks).astype(np.float32)
    valid_log_L = np.cumsum(increments).astype(np.float32)
    valid_log_L = valid_log_L - np.mean(valid_log_L) + np.float32(offset)

    total = num_blocks + pad_invalid
    log_L_classic = np.zeros((total,), dtype=np.float32)
    log_L_classic[:num_blocks] = valid_log_L
    K_classic = np.ones((total,), dtype=np.int32)
    K_classic[:num_blocks] = rng.integers(15, 60, size=num_blocks, dtype=np.int32)

    log_L_constraints = np.zeros((total,), dtype=np.float32)
    log_L_constraints[0] = -np.inf
    log_L_constraints[1:num_blocks] = valid_log_L[:-1]

    if valid_mask is None:
        valid_phantom_valid = rng.uniform(size=num_blocks) > 0.3
    else:
        valid_phantom_valid = np.array(valid_mask, dtype=bool)
    valid_phantom = np.zeros((total,), dtype=bool)
    valid_phantom[:num_blocks] = valid_phantom_valid

    log_L_phantom = np.zeros((total, num_phantom), dtype=np.float32)
    for i in range(num_blocks):
        base = log_L_constraints[i]
        if not np.isfinite(base):
            base = valid_log_L[0] - np.float32(1.0)
        jitter = np.abs(rng.normal(loc=0.6, scale=0.35, size=num_phantom)).astype(np.float32)
        log_L_phantom[i] = base + jitter

    return log_L_constraints, log_L_classic, K_classic, valid_phantom, log_L_phantom, np.int32(num_blocks)


SCENARIOS = [
    _make_scenario(seed=0, num_blocks=3, num_phantom=6, valid_mask=[False, False, False], offset=-0.5),
    _make_scenario(seed=1, num_blocks=4, num_phantom=7, valid_mask=[True, False, True, True], offset=-1.0),
    _make_scenario(seed=2, num_blocks=5, num_phantom=9, valid_mask=[True, True, True, True, True], offset=0.3),
    _make_scenario(seed=3, num_blocks=5, num_phantom=8, valid_mask=[True, False, True, True, False], offset=0.0, pad_invalid=3),
]


def _evaluate_both(scenario):
    (log_L_constraints,
     log_L_classic,
     K_classic,
     valid_phantom,
     log_L_phantom,
     num_samples) = scenario

    return (
        log_L_constraints,
        log_L_classic,
        K_classic,
        valid_phantom,
        log_L_phantom,
        num_samples,
    )


def _assert_logz_stats_close(ref_out, jax_out, *, rtol, atol):
    ref_mean = float(np.mean(ref_out.log_Z_samples))
    jax_mean = float(np.mean(jax_out.log_Z_samples))
    np.testing.assert_allclose(jax_mean, ref_mean, rtol=rtol, atol=atol)

    ref_std = float(np.std(ref_out.log_Z_samples))
    jax_std = float(np.std(jax_out.log_Z_samples))
    np.testing.assert_allclose(jax_std, ref_std, rtol=rtol, atol=atol)

    np.testing.assert_allclose(jax_out.log_dZ_mean, ref_out.log_dZ_mean, rtol=rtol, atol=atol)
    np.testing.assert_allclose(jax_out.log_dZ_var, ref_out.log_dZ_var, rtol=rtol, atol=atol)


def test_sample_mc_shrinkage_stats_close():
    num_Z_samples = 128
    for scenario in SCENARIOS:
        (log_L_constraints,
         log_L_classic,
         K_classic,
         valid_phantom,
         log_L_phantom,
         num_samples) = _evaluate_both(scenario)

        ref_out = ref_phantom.sample_mc_shrinkage(
            seed=321,
            log_L_constraints=log_L_constraints,
            log_L_classic=log_L_classic,
            K_classic=K_classic,
            valid_phantom=valid_phantom,
            log_L_phantom=log_L_phantom,
            num_samples=num_samples,
            num_Z_samples=num_Z_samples,
        )
        jax_out = jax_phantom.sample_mc_shrinkage(
            seed=321,
            log_L_constraints=jnp.array(log_L_constraints),
            log_L_classic=jnp.array(log_L_classic),
            K_classic=jnp.array(K_classic),
            valid_phantom=jnp.array(valid_phantom),
            log_L_phantom=jnp.array(log_L_phantom),
            num_samples=jnp.asarray(num_samples, dtype=jnp.int32),
            num_Z_samples=num_Z_samples,
            batch_size=32,
            rho_prior="none",
        )
        jax_out = _to_numpy(jax_out)
        _assert_logz_stats_close(ref_out, jax_out, rtol=0.2, atol=0.2)

        block_mask = np.isfinite(ref_out.log_L_blocks)
        assert np.all(ref_out.block_first_idx[block_mask] >= 0)
        assert np.all(ref_out.block_first_idx[~block_mask] == -1)
        np.testing.assert_array_equal(jax_out.block_first_idx, ref_out.block_first_idx)
        np.testing.assert_allclose(jax_out.log_L_blocks, ref_out.log_L_blocks, equal_nan=True)

        ref_rho_mean = float(np.mean(ref_out.rho_samples))
        jax_rho_mean = float(np.mean(jax_out.rho_samples))
        np.testing.assert_allclose(jax_rho_mean, ref_rho_mean, rtol=0.2, atol=0.05)
