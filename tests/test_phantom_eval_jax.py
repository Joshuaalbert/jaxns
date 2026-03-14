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
    _make_scenario(seed=4, num_blocks=4, num_phantom=0, valid_mask=[False, False, False, False], offset=0.1),
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


def _boundary_counts_reference(
        cluster_multiplicity,
        start_idx,
        event_cluster_idx,
        event_a_hi,
        event_b_hi,
        event_A_active,
        event_B_active,
        event_eq_idx,
        event_eq_active,
        num_blocks,
):
    A = np.zeros((num_blocks,), dtype=float)
    B = np.zeros((num_blocks,), dtype=float)
    E = np.zeros((num_blocks,), dtype=float)

    num_events = event_cluster_idx.shape[0]
    for e in range(num_events):
        c = int(event_cluster_idx[e])
        mult = float(cluster_multiplicity[c])
        g_start = int(start_idx[c])

        if event_A_active[e]:
            g_stop = min(int(event_a_hi[e]), num_blocks)
            for g in range(g_start, g_stop):
                A[g] += mult

        if event_B_active[e]:
            g_stop = min(int(event_b_hi[e]), num_blocks)
            for g in range(g_start, g_stop):
                B[g] += mult

        if event_eq_active[e]:
            g_eq = int(event_eq_idx[e])
            if 0 <= g_eq < num_blocks:
                E[g_eq] += mult

    return A, B, E


def _boundary_counts_direct_from_raw(
        log_L_blocks,
        log_L_constraints,
        log_L_phantom,
        effective_valid_phantom,
        cluster_multiplicity,
        num_valid_blocks,
):
    num_blocks = int(log_L_blocks.shape[0])
    num_phantom = int(log_L_phantom.shape[1])
    A = np.zeros((num_blocks,), dtype=float)
    B = np.zeros((num_blocks,), dtype=float)
    E = np.zeros((num_blocks,), dtype=float)

    for g in range(int(num_valid_blocks)):
        a = -np.inf if g == 0 else float(log_L_blocks[g - 1])
        b = float(log_L_blocks[g])
        for c in range(int(log_L_constraints.shape[0])):
            if not bool(effective_valid_phantom[c]):
                continue
            m = float(cluster_multiplicity[c])
            if m == 0.0:
                continue
            if float(log_L_constraints[c]) > a:
                continue

            for p in range(num_phantom):
                v = float(log_L_phantom[c, p])
                if v > a:
                    A[g] += m
                if v > b:
                    B[g] += m
                if v == b:
                    E[g] += m
    return A, B, E


def _precompute_from_sample_mc_snippet(
        log_L_blocks,
        log_L_constraints,
        valid_phantom,
        effective_sample_mask,
        log_L_phantom,
        num_valid_blocks,
):
    dtype = log_L_blocks.dtype
    num_clusters = log_L_constraints.shape[0]
    num_blocks = log_L_blocks.shape[0]
    num_phantom = log_L_phantom.shape[1]

    effective_valid_phantom = valid_phantom & effective_sample_mask
    left_c = jnp.searchsorted(log_L_blocks, log_L_constraints, side='left')
    start_idx = jnp.where(jnp.isneginf(log_L_constraints), 0, left_c + 1)
    start_idx = jnp.minimum(start_idx, num_valid_blocks)
    start_idx = jnp.where(effective_valid_phantom, start_idx, 0)

    event_cluster_idx = jnp.repeat(jnp.arange(num_clusters, dtype=jnp.int32), repeats=num_phantom)
    event_start = start_idx[event_cluster_idx]
    event_log_l = log_L_phantom.reshape((-1,))
    left_l = jnp.searchsorted(log_L_blocks, event_log_l, side='left')
    event_a_hi = jnp.minimum(left_l + 1, num_valid_blocks)
    event_b_hi = jnp.minimum(left_l, num_valid_blocks)
    event_a_active = event_a_hi > event_start
    event_b_active = event_b_hi > event_start

    count_a_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_a_active, dtype=dtype),
        length=num_clusters,
    )
    count_b_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_b_active, dtype=dtype),
        length=num_clusters,
    )

    eq_ok = jnp.logical_and(left_l < num_valid_blocks, log_L_blocks[left_l] == event_log_l)
    event_eq_idx = jnp.where(eq_ok, left_l, 0)
    event_eq_active = jnp.logical_and(eq_ok, event_eq_idx >= event_start)
    event_eq_active = jnp.logical_and(event_eq_active, effective_valid_phantom[event_cluster_idx])

    return {
        "effective_valid_phantom": effective_valid_phantom,
        "start_idx": start_idx,
        "event_cluster_idx": event_cluster_idx,
        "event_a_hi": event_a_hi,
        "event_b_hi": event_b_hi,
        "event_a_active": event_a_active,
        "event_b_active": event_b_active,
        "count_a_start_per_cluster": count_a_start_per_cluster,
        "count_b_start_per_cluster": count_b_start_per_cluster,
        "event_eq_idx": event_eq_idx,
        "event_eq_active": event_eq_active,
        "num_blocks": num_blocks,
    }


def _fast_counts_from_precompute(precomputed, cluster_multiplicity):
    return jax_phantom._boundary_counts_from_multiplicity(
        cluster_multiplicity=cluster_multiplicity,
        start_idx=precomputed["start_idx"],
        count_A_start_per_cluster=precomputed["count_a_start_per_cluster"],
        count_B_start_per_cluster=precomputed["count_b_start_per_cluster"],
        event_cluster_idx=precomputed["event_cluster_idx"],
        event_a_hi=precomputed["event_a_hi"],
        event_b_hi=precomputed["event_b_hi"],
        event_A_active=precomputed["event_a_active"],
        event_B_active=precomputed["event_b_active"],
        event_eq_idx=precomputed["event_eq_idx"],
        event_eq_active=precomputed["event_eq_active"],
        num_blocks=int(precomputed["num_blocks"]),
    )


def test_boundary_counts_from_multiplicity_matches_reference_toy_case():
    cluster_multiplicity = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    start_idx = np.array([0, 1, 2], dtype=np.int32)

    event_cluster_idx = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    event_a_hi = np.array([3, 2, 4, 1, 4], dtype=np.int32)
    event_b_hi = np.array([2, 1, 3, 1, 2], dtype=np.int32)
    event_A_active = np.array([True, False, True, True, True], dtype=bool)
    event_B_active = np.array([True, True, True, False, True], dtype=bool)
    event_eq_idx = np.array([1, 0, 3, 2, 1], dtype=np.int32)
    event_eq_active = np.array([True, False, True, False, True], dtype=bool)
    num_blocks = 4

    count_A_start_per_cluster = np.bincount(
        event_cluster_idx,
        weights=event_A_active.astype(np.float32),
        minlength=cluster_multiplicity.shape[0],
    ).astype(np.float32)
    count_B_start_per_cluster = np.bincount(
        event_cluster_idx,
        weights=event_B_active.astype(np.float32),
        minlength=cluster_multiplicity.shape[0],
    ).astype(np.float32)

    ref_A, ref_B, ref_E = _boundary_counts_reference(
        cluster_multiplicity=cluster_multiplicity,
        start_idx=start_idx,
        event_cluster_idx=event_cluster_idx,
        event_a_hi=event_a_hi,
        event_b_hi=event_b_hi,
        event_A_active=event_A_active,
        event_B_active=event_B_active,
        event_eq_idx=event_eq_idx,
        event_eq_active=event_eq_active,
        num_blocks=num_blocks,
    )

    got_A, got_B, got_E = jax_phantom._boundary_counts_from_multiplicity(
        cluster_multiplicity=jnp.asarray(cluster_multiplicity),
        start_idx=jnp.asarray(start_idx),
        count_A_start_per_cluster=jnp.asarray(count_A_start_per_cluster),
        count_B_start_per_cluster=jnp.asarray(count_B_start_per_cluster),
        event_cluster_idx=jnp.asarray(event_cluster_idx),
        event_a_hi=jnp.asarray(event_a_hi),
        event_b_hi=jnp.asarray(event_b_hi),
        event_A_active=jnp.asarray(event_A_active),
        event_B_active=jnp.asarray(event_B_active),
        event_eq_idx=jnp.asarray(event_eq_idx),
        event_eq_active=jnp.asarray(event_eq_active),
        num_blocks=num_blocks,
    )

    np.testing.assert_allclose(np.asarray(got_A), ref_A)
    np.testing.assert_allclose(np.asarray(got_B), ref_B)
    np.testing.assert_allclose(np.asarray(got_E), ref_E)


def test_boundary_counts_from_multiplicity_matches_reference_randomized():
    rng = np.random.default_rng(12)

    num_clusters = 6
    num_blocks = 7
    num_events = 18

    cluster_multiplicity = rng.integers(0, 4, size=num_clusters).astype(np.float32)
    start_idx = rng.integers(0, num_blocks + 1, size=num_clusters, dtype=np.int32)

    event_cluster_idx = rng.integers(0, num_clusters, size=num_events, dtype=np.int32)
    event_A_active = rng.uniform(size=num_events) > 0.35
    event_B_active = rng.uniform(size=num_events) > 0.45

    event_a_hi = np.empty((num_events,), dtype=np.int32)
    event_b_hi = np.empty((num_events,), dtype=np.int32)
    for e in range(num_events):
        c = int(event_cluster_idx[e])
        start = int(start_idx[c])
        event_a_hi[e] = rng.integers(start, num_blocks + 1, dtype=np.int32)
        event_b_hi[e] = rng.integers(start, num_blocks + 1, dtype=np.int32)

    event_eq_idx = rng.integers(0, num_blocks, size=num_events, dtype=np.int32)
    event_eq_active = rng.uniform(size=num_events) > 0.5

    count_A_start_per_cluster = np.bincount(
        event_cluster_idx,
        weights=event_A_active.astype(np.float32),
        minlength=num_clusters,
    ).astype(np.float32)
    count_B_start_per_cluster = np.bincount(
        event_cluster_idx,
        weights=event_B_active.astype(np.float32),
        minlength=num_clusters,
    ).astype(np.float32)

    ref_A, ref_B, ref_E = _boundary_counts_reference(
        cluster_multiplicity=cluster_multiplicity,
        start_idx=start_idx,
        event_cluster_idx=event_cluster_idx,
        event_a_hi=event_a_hi,
        event_b_hi=event_b_hi,
        event_A_active=event_A_active,
        event_B_active=event_B_active,
        event_eq_idx=event_eq_idx,
        event_eq_active=event_eq_active,
        num_blocks=num_blocks,
    )

    got_A, got_B, got_E = jax_phantom._boundary_counts_from_multiplicity(
        cluster_multiplicity=jnp.asarray(cluster_multiplicity),
        start_idx=jnp.asarray(start_idx),
        count_A_start_per_cluster=jnp.asarray(count_A_start_per_cluster),
        count_B_start_per_cluster=jnp.asarray(count_B_start_per_cluster),
        event_cluster_idx=jnp.asarray(event_cluster_idx),
        event_a_hi=jnp.asarray(event_a_hi),
        event_b_hi=jnp.asarray(event_b_hi),
        event_A_active=jnp.asarray(event_A_active),
        event_B_active=jnp.asarray(event_B_active),
        event_eq_idx=jnp.asarray(event_eq_idx),
        event_eq_active=jnp.asarray(event_eq_active),
        num_blocks=num_blocks,
    )

    np.testing.assert_allclose(np.asarray(got_A), ref_A)
    np.testing.assert_allclose(np.asarray(got_B), ref_B)
    np.testing.assert_allclose(np.asarray(got_E), ref_E)


def test_precompute_pipeline_matches_direct_reference_randomized():
    rng = np.random.default_rng(123)

    for seed in range(8):
        num_clusters = int(rng.integers(4, 9))
        num_phantom = int(rng.integers(1, 8))
        num_valid_blocks = int(rng.integers(2, num_clusters + 1))

        increments = np.random.default_rng(seed).uniform(0.2, 0.9, size=num_valid_blocks).astype(np.float32)
        valid_blocks = np.cumsum(increments).astype(np.float32)
        log_l_blocks = np.full((num_clusters,), np.inf, dtype=np.float32)
        log_l_blocks[:num_valid_blocks] = valid_blocks

        log_l_constraints = np.full((num_clusters,), valid_blocks[0] - 1.0, dtype=np.float32)
        if num_valid_blocks > 1:
            parent_idx = rng.integers(0, num_valid_blocks - 1, size=num_clusters)
            log_l_constraints = valid_blocks[parent_idx].astype(np.float32)
        log_l_constraints[0] = -np.inf

        num_samples = int(rng.integers(1, num_clusters + 1))
        effective_sample_mask = np.arange(num_clusters, dtype=np.int32) < num_samples
        valid_phantom = rng.uniform(size=num_clusters) > 0.25
        effective_valid_phantom = valid_phantom & effective_sample_mask

        log_l_phantom = np.empty((num_clusters, num_phantom), dtype=np.float32)
        for c in range(num_clusters):
            for p in range(num_phantom):
                draw = float(rng.uniform())
                if draw < 0.25:
                    g = int(rng.integers(0, num_valid_blocks))
                    log_l_phantom[c, p] = valid_blocks[g]
                elif draw < 0.50:
                    log_l_phantom[c, p] = valid_blocks[-1] + np.float32(rng.uniform(0.05, 1.0))
                else:
                    g = int(rng.integers(0, num_valid_blocks))
                    base = valid_blocks[g]
                    log_l_phantom[c, p] = base - np.float32(rng.uniform(0.05, 0.4))

        cluster_multiplicity = rng.integers(0, 4, size=num_clusters).astype(np.float32)
        cluster_multiplicity = np.where(effective_valid_phantom, cluster_multiplicity, 0.0).astype(np.float32)

        precomputed = _precompute_from_sample_mc_snippet(
            log_L_blocks=jnp.asarray(log_l_blocks),
            log_L_constraints=jnp.asarray(log_l_constraints),
            valid_phantom=jnp.asarray(valid_phantom),
            effective_sample_mask=jnp.asarray(effective_sample_mask),
            log_L_phantom=jnp.asarray(log_l_phantom),
            num_valid_blocks=jnp.asarray(num_valid_blocks, dtype=jnp.int32),
        )

        got_a, got_b, got_e = _fast_counts_from_precompute(precomputed, jnp.asarray(cluster_multiplicity))
        ref_a, ref_b, ref_e = _boundary_counts_direct_from_raw(
            log_L_blocks=log_l_blocks,
            log_L_constraints=log_l_constraints,
            log_L_phantom=log_l_phantom,
            effective_valid_phantom=effective_valid_phantom,
            cluster_multiplicity=cluster_multiplicity,
            num_valid_blocks=num_valid_blocks,
        )

        np.testing.assert_allclose(np.asarray(got_a), ref_a)
        np.testing.assert_allclose(np.asarray(got_b), ref_b)
        np.testing.assert_allclose(np.asarray(got_e), ref_e)


def test_precompute_respects_effective_valid_phantom_mask():
    log_l_blocks = np.array([0.0, 1.0, 2.0, np.inf, np.inf], dtype=np.float32)
    num_valid_blocks = 3
    log_l_constraints = np.array([-np.inf, 0.0, 1.0, 0.0, -np.inf], dtype=np.float32)
    valid_phantom = np.array([True, True, True, True, True], dtype=bool)
    effective_sample_mask = np.array([True, True, False, False, True], dtype=bool)
    log_l_phantom = np.array(
        [
            [0.0, 0.2, 1.0],
            [1.0, 1.5, 2.2],
            [2.0, 2.5, 3.0],
            [0.0, 1.0, 2.0],
            [0.1, 1.0, 1.9],
        ],
        dtype=np.float32,
    )

    precomputed = _precompute_from_sample_mc_snippet(
        log_L_blocks=jnp.asarray(log_l_blocks),
        log_L_constraints=jnp.asarray(log_l_constraints),
        valid_phantom=jnp.asarray(valid_phantom),
        effective_sample_mask=jnp.asarray(effective_sample_mask),
        log_L_phantom=jnp.asarray(log_l_phantom),
        num_valid_blocks=jnp.asarray(num_valid_blocks, dtype=jnp.int32),
    )

    effective_valid_phantom = np.asarray(precomputed["effective_valid_phantom"])
    start_idx = np.asarray(precomputed["start_idx"])
    event_cluster_idx = np.asarray(precomputed["event_cluster_idx"])
    event_eq_active = np.asarray(precomputed["event_eq_active"])

    np.testing.assert_array_equal(effective_valid_phantom, valid_phantom & effective_sample_mask)
    assert np.all(start_idx[~effective_valid_phantom] == 0)
    assert np.all(~event_eq_active[~effective_valid_phantom[event_cluster_idx]])

    cluster_multiplicity = np.array([2.0, 1.0, 0.0, 0.0, 3.0], dtype=np.float32)
    got_a, got_b, got_e = _fast_counts_from_precompute(precomputed, jnp.asarray(cluster_multiplicity))
    ref_a, ref_b, ref_e = _boundary_counts_direct_from_raw(
        log_L_blocks=log_l_blocks,
        log_L_constraints=log_l_constraints,
        log_L_phantom=log_l_phantom,
        effective_valid_phantom=effective_valid_phantom,
        cluster_multiplicity=cluster_multiplicity,
        num_valid_blocks=num_valid_blocks,
    )

    np.testing.assert_allclose(np.asarray(got_a), ref_a)
    np.testing.assert_allclose(np.asarray(got_b), ref_b)
    np.testing.assert_allclose(np.asarray(got_e), ref_e)


def test_precompute_equality_boundary_logic_matches_reference():
    log_l_blocks = np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float32)
    num_valid_blocks = 3
    log_l_constraints = np.array([-np.inf, 0.0, 1.0, 0.0], dtype=np.float32)
    valid_phantom = np.array([True, True, True, False], dtype=bool)
    effective_sample_mask = np.array([True, True, True, True], dtype=bool)
    log_l_phantom = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 0.2, 2.0],
            [2.0, 1.5, 2.0, 0.5],
            [0.0, 1.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    cluster_multiplicity = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    effective_valid_phantom = valid_phantom & effective_sample_mask
    cluster_multiplicity = np.where(effective_valid_phantom, cluster_multiplicity, 0.0).astype(np.float32)

    precomputed = _precompute_from_sample_mc_snippet(
        log_L_blocks=jnp.asarray(log_l_blocks),
        log_L_constraints=jnp.asarray(log_l_constraints),
        valid_phantom=jnp.asarray(valid_phantom),
        effective_sample_mask=jnp.asarray(effective_sample_mask),
        log_L_phantom=jnp.asarray(log_l_phantom),
        num_valid_blocks=jnp.asarray(num_valid_blocks, dtype=jnp.int32),
    )

    got_a, got_b, got_e = _fast_counts_from_precompute(precomputed, jnp.asarray(cluster_multiplicity))
    ref_a, ref_b, ref_e = _boundary_counts_direct_from_raw(
        log_L_blocks=log_l_blocks,
        log_L_constraints=log_l_constraints,
        log_L_phantom=log_l_phantom,
        effective_valid_phantom=effective_valid_phantom,
        cluster_multiplicity=cluster_multiplicity,
        num_valid_blocks=num_valid_blocks,
    )

    np.testing.assert_allclose(np.asarray(got_a), ref_a)
    np.testing.assert_allclose(np.asarray(got_b), ref_b)
    np.testing.assert_allclose(np.asarray(got_e), ref_e)


def test_precompute_pipeline_jit_matches_reference():
    log_l_blocks = jnp.asarray([0.0, 1.0, 2.0, jnp.inf, jnp.inf], dtype=jnp.float32)
    log_l_constraints = jnp.asarray([-jnp.inf, 0.0, 1.0, 0.0, -jnp.inf], dtype=jnp.float32)
    valid_phantom = jnp.asarray([True, True, True, True, False])
    effective_sample_mask = jnp.asarray([True, True, True, False, True])
    log_l_phantom = jnp.asarray(
        [
            [0.0, 0.1, 1.0],
            [1.0, 1.2, 2.5],
            [2.0, 2.0, 0.5],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    cluster_multiplicity = jnp.asarray([1.0, 2.0, 3.0, 0.0, 0.0], dtype=jnp.float32)
    num_valid_blocks = jnp.asarray(3, dtype=jnp.int32)

    @jax.jit
    def _run(log_l_blocks_, log_l_constraints_, valid_phantom_, effective_sample_mask_, log_l_phantom_, num_valid_blocks_,
             cluster_multiplicity_):
        precomputed_ = _precompute_from_sample_mc_snippet(
            log_L_blocks=log_l_blocks_,
            log_L_constraints=log_l_constraints_,
            valid_phantom=valid_phantom_,
            effective_sample_mask=effective_sample_mask_,
            log_L_phantom=log_l_phantom_,
            num_valid_blocks=num_valid_blocks_,
        )
        return _fast_counts_from_precompute(precomputed_, cluster_multiplicity_)

    got_a, got_b, got_e = _run(
        log_l_blocks,
        log_l_constraints,
        valid_phantom,
        effective_sample_mask,
        log_l_phantom,
        num_valid_blocks,
        cluster_multiplicity,
    )

    ref_a, ref_b, ref_e = _boundary_counts_direct_from_raw(
        log_L_blocks=np.asarray(log_l_blocks),
        log_L_constraints=np.asarray(log_l_constraints),
        log_L_phantom=np.asarray(log_l_phantom),
        effective_valid_phantom=np.asarray(valid_phantom & effective_sample_mask),
        cluster_multiplicity=np.asarray(cluster_multiplicity),
        num_valid_blocks=int(np.asarray(num_valid_blocks)),
    )

    np.testing.assert_allclose(np.asarray(got_a), ref_a)
    np.testing.assert_allclose(np.asarray(got_b), ref_b)
    np.testing.assert_allclose(np.asarray(got_e), ref_e)


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
            key=jax.random.PRNGKey(321),
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

        ref_eta_mean = float(np.mean(ref_out.eta_samples))
        jax_eta_mean = float(np.mean(jax_out.eta_samples))
        np.testing.assert_allclose(jax_eta_mean, ref_eta_mean, rtol=0.25, atol=0.05)

        ref_rho_eta_mean = float(np.mean(ref_out.rho_eta_samples))
        jax_rho_eta_mean = float(np.mean(jax_out.rho_eta_samples))
        np.testing.assert_allclose(jax_rho_eta_mean, ref_rho_eta_mean, rtol=0.25, atol=0.05)

        np.testing.assert_allclose(jax_out.rho_eta_samples, jax_out.rho_samples * jax_out.eta_samples, rtol=1e-6, atol=1e-6)
