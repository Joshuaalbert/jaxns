"""Composed scientific contracts that were missing from the release gate."""

import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.core_fixtures import make_state
from cicd.tests.distributed_support import make_toy_model
from jaxns.algorithm.race_tree import build_block_state
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.shrinkage.classic import (
    classic_dirichlet_concentrations,
    expected_evidence_summary,
)


@pytest.mark.parametrize("order", [
    (5, 2, 0, 4, 1, 3),
    (0, 2, 1, 4, 3, 5),
])
def test_stored_sample_permutation_preserves_evidence_and_posterior(order):
    state = make_state(
        root_out_degree=3,
        log_likelihoods=(1.0, 2.0, 2.0, 3.0, 3.0, 4.0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf, 1.0, 2.0, 3.0),
        out_degree=(1, 1, 0, 1, 0, 0),
        max_samples=6,
    )
    permuted = dataclasses.replace(
        state, samples=state.samples[jnp.asarray(order)],
    )
    expected = state.to_result()
    actual = permuted.to_result()
    # The second permutation only swaps members of equality blocks. Their
    # parent contours and degrees differ, so a test using identical rows would
    # miss an accidental dependence on the first arrival in a plateau.
    jax.tree.map(
        np.testing.assert_array_equal,
        (actual.block_data.log_L, actual.block_data.size,
         actual.block_data.incoming_K, actual.block_data.out_degree),
        (expected.block_data.log_L, expected.block_data.size,
         expected.block_data.incoming_K, expected.block_data.out_degree),
    )
    np.testing.assert_allclose(actual.log_Z_mean, expected.log_Z_mean)
    np.testing.assert_allclose(actual.log_Z_uncert, expected.log_Z_uncert)
    np.testing.assert_allclose(actual.log_dp, expected.log_dp[jnp.asarray(order)])
    np.testing.assert_allclose(
        actual.log_X_mean, expected.log_X_mean[jnp.asarray(order)],
    )
    key = jax.random.PRNGKey(303)
    np.testing.assert_array_equal(
        actual.sample_evidence(32, key=key).log_Z_samples,
        expected.sample_evidence(32, key=key).log_Z_samples,
    )


def test_evidence_mass_is_nonnegative_and_posterior_is_normalized():
    state = make_state(
        root_out_degree=5,
        log_likelihoods=(-np.inf, -1.0, -1.0, 2.0, 4.0),
        out_degree=(0, 0, 0, 0, 0),
        max_samples=8,
    )
    blocks = build_block_state(state.samples, state.root_out_degree, state.num_samples)
    summary = expected_evidence_summary(blocks, classic_dirichlet_concentrations(blocks))
    contribution = np.exp(np.asarray(summary.log_dZ_mean))
    assert np.all(np.isfinite(contribution))
    assert np.all(contribution >= 0.0)
    np.testing.assert_array_equal(contribution[~np.asarray(blocks.valid)], 0.0)
    assert contribution[0] == 0.0
    result = state.to_result()
    mass = np.exp(np.asarray(result.log_dp))
    assert np.all(np.isfinite(mass))
    assert np.all(mass >= 0.0)
    np.testing.assert_allclose(mass.sum(), 1.0, atol=1e-14)
    assert mass[0] == 0.0
    np.testing.assert_array_equal(mass[5:], 0.0)


@pytest.mark.parametrize("allocation_target", [
    "uniform", "evidence_improving", "posterior_improving",
])
def test_completed_race_law_depends_on_edges_not_allocation_policy(allocation_target):
    runner = NestedSampler(
        model=make_toy_model(), root_allocation_degree=8, replacement_width=2,
        delta_K=8, max_samples=128, initial_capacity=128,
        sampler=UniDimSliceSampler(num_slices=2),
        depth_condition=DepthCondition(dlogZ=jnp.asarray(0.2)),
        allocation_target=allocation_target,
    )
    state = runner.run_until_goal(
        lambda value: int(value.goal_loop_iter) >= 2,
        key=jax.random.PRNGKey(306),
    ).trim()
    state.ensure_consistency()
    blocks = build_block_state(state.samples, state.root_out_degree, state.num_samples)
    log_l = np.asarray(state.samples.log_likelihoods)
    parent = np.asarray(state.samples.log_L_constraints)
    valid = np.asarray(blocks.valid)
    levels = np.asarray(blocks.log_L_blocks)[valid]
    # Count actual contour crossings independently of the prefix recurrence.
    # Allocation may change these edges, but never the law of the stored tree.
    crossings = np.sum(
        (parent[:, None] < levels) & (log_l[:, None] >= levels), axis=0,
    )
    np.testing.assert_array_equal(np.asarray(blocks.incoming_K)[valid], crossings)
    concentrations = classic_dirichlet_concentrations(blocks)
    # This continuous model has singleton arrivals. Their strict shrinkage
    # must be Beta(K, 1) for every scheduling policy.
    np.testing.assert_array_equal(np.asarray(blocks.block_size)[valid], 1)
    np.testing.assert_array_equal(np.asarray(concentrations.alpha_gt)[valid], crossings)
    np.testing.assert_array_equal(np.asarray(concentrations.alpha_eq)[valid], 0.0)
    np.testing.assert_array_equal(np.asarray(concentrations.alpha_lt)[valid], 1.0)
    edges = np.asarray(state.determine_parent_graph())
    parents, children = edges.T
    np.testing.assert_array_equal(np.sort(children), np.arange(log_l.size))
    np.testing.assert_array_equal(
        np.bincount(parents + 1, minlength=log_l.size + 1),
        np.r_[int(state.root_out_degree), np.asarray(state.samples.out_degree)],
    )
    assert int(state.root_out_degree) > 0


def test_final_phantom_inference_leaves_state_and_continuation_unchanged():
    runner = NestedSampler(
        model=make_toy_model(), root_allocation_degree=16, replacement_width=2,
        sampler=UniDimSliceSampler(num_slices=4), collect_phantom_samples=True,
        max_samples=256, initial_capacity=256,
        depth_condition=DepthCondition(dlogZ=jnp.asarray(0.2)),
    )
    state = runner.run(key=jax.random.PRNGKey(307))
    before = [np.array(value) for value in jax.tree.leaves(state)]
    result_before = state.to_result()
    conditioned = state.sample_evidence(
        32, key=jax.random.PRNGKey(308), phantom_conditioning=True, C_min=1,
    )
    assert np.any(np.asarray(conditioned.phantom_gate_active))
    for actual, expected in zip(jax.tree.leaves(state), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    result_after = state.to_result()
    np.testing.assert_array_equal(result_after.log_dp, result_before.log_dp)
    np.testing.assert_array_equal(result_after.log_Z_mean, result_before.log_Z_mean)
    replay = runner.run(key=jax.random.PRNGKey(307))
    jax.tree.map(np.testing.assert_array_equal, replay, state)


def test_phantom_evidence_requires_collected_metadata():
    state = NestedSampler(
        model=make_toy_model(), root_allocation_degree=4,
    ).initialise(jax.random.PRNGKey(309))
    with pytest.raises(ValueError, match="no phantom slots were collected"):
        state.sample_evidence(4, key=jax.random.PRNGKey(310), phantom_conditioning=True)
