"""Verify that allocation observations preserve the actual sampled process."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfpd

import jaxns.core as core_module
from benchmarks.issue_312.allocation_observer import run_with_observer
from jaxns.algorithm.depth import _sample_work_batch
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.priors import Prior


def _allocation_observer_model():
    point = Prior(tfpd.Uniform(jnp.zeros(2), jnp.ones(2)), name="x").realise()
    return -jnp.sum(((point - 0.5) / 0.1) ** 2) / 2


@pytest.mark.parametrize("allocation", ["uniform", "evidence_improving", "posterior_improving"])
def test_allocation_observer_replays_seeds_and_preserves_complete_state(allocation):
    runner = NestedSampler(
        model=Model(prior_model=_allocation_observer_model),
        sampler=UniDimSliceSampler(num_slices=4), root_allocation_degree=12,
        replacement_width=4, delta_K=12, initial_capacity=512, unlimited_samples=True,
        allocation_target=allocation,
    )
    initial = runner.initialise(jax.random.PRNGKey(31241))
    configured = runner._resolve_config(initial.model, initial.args, initial.params).sampler
    reference = runner.resume_until_goal(initial, lambda state: int(state.goal_loop_iter) >= 1)
    observed = []

    def observer(before, after, work):
        start, stop = int(before.num_samples), int(after.num_samples)
        assert stop - start == int(work.num_valid)
        if len(observed) < 3:
            # Independent replay of the actual child generation verifies seed
            # identities, not merely that parent contours were reproduced.
            new_depth_key, _ = jax.random.split(before.random_key)
            key = jnp.where(before.depth_reached, new_depth_key, before.random_key)
            _, sample_key, _ = jax.random.split(key, 3)
            batch = jax.jit(_sample_work_batch)(sample_key, before, configured, work)
            np.testing.assert_array_equal(
                jax.tree.leaves(after.samples.U_samples)[0][start:stop],
                jax.tree.leaves(batch.U_samples)[0][:stop - start],
            )
            np.testing.assert_allclose(
                after.samples.log_likelihoods[start:stop], batch.log_likelihoods[:stop - start],
                rtol=1e-6, atol=1e-6,
            )
        observed.append(stop)
        return False

    original_depth = core_module._run_depth
    actual = run_with_observer(
        runner, initial, observer, lambda state: int(state.goal_loop_iter) >= 1,
    )
    assert core_module._run_depth is original_depth and len(observed) > 3
    assert jax.tree.structure(actual) == jax.tree.structure(reference)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True):
        np.testing.assert_array_equal(left, right)


def test_allocation_budget_return_resumes_without_an_extra_goal_boundary():
    runner = NestedSampler(
        model=Model(prior_model=_allocation_observer_model),
        sampler=UniDimSliceSampler(num_slices=4), root_allocation_degree=12,
        replacement_width=4, initial_capacity=64, unlimited_samples=True,
    )
    initial = runner.initialise(jax.random.PRNGKey(31242))
    reference = runner.resume_until_goal(initial, lambda state: int(state.goal_loop_iter) >= 1)
    original_depth = core_module._run_depth
    stopped = run_with_observer(runner, initial, lambda before, after, work: True, lambda s: False)
    assert core_module._run_depth is original_depth
    assert int(stopped.num_samples) > int(initial.num_samples)
    assert int(stopped.goal_loop_iter) == 0 and int(stopped.termination_reason) == 0
    resumed = runner.resume_until_goal(stopped, lambda state: int(state.goal_loop_iter) >= 1)
    for left, right in zip(jax.tree.leaves(resumed), jax.tree.leaves(reference), strict=True):
        np.testing.assert_array_equal(left, right)
