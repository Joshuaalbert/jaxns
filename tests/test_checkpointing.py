import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import random
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.checkpointing import CheckpointValidationError, append_checkpoint, initialise_archive
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler, _check_termination, _flatten_step_deltas, _run_ns_chunk, _sample_init_state
from jaxns.model import Model


tfpd = tfp.distributions


def _make_basic_model() -> Model:
    """Create a tiny one-parameter model for checkpoint tests."""

    def prior_model():
        x = Prior(tfpd.Uniform(low=0.0, high=1.0), name='x').realise()
        return -x ** 2

    return Model(prior_model=prior_model)


def _make_sampler(model: Model, *, max_samples: int = 28) -> NestedSampler:
    """Build a small nested sampler configuration for checkpoint tests."""

    return NestedSampler(
        model=model,
        target_num_live_points=10,
        max_samples=max_samples,
        shell_size=2,
    )


def _assert_states_match(left, right):
    """Assert that two states match on the checkpointed payload surface."""

    assert int(left.root_out_degree) == int(right.root_out_degree)
    assert int(left.num_samples) == int(right.num_samples)
    assert int(left.termination_reason) == int(right.termination_reason)
    np.testing.assert_allclose(np.asarray(left.samples.log_likelihoods), np.asarray(right.samples.log_likelihoods))
    np.testing.assert_array_equal(np.asarray(left.samples.sample_ids), np.asarray(right.samples.sample_ids))
    np.testing.assert_array_equal(np.asarray(left.samples.out_degree), np.asarray(right.samples.out_degree))
    np.testing.assert_array_equal(
        np.asarray(left.samples.num_likelihood_evaluations),
        np.asarray(right.samples.num_likelihood_evaluations),
    )
    np.testing.assert_allclose(np.asarray(left.log_L_supremum), np.asarray(right.log_L_supremum))

    left_u_supremum, _ = jax.tree.flatten(left.U_supremum)
    right_u_supremum, _ = jax.tree.flatten(right.U_supremum)
    for left_leaf, right_leaf in zip(left_u_supremum, right_u_supremum, strict=True):
        np.testing.assert_allclose(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_phantom_payload_match(left, right):
    """Assert that phantom sample payloads match exactly between two states."""

    np.testing.assert_array_equal(
        np.asarray(left.samples.phantom_samples.valid_mask),
        np.asarray(right.samples.phantom_samples.valid_mask),
    )
    np.testing.assert_allclose(
        np.asarray(left.samples.phantom_samples.log_L),
        np.asarray(right.samples.phantom_samples.log_L),
    )

    left_u_samples = left.samples.phantom_samples.U_samples
    right_u_samples = right.samples.phantom_samples.U_samples
    if left_u_samples is None or right_u_samples is None:
        assert left_u_samples is None
        assert right_u_samples is None
        return

    left_leaves, _ = jax.tree.flatten(left_u_samples)
    right_leaves, _ = jax.tree.flatten(right_u_samples)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(left_leaf), np.asarray(right_leaf))


def test_checkpointed_run_matches_direct_run(tmp_path: Path):
    """A fresh checkpointed run should match the direct non-checkpointed run."""

    model = _make_basic_model()
    sampler = _make_sampler(model)
    key = random.PRNGKey(0)
    archive_path = tmp_path / 'checkpoint.h5'

    direct_state = sampler.run(key)
    checkpointed_state = sampler.run(key, archive_path=archive_path, checkpoint_every=2)
    resumed_state = sampler.run(resume=True, archive_path=archive_path)

    _assert_states_match(checkpointed_state, direct_state)
    _assert_states_match(resumed_state, checkpointed_state)


def test_resume_from_partial_archive_matches_direct_run(tmp_path: Path):
    """Resuming from a partial archive should reproduce the uninterrupted run."""

    model = _make_basic_model()
    sampler = _make_sampler(model, max_samples=32)
    archive_path = tmp_path / 'partial-checkpoint.h5'
    key = random.PRNGKey(1)

    run_key, init_key = random.split(key)
    initial_state = _sample_init_state(
        key=init_key,
        num_live_points=int(sampler.target_num_live_points),
        max_samples=int(sampler.max_samples),
        model=sampler.model,
        num_phantom=int(sampler.sampler.num_phantom()),
        args=sampler.args,
        params=sampler.params,
        store_phantom_samples=sampler.store_phantom_samples,
        batch_size=sampler.batch_size,
    )
    done, termination_reason = _check_termination(initial_state, int(sampler.target_num_live_points), sampler.termination_condition)
    done = bool(np.asarray(jax.device_get(done)))
    if done:
        initial_state = dataclasses.replace(initial_state, termination_reason=termination_reason)

    initialise_archive(
        archive_path,
        nested_sampler=sampler,
        state=initial_state,
        current_key=run_key,
        checkpoint_every=2,
        completed=done,
    )

    assert not done

    next_key, partial_state, step_deltas, steps_executed, done, termination_reason = _run_ns_chunk(
        key=run_key,
        state=initial_state,
        target_num_live_points=int(sampler.target_num_live_points),
        shell_size=int(sampler.shell_size),
        checkpoint_every=2,
        args=sampler.args,
        sampler=sampler.sampler,
        params=sampler.params,
        termination_condition=sampler.termination_condition,
        batch_size=sampler.batch_size,
    )
    steps_executed = int(np.asarray(jax.device_get(steps_executed)))
    flat_samples, parent_sample_ids = _flatten_step_deltas(step_deltas, steps_executed)
    append_checkpoint(
        archive_path,
        samples=flat_samples,
        parent_sample_ids=parent_sample_ids,
        current_key=next_key,
        state=partial_state,
        checkpoint_index=1,
        checkpoint_every=2,
        completed=bool(np.asarray(jax.device_get(done))),
    )

    direct_state = sampler.run(key)
    resumed_state = sampler.run(resume=True, archive_path=archive_path)

    _assert_states_match(resumed_state, direct_state)


def test_resume_rejects_incompatible_sampler_configuration(tmp_path: Path):
    """Resume should fail fast when the sampler configuration no longer matches."""

    model = _make_basic_model()
    sampler = _make_sampler(model)
    archive_path = tmp_path / 'mismatch-checkpoint.h5'

    sampler.run(random.PRNGKey(2), archive_path=archive_path, checkpoint_every=2)

    incompatible_sampler = NestedSampler(
        model=model,
        target_num_live_points=12,
        max_samples=28,
        shell_size=2,
    )

    with pytest.raises(CheckpointValidationError):
        incompatible_sampler.run(resume=True, archive_path=archive_path)


def test_checkpointing_persists_phantom_samples(tmp_path: Path):
    """Checkpoint archives should round-trip collected phantom payloads, including U-samples."""

    model = _make_basic_model()
    sampler = NestedSampler(
        model=model,
        target_num_live_points=10,
        max_samples=28,
        shell_size=2,
        sampler=UniDimSliceSampler(
            model=model,
            num_slices=5,
            collect_phantom_samples=True,
            phantom_burn_in=1,
        ),
        collect_phantom_samples=True,
        store_phantom_samples=True,
    )
    archive_path = tmp_path / 'phantom-checkpoint.h5'
    key = random.PRNGKey(3)

    direct_state = sampler.run(key)
    checkpointed_state = sampler.run(key, archive_path=archive_path, checkpoint_every=2)
    resumed_state = sampler.run(resume=True, archive_path=archive_path)

    assert checkpointed_state.samples.phantom_samples.U_samples is not None
    assert checkpointed_state.samples.phantom_samples.valid_mask.shape[1] > 0
    assert int(np.sum(np.asarray(checkpointed_state.samples.phantom_samples.valid_mask))) > 0

    _assert_states_match(checkpointed_state, direct_state)
    _assert_phantom_payload_match(checkpointed_state, direct_state)
    _assert_states_match(resumed_state, checkpointed_state)
    _assert_phantom_payload_match(resumed_state, checkpointed_state)