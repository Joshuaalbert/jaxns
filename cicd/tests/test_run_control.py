"""Interruptions preserve scientific work and Python signal ownership."""

import signal

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.distributed_support import make_toy_model
from jaxns import core
from jaxns.checkpoint import CheckpointManager
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.state import State


@pytest.mark.parametrize("batch_budget", [1, 128])
def test_local_sigint_saves_partial_depth_and_replays_exactly(
        tmp_path, monkeypatch, batch_budget,
):
    runner = NestedSampler(
        model=make_toy_model(), root_allocation_degree=4, delta_K=4,
        replacement_width=2, initial_capacity=128, max_samples=128,
        sampler=UniDimSliceSampler(num_slices=2),
        depth_condition=DepthCondition(dlogZ=jnp.asarray(0.1)),
    )
    initial = runner.initialise(jax.random.PRNGKey(302))
    expected = runner.resume_until_goal(
        initial, lambda state: int(state.goal_loop_iter) >= 2,
    )
    calls = []

    def goal(state):
        calls.append(int(state.goal_loop_iter))
        return int(state.goal_loop_iter) >= 2

    run_depth = core._run_depth
    interrupted = False

    def stop_during_depth(state, *args, **kwargs):
        nonlocal interrupted
        advanced = run_depth(state, *args, **kwargs)
        if (
            int(advanced.num_samples) > int(initial.num_samples)
            and not interrupted
        ):
            interrupted = True
            signal.raise_signal(signal.SIGINT)
        return advanced

    previous_handler = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(core, "INTERRUPT_BATCHES", batch_budget)
    monkeypatch.setattr(core, "_run_depth", stop_during_depth)
    with pytest.raises(KeyboardInterrupt):
        runner.resume_until_goal(initial, goal, checkpoint_dir=tmp_path)
    assert signal.getsignal(signal.SIGINT) == previous_handler
    with CheckpointManager[State](tmp_path) as manager:
        saved = manager.load()
    assert int(saved.num_samples) > int(initial.num_samples)
    if batch_budget == 1:
        assert saved.scheduler_data is not None
        assert not bool(saved.depth_reached)
    resumed = runner.resume_until_goal(initial, goal, checkpoint_dir=tmp_path)
    assert calls == [0, 1, 2]
    for reference, actual in zip(
            jax.tree.leaves(expected), jax.tree.leaves(resumed), strict=True,
    ):
        np.testing.assert_array_equal(reference, actual)


def test_local_sigint_without_checkpoint_bubbles_up(monkeypatch):
    runner = NestedSampler(model=make_toy_model(), root_allocation_degree=3)

    def stop(state):
        signal.raise_signal(signal.SIGINT)
        return False

    def unexpected_save(*args, **kwargs):
        pytest.fail("Checkpointing was not requested.")

    monkeypatch.setattr(CheckpointManager, "save_if_changed", unexpected_save)
    previous_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        runner.run_until_goal(stop, key=jax.random.PRNGKey(302))
    assert signal.getsignal(signal.SIGINT) == previous_handler


@pytest.mark.parametrize("batch_budget", [1, 128])
def test_single_iteration_sigint_saves_and_resumes_exact_depth(
        tmp_path, monkeypatch, batch_budget,
):
    runner = NestedSampler(
        model=make_toy_model(), root_allocation_degree=4, delta_K=4,
        replacement_width=2, initial_capacity=128, max_samples=128,
        sampler=UniDimSliceSampler(num_slices=2),
        depth_condition=DepthCondition(dlogZ=jnp.asarray(0.1)),
    )
    initial = runner.initialise(jax.random.PRNGKey(307))
    expected = runner.run_single_iteration(initial)
    run_depth = core._run_depth
    interrupted = False

    def stop_after_batch(state, *args, **kwargs):
        nonlocal interrupted
        advanced = run_depth(state, *args, **kwargs)
        if int(advanced.num_samples) > int(initial.num_samples) and not interrupted:
            interrupted = True
            signal.raise_signal(signal.SIGINT)
        return advanced

    monkeypatch.setattr(core, "INTERRUPT_BATCHES", batch_budget)
    monkeypatch.setattr(core, "_run_depth", stop_after_batch)
    previous_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        runner.run_single_iteration(initial, checkpoint_dir=tmp_path)
    assert signal.getsignal(signal.SIGINT) == previous_handler
    with CheckpointManager[State](tmp_path) as manager:
        saved = manager.load()
    assert int(saved.num_samples) > int(initial.num_samples)
    if bool(saved.depth_reached):
        # The signal arrived at the completed boundary. Resuming would start
        # the next iteration, so the checkpoint itself is the completed result.
        actual = saved
    else:
        actual = runner.run_single_iteration(initial, checkpoint_dir=tmp_path)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(left, right)


def test_verbose_reports_goals_without_computing_results(monkeypatch):
    messages = []
    monkeypatch.setattr(
        core.jaxns_logger, "info", lambda *args: messages.append(args),
    )

    def unexpected_analysis(*args, **kwargs):
        pytest.fail("Progress must not perform evidence/posterior analysis.")

    monkeypatch.setattr(State, "to_result", unexpected_analysis)
    monkeypatch.setattr(State, "sample_evidence", unexpected_analysis)
    monkeypatch.setattr(
        State, "expected_log_Z_mean", property(unexpected_analysis),
    )
    monkeypatch.setattr(
        State, "expected_log_Z_uncert", property(unexpected_analysis),
    )
    runner = NestedSampler(
        model=make_toy_model(), root_allocation_degree=3, max_samples=32,
        sampler=UniDimSliceSampler(num_slices=2), verbose=True,
        depth_condition=DepthCondition(dlogZ=jnp.asarray(0.5)),
    )
    completed = runner.run_until_goal(
        lambda state: int(state.goal_loop_iter) >= 2,
        key=jax.random.PRNGKey(302),
    )
    progress = [
        message for message in messages if message[0].startswith("Local goal")
    ]
    assert [message[1] for message in progress] == [1, 2]
    assert progress[-1][2] == int(completed.num_samples)
