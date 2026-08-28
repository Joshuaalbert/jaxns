"""Durability and automatic-resume contracts for run checkpoints."""

import dataclasses
import json
import multiprocessing
import os
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jaxns.checkpoint as checkpoint_module
from cicd.tests.core_fixtures import make_state
from cicd.tests.distributed_support import make_toy_model
from jaxns.checkpoint import (
    CHECKPOINT_CADENCE_SECONDS,
    CheckpointCorruptionError,
    CheckpointInUseError,
    CheckpointManager,
)
from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    UniDimSliceSampler,
)
from jaxns.core import CoreWorkBatch, NestedSampler
from jaxns.distributed_core import (
    DistributedState,
    PendingTask,
    ReservationState,
)
from jaxns.pytree import PureDataclassPytree, Pytree
from jaxns.samples import SeedPoint
from jaxns.termination_condition import TerminationCondition


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointValue(PureDataclassPytree):
    """Small complete Pytree used to isolate filesystem behavior."""

    value: jax.Array  # []


CheckpointValue.register_pytree()


class ManualClock:
    """Controllable monotonic clock for cadence tests."""

    __slots__ = ("seconds",)

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


def _value(value: int) -> CheckpointValue:
    return CheckpointValue(jnp.asarray(value, dtype=jnp.int32))


def _manifest(checkpoint_dir: Path) -> dict[str, object]:
    return json.loads(
        (checkpoint_dir / "CHECKPOINT").read_text(encoding="utf-8")
    )


def _assert_same_tree(left, right) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
        )


def _attempt_checkpoint_lock(checkpoint_dir: str, outcomes) -> None:
    try:
        with CheckpointManager[CheckpointValue](checkpoint_dir):
            outcomes.put("acquired")
    except CheckpointInUseError:
        outcomes.put("blocked")


def test_checkpoint_round_trip_publishes_manifest_after_complete_state(
        tmp_path,
):
    state = _value(7)
    with CheckpointManager[CheckpointValue](tmp_path) as manager:
        manager.save(state)

    manifest = _manifest(tmp_path)
    assert manifest["schema_version"] == 1
    assert manifest["generation"] == 1
    assert manifest["checksum_algorithm"] == "sha256"
    assert len(manifest["checksum"]) == 64
    assert (tmp_path / manifest["state_file"]).is_file()

    with CheckpointManager[CheckpointValue](tmp_path) as manager:
        restored = manager.load()
    _assert_same_tree(restored, state)


def test_checkpoint_cadence_uses_monotonic_seconds_and_retains_two_generations(
        tmp_path,
):
    clock = ManualClock()
    with CheckpointManager[CheckpointValue](
        tmp_path,
        cadence_seconds=CHECKPOINT_CADENCE_SECONDS,
        clock=clock,
    ) as manager:
        clock.seconds = CHECKPOINT_CADENCE_SECONDS - 1.0
        assert not manager.maybe_save(_value(1))
        clock.seconds = CHECKPOINT_CADENCE_SECONDS
        assert manager.maybe_save(_value(1))
        clock.seconds = 2.0 * CHECKPOINT_CADENCE_SECONDS
        assert manager.maybe_save(_value(2))
        clock.seconds = 3.0 * CHECKPOINT_CADENCE_SECONDS
        assert manager.maybe_save(_value(3))

    generations = sorted(tmp_path.glob("state-*.pkl"))
    assert [path.name for path in generations] == [
        "state-00000000000000000002.pkl",
        "state-00000000000000000003.pkl",
    ]


def test_checksum_mismatch_fails_closed_before_deserialization(
        tmp_path,
        monkeypatch,
):
    with CheckpointManager[CheckpointValue](tmp_path) as manager:
        manager.save(_value(1))
        manager.save(_value(2))
    manifest = _manifest(tmp_path)
    state_path = tmp_path / manifest["state_file"]
    state_path.write_bytes(state_path.read_bytes() + b"corruption")

    deserialized = []

    def forbidden_load(filename):
        deserialized.append(filename)
        raise AssertionError("Corrupted bytes reached pickle.load.")

    monkeypatch.setattr(Pytree, "load", staticmethod(forbidden_load))
    with (
        CheckpointManager[CheckpointValue](tmp_path) as manager,
        pytest.raises(CheckpointCorruptionError, match="checksum mismatch"),
    ):
        manager.load()
    assert not deserialized
    # The prior generation is retained for explicit operator recovery, but a
    # corrupt newest commit never causes a silent scientific rollback.
    assert len(tuple(tmp_path.glob("state-*.pkl"))) == 2


def test_interrupted_state_write_preserves_previous_commit(
        tmp_path,
        monkeypatch,
):
    first = _value(1)
    with CheckpointManager[CheckpointValue](tmp_path) as manager:
        manager.save(first)

        def interrupted_save(self, filename):
            del self
            Path(filename).write_bytes(b"partial pickle")
            raise OSError("injected state interruption")

        with monkeypatch.context() as patcher:
            patcher.setattr(CheckpointValue, "save", interrupted_save)
            with pytest.raises(OSError, match="state interruption"):
                manager.save(_value(2))

    assert not tuple(tmp_path.glob(".*.tmp.pkl"))
    with CheckpointManager[CheckpointValue](tmp_path) as manager:
        restored = manager.load()
    _assert_same_tree(restored, first)


def test_interrupted_manifest_publication_preserves_previous_commit(
        tmp_path,
        monkeypatch,
):
    first = _value(1)
    with CheckpointManager[CheckpointValue](tmp_path) as manager:
        manager.save(first)
        replace = os.replace

        def interrupt_manifest(source, target):
            if Path(target).name == "CHECKPOINT":
                raise OSError("injected manifest interruption")
            replace(source, target)

        with monkeypatch.context() as patcher:
            patcher.setattr(
                checkpoint_module.os,
                "replace",
                interrupt_manifest,
            )
            with pytest.raises(OSError, match="manifest interruption"):
                manager.save(_value(2))

    with CheckpointManager[CheckpointValue](tmp_path) as manager:
        restored = manager.load()
    _assert_same_tree(restored, first)


def test_missing_manifest_with_state_file_is_incomplete_checkpoint(tmp_path):
    (tmp_path / "state-00000000000000000001.pkl").write_bytes(b"partial")
    with (
        CheckpointManager[CheckpointValue](tmp_path) as manager,
        pytest.raises(CheckpointCorruptionError, match="without a committed"),
    ):
        manager.load()


def test_malformed_manifest_fails_before_state_lookup(tmp_path):
    (tmp_path / "CHECKPOINT").write_text("{not-json", encoding="utf-8")
    with (
        CheckpointManager[CheckpointValue](tmp_path) as manager,
        pytest.raises(CheckpointCorruptionError, match="valid UTF-8 JSON"),
    ):
        manager.load()


def test_missing_state_file_and_unsupported_schema_fail_clearly(tmp_path):
    manifest_path = tmp_path / "CHECKPOINT"
    manifest_path.write_text(
        json.dumps({
            "schema_version": 1,
            "generation": 1,
            "state_file": "state-00000000000000000001.pkl",
            "checksum_algorithm": "sha256",
            "checksum": "0" * 64,
        }),
        encoding="utf-8",
    )
    with (
        CheckpointManager[CheckpointValue](tmp_path) as manager,
        pytest.raises(CheckpointCorruptionError, match="is missing"),
    ):
        manager.load()

    manifest = _manifest(tmp_path)
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with (
        CheckpointManager[CheckpointValue](tmp_path) as manager,
        pytest.raises(CheckpointCorruptionError, match="schema version"),
    ):
        manager.load()


def test_checkpoint_directory_has_one_writer_for_whole_run(tmp_path):
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    with CheckpointManager[CheckpointValue](tmp_path):
        process = context.Process(
            target=_attempt_checkpoint_lock,
            args=(str(tmp_path), outcomes),
        )
        process.start()
        process.join(timeout=15.0)
    assert not process.is_alive()
    assert process.exitcode == 0
    outcome = outcomes.get(timeout=1.0)
    outcomes.close()
    outcomes.join_thread()
    assert outcome == "blocked"


@pytest.mark.parametrize("cadence", [-1.0, float("nan"), float("inf")])
def test_checkpoint_cadence_rejects_invalid_seconds(tmp_path, cadence):
    with pytest.raises(ValueError, match="finite and non-negative"):
        CheckpointManager[CheckpointValue](tmp_path, cadence_seconds=cadence)


def test_local_automatic_resume_matches_uninterrupted_random_stream(tmp_path):
    model = make_toy_model()
    sampler = UniDimSliceSampler(model=model, num_slices=2)
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=2,
        shell_size=1,
        max_samples=10,
        initial_capacity=4,
        sampler=sampler,
        termination_condition=TerminationCondition(max_samples=10),
    )

    def first_goal(state):
        return int(state.goal_loop_iter) >= 1

    def final_goal(state):
        return int(state.goal_loop_iter) >= 3

    key = jax.random.PRNGKey(17)
    uninterrupted = nested_sampler.run_until_goal(final_goal, key=key)
    nested_sampler.run_until_goal(
        first_goal,
        key=key,
        checkpoint_dir=tmp_path,
    )
    resumed = nested_sampler.run_until_goal(
        final_goal,
        # A loaded checkpoint owns its continuation key.
        key=jax.random.PRNGKey(999),
        checkpoint_dir=tmp_path,
    )

    _assert_same_tree(resumed, uninterrupted)


def test_run_single_iteration_automatically_loads_and_advances_checkpoint(
        tmp_path,
):
    model = make_toy_model()
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=2,
        shell_size=1,
        max_samples=6,
        initial_capacity=6,
        sampler=UniDimSliceSampler(model=model, num_slices=2),
    )
    first = nested_sampler.run_single_iteration(
        key=jax.random.PRNGKey(21),
        checkpoint_dir=tmp_path,
    )
    expected = nested_sampler.run_single_iteration(first)
    unrelated = nested_sampler.initialise(jax.random.PRNGKey(999))
    resumed = nested_sampler.run_single_iteration(
        unrelated,
        checkpoint_dir=tmp_path,
    )
    _assert_same_tree(resumed, expected)


def test_complete_distributed_pending_state_round_trips_without_task_loss(
        tmp_path,
):
    state = make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0),
        out_degree=(0, 0),
        max_samples=5,
    )
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(31), 1),
        valid=jnp.asarray([True]),
        log_L_constraints=jnp.asarray([0.0]),
        seed_points=SeedPoint(
            U0=jnp.asarray([0.4]),
            log_L0=jnp.asarray([0.0]),
        ),
        sampler_data=None,
    )
    task_work = CoreWorkBatch(
        valid=jnp.asarray([True]),
        parent_idx=jnp.asarray([0], dtype=jnp.int32),
        log_L_constraint=jnp.asarray([0.0]),
        seed_idx=jnp.asarray([1], dtype=jnp.int32),
    )
    reservations = ReservationState(
        parent_delta=jnp.asarray([1, 0, 0, 0, 0], dtype=jnp.int32),
        root_delta=jnp.asarray(0, dtype=jnp.int32),
        num_reserved=jnp.asarray(1, dtype=jnp.int32),
    )
    distributed = DistributedState(
        state=state,
        reservations=reservations,
        pending=(PendingTask(7, task_work, request),),
        next_task_id=8,
        session_id="checkpoint-session",
        depth_active=True,
        goal_key=jax.random.PRNGKey(32),
    )

    with CheckpointManager[DistributedState](tmp_path) as manager:
        manager.save(distributed)
    with CheckpointManager[DistributedState](tmp_path) as manager:
        restored = manager.load()

    _assert_same_tree(restored, distributed)
    assert restored.pending[0].task_id == 7
    assert restored.next_task_id == 8
    assert int(restored.reservations.num_reserved) == 1
