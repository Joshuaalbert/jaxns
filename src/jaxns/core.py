"""Depth-first nested sampling core described by the paper."""

import dataclasses
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxctx import CtxParams

from jaxns.algorithm.depth import (
    MAX_SAMPLES_REACHED,
    _continuation_storage_full,
    _continue_schedule_round,
    _depth_condition_reached,
    _publish_seed_source,
    _refresh_likelihood_order,
    _resize_depth_state,
    _run_depth,
    _seed_source_refresh_due,
    _start_schedule_round,
    _start_seed_storage_full,
)
from jaxns.algorithm.initialisation import _sample_init_state
from jaxns.algorithm.schedule_storage import (
    _grow_continuation_storage,
    _grow_start_seed_storage,
)
from jaxns.checkpoint import (
    CHECKPOINT_CADENCE_SECONDS,
    CheckpointManager,
)
from jaxns.constrained_sampler import (
    AbstractSampler,
)
from jaxns.depth_condition import DepthCondition
from jaxns.logging import jaxns_logger
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.run_config import (
    ResolvedRunConfig,
    default_depth_condition,
    resolve_run_config,
)
from jaxns.run_control import INTERRUPT_BATCHES, DeferredSIGINT
from jaxns.state import State
from jaxns.types import PRNGKey


def _ensure_thread_schedule(
        state: State,
        depth_cond: DepthCondition,
        *,
        replacement_width: int,
        allocation_target: str,
        root_degree: int,
        delta_K: int,
) -> State:
    """Create the small planning object before entering the large JAX loop.

    Keeping scheduler_data present on every `_run_depth` call gives each sample
    capacity one stable Pytree signature. Otherwise JAX compiles separate large
    executables for new and resumed rounds even though both execute the same
    replacement body after planning.
    """
    if state.scheduler_data is not None:
        return state
    state = _start_schedule_round(
        state,
        depth_cond,
        replacement_width=replacement_width,
        allocation_target=allocation_target,
        root_degree=root_degree,
        delta_K=delta_K,
    )
    return state


@dataclasses.dataclass(slots=True)
class NestedSampler(PureDataclassPytree):
    """Object-oriented configuration and Python goal-loop driver.

    Sample arrays have a static leading dimension during each compiled depth
    call. Finite storage is the default: ``max_samples`` is a hard maximum and
    physical buffers grow only up to it. Set ``unlimited_samples=True`` to opt
    into unbounded geometric growth and its associated memory use and one-time
    recompilation pause for each new shape.

    When ``collect_phantom_samples=True``, every intermediate chain state is
    retained: num_slices - 1 phantoms per classic replacement. The runner owns
    this choice even with an explicit sampler. A shorter prefix can later be
    selected by ``sample_evidence`` without changing the stored observations.

    ``verbose=True`` reports existing scalar counters and timings after each
    goal iteration without constructing results. With checkpointing enabled,
    Ctrl-C saves a continuation after the current bounded batch of compiled
    work, then propagates KeyboardInterrupt. Active chains must finish first.
    """

    model: Model
    root_allocation_degree: int | None = None
    max_samples: int | None = None
    replacement_width: int | None = None
    sampler: AbstractSampler | None = None
    depth_condition: DepthCondition | None = None
    collect_phantom_samples: bool = False
    allocation_target: Literal[
        "uniform",
        "evidence_improving",
        "posterior_improving",
    ] = "uniform"
    delta_K: int | None = None
    initial_capacity: int | None = None
    unlimited_samples: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.depth_condition is None:
            self.depth_condition = default_depth_condition()

    def _resolve_config(
            self,
            model: Model,
            args: tuple,
            params: CtxParams | None,
    ) -> ResolvedRunConfig:
        # Resolve from the active inputs without overwriting requested defaults.
        # The same runner may start runs with different parameter dimensions.
        return resolve_run_config(
            execution="local",
            model=model,
            args=args,
            params=params,
            root_allocation_degree=self.root_allocation_degree,
            replacement_width=self.replacement_width,
            max_samples=self.max_samples,
            sampler=self.sampler,
            collect_phantom_samples=self.collect_phantom_samples,
            allocation_target=self.allocation_target,
            delta_K=self.delta_K,
            initial_capacity=self.initial_capacity,
            unlimited_samples=self.unlimited_samples,
        )

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(
            this,
            [
                "root_allocation_degree",
                "max_samples",
                "replacement_width",
                "collect_phantom_samples",
                "allocation_target",
                "delta_K",
                "initial_capacity",
                "unlimited_samples",
                "verbose",
            ],
        )

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def initialise(
            self,
            key: PRNGKey | None = None,
            *,
            args: tuple = (),
            params: CtxParams | None = None,
    ) -> State:
        """Create a root state that owns this run's model inputs.

        Args:
            key: Root sampling key, with a deterministic default if omitted.
            args: Model arguments stored on the new State.
            params: Model parameters stored on the new State.

        Returns:
            An immutable state containing the root samples and inputs.
        """
        config = self._resolve_config(self.model, args, params)
        return self._initialise(key, config, args=args, params=params)

    def _initialise(
            self,
            key: PRNGKey | None,
            config: ResolvedRunConfig,
            *,
            args: tuple,
            params: CtxParams | None,
    ) -> State:
        """Sample roots after resolving the input-dependent configuration."""
        if key is None:
            key = jax.random.PRNGKey(42)
        init_key, run_key = jax.random.split(key)
        state = _sample_init_state(
            init_key,
            self.model,
            args,
            params,
            root_degree=config.root_allocation_degree,
            sample_capacity=config.initial_capacity,
            num_phantom=config.num_phantom_samples,
        )
        return dataclasses.replace(
            state,
            random_key=run_key,
            goal_key=run_key,
            # Initialisation is a Python goal boundary. Marking it this way
            # makes the first compiled call perform the ordinary per-depth key
            # split, while a capacity resume remains distinguishable.
            depth_reached=jnp.asarray(True, mp_policy.bool_dtype),
        )

    def run(
            self,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
            *,
            args: tuple = (),
            params: CtxParams | None = None,
    ) -> State:
        """Run until the configured expected-depth condition is reached.

        Args:
            args: Model arguments used only when starting a new run.
            params: Model parameters used only when starting a new run.
            key: Random key used only when starting a new run.
            checkpoint_dir: Optional directory for automatic full-state
                checkpointing and resume.
            checkpoint_cadence: Minimum seconds between depth-boundary
                checkpoints. The final state is always saved when changed.

        Returns:
            The completed or terminal immutable state.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        def default_goal(state: State) -> bool:
            if int(state.goal_loop_iter) == 0:
                return False
            return bool(_depth_condition_reached(
                state,
                self.depth_condition,
            ))

        return self.run_until_goal(
            default_goal,
            key=key,
            args=args,
            params=params,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
        )

    def run_until_goal(
            self,
            goal_cond: Callable[[State], bool],
            depth_cond: DepthCondition | None = None,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
            *,
            args: tuple = (),
            params: CtxParams | None = None,
    ) -> State:
        """Run a Python goal loop around compiled JAX depth epochs.

        A valid checkpoint in ``checkpoint_dir`` takes precedence over
        ``key``, ``args``, and ``params`` and resumes its saved model and random
        stream. JAXNS verifies the checkpoint schema and checksum. The caller
        is responsible for compatible sampler and run configuration.

        Args:
            args: Model arguments used only when starting a new run.
            params: Model parameters used only when starting a new run.
            goal_cond: Python goal evaluated at complete depth boundaries.
            depth_cond: Optional condition bounding one allocation epoch.
            key: Random key used only when no checkpoint exists.
            checkpoint_dir: Optional directory for automatic full-state
                checkpointing and resume.
            checkpoint_cadence: Minimum seconds between depth-boundary
                checkpoints. The default is one hour.

        Returns:
            The completed or terminal immutable state.
        """
        return self._execute(
            None,
            goal_cond,
            depth_cond=depth_cond,
            key=key,
            args=args,
            params=params,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
        )

    def resume_until_goal(
            self,
            state: State,
            goal_cond: Callable[[State], bool],
            depth_cond: DepthCondition | None = None,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
    ) -> State:
        """Resume an immutable state under a user-provided Python goal.

        If ``checkpoint_dir`` already contains a valid checkpoint, its state
        takes precedence over the explicit ``state`` and ``key``.

        Args:
            state: Explicit state used when no checkpoint exists.
            goal_cond: Python goal evaluated at complete depth boundaries.
            depth_cond: Optional condition bounding one allocation epoch.
            key: Optional replacement continuation key when no checkpoint
                exists.
            checkpoint_dir: Optional directory for automatic full-state
                checkpointing and resume.
            checkpoint_cadence: Minimum seconds between depth-boundary
                checkpoints. The default is one hour.

        Returns:
            The completed or terminal immutable state.
        """
        return self._execute(
            state,
            goal_cond,
            depth_cond=depth_cond,
            key=key,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
        )

    def _execute(
            self,
            state: State | None,
            goal_cond: Callable[[State], bool] | None,
            *,
            depth_cond: DepthCondition | None,
            key: PRNGKey | None,
            checkpoint_dir: str | Path | None,
            checkpoint_cadence: float,
            args: tuple = (),
            params: CtxParams | None = None,
    ) -> State:
        """Resolve checkpoint and input ownership for every local run entry point."""
        checkpoint_context = (
            CheckpointManager[State](
                checkpoint_dir,
                checkpoint_cadence,
            )
            if checkpoint_dir is not None
            else nullcontext()
        )
        with checkpoint_context as checkpoint_manager:
            if checkpoint_manager is not None:
                restored = checkpoint_manager.load()
                if restored is not None:
                    state = restored
                    key = None
            if state is None:
                config = self._resolve_config(self.model, args, params)
                state = self._initialise(key, config, args=args, params=params)
                key = None
            else:
                config = self._resolve_config(
                    state.model, state.args, state.params,
                )
                saved_phantoms = state.samples.phantom_samples.log_L.shape[1]
                if saved_phantoms != config.num_phantom_samples:
                    raise ValueError(
                        "Saved phantom count differs from the runner's collection policy. "
                        "Resume using compatible collection settings and code."
                    )
            completed = self._run_depth_epochs(
                state,
                goal_cond,
                depth_cond=depth_cond,
                key=key,
                checkpoint_manager=checkpoint_manager,
                config=config,
            )
            if checkpoint_manager is not None:
                checkpoint_manager.save_if_changed(completed)
            return completed

    def _run_depth_epochs(
            self,
            state: State,
            goal_cond: Callable[[State], bool] | None,
            *,
            depth_cond: DepthCondition | None,
            key: PRNGKey | None,
            checkpoint_manager: CheckpointManager[State] | None,
            config: ResolvedRunConfig,
    ) -> State:
        """Advance coherent depths with one shared growth/interrupt lifecycle.

        A missing goal requests one depth boundary and leaves sample growth to
        the caller. Both entry points retain the same schedule and random keys
        at physical interruptions, and commit a finished depth exactly once.
        """
        if depth_cond is None:
            depth_cond = self.depth_condition
        if key is not None:
            state = dataclasses.replace(
                state,
                random_key=key,
                goal_key=key,
                depth_reached=jnp.asarray(True, mp_policy.bool_dtype),
            )
        elif state.random_key is None:
            state = dataclasses.replace(
                state,
                random_key=jax.random.PRNGKey(42),
                goal_key=jax.random.PRNGKey(42),
                depth_reached=jnp.asarray(True, mp_policy.bool_dtype),
            )
        elif state.goal_key is None:
            state = dataclasses.replace(
                state,
                goal_key=state.random_key,
                depth_reached=jnp.asarray(True, mp_policy.bool_dtype),
            )
        if bool(state.depth_reached) and state.scheduler_data is not None:
            state = dataclasses.replace(state, scheduler_data=None)
        started_s = time.monotonic() if self.verbose else 0.0
        iteration_started_s = started_s
        previous_count = int(state.num_samples) if self.verbose else 0
        with DeferredSIGINT(enabled=checkpoint_manager is not None) as interrupt:
            try:
                while int(state.termination_reason) == 0:
                    interrupt.raise_if_requested()
                    # Capacity and interrupt-control returns resume the same
                    # logical depth, without evaluating the user's goal early.
                    if (goal_cond is not None and bool(state.depth_reached)
                            and bool(goal_cond(state))):
                        break
                    batch_start = state.depth_loop_iter
                    state = _ensure_thread_schedule(
                        state,
                        depth_cond,
                        replacement_width=int(config.replacement_width),
                        allocation_target=self.allocation_target,
                        root_degree=int(config.root_allocation_degree),
                        delta_K=int(config.delta_K),
                    )
                    state = _run_depth(
                        state,
                        config.sampler,
                        depth_cond,
                        max_samples=config.max_samples,
                        max_batches=(
                            INTERRUPT_BATCHES if checkpoint_manager is not None else None
                        ),
                    )
                    if not bool(state.depth_reached):
                        interrupt.raise_if_requested()
                    # A drained compiled schedule still needs its Python depth
                    # check and key/counter commit before it is a goal boundary.
                    if bool(state.needs_growth):
                        if goal_cond is None:
                            break
                        capacity = state.samples.log_likelihoods.shape[0]
                        required_capacity = int(state.num_samples) + int(
                            config.replacement_width
                        )
                        new_capacity = max(2 * capacity, required_capacity)
                        if config.max_samples is not None:
                            new_capacity = min(new_capacity, config.max_samples)
                        if new_capacity <= capacity:
                            # This branch is defensive: the compiled classifier should
                            # already report a finite hard maximum as terminal.
                            state = dataclasses.replace(
                                state,
                                termination_reason=jnp.asarray(
                                    MAX_SAMPLES_REACHED,
                                    mp_policy.count_dtype,
                                ),
                                needs_growth=jnp.asarray(
                                    False,
                                    mp_policy.bool_dtype,
                                ),
                                depth_reached=jnp.asarray(
                                    False,
                                    mp_policy.bool_dtype,
                                ),
                            )
                            break
                        # Growth resumes the same allocation target and key. Clearing
                        # only the transient request prevents this implementation
                        # boundary from becoming a logical goal iteration.
                        state = dataclasses.replace(
                            _resize_depth_state(state, new_capacity),
                            needs_growth=jnp.asarray(False, mp_policy.bool_dtype),
                            depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
                        )
                        continue
                    if int(state.termination_reason) != 0:
                        state = _refresh_likelihood_order(state)
                        break
                    if (
                        state.scheduler_data is not None
                        and bool(_continuation_storage_full(
                            state.scheduler_data,
                        ))
                    ):
                        state = _grow_continuation_storage(
                            state,
                            int(config.replacement_width),
                        )
                        continue
                    if (
                        state.scheduler_data is not None
                        and bool(_start_seed_storage_full(
                            state.scheduler_data,
                        ))
                    ):
                        state = _grow_start_seed_storage(
                            state,
                            int(config.replacement_width),
                        )
                        continue
                    source_published = False
                    if (
                        state.scheduler_data is not None
                        and bool(state.scheduler_data.active)
                        and bool(_seed_source_refresh_due(
                            state,
                            state.scheduler_data,
                        ))
                    ):
                        # Promote stationary seeds without changing the frozen target,
                        # maximal thread runs, active heads, or continuation heap.
                        # Every accepted edge already covers intervening new contours;
                        # re-decomposing the refined race would duplicate that work.
                        state = _publish_seed_source(state)
                        source_published = True
                        if bool(state.scheduler_data.active):
                            continue
                        else:
                            state = dataclasses.replace(
                                state,
                                depth_reached=jnp.asarray(
                                    True,
                                    mp_policy.bool_dtype,
                                ),
                            )
                    if bool(state.depth_reached):
                        if not source_published:
                            state = _refresh_likelihood_order(state)
                        reached_expected_depth = bool(_depth_condition_reached(
                            state,
                            depth_cond,
                        ))
                        if not reached_expected_depth:
                            previous = state.scheduler_data
                            state = _continue_schedule_round(
                                state,
                                previous,
                                depth_cond,
                                replacement_width=int(config.replacement_width),
                            )
                            schedule = state.scheduler_data
                            if schedule is None:
                                raise RuntimeError(
                                    "Continuation planning did not create a schedule."
                                )
                            if bool(schedule.active):
                                state = dataclasses.replace(
                                    state,
                                    depth_reached=jnp.asarray(
                                        False,
                                        mp_policy.bool_dtype,
                                    ),
                                )
                                continue
                            # The frozen target is now full but expected depth is not
                            # reached. Advance allocation internally without exposing
                            # this intermediate state to the user-provided goal.
                            state = dataclasses.replace(
                                state,
                                allocation_loop_iter=(
                                    state.allocation_loop_iter
                                    + jnp.asarray(
                                        1,
                                        state.allocation_loop_iter.dtype,
                                    )
                                ),
                                depth_reached=jnp.asarray(
                                    False,
                                    mp_policy.bool_dtype,
                                ),
                                scheduler_data=None,
                            )
                            continue
                        state = dataclasses.replace(
                            state,
                            random_key=state.goal_key,
                            goal_loop_iter=(
                                state.goal_loop_iter
                                + jnp.asarray(goal_cond is not None, state.goal_loop_iter.dtype)
                            ),
                            allocation_loop_iter=(
                                state.allocation_loop_iter
                                + jnp.asarray(1, state.allocation_loop_iter.dtype)
                            ),
                            # Completed thread schedules are implementation-only
                            # continuation state. Clear them before the next outer
                            # iteration constructs a fresh planning domain.
                            scheduler_data=None,
                        )
                        if self.verbose and goal_cond is not None:
                            now = time.monotonic()
                            count = int(state.num_samples)
                            jaxns_logger.info(
                                "Local goal %d: samples=%d (+%d), capacity=%d, "
                                "max(logL)=%.6g, iteration=%.2fs, elapsed=%.2fs",
                                int(state.goal_loop_iter), count, count - previous_count,
                                state.samples.log_likelihoods.shape[0],
                                float(state.log_L_supremum), now - iteration_started_s,
                                now - started_s,
                            )
                            previous_count = count
                            iteration_started_s = now
                        # Checkpoint only after the logical depth is complete. A
                        # capacity return is a physical interruption of the same
                        # epoch and must not become a persisted goal boundary.
                        if checkpoint_manager is not None:
                            checkpoint_manager.maybe_save(state)
                        if goal_cond is None:
                            break
                        continue
                    if (
                        checkpoint_manager is not None
                        and int(state.depth_loop_iter - batch_start) >= INTERRUPT_BATCHES
                    ):
                        # A control return preserves the active schedule and keys. It
                        # must not advance allocation or evaluate the scientific goal.
                        continue
                    raise RuntimeError(
                        "Compiled planning round returned without termination, "
                        "growth, or a drained schedule."
                    )
                interrupt.raise_if_requested()
            except KeyboardInterrupt:
                if checkpoint_manager is not None:
                    # The immutable state includes an unfinished schedule and
                    # its exact keys when Ctrl-C interrupts a logical depth.
                    checkpoint_manager.save_if_changed(state)
                raise

        return state

    def run_single_iteration(
            self,
            state: State | None = None,
            depth_cond: DepthCondition | None = None,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
            *,
            args: tuple = (),
            params: CtxParams | None = None,
    ) -> State:
        """Run exactly one compiled depth epoch.

        When checkpointing is enabled, an existing committed state takes
        precedence over ``state`` and the returned state is persisted. The
        cadence does not defer that final save because this method has only
        one Python depth boundary.

        Args:
            args: Model arguments used only when starting a new run.
            params: Model parameters used only when starting a new run.
            state: Optional explicit continuation state.
            depth_cond: Optional condition bounding the allocation epoch.
            key: Random key used only when starting or explicitly overriding
                a state without a checkpoint.
            checkpoint_dir: Optional directory for automatic full-state
                checkpointing and resume.
            checkpoint_cadence: Checkpoint cadence in seconds, retained for a
                consistent run API.

        Returns:
            The immutable state returned by one compiled depth epoch.
        """
        return self._execute(
            state,
            None,
            depth_cond=depth_cond,
            key=key,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
            args=args,
            params=params,
        )


NestedSampler.register_pytree()
