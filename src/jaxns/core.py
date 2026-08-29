"""Depth-first nested sampling core described by the paper."""

import dataclasses
import operator
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxctx import CtxParams

from jaxns.algorithm.depth import MAX_SAMPLES_REACHED, _run_depth
from jaxns.algorithm.initialisation import _sample_init_state
from jaxns.checkpoint import (
    CHECKPOINT_CADENCE_SECONDS,
    CheckpointManager,
)
from jaxns.constrained_sampler import (
    AbstractSampler,
    UniDimSliceSampler,
)
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.state import State
from jaxns.termination_condition import (
    TerminationCondition,
)
from jaxns.types import PRNGKey

# The default finite ceiling permits substantial runs without silently opting
# into unlimited memory. Physical storage starts smaller and grows on demand;
# keeping both policies singular lets benchmark evidence revise either one.
SAMPLES_PER_ROOT = 1000
INITIAL_BATCHES = 64


@dataclasses.dataclass(slots=True)
class NestedSampler(PureDataclassPytree):
    """Object-oriented configuration and Python goal-loop driver.

    Sample arrays have a static leading dimension during each compiled depth
    call. Finite storage is the default: ``max_samples`` is a hard maximum and
    physical buffers grow only up to it. Set ``unlimited_samples=True`` to opt
    into unbounded geometric growth and its associated memory use and one-time
    recompilation pause for each new shape.

    When ``collect_phantom_samples=True``, ``max_phantom_samples`` bounds the
    leading stationary chain prefix stored per classic replacement. ``None``
    resolves to ``min(model dimension, num_slices - 1)`` for the default slice
    sampler. The retained width is independent of the shorter prefix that can
    later be selected by ``sample_evidence_mc``.
    """

    model: Model
    target_num_live_points: int | None = None
    root_allocation_degree: int | None = None
    max_samples: int | None = None
    shell_size: int | None = None
    batch_size: int | None = None
    args: tuple = ()
    params: CtxParams | None = None
    sampler: AbstractSampler | None = None
    termination_condition: TerminationCondition | None = None
    # Constructor-compatible legacy option. The evidence model stores phantom
    # likelihoods only; high-dimensional phantom coordinates are not retained.
    store_phantom_samples: bool = False
    collect_phantom_samples: bool = False
    max_phantom_samples: int | None = None
    allocation_target: Literal[
        "uniform",
        "evidence_improving",
        "posterior_improving",
    ] = "uniform"
    delta_K: int | None = None
    initial_capacity: int | None = None
    unlimited_samples: bool = False

    def __post_init__(self):
        periodic = self.model._periodic_coordinates(self.args, self.params)
        # The aligned metadata already carries one flag per scalar base-space
        # coordinate, so it also supplies dimension without a second model
        # init trace. Collapse all-false flags before sampler construction to
        # preserve the exact pre-periodic key schedule and compiled hot path.
        U_ndims = len(periodic)
        if not any(periodic):
            periodic = ()
        root_degree = self.root_allocation_degree
        if root_degree is None:
            root_degree = self.target_num_live_points
        if root_degree is None:
            # Match v2's robust default number of independent Markov chains.
            # Merely recording phantoms must not change the sampled race tree.
            root_degree = max(1, 30 * U_ndims)
        if (
            self.root_allocation_degree is not None
            and self.target_num_live_points is not None
            and self.root_allocation_degree != self.target_num_live_points
        ):
            raise ValueError(
                "root_allocation_degree and target_num_live_points disagree."
            )
        shell_size = self.shell_size
        if shell_size is None:
            # A wider vmap increases exposure to the slowest data-dependent
            # rejection loop in the batch. Ten chains per dimension retains
            # useful CPU batching without the long-tail cost observed at the
            # former half-root width on multimodal problems.
            shell_size = min(root_degree, max(1, 10 * U_ndims))
        max_samples = self.max_samples
        if self.unlimited_samples and max_samples is not None:
            raise ValueError(
                "unlimited_samples=True conflicts with a finite max_samples."
            )
        if not self.unlimited_samples:
            if max_samples is None:
                max_samples = max(
                    root_degree + shell_size,
                    SAMPLES_PER_ROOT * root_degree,
                )
            max_samples = int(max_samples)
            if max_samples < root_degree:
                raise ValueError("max_samples must hold all root samples.")
        delta_K = self.delta_K
        if delta_K is None:
            # One outer allocation step should normally create one full
            # replacement batch. A unit increment leaves S-1 vmapped sampler
            # lanes idle and adds a Python/device round trip per sample.
            delta_K = shell_size
        if shell_size <= 0 or delta_K <= 0:
            raise ValueError("shell_size and delta_K must be positive.")

        sampler = self.sampler
        if sampler is None:
            num_slices = max(1, 5 * U_ndims)
            sampler = UniDimSliceSampler(
                model=self.model,
                num_slices=num_slices,
                no_step_out=True,
                gradient_guided=False,
                collect_phantom_samples=self.collect_phantom_samples,
            )
        max_phantom_samples = self.max_phantom_samples
        if max_phantom_samples is not None:
            try:
                max_phantom_samples = operator.index(max_phantom_samples)
            except TypeError as error:
                raise TypeError(
                    "max_phantom_samples must be an integer or None."
                ) from error
        sampler = sampler._with_phantom_capacity(
            max_phantom_samples,
            U_ndims,
        )
        sampler = sampler._with_periodic(periodic)
        sampler.validate_core(U_ndims)

        termination_condition = self.termination_condition
        if termination_condition is None:
            termination_condition = TerminationCondition(
                # Match the released v2 scientific stopping goal exactly so
                # accuracy/performance comparisons cannot benefit from an
                # earlier termination threshold.
                dlogZ=jnp.log1p(
                    jnp.asarray(1e-3, mp_policy.measure_dtype)
                ),
                max_samples=(
                    None
                    if max_samples is None
                    else jnp.asarray(max_samples, mp_policy.count_dtype)
                ),
            )
        elif (
            max_samples is not None
            and termination_condition.max_samples is None
        ):
            termination_condition = dataclasses.replace(
                termination_condition,
                max_samples=jnp.asarray(max_samples, mp_policy.count_dtype),
            )
        initial_capacity = self.initial_capacity
        if initial_capacity is None:
            # Preallocating the full default maximum makes every fixed-shape
            # block scan pay for unused padding. Start with enough room for a
            # useful number of replacement batches, then grow geometrically.
            initial_capacity = root_degree + INITIAL_BATCHES * shell_size
        initial_capacity = int(initial_capacity)
        if initial_capacity < root_degree:
            raise ValueError("initial_capacity must hold all root samples.")
        if max_samples is not None:
            initial_capacity = min(initial_capacity, max_samples)

        self.target_num_live_points = root_degree
        self.root_allocation_degree = root_degree
        self.shell_size = int(shell_size)
        self.max_samples = max_samples
        self.sampler = sampler
        # Publish the resolved static width, including custom samplers, so a
        # caller can distinguish retained capacity from later MC prefix use.
        self.max_phantom_samples = int(sampler.num_phantom())
        self.termination_condition = termination_condition
        self.initial_capacity = initial_capacity
        self.delta_K = int(delta_K)

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(
            this,
            [
                "target_num_live_points",
                "root_allocation_degree",
                "max_samples",
                "shell_size",
                "batch_size",
                "store_phantom_samples",
                "collect_phantom_samples",
                "max_phantom_samples",
                "allocation_target",
                "delta_K",
                "initial_capacity",
                "unlimited_samples",
            ],
        )

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def initialise(self, key: PRNGKey | None = None) -> State:
        """Create a resumable immutable root state."""
        if key is None:
            key = jax.random.PRNGKey(42)
        init_key, run_key = jax.random.split(key)
        state = _sample_init_state(
            init_key,
            self.model,
            self.args,
            self.params,
            root_degree=int(self.root_allocation_degree),
            sample_capacity=int(self.initial_capacity),
            num_phantom=int(self.sampler.num_phantom()),
        )
        sampler_data = self.sampler.initial_sampler_data(
            int(self.model.U_ndims(self.args, self.params))
        )
        return dataclasses.replace(
            state,
            sampler_data=sampler_data,
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
    ) -> State:
        """Run until the default expectation-based goal is satisfied.

        Args:
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
            done, _ = state.compute_termination_register().is_done(
                self.termination_condition
            )
            return bool(done)

        return self.run_until_goal(
            default_goal,
            key=key,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
        )

    def run_until_goal(
            self,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
    ) -> State:
        """Run a Python goal loop around compiled JAX depth epochs.

        A valid checkpoint in ``checkpoint_dir`` takes precedence over
        ``key`` and resumes its stored random stream. JAXNS verifies the
        checkpoint schema and checksum; the caller is responsible for using a
        compatible model, sampler, arguments, and run configuration.

        Args:
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
        return self._resume_until_goal(
            None,
            goal_cond,
            depth_cond=depth_cond,
            key=key,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
        )

    def resume_until_goal(
            self,
            state: State,
            goal_cond: Callable[[State], bool],
            depth_cond: TerminationCondition | None = None,
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
        return self._resume_until_goal(
            state,
            goal_cond,
            depth_cond=depth_cond,
            key=key,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=checkpoint_cadence,
        )

    def _resume_until_goal(
            self,
            state: State | None,
            goal_cond: Callable[[State], bool],
            *,
            depth_cond: TerminationCondition | None,
            key: PRNGKey | None,
            checkpoint_dir: str | Path | None,
            checkpoint_cadence: float,
    ) -> State:
        """Resolve checkpoint precedence, then continue one goal loop."""
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
                state = self.initialise(key)
                key = None
            completed = self._run_goal_loop(
                state,
                goal_cond,
                depth_cond=depth_cond,
                key=key,
                checkpoint_manager=checkpoint_manager,
            )
            if checkpoint_manager is not None:
                checkpoint_manager.save_if_changed(completed)
            return completed

    def _run_goal_loop(
            self,
            state: State,
            goal_cond: Callable[[State], bool],
            *,
            depth_cond: TerminationCondition | None,
            key: PRNGKey | None,
            checkpoint_manager: CheckpointManager[State] | None,
    ) -> State:
        """Continue compiled depths after checkpoint ownership is resolved."""
        if depth_cond is None:
            depth_cond = self.termination_condition
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
        while (
            int(state.termination_reason) == 0
            # A resized sample buffer resumes the same compiled depth epoch.
            # Only let the Python goal observe completed allocation rounds so
            # physical capacity cannot change the scientific stopping point.
            and (
                not bool(state.depth_reached)
                or not bool(goal_cond(state))
            )
        ):
            state = _run_depth(
                state,
                self.sampler,
                depth_cond,
                shell_size=int(self.shell_size),
                allocation_target=self.allocation_target,
                root_degree=int(self.root_allocation_degree),
                delta_K=int(self.delta_K),
                max_samples=self.max_samples,
            )
            if bool(state.needs_growth):
                capacity = state.samples.log_likelihoods.shape[0]
                required_capacity = int(state.num_samples) + int(
                    self.shell_size
                )
                new_capacity = max(2 * capacity, required_capacity)
                if self.max_samples is not None:
                    new_capacity = min(new_capacity, self.max_samples)
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
                    state.resize(new_capacity),
                    needs_growth=jnp.asarray(False, mp_policy.bool_dtype),
                    depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
                )
                continue
            if int(state.termination_reason) != 0:
                break
            if bool(state.depth_reached):
                state = dataclasses.replace(
                    state,
                    goal_loop_iter=(
                        state.goal_loop_iter
                        + jnp.asarray(1, state.goal_loop_iter.dtype)
                    ),
                )
                # Checkpoint only after the logical depth is complete. A
                # capacity return is a physical interruption of the same
                # epoch and must not become a persisted goal boundary.
                if checkpoint_manager is not None:
                    checkpoint_manager.maybe_save(state)
                continue
            raise RuntimeError(
                "Compiled depth returned without termination, growth, or "
                "normal depth completion."
            )

        if int(state.termination_reason) == 0:
            done, reason = state.compute_termination_register().is_done(
                self.termination_condition
            )
            if bool(done):
                state = dataclasses.replace(
                    state,
                    termination_reason=reason,
                    needs_growth=jnp.asarray(False, mp_policy.bool_dtype),
                    depth_reached=jnp.asarray(False, mp_policy.bool_dtype),
                )
        return state

    def run_single_iteration(
            self,
            state: State | None = None,
            depth_cond: TerminationCondition | None = None,
            key: PRNGKey | None = None,
            checkpoint_dir: str | Path | None = None,
            checkpoint_cadence: float = CHECKPOINT_CADENCE_SECONDS,
    ) -> State:
        """Run exactly one compiled depth epoch.

        When checkpointing is enabled, an existing committed state takes
        precedence over ``state`` and the returned state is persisted. The
        cadence does not defer that final save because this method has only
        one Python depth boundary.

        Args:
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
        if checkpoint_dir is not None:
            with CheckpointManager[State](
                checkpoint_dir,
                checkpoint_cadence,
            ) as checkpoint_manager:
                restored = checkpoint_manager.load()
                if restored is not None:
                    state = restored
                    key = None
                state = self._run_single_iteration(
                    state=state,
                    depth_cond=depth_cond,
                    key=key,
                )
                checkpoint_manager.save_if_changed(state)
                return state
        return self._run_single_iteration(
            state=state,
            depth_cond=depth_cond,
            key=key,
        )

    def _run_single_iteration(
            self,
            state: State | None,
            depth_cond: TerminationCondition | None,
            key: PRNGKey | None,
    ) -> State:
        """Execute one depth epoch after checkpoint ownership is resolved."""
        if state is None:
            state = self.initialise(key)
        elif key is not None:
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
        if depth_cond is None:
            depth_cond = self.termination_condition
        return _run_depth(
            state,
            self.sampler,
            depth_cond,
            shell_size=int(self.shell_size),
            allocation_target=self.allocation_target,
            root_degree=int(self.root_allocation_degree),
            delta_K=int(self.delta_K),
            max_samples=self.max_samples,
        )


NestedSampler.register_pytree()
