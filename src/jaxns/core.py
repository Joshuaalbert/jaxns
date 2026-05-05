import dataclasses
from functools import partial
from pathlib import Path
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.random
import numpy as np
from jaxctx import CtxParams

from jaxns.constrained_sampler import AbstractSampler, UniDimSliceSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample_indicies
from jaxns.samples import Samples, SeedPoint, PhantomSamples
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition
from jaxns.types import IntArray, BoolArray, PRNGKey


class StepDelta(NamedTuple):
    """Checkpoint delta emitted by one outer nested-sampling step."""

    samples: Samples
    parent_sample_ids: IntArray


@partial(jax.jit, inline=True, static_argnames=['num_live_points', 'num_phantom', 'max_samples', 'store_phantom_samples', 'batch_size'])
def _sample_init_state(key, num_live_points: int, max_samples: int, model: Model, num_phantom: int = 0, args=(),
                       params: CtxParams | None = None, store_phantom_samples: bool = False, batch_size: int | None = None) -> State:
    """Draw the initial live points and allocate the padded sample buffers."""

    def single_sample(key):
        key, subkey = jax.random.split(key)
        U_sample = model.sample_U(subkey)
        log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False).astype(mp_policy.measure_dtype)
        num_likelihood_evaluations = jnp.array(1, dtype=mp_policy.count_dtype)
        carry = (key, U_sample, log_L, num_likelihood_evaluations)

        def cond_fn(carry):
            _, _, log_L, _ = carry
            return log_L <= -jnp.inf

        def body_fn(carry):
            key, _, _, num_likelihood_evaluations = carry
            key, subkey = jax.random.split(key)
            U_sample = model.sample_U(subkey)
            log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False).astype(mp_policy.measure_dtype)
            num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
            return (key, U_sample, log_L, num_likelihood_evaluations)

        key, U_sample, log_L, num_likelihood_evaluations = jax.lax.while_loop(cond_fn, body_fn, carry)
        return U_sample, log_L, num_likelihood_evaluations

    U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
        single_sample,
        jax.random.split(key, num_live_points),
        batch_size=batch_size
    )
    out_degree = jnp.zeros((num_live_points,), dtype=mp_policy.count_dtype)

    # extend each to max_samples
    def _concat(x, fill_value):
        return jax.tree.map(
            lambda x, y: jnp.concatenate([
                x,
                jnp.repeat(y[None, ...], repeats=(max_samples - num_live_points), axis=0)
            ], axis=0),
            x,
            fill_value
        )

    phantom_samples = PhantomSamples(
        valid_mask=jnp.full((num_live_points, num_phantom), False, dtype=mp_policy.bool_dtype),
        U_samples=jax.tree.map(lambda u: jnp.zeros((num_live_points, num_phantom) + u[0].shape, u.dtype), U_samples),
        log_L=jnp.full((num_live_points, num_phantom), -jnp.inf, dtype=mp_policy.measure_dtype)
    )
    if not store_phantom_samples:
        phantom_samples.U_samples = None
    samples = Samples(
        log_L_constraints=jnp.full((num_live_points,), -jnp.inf, dtype=mp_policy.measure_dtype),
        log_likelihoods=log_likelihoods,
        sample_ids=jnp.arange(num_live_points, dtype=mp_policy.count_dtype),
        U_samples=U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations,
        phantom_samples=phantom_samples
    )

    sample_atom = Samples(
        log_likelihoods=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
        sample_ids=jnp.asarray(0, mp_policy.count_dtype),
        out_degree=jnp.asarray(0, mp_policy.count_dtype),
        num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype),
        U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), samples.U_samples),
        phantom_samples=PhantomSamples(
            U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), samples.phantom_samples.U_samples),
            log_L=jnp.zeros_like(samples.phantom_samples.log_L[0]),
            valid_mask=jnp.zeros_like(samples.phantom_samples.valid_mask[0])
        ),
        log_L_constraints=jnp.asarray(jnp.inf, mp_policy.measure_dtype)
    )
    if not store_phantom_samples:
        sample_atom.phantom_samples.U_samples = None

    samples = _concat(samples, sample_atom).sort()
    samples.sample_ids = jnp.arange(max_samples, dtype=mp_policy.count_dtype)

    log_L_supremum_idx = jnp.argmax(log_likelihoods)
    log_L_supremum = log_likelihoods[log_L_supremum_idx]
    U_supremum = jax.tree.map(lambda u: u[log_L_supremum_idx], U_samples)
    # Sort samples into increasing log-likelihood order
    state = State(
        root_out_degree=jnp.array(num_live_points, dtype=mp_policy.count_dtype),
        samples=samples,
        num_samples=jnp.array(num_live_points, dtype=mp_policy.count_dtype),
        log_L_supremum=log_L_supremum,
        U_supremum=U_supremum,
        model=model,
        args=args,
        params=params,
        termination_reason=jnp.array(0, dtype=mp_policy.index_dtype)
    )
    return state


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points'])
def _check_termination(state: State, target_num_live_points: int,
                       termination_condition: TerminationCondition | None = None) -> tuple[BoolArray, IntArray]:
    """Evaluate whether the current state satisfies the termination condition."""

    register = state.compute_termination_register(target_num_live_points=target_num_live_points)
    return register.is_done(termination_condition)


@partial(jax.jit, inline=True)
def _set_termination_reason(state: State, termination_reason: IntArray) -> State:
    """Return a copy of ``state`` with an updated termination reason."""

    return State(
        root_out_degree=state.root_out_degree,
        samples=state.samples,
        num_samples=state.num_samples,
        log_L_supremum=state.log_L_supremum,
        U_supremum=state.U_supremum,
        model=state.model,
        args=state.args,
        params=state.params,
        termination_reason=termination_reason,
    )


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points', 'shell_size', 'batch_size'])
def _run_ns_step(key, state: State, target_num_live_points: int, shell_size: int, args=(),
                 sampler: AbstractSampler | None = None,
                 params=None,
                 batch_size: int | None = None) -> tuple[jax.Array, State, StepDelta]:
    """Execute one replenishment step and emit the corresponding checkpoint delta."""

    K_per_sample = state.samples.compute_num_live_points_per_sample(
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
    )
    K_next_sample = K_per_sample - 1 + state.samples.out_degree
    select_weights = jnp.where(
        jnp.logical_and(
            jnp.arange(K_next_sample.shape[0]) < state.num_samples,
            K_next_sample < target_num_live_points,
        ),
        0,
        -jnp.inf,
    )
    select_contours_key, key = jax.random.split(key, 2)
    parent_idxs = resample_indicies(select_contours_key, log_weights=select_weights, S=shell_size, replace=False)
    proposed_log_L_constraints = state.samples.log_likelihoods[parent_idxs]

    def get_sample(sample_key, log_L_constraint, parent_idx: IntArray):
        seed_key, constrained_sample_key = jax.random.split(sample_key)
        i_start = jax.lax.while_loop(
            lambda i: (i < state.num_samples) & (state.samples.log_likelihoods[i] <= log_L_constraint),
            lambda i: i + 1,
            parent_idx + 1,
        )
        no_seeds = i_start == state.num_samples
        log_L_constraint = jnp.where(no_seeds, jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype), log_L_constraint)
        delta_root_out_degree = jnp.where(no_seeds, 1, 0).astype(mp_policy.count_dtype)
        delta_parent_out_degree = jnp.where(no_seeds, 0, 1).astype(mp_policy.count_dtype)
        parent_sample_id = jnp.where(
            no_seeds,
            -jnp.ones((), dtype=state.samples.sample_ids.dtype),
            state.samples.sample_ids[parent_idx],
        )
        i_start = jnp.where(no_seeds, 0, i_start).astype(mp_policy.index_dtype)
        seed_select_idx = jax.random.randint(seed_key, (), i_start, state.num_samples)
        seed_point = SeedPoint(
            U0=jax.tree.map(lambda u: u[seed_select_idx], state.samples.U_samples),
            log_L0=state.samples.log_likelihoods[seed_select_idx],
        )
        return sampler.get_sample(
            constrained_sample_key,
            log_L_constraint,
            seed_point,
            args=args,
            params=params,
        ), (delta_root_out_degree, delta_parent_out_degree, log_L_constraint, parent_sample_id)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, shell_size)
    (U_samples, log_likelihoods, num_likelihood_evaluations, phantom_samples), (
        delta_root_out_degree,
        delta_parent_out_degree,
        log_L_constraints,
        parent_sample_ids,
    ) = jax.lax.map(
        lambda x: get_sample(x[0], x[1], x[2]),
        (keys, proposed_log_L_constraints, parent_idxs),
        batch_size=batch_size,
    )

    new_samples = Samples(
        log_L_constraints=log_L_constraints,
        log_likelihoods=log_likelihoods,
        sample_ids=state.num_samples + jnp.arange(shell_size, dtype=mp_policy.count_dtype),
        U_samples=U_samples,
        out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
        num_likelihood_evaluations=num_likelihood_evaluations,
        phantom_samples=phantom_samples,
    )
    if state.samples.phantom_samples.U_samples is None:
        new_samples.phantom_samples.U_samples = None

    candidate_supremum_candidate_iid = jnp.argmax(new_samples.log_likelihoods)
    candidate_log_L_supremum = new_samples.log_likelihoods[candidate_supremum_candidate_iid]
    candidate_U_supremum = jax.tree.map(lambda u: u[candidate_supremum_candidate_iid], new_samples.U_samples)

    log_L_supremum = jnp.where(
        candidate_log_L_supremum > state.log_L_supremum,
        candidate_log_L_supremum,
        state.log_L_supremum,
    )
    U_supremum = jax.tree.map(
        lambda u_new, u_old: jnp.where(candidate_log_L_supremum > state.log_L_supremum, u_new, u_old),
        candidate_U_supremum,
        state.U_supremum,
    )
    if new_samples.phantom_samples.U_samples is not None:
        valid_mask = new_samples.phantom_samples.valid_mask.reshape((-1,))
        phantom_log_L = new_samples.phantom_samples.log_L.reshape((-1,))
        phantom_U_samples = jax.tree.map(lambda u: u.reshape((-1,) + u.shape[2:]), new_samples.phantom_samples.U_samples)
        phantom_log_L = jnp.where(valid_mask, phantom_log_L, -jnp.inf)
        candidate_supremum_candidate_phantom = jnp.argmax(phantom_log_L)
        candidate_log_L_supremum_phantom = phantom_log_L[candidate_supremum_candidate_phantom]
        candidate_U_supremum_phantom = jax.tree.map(lambda u: u[candidate_supremum_candidate_phantom], phantom_U_samples)
        log_L_supremum = jnp.where(
            candidate_log_L_supremum_phantom > log_L_supremum,
            candidate_log_L_supremum_phantom,
            log_L_supremum,
        )
        U_supremum = jax.tree.map(
            lambda u_phantom, u_current: jnp.where(candidate_log_L_supremum_phantom > log_L_supremum, u_phantom, u_current),
            candidate_U_supremum_phantom,
            U_supremum,
        )

    next_state = State(
        root_out_degree=state.root_out_degree + jnp.sum(delta_root_out_degree),
        samples=state.samples.append_samples(
            insert_idx=state.num_samples,
            parent_idxs=parent_idxs,
            samples=new_samples,
            delta_parent_out_degree=delta_parent_out_degree,
        ).sort(),
        num_samples=state.num_samples + len(new_samples),
        log_L_supremum=log_L_supremum,
        U_supremum=U_supremum,
        model=state.model,
        args=state.args,
        params=state.params,
        termination_reason=state.termination_reason,
    )

    return key, next_state, StepDelta(samples=new_samples, parent_sample_ids=parent_sample_ids)


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points', 'shell_size', 'batch_size'])
def _run_ns(key, state: State, target_num_live_points: int, shell_size: int, args=(),
            sampler: AbstractSampler | None = None,
            params=None,
            termination_condition: TerminationCondition | None = None,
            batch_size: int | None = None) -> State:
    """
    Perform a single nested sampling run.

    Args:
        key: PRNG key.
        state: State to advance until the termination condition is met.
        target_num_live_points: The number of live points to maintain off root.
        shell_size: The number of samples to discard and replenish per iteration.
        args: Arguments to pass to the model.
        sampler: Sampler used to produce i.i.d. samples within likelihood constraints.
        params: Parameters to pass to the model.
        termination_condition: Termination condition to use.
        batch_size: How many likelihood evaluations to batch together.

    Returns:
        A final state object containing all samples, with the final termination reason
        recorded for evidence calculation or later in-memory resume.
    """

    class OuterCarry(NamedTuple):
        key: jax.Array
        state: State

    def outer_cond_fn(carry: OuterCarry) -> BoolArray:
        done, _ = _check_termination(carry.state, target_num_live_points, termination_condition)
        return jnp.logical_not(done)

    def outer_body_fn(carry: OuterCarry) -> OuterCarry:
        key, state, _ = _run_ns_step(
            key=carry.key,
            state=carry.state,
            target_num_live_points=target_num_live_points,
            shell_size=shell_size,
            args=args,
            sampler=sampler,
            params=params,
            batch_size=batch_size,
        )
        return OuterCarry(key=key, state=state)

    carry = jax.lax.while_loop(outer_cond_fn, outer_body_fn, OuterCarry(key=key, state=state))
    _, termination_reason = _check_termination(carry.state, target_num_live_points, termination_condition)
    return _set_termination_reason(carry.state, termination_reason)


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points', 'shell_size', 'checkpoint_every', 'batch_size'])
def _run_ns_chunk(key, state: State, target_num_live_points: int, shell_size: int, checkpoint_every: int, args=(),
                  sampler: AbstractSampler | None = None,
                  params=None,
                  termination_condition: TerminationCondition | None = None,
                  batch_size: int | None = None):
    """Run up to one checkpoint chunk of nested-sampling steps."""

    class ChunkCarry(NamedTuple):
        key: jax.Array
        state: State
        steps_executed: IntArray

    empty_delta = StepDelta(
        samples=state.samples.slice(0, shell_size),
        parent_sample_ids=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
    )

    def scan_fn(carry: ChunkCarry, _):
        done, _ = _check_termination(carry.state, target_num_live_points, termination_condition)

        def skip_step(_):
            return carry, empty_delta

        def run_step(_):
            key, next_state, step_delta = _run_ns_step(
                key=carry.key,
                state=carry.state,
                target_num_live_points=target_num_live_points,
                shell_size=shell_size,
                args=args,
                sampler=sampler,
                params=params,
                batch_size=batch_size,
            )
            return ChunkCarry(
                key=key,
                state=next_state,
                steps_executed=carry.steps_executed + jnp.ones((), dtype=mp_policy.count_dtype),
            ), step_delta

        return jax.lax.cond(done, skip_step, run_step, operand=None)

    carry, step_deltas = jax.lax.scan(
        scan_fn,
        ChunkCarry(key=key, state=state, steps_executed=jnp.zeros((), dtype=mp_policy.count_dtype)),
        xs=None,
        length=checkpoint_every,
    )
    done, termination_reason = _check_termination(carry.state, target_num_live_points, termination_condition)
    final_state = jax.lax.cond(
        done,
        lambda current_state: _set_termination_reason(current_state, termination_reason),
        lambda current_state: current_state,
        carry.state,
    )
    return carry.key, final_state, step_deltas, carry.steps_executed, done, termination_reason


@dataclasses.dataclass(slots=True)
class NestedSampler(PureDataclassPytree):
    model: Model
    target_num_live_points: int | None = None
    max_samples: int | None = None
    shell_size: int | None = None
    args: tuple = ()
    params: CtxParams | None = None
    sampler: AbstractSampler | None = None
    termination_condition: TerminationCondition | None = None
    store_phantom_samples: bool = False
    collect_phantom_samples: bool = False
    batch_size: int | None = None

    def __post_init__(self):
        U_ndims = 0
        if self.target_num_live_points is None or self.max_samples is None or self.shell_size is None or self.sampler is None:
            U_ndims = int(self.model.U_ndims(self.args, self.params))
        if self.target_num_live_points is None:
            self.target_num_live_points = 20 * U_ndims
        if self.max_samples is None:
            self.max_samples = 10000 * U_ndims
        if self.shell_size is None:
            self.shell_size = max(1, self.target_num_live_points // 2)
        max_samples = jnp.asarray(self.max_samples, dtype=mp_policy.count_dtype)
        if self.termination_condition is None:
            self.termination_condition = TerminationCondition(dlogZ=1e-2, max_samples=max_samples)
        elif self.termination_condition.max_samples is None:
            self.termination_condition.max_samples = max_samples
        else:
            self.termination_condition.max_samples = jnp.minimum(self.termination_condition.max_samples, max_samples)
        if self.sampler is None:
            self.sampler = UniDimSliceSampler(model=self.model, num_slices=max(1, 100 * U_ndims), phantom_burn_in=max(1, 20 * U_ndims), no_step_out=True,
                                              collect_phantom_samples=self.collect_phantom_samples)

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['target_num_live_points', 'max_samples', 'shell_size', 'store_phantom_samples', 'batch_size'])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def run(self, key: PRNGKey | None = None, *, resume: bool = False,
            archive_path: str | Path | None = None,
            checkpoint_every: int | None = None) -> State:
        """
        Create or load a state and perform sampling until the termination condition is met.

        Args:
            key: PRNGKey to use for sampling. When ``archive_path`` is omitted and ``key``
                is ``None``, a default key of ``jax.random.PRNGKey(42)`` is used. When
                ``resume=True``, the committed key stored in the archive is used instead.
            resume: If true, resume from an existing checkpoint archive after validating
                that the current sampler configuration matches the archived metadata.
            archive_path: Optional HDF5 archive used for checkpointing and resume. When
                provided with ``resume=False``, any existing archive at this path is
                overwritten with a fresh run.
            checkpoint_every: Number of outer nested-sampling steps to execute per
                checkpoint commit. If omitted for a checkpointed run, a built-in
                heuristic chooses a conservative chunk size automatically.

        Returns:
            The final state after running nested sampling, which can be used for
            evidence calculation or later in-memory resume.

        Raises:
            ValueError: If ``resume`` or ``checkpoint_every`` is provided without ``archive_path``.
            FileNotFoundError: If ``resume=True`` and the requested archive does not exist.
            CheckpointValidationError: If ``resume=True`` and the archive metadata is
                incompatible with the current sampler configuration.
        """
        if archive_path is None:
            if resume:
                raise ValueError('resume=True requires archive_path to be set.')
            if checkpoint_every is not None:
                raise ValueError('checkpoint_every requires archive_path to be set.')
            if key is None:
                key = jax.random.PRNGKey(42)
            return _run(self, key)
        return _run_with_checkpointing(self, key, archive_path, resume=resume, checkpoint_every=checkpoint_every)

    def resume(self, state: State, key: PRNGKey | None = None) -> State:
        """
        Performs sampling until the termination condition is met, starting from the provided state. Can be used to resume a nested sampling run.

        Args:
            state: the state to resume from, which should be a valid state returned by a previous call to run or resume. The state should not have met the termination condition yet.
            key: the PRNGKey to use for sampling

        Returns:
            the final state after running nested sampling, which can be used for evidence calculation or resuming.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        return _run_ns(
            key=key,
            state=state,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            args=self.args,
            sampler=self.sampler,
            params=self.params,
            termination_condition=self.termination_condition,
            batch_size=self.batch_size
        )


NestedSampler.register_pytree()


@partial(jax.jit, inline=True)
def _run(self: NestedSampler, key) -> State:
    """Initialise a fresh state and execute the non-checkpointed fast path."""

    key, init_key = jax.random.split(key)
    state = _sample_init_state(
        key=init_key,
        num_live_points=int(self.target_num_live_points),
        max_samples=int(self.max_samples),
        model=self.model,
        num_phantom=int(self.sampler.num_phantom()) if self.sampler is not None else 0,
        args=self.args,
        params=self.params,
        store_phantom_samples=self.store_phantom_samples,
        batch_size=self.batch_size
    )
    state = _run_ns(
        key=key,
        state=state,
        target_num_live_points=int(self.target_num_live_points),
        shell_size=int(self.shell_size),
        args=self.args,
        sampler=self.sampler,
        params=self.params,
        termination_condition=self.termination_condition,
        batch_size=self.batch_size
    )
    return state


def _default_checkpoint_every(self: NestedSampler) -> int:
    """Choose a conservative default chunk size for checkpointed runs."""

    shell_size = max(1, int(self.shell_size))
    target_num_live_points = max(shell_size, int(self.target_num_live_points))
    return max(1, min(16, (4 * target_num_live_points) // shell_size))


def _flatten_step_deltas(step_deltas: StepDelta, steps_executed: int) -> tuple[Samples | None, np.ndarray | None]:
    """Flatten a scanned chunk delta into append-ready journal rows."""

    if steps_executed <= 0:
        return None, None

    host_deltas = jax.device_get(step_deltas)
    row_count = steps_executed * host_deltas.parent_sample_ids.shape[1]

    def flatten_leaf(leaf):
        leaf = np.asarray(leaf[:steps_executed])
        return leaf.reshape((row_count,) + leaf.shape[2:])

    phantom_U_samples = None
    if host_deltas.samples.phantom_samples.U_samples is not None:
        phantom_U_samples = jax.tree.map(flatten_leaf, host_deltas.samples.phantom_samples.U_samples)

    flat_samples = Samples(
        log_L_constraints=flatten_leaf(host_deltas.samples.log_L_constraints),
        log_likelihoods=flatten_leaf(host_deltas.samples.log_likelihoods),
        sample_ids=flatten_leaf(host_deltas.samples.sample_ids),
        U_samples=jax.tree.map(flatten_leaf, host_deltas.samples.U_samples),
        out_degree=flatten_leaf(host_deltas.samples.out_degree),
        num_likelihood_evaluations=flatten_leaf(host_deltas.samples.num_likelihood_evaluations),
        phantom_samples=PhantomSamples(
            U_samples=phantom_U_samples,
            valid_mask=flatten_leaf(host_deltas.samples.phantom_samples.valid_mask),
            log_L=flatten_leaf(host_deltas.samples.phantom_samples.log_L),
        ),
    )
    parent_sample_ids = np.asarray(host_deltas.parent_sample_ids[:steps_executed]).reshape((row_count,))
    return flat_samples, parent_sample_ids


def _run_with_checkpointing(self: NestedSampler, key: PRNGKey | None, archive_path: str | Path, *, resume: bool,
                            checkpoint_every: int | None) -> State:
    """Drive checkpointed sampling from Python while keeping chunk execution jitted."""

    from jaxns.checkpointing import append_checkpoint, initialise_archive, load_checkpoint

    archive_path = Path(archive_path)

    if resume:
        loaded_checkpoint = load_checkpoint(archive_path, nested_sampler=self)
        state = loaded_checkpoint.state
        key = loaded_checkpoint.key
        checkpoint_index = loaded_checkpoint.checkpoint_index
        if checkpoint_every is None:
            checkpoint_every = loaded_checkpoint.checkpoint_every
        if loaded_checkpoint.completed:
            return state
    else:
        if key is None:
            key = jax.random.PRNGKey(42)
        if checkpoint_every is None:
            checkpoint_every = _default_checkpoint_every(self)
        key, init_key = jax.random.split(key)
        state = _sample_init_state(
            key=init_key,
            num_live_points=int(self.target_num_live_points),
            max_samples=int(self.max_samples),
            model=self.model,
            num_phantom=int(self.sampler.num_phantom()) if self.sampler is not None else 0,
            args=self.args,
            params=self.params,
            store_phantom_samples=self.store_phantom_samples,
            batch_size=self.batch_size,
        )
        done, termination_reason = _check_termination(state, int(self.target_num_live_points), self.termination_condition)
        done = bool(np.asarray(jax.device_get(done)))
        if done:
            state = dataclasses.replace(state, termination_reason=termination_reason)
        initialise_archive(
            archive_path,
            nested_sampler=self,
            state=state,
            current_key=key,
            checkpoint_every=checkpoint_every,
            completed=done,
        )
        checkpoint_index = 0
        if done:
            return state

    if checkpoint_every is None or checkpoint_every < 1:
        raise ValueError(f'checkpoint_every must be >= 1, got {checkpoint_every}.')

    while True:
        key, state, step_deltas, steps_executed, done, termination_reason = _run_ns_chunk(
            key=key,
            state=state,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            checkpoint_every=int(checkpoint_every),
            args=self.args,
            sampler=self.sampler,
            params=self.params,
            termination_condition=self.termination_condition,
            batch_size=self.batch_size,
        )
        done = bool(np.asarray(jax.device_get(done)))
        if done:
            state = dataclasses.replace(state, termination_reason=termination_reason)
        steps_executed_host = int(np.asarray(jax.device_get(steps_executed)))
        flat_samples, parent_sample_ids = _flatten_step_deltas(step_deltas, steps_executed_host)
        checkpoint_index += 1
        append_checkpoint(
            archive_path,
            samples=flat_samples,
            parent_sample_ids=parent_sample_ids,
            current_key=key,
            state=state,
            checkpoint_index=checkpoint_index,
            checkpoint_every=int(checkpoint_every),
            completed=done,
        )
        if done:
            return state
