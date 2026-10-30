"""Single-process observer ownership of policy versions, feedback and raw ledgers."""

from dataclasses import dataclass
import time

import jax
import numpy as np
from jax import numpy as jnp

from benchmarks.issue_312.online_controller import OnlineController, observe_completed
from benchmarks.issue_312.online_frozen import FrozenLikelihoodPolicy
from benchmarks.issue_312.online_policy import ChosenParameters, OnlinePolicy
from benchmarks.issue_312.online_sampler import OnlineReflectionSampler, prepare_contours
from benchmarks.issue_312.reflection_sampler import ReflectionSampler
from jaxns.algorithm.depth import CoreWorkBatch
from jaxns.constrained_sampler import AbstractSampler
from jaxns.state import State


@jax.jit
def dispatch_parameters(sampler: OnlineReflectionSampler, state: State, work: CoreWorkBatch):
    """Reconstruct existing chain keys without consuming another sampling stream."""
    new_depth_key, _ = jax.random.split(state.random_key)
    depth_key = jnp.where(state.depth_reached, new_depth_key, state.random_key)
    _, sample_key, _ = jax.random.split(depth_key, 3)
    keys = jax.random.split(sample_key, work.seed_idx.shape[0])  # [width, 2]
    chosen = jax.vmap(sampler.parameters)(keys, work.log_L_constraint)
    positions = jnp.minimum(
        jnp.searchsorted(sampler.prepared.contours, work.log_L_constraint, side='left'),
        work.seed_idx.shape[0] - 1,
    )
    return (keys, chosen, sampler.prepared.compression[positions],
            sampler.prepared.incoming_k[positions])


@jax.jit
def completed_feedback(before: State, current: State, work: CoreWorkBatch):
    """Transfer a width-sized observer summary, not the full sample capacity."""
    indices = before.num_samples + jnp.arange(work.seed_idx.shape[0])
    indices = jnp.minimum(indices, current.samples.log_likelihoods.shape[0] - 1)
    # This experiment's actual unit cube is one [capacity, 18] model leaf.
    starts = jax.tree.leaves(before.samples.U_samples)[0][jnp.maximum(work.seed_idx, 0)]
    ends = jax.tree.leaves(current.samples.U_samples)[0][indices]
    squared_jump = jnp.sum((ends - starts)**2, axis=1)  # [width], ALL unit coordinates
    child = work.valid & (work.seed_idx >= 0)
    return (jnp.where(child, squared_jump, 0.),
            current.samples.num_likelihood_evaluations[indices])


@dataclass(frozen=True, slots=True)
class OnlineCheckpoint:
    """One atomic experiment continuation; a bare production State is insufficient."""

    state: State
    controller: OnlineController
    kernel: ReflectionSampler
    adapt: bool
    explore: bool
    frozen_policy: FrozenLikelihoodPolicy | None
    rows: tuple[dict[str, np.ndarray], ...]
    receipts: tuple[dict, ...]
    policy_history: tuple[OnlinePolicy, ...]
    budgets: tuple[int, ...]
    next_budget: int
    snapshots: tuple[dict, ...]
    sampling_elapsed_seconds: float
    controller_seconds: float
    prepare_seconds: float
    feedback_seconds: float
    source_sha256: dict[str, str]


class OnlineSession:
    """Explicit host state for one run; no global or hidden sampler mutation.

    The runner saves this controller and committed ledger with its State. All
    arrays appended here have only valid rows, and policy snapshots precede use.
    """

    def __init__(self, initial_roots: int, *, adapt: bool = True, explore: bool = True,
                 frozen_policy: FrozenLikelihoodPolicy | None = None):
        self.controller = OnlineController.initialise(initial_roots)
        self.adapt = adapt
        self.explore = explore
        self.frozen_policy = frozen_policy
        self.kernel = ReflectionSampler(num_cycles=9, slices_per_cycle=1,
                                        num_bounces=10, step_size=.003)
        self.active: OnlineReflectionSampler | None = None
        self.prepared_start = -1
        self.parameters: tuple | None = None
        self.rows: list[dict[str, np.ndarray]] = []
        self.receipts: list[dict] = []
        self.policy_history = [jax.device_get(self.controller.policy)]
        self.controller_seconds = 0.
        self.prepare_seconds = 0.
        self.feedback_seconds = 0.

    def prepare(self, before: State, work: CoreWorkBatch, original: AbstractSampler):
        started = time.monotonic()
        prepared = prepare_contours(before, work)
        self.active = OnlineReflectionSampler(
            self.kernel._with_phantom_count(original.num_phantom()),
            self.controller.policy, prepared, explore=self.explore,
            frozen_policy=self.frozen_policy,
        )
        self.parameters = jax.device_get(dispatch_parameters(self.active, before, work))
        self.prepared_start = int(before.num_samples)
        self.prepare_seconds += time.monotonic() - started
        return self.active

    def observe(self, before: State, current: State, work: CoreWorkBatch) -> dict[str, np.ndarray]:
        started = time.monotonic()
        start, stop = int(before.num_samples), int(current.num_samples)
        if (self.active is None or self.parameters is None or self.prepared_start != start):
            raise RuntimeError('Feedback lacks the exact preceding dispatch preparation.')
        size = stop - start
        keys, parameters, compression, incoming_k = self.parameters
        parameters = ChosenParameters(*(np.asarray(item)[:size] for item in parameters))
        seed_indices = np.asarray(work.seed_idx)[:size]
        child = seed_indices >= 0
        jumps, calls = jax.device_get(completed_feedback(before, current, work))
        version = int(self.active.policy.version)
        record = dict(
            sample_indices=np.arange(start, stop, dtype=np.int64),
            seed_indices=seed_indices,
            parent_indices=np.asarray(work.parent_idx)[:size],
            chain_keys=np.asarray(keys)[:size],
            compression=np.asarray(compression)[:size],
            incoming_k=np.asarray(incoming_k)[:size],
            controller_versions=np.full(size, version, np.int64),
            log_L_constraints=np.asarray(work.log_L_constraint)[:size],
            squared_jumps=np.asarray(jumps)[:size],
            work=np.asarray(calls)[:size],
            batch_indices=np.full(size, int(current.depth_loop_iter), np.int64),
        )
        strata = np.clip(np.floor(record['compression'] / 2).astype(np.int64), 0, 63)
        record['controller_generations'] = self.controller.generations[strata].copy()
        for name, values in zip(ChosenParameters._fields, parameters, strict=True):
            record[name] = np.where(child, values, -1 if name == 'choice' else 0)
        record['gradient_evaluations'] = 9 * record['num_bounces']
        record['value_only_evaluations'] = record['work'] - record['gradient_evaluations']
        if not np.all(record['value_only_evaluations'][~child] == 1):
            raise RuntimeError('New root rows must have one value call and zero gradients.')
        self.feedback_seconds += time.monotonic() - started
        started = time.monotonic()
        self.controller, receipt = observe_completed(
            self.controller, start=start, seed_indices=seed_indices,
            compression=record['compression'], incoming_k=record['incoming_k'],
            choices=record['choice'], squared_jumps=record['squared_jumps'],
            work=record['work'], policy_version=version, adapt=self.adapt,
        )
        if receipt['changes']:
            self.policy_history.append(jax.device_get(self.controller.policy))
        self.receipts.append(receipt)
        self.rows.append(record)
        self.controller_seconds += time.monotonic() - started
        return record

    def checkpoint(
        self, state: State, *, budgets: tuple[int, ...], next_budget: int,
        snapshots: tuple[dict, ...], sampling_elapsed_seconds: float,
        source_sha256: dict[str, str],
    ) -> OnlineCheckpoint:
        """Capture all continuation owners together at a committed batch boundary."""
        if int(state.num_samples) != self.controller.next_sample:
            raise ValueError('Core and controller do not describe the same committed prefix.')
        return OnlineCheckpoint(
            jax.device_get(state), self.controller, self.kernel, self.adapt, self.explore,
            self.frozen_policy, tuple(self.rows), tuple(self.receipts),
            tuple(self.policy_history), budgets,
            next_budget, snapshots, sampling_elapsed_seconds, self.controller_seconds,
            self.prepare_seconds, self.feedback_seconds, source_sha256,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: OnlineCheckpoint) -> 'OnlineSession':
        """Construct a fresh host session; no previous Python ledgers are consulted."""
        if int(checkpoint.state.num_samples) != checkpoint.controller.next_sample:
            raise ValueError('Checkpoint core/controller prefixes disagree.')
        session = cls(
            0, adapt=checkpoint.adapt, explore=checkpoint.explore,
            frozen_policy=checkpoint.frozen_policy,
        )
        session.controller = checkpoint.controller
        session.kernel = checkpoint.kernel
        session.rows = list(checkpoint.rows)
        session.receipts = list(checkpoint.receipts)
        session.policy_history = list(checkpoint.policy_history)
        session.controller_seconds = checkpoint.controller_seconds
        session.prepare_seconds = checkpoint.prepare_seconds
        session.feedback_seconds = checkpoint.feedback_seconds
        return session

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {}
        return {name: np.concatenate([row[name] for row in self.rows]) for name in self.rows[0]}
