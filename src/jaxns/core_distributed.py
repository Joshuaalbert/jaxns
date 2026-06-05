from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx import CtxParams

from jaxns.constrained_sampler_distributed import DistributedUniDimSliceSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.random_utils import resample_indicies
from jaxns.samples import PhantomSamples, Samples, SeedPoint
from jaxns.state import State
from jaxns.termination_condition import DepthCondition


def _map_parallel(items: Iterable[Any], fn, max_workers: int | None):
    items = list(items)
    if len(items) == 0:
        return []
    if max_workers is None or max_workers <= 1 or len(items) == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(fn, items))


def _stack_pytrees(items):
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *items)


def _build_empty_phantom(U_samples, outer_size: int, num_phantom: int) -> PhantomSamples:
    return PhantomSamples(
        valid_mask=jnp.full((outer_size, num_phantom), False, dtype=mp_policy.bool_dtype),
        U_samples=jax.tree.map(
            lambda u: jnp.zeros((outer_size, num_phantom) + u.shape[1:], dtype=u.dtype),
            U_samples,
        ),
        log_L=jnp.full((outer_size, num_phantom), -jnp.inf, dtype=mp_policy.measure_dtype),
    )


def _pad_samples(samples: Samples, sample_atom: Samples, current_size: int, max_samples: int) -> Samples:
    if current_size == max_samples:
        return samples
    return jax.tree.map(
        lambda x, y: jnp.concatenate(
            [x, jnp.repeat(y[None, ...], repeats=max_samples - current_size, axis=0)],
            axis=0,
        ),
        samples,
        sample_atom,
    )


def _sample_init_state_distributed(
    key,
    *,
    num_live_points: int,
    max_samples: int,
    model: Model,
    evaluator,
    num_phantom: int = 0,
    args=(),
    params: CtxParams | None = None,
    store_phantom_samples: bool = False,
    num_parallel_workers: int | None = None,
) -> State:
    def single_sample(sample_key):
        key_inner = sample_key
        num_likelihood_evaluations = 0
        while True:
            key_inner, subkey = jax.random.split(key_inner)
            U_sample = model.sample_U(subkey, args=args, params=params)
            log_L = jnp.asarray(
                evaluator.evaluate(U_sample),
                dtype=mp_policy.measure_dtype,
            )
            num_likelihood_evaluations += 1
            if bool(log_L > -jnp.inf):
                return U_sample, log_L, jnp.asarray(num_likelihood_evaluations, mp_policy.count_dtype)

    init_keys = jax.random.split(key, num_live_points)
    results = [single_sample(sample_key) for sample_key in init_keys]
    U_samples = _stack_pytrees([result[0] for result in results])
    log_likelihoods = jnp.stack([result[1] for result in results], axis=0)
    num_likelihood_evaluations = jnp.stack([result[2] for result in results], axis=0)
    out_degree = jnp.zeros((num_live_points,), dtype=mp_policy.count_dtype)
    phantom_samples = _build_empty_phantom(U_samples, num_live_points, num_phantom)
    samples = Samples(
        log_L_constraints=jnp.full((num_live_points,), -jnp.inf, dtype=mp_policy.measure_dtype),
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations,
        phantom_samples=phantom_samples,
    )

    sample_atom = Samples(
        log_likelihoods=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
        out_degree=jnp.asarray(0, mp_policy.count_dtype),
        num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype),
        U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), U_samples),
        phantom_samples=PhantomSamples(
            U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), phantom_samples.U_samples),
            log_L=jnp.zeros_like(phantom_samples.log_L[0]),
            valid_mask=jnp.zeros_like(phantom_samples.valid_mask[0]),
        ),
        log_L_constraints=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
    )
    samples = _pad_samples(samples, sample_atom, num_live_points, max_samples).sort()
    if not store_phantom_samples:
        samples.phantom_samples.U_samples = None

    log_L_supremum_idx = int(jnp.argmax(log_likelihoods))
    log_L_supremum = log_likelihoods[log_L_supremum_idx]
    U_supremum = jax.tree.map(lambda u: u[log_L_supremum_idx], U_samples)
    return State(
        root_out_degree=jnp.asarray(num_live_points, dtype=mp_policy.count_dtype),
        samples=samples,
        num_samples=jnp.asarray(num_live_points, dtype=mp_policy.count_dtype),
        log_L_supremum=log_L_supremum,
        U_supremum=U_supremum,
        model=model,
        args=args,
        params=params,
        termination_reason=jnp.asarray(0, dtype=mp_policy.index_dtype),
    )


def _run_ns_distributed(
    key,
    *,
    state: State,
    target_num_live_points: int,
    shell_size: int,
    args=(),
    sampler=None,
    params=None,
    termination_condition: DepthCondition | None = None,
    num_parallel_workers: int | None = None,
) -> State:
    while True:
        register = state.compute_termination_register(target_num_live_points=target_num_live_points)
        done, termination_reason = register.is_done(termination_condition)
        if bool(done):
            state.termination_reason = termination_reason
            return state

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
            0.0,
            -jnp.inf,
        )
        select_contours_key, key = jax.random.split(key)
        parent_idxs = resample_indicies(
            select_contours_key,
            log_weights=select_weights,
            S=shell_size,
            replace=False,
        )
        proposed_log_L_constraints = state.samples.log_likelihoods[parent_idxs]
        shell_keys = jax.random.split(key, shell_size + 1)
        key = shell_keys[0]
        proposal_keys = shell_keys[1:]
        num_samples = int(state.num_samples)
        sorted_log_l = np.asarray(state.samples.log_likelihoods[:num_samples])

        def get_sample(task):
            task_key, log_L_constraint, parent_idx = task
            seed_key, sample_key = jax.random.split(task_key)
            i_start = int(np.searchsorted(sorted_log_l, float(log_L_constraint), side="right"))
            no_seeds = i_start == num_samples
            effective_log_L_constraint = (
                jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype)
                if no_seeds
                else jnp.asarray(log_L_constraint, dtype=mp_policy.measure_dtype)
            )
            delta_root_out_degree = jnp.asarray(1 if no_seeds else 0, dtype=mp_policy.count_dtype)
            delta_parent_out_degree = jnp.asarray(0 if no_seeds else 1, dtype=mp_policy.count_dtype)
            seed_low = 0 if no_seeds else i_start
            seed_select_idx = int(jax.random.randint(seed_key, (), seed_low, num_samples))
            seed_point = SeedPoint(
                U0=jax.tree.map(lambda u: u[seed_select_idx], state.samples.U_samples),
                log_L0=state.samples.log_likelihoods[seed_select_idx],
            )
            sample = sampler.get_sample(
                sample_key,
                effective_log_L_constraint,
                seed_point,
                args=args,
                params=params,
            )
            return sample, (delta_root_out_degree, delta_parent_out_degree, effective_log_L_constraint, parent_idx)

        tasks = list(zip(proposal_keys, proposed_log_L_constraints, parent_idxs))
        sampled = _map_parallel(
            tasks,
            get_sample,
            max_workers=min(shell_size, num_parallel_workers or shell_size),
        )
        sample_results = [item[0] for item in sampled]
        metadata = [item[1] for item in sampled]

        U_samples = _stack_pytrees([item[0] for item in sample_results])
        log_likelihoods = jnp.stack([item[1] for item in sample_results], axis=0)
        num_likelihood_evaluations = jnp.stack([item[2] for item in sample_results], axis=0)
        phantom_samples = PhantomSamples(
            U_samples=_stack_pytrees([item[3].U_samples for item in sample_results]),
            log_L=jnp.stack([item[3].log_L for item in sample_results], axis=0),
            valid_mask=jnp.stack([item[3].valid_mask for item in sample_results], axis=0),
        )
        delta_root_out_degree = jnp.stack([item[0] for item in metadata], axis=0)
        delta_parent_out_degree = jnp.stack([item[1] for item in metadata], axis=0)
        log_L_constraints = jnp.stack([item[2] for item in metadata], axis=0)
        parent_idxs = jnp.stack([item[3] for item in metadata], axis=0)

        new_samples = Samples(
            log_L_constraints=log_L_constraints,
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations,
            phantom_samples=phantom_samples,
        )
        if state.samples.phantom_samples.U_samples is None:
            new_samples.phantom_samples.U_samples = None

        candidate_idx = int(jnp.argmax(new_samples.log_likelihoods))
        candidate_log_L_supremum = new_samples.log_likelihoods[candidate_idx]
        candidate_U_supremum = jax.tree.map(lambda u: u[candidate_idx], new_samples.U_samples)

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
            phantom_U_samples = jax.tree.map(
                lambda u: u.reshape((-1,) + u.shape[2:]),
                new_samples.phantom_samples.U_samples,
            )
            phantom_log_L = jnp.where(valid_mask, phantom_log_L, -jnp.inf)
            phantom_candidate_idx = int(jnp.argmax(phantom_log_L))
            phantom_candidate_log_L = phantom_log_L[phantom_candidate_idx]
            phantom_candidate_U = jax.tree.map(lambda u: u[phantom_candidate_idx], phantom_U_samples)
            better_phantom = phantom_candidate_log_L > log_L_supremum
            log_L_supremum = jnp.where(better_phantom, phantom_candidate_log_L, log_L_supremum)
            U_supremum = jax.tree.map(
                lambda u_new, u_old: jnp.where(better_phantom, u_new, u_old),
                phantom_candidate_U,
                U_supremum,
            )

        state = State(
            root_out_degree=state.root_out_degree + jnp.sum(delta_root_out_degree),
            samples=state.samples.append_samples(
                insert_idx=state.num_samples,
                parent_idxs=parent_idxs,
                samples=new_samples,
                delta_parent_out_degree=delta_parent_out_degree,
            ).sort(),
            num_samples=state.num_samples + shell_size,
            log_L_supremum=log_L_supremum,
            U_supremum=U_supremum,
            model=state.model,
            args=state.args,
            params=state.params,
            termination_reason=state.termination_reason,
        )


@dataclasses.dataclass(slots=True)
class NestedSamplerDistributed:
    model: Model
    evaluator: Any | None = None
    target_num_live_points: int | None = None
    max_samples: int | None = None
    shell_size: int | None = None
    args: tuple = ()
    params: CtxParams | None = None
    sampler: Any | None = None
    termination_condition: DepthCondition | None = None
    store_phantom_samples: bool = False
    collect_phantom_samples: bool = False
    batch_size: int | None = None
    num_parallel_workers: int | None = None

    def __post_init__(self):
        U_ndims = 0
        if (
            self.target_num_live_points is None
            or self.max_samples is None
            or self.shell_size is None
            or self.sampler is None
        ):
            U_ndims = int(self.model.U_ndims(self.args, self.params))
        if self.target_num_live_points is None:
            self.target_num_live_points = 20 * U_ndims
        if self.max_samples is None:
            self.max_samples = 10000 * U_ndims
        if self.shell_size is None:
            self.shell_size = max(1, self.target_num_live_points // 2)
        if self.max_samples < self.target_num_live_points:
            raise ValueError("max_samples must be >= target_num_live_points.")
        max_samples = jnp.asarray(self.max_samples, dtype=mp_policy.count_dtype)
        if self.termination_condition is None:
            self.termination_condition = DepthCondition(dlogZ=1e-2, max_samples=max_samples)
        elif self.termination_condition.max_samples is None:
            self.termination_condition.max_samples = max_samples
        else:
            self.termination_condition.max_samples = jnp.minimum(self.termination_condition.max_samples, max_samples)
        if self.num_parallel_workers is None:
            self.num_parallel_workers = int(self.shell_size)
        if self.sampler is None:
            if self.evaluator is None:
                raise ValueError("Provide evaluator when sampler is not supplied.")
            self.sampler = DistributedUniDimSliceSampler(
                model=self.model,
                evaluator=self.evaluator,
                num_slices=max(1, 100 * U_ndims),
                phantom_burn_in=max(1, 20 * U_ndims),
                no_step_out=True,
                collect_phantom_samples=self.collect_phantom_samples,
            )
        elif self.evaluator is None:
            self.evaluator = getattr(self.sampler, "evaluator", None)

    def run(self, key=None) -> State:
        if key is None:
            key = jax.random.PRNGKey(42)
        key, init_key = jax.random.split(key)
        state = _sample_init_state_distributed(
            key=init_key,
            num_live_points=int(self.target_num_live_points),
            max_samples=int(self.max_samples),
            model=self.model,
            evaluator=self.evaluator,
            num_phantom=int(self.sampler.num_phantom()) if self.sampler is not None else 0,
            args=self.args,
            params=self.params,
            store_phantom_samples=self.store_phantom_samples,
            num_parallel_workers=self.num_parallel_workers,
        )
        return _run_ns_distributed(
            key=key,
            state=state,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            args=self.args,
            sampler=self.sampler,
            params=self.params,
            termination_condition=self.termination_condition,
            num_parallel_workers=self.num_parallel_workers,
        )

    def resume(self, state: State, key=None) -> State:
        if key is None:
            key = jax.random.PRNGKey(42)
        return _run_ns_distributed(
            key=key,
            state=state,
            target_num_live_points=int(self.target_num_live_points),
            shell_size=int(self.shell_size),
            args=self.args,
            sampler=self.sampler,
            params=self.params,
            termination_condition=self.termination_condition,
            num_parallel_workers=self.num_parallel_workers,
        )


DistributedNestedSampler = NestedSamplerDistributed
