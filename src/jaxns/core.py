import dataclasses
from abc import abstractmethod, ABC
from functools import partial
from typing import NamedTuple, Any, Callable, Literal

import jax
import jax.numpy as jnp
import jax.random
from jaxctx import CtxParams

from jaxns.constrained_sampler import AbstractSampler, UniDimSliceSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample_indicies
from jaxns.samples import Samples, SeedPoint, PhantomSamples
from jaxns.state import State
from jaxns.termination_condition import DepthCondition
from jaxns.types import IntArray, BoolArray, PRNGKey, FloatArray


class AbstractAllocationTargetFn(ABC):
    @abstractmethod
    def compute_target_active_lineage_count(self, log_likelihood: FloatArray) -> FloatArray:
        """
        Compute the target active lineage count as a function of the log likelihood.

        Args:
            log_likelihood: the log likelihood value

        Returns:
            the target active lineage count
        """
        ...


@dataclasses.dataclass(slots=True, frozen=True)
class UniformDepth(PureDataclassPytree, AbstractAllocationTargetFn):
    goal_loop_iter: IntArray
    delta_allocation_degree: FloatArray
    root_allocation_degree: FloatArray

    def compute_target_active_lineage_count(self, log_likelihood: FloatArray) -> FloatArray:
        return self.delta_allocation_degree * self.goal_loop_iter + self.root_allocation_degree


UniformDepth.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class InterpolatedAllocationTargetFn(PureDataclassPytree, AbstractAllocationTargetFn):
    log_likelihoods_inputs: FloatArray
    allocations: FloatArray

    def add(self, other: 'InterpolatedAllocationTargetFn'):
        # Combine the two by summing their allocations at the union of their log_likelihoods_inputs
        new_allocations = (self.compute_target_active_lineage_count(other.log_likelihoods_inputs) +
                           other.compute_target_active_lineage_count(other.log_likelihoods_inputs))
        return InterpolatedAllocationTargetFn(
            log_likelihoods_inputs=other.log_likelihoods_inputs,
            allocations=new_allocations
        )

    def compute_target_active_lineage_count(self, log_likelihood: FloatArray) -> FloatArray:
        return jnp.interp(log_likelihood, self.log_likelihoods_inputs, self.allocations)


InterpolatedAllocationTargetFn.register_pytree()


@partial(jax.jit, inline=True,
         static_argnames=['root_degree', 'num_phantom', 'max_samples', 'store_phantom_samples', 'batch_size'])
def _sample_init_state(key, root_degree: int, max_samples: int, model: Model, num_phantom: int = 0, args=(),
                       params: CtxParams | None = None, store_phantom_samples: bool = False,
                       batch_size: int | None = None) -> State:
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
            log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False).astype(
                mp_policy.measure_dtype)
            num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
            return (key, U_sample, log_L, num_likelihood_evaluations)

        key, U_sample, log_L, num_likelihood_evaluations = jax.lax.while_loop(cond_fn, body_fn, carry)
        return U_sample, log_L, num_likelihood_evaluations

    U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
        single_sample,
        jax.random.split(key, root_degree),
        batch_size=batch_size
    )
    out_degree = jnp.zeros((root_degree,), dtype=mp_policy.count_dtype)

    # extend each to max_samples
    def _concat(x, fill_value):
        return jax.tree.map(
            lambda x, y: jnp.concatenate([
                x,
                jnp.repeat(y[None, ...], repeats=(max_samples - root_degree), axis=0)
            ], axis=0),
            x,
            fill_value
        )

    phantom_samples = PhantomSamples(
        # TODO: make valid_mask shaped [max_samples] since we either have all phantoms for a sample or none valid.
        valid_mask=jnp.full((root_degree, num_phantom), False, dtype=mp_policy.bool_dtype),
        U_samples=jax.tree.map(lambda u: jnp.zeros((root_degree, num_phantom) + u[0].shape, u.dtype), U_samples),
        log_L=jnp.full((root_degree, num_phantom), -jnp.inf, dtype=mp_policy.measure_dtype)
    )
    if not store_phantom_samples:
        phantom_samples.U_samples = None
    samples = Samples(
        log_L_constraints=jnp.full((root_degree,), -jnp.inf, dtype=mp_policy.measure_dtype),
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations,
        phantom_samples=phantom_samples
    )

    sample_atom = Samples(
        log_likelihoods=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
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

    log_L_supremum_idx = jnp.argmax(log_likelihoods)
    log_L_supremum = log_likelihoods[log_L_supremum_idx]
    U_supremum = jax.tree.map(lambda u: u[log_L_supremum_idx], U_samples)
    # Sort samples into increasing log-likelihood order
    state = State(
        root_out_degree=jnp.array(root_degree, dtype=mp_policy.count_dtype),
        samples=samples,
        num_samples=jnp.array(root_degree, dtype=mp_policy.count_dtype),
        log_L_supremum=log_L_supremum,
        U_supremum=U_supremum,
        model=model,
        args=args,
        params=params,
        termination_reason=jnp.array(0, dtype=mp_policy.index_dtype),
        goal_loop_iter=jnp.array(0, dtype=mp_policy.index_dtype)
    )
    return state


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points', 'shell_size', 'batch_size'])
def _run_single_iteration(
        key, state: State, allocation_target_fn: AbstractAllocationTargetFn, shell_size: int,
        sampler: AbstractSampler,
        depth_cond: DepthCondition,
        batch_size: int | None = None
) -> State:
    """
    Perform a single nested sampling run.

    Args:
        key: PRNG key
        allocation_target_fn: the target number of live lineages to allocate wrt log-likelihood.
        shell_size: the number of samples to discard and replenish per iteration
        sampler: the sampler to use to produce i.i.d. samples within likelihood constraints
        depth_cond: the termination condition to use
        batch_size: how many likelihood evaluations to batch together

    Returns:
        A final state object containing all samples and relevant information for resuming, or evidence calculation.
    """

    # Algorithm
    # repeat until depth condition:
    # 1. count lineages
    # 2. choose parents
    # 3. choose seed for each parent
    # 4. sample from within parent contours
    # 5. compute depth condition

    args = state.args
    params = state.params

    class OuterCarry(NamedTuple):
        key: jax.Array
        state: State

    def outer_cond_fn(carry: OuterCarry) -> BoolArray:
        register = carry.state.compute_termination_register()
        done, _ = register.is_done(depth_cond)
        return jnp.logical_not(done)

    def outer_body_fn(outer_carry: OuterCarry) -> OuterCarry:
        # Select likelihood constraints to achieve minimum K[i]>target (randomly without replacement for simplicity)
        active_lineage_count = outer_carry.state.samples.compute_num_live_points_per_sample(
            root_out_degree=outer_carry.state.root_out_degree,
            num_samples=outer_carry.state.num_samples
        ) # [N]
        target_active_lineage_count = allocation_target_fn.compute_target_active_lineage_count(
            outer_carry.state.samples.log_likelihoods
        ) # [S] or scalar
        K_next_sample = active_lineage_count - 1 + outer_carry.state.samples.out_degree
        select_weights = jnp.where(
            jnp.logical_and(
                jnp.arange(K_next_sample.shape[0]) < outer_carry.state.num_samples,
                K_next_sample < target_active_lineage_count
            ), 0, -jnp.inf)
        select_contours_key, key = jax.random.split(outer_carry.key, 2)
        parent_idxs = resample_indicies(select_contours_key, log_weights=select_weights, S=shell_size,
                                        replace=False)  # [S]
        proposed_log_L_constraints = outer_carry.state.samples.log_likelihoods[parent_idxs]  # [S]

        # TODO: give sampling a multi-ellipsoidal clustering, then use to guide sampling along preferential axes.

        def get_sample(key, log_L_constraint, parent_idx: IntArray):
            seed_key, sample_key = jax.random.split(key)
            # Get seed from samples
            i_start = jax.lax.while_loop(
                lambda i: (i < outer_carry.state.num_samples) & (
                        outer_carry.state.samples.log_likelihoods[i] <= log_L_constraint),
                lambda i: i + 1,
                parent_idx + 1
            )
            no_seeds = i_start == outer_carry.state.num_samples
            log_L_constraint = jnp.where(no_seeds, jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype),
                                         log_L_constraint)
            delta_root_out_degree = jnp.where(no_seeds, 1, 0).astype(mp_policy.count_dtype)
            delta_parent_out_degree = jnp.where(no_seeds, 0, 1).astype(mp_policy.count_dtype)
            i_start = jnp.where(no_seeds, 0, i_start).astype(mp_policy.index_dtype)
            seed_select_idx = jax.random.randint(seed_key, (), i_start, outer_carry.state.num_samples)
            seed_point = SeedPoint(
                U0=jax.tree.map(lambda u: u[seed_select_idx], outer_carry.state.samples.U_samples),
                log_L0=outer_carry.state.samples.log_likelihoods[seed_select_idx]
            )
            return sampler.get_sample(
                sample_key, log_L_constraint, seed_point, args=args, params=params,
            ), (delta_root_out_degree, delta_parent_out_degree, log_L_constraint)

        key, subkey = jax.random.split(outer_carry.key)
        keys = jax.random.split(subkey, shell_size)
        (U_samples, log_likelihoods, num_likelihood_evaluations, phantom_samples), (
            delta_root_out_degree,
            delta_parent_out_degree,
            log_L_constraints
        ) = jax.lax.map(
            lambda x: get_sample(x[0], x[1], x[2]),
            (keys, proposed_log_L_constraints, parent_idxs),
            batch_size=batch_size
        )

        new_samples = Samples(
            log_L_constraints=log_L_constraints,
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations,
            phantom_samples=phantom_samples
        )
        if outer_carry.state.samples.phantom_samples.U_samples is None:
            new_samples.phantom_samples.U_samples = None

        candidate_supremum_candidate_iid = jnp.argmax(new_samples.log_likelihoods)
        candidate_log_L_supremum = new_samples.log_likelihoods[candidate_supremum_candidate_iid]
        candidate_U_supremum = jax.tree.map(lambda u: u[candidate_supremum_candidate_iid], new_samples.U_samples)

        log_L_supremum = jnp.where(candidate_log_L_supremum > outer_carry.state.log_L_supremum,
                                   candidate_log_L_supremum, outer_carry.state.log_L_supremum)
        U_supremum = jax.tree.map(
            lambda u_new, u_old: jnp.where(candidate_log_L_supremum > outer_carry.state.log_L_supremum, u_new, u_old),
            candidate_U_supremum, outer_carry.state.U_supremum)
        if new_samples.phantom_samples.U_samples is not None:
            # flatten, and select only valid ones
            valid_mask = new_samples.phantom_samples.valid_mask.reshape((-1,))
            phantom_log_L = new_samples.phantom_samples.log_L.reshape((-1,))
            phantom_U_samples = jax.tree.map(lambda u: u.reshape((-1,) + u.shape[2:]),
                                             new_samples.phantom_samples.U_samples)
            phantom_log_L = jnp.where(valid_mask, phantom_log_L, -jnp.inf)
            candidate_supremum_candidate_phantom = jnp.argmax(phantom_log_L)
            candidate_log_L_supremum_phantom = phantom_log_L[candidate_supremum_candidate_phantom]
            candidate_U_supremum_phantom = jax.tree.map(lambda u: u[candidate_supremum_candidate_phantom],
                                                        phantom_U_samples)
            log_L_supremum = jnp.where(candidate_log_L_supremum_phantom > log_L_supremum,
                                       candidate_log_L_supremum_phantom, log_L_supremum)
            U_supremum = jax.tree.map(
                lambda u_phantom, u_current: jnp.where(candidate_log_L_supremum_phantom > log_L_supremum, u_phantom,
                                                       u_current),
                candidate_U_supremum_phantom, U_supremum)

        state = State(
            root_out_degree=outer_carry.state.root_out_degree + jnp.sum(delta_root_out_degree),
            samples=outer_carry.state.samples.append_samples(
                insert_idx=outer_carry.state.num_samples,
                parent_idxs=parent_idxs,
                samples=new_samples,
                delta_parent_out_degree=delta_parent_out_degree,
            ).sort(),
            num_samples=outer_carry.state.num_samples + len(new_samples),
            log_L_supremum=log_L_supremum,
            U_supremum=U_supremum,
            model=outer_carry.state.model,
            args=outer_carry.state.args,
            params=outer_carry.state.params,
            termination_reason=outer_carry.state.termination_reason
        )

        return OuterCarry(key=key, state=state)

    init_outer_carry = OuterCarry(
        key=key,
        state=state
    )

    carry = jax.lax.while_loop(outer_cond_fn, outer_body_fn, init_outer_carry)
    return carry.state


@dataclasses.dataclass(slots=True)
class NestedSampler(PureDataclassPytree):
    model: Model
    root_allocation_degree: int | None = None
    max_samples: int | None = None  # how many initial samples to allow
    shell_size: int | None = None  # how many parents to select per iteration
    batch_size: int | None = None  # how many samples to replace at once with vmap
    args: tuple = ()
    params: CtxParams | None = None
    sampler: AbstractSampler | None = None
    allocation_target: Literal['uniform', 'evidence_improving', 'posterior_improving'] = 'uniform'

    def __post_init__(self):
        U_ndims = int(self.model.U_ndims(self.args, self.params))
        if self.root_allocation_degree is None:
            self.root_allocation_degree = 10 * U_ndims
        if self.shell_size is None:
            self.shell_size = max(1, self.root_allocation_degree // 2)
        if self.max_samples is None:
            self.max_samples = 100 * self.shell_size
        if self.sampler is None:
            self.sampler = UniDimSliceSampler(
                model=self.model,
                num_slices=max(1, 100 * U_ndims),
                phantom_burn_in=max(1, 20 * U_ndims),
                no_step_out=True,
                collect_phantom_samples=self.collect_phantom_samples
            )

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['target_num_live_points', 'max_samples', 'shell_size', 'store_phantom_samples',
                                        'batch_size'])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def run_until_goal(self, goal_cond: Callable[[State], bool], depth_cond: DepthCondition | None = None,
                       key: PRNGKey | None = None) -> State:
        """
        Perform nested sampling until goal condition is met.

        Args:
            goal_cond: a goal condition, any function of the state.
            depth_cond: a depth condition.

        Returns:
            the final state after running nested sampling, which can be used for evidence calculation or resuming.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        if depth_cond is None:
            depth_cond = DepthCondition(
                dlogZ=jnp.asarray(1e-2, dtype=mp_policy.measure_dtype),
                max_samples=jnp.asarray(self.max_samples, dtype=mp_policy.count_dtype)
            )
        return _run_until_goal(self, None, goal_cond, depth_cond, key)

    def resume_until_goal(self, state: State, goal_cond: Callable[[State], bool],
                          depth_cond: DepthCondition | None = None, key: PRNGKey | None = None) -> State:
        """
        Perform nested sampling until goal condition is met.

        Args:
            state: the current state to resume from.
            goal_cond: a goal condition, any function of the state.
            depth_cond: a depth condition.

        Returns:
            the final state after running nested sampling, which can be used for evidence calculation or resuming.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        if depth_cond is None:
            depth_cond = DepthCondition(
                dlogZ=jnp.asarray(1e-2, dtype=mp_policy.measure_dtype),
                max_samples=jnp.asarray(self.max_samples, dtype=mp_policy.count_dtype)
            )
        return _run_until_goal(self, state, goal_cond, depth_cond, key)

    def run_single_iteration(self, state: State | None, depth_cond: DepthCondition | None = None,
                             key: PRNGKey | None = None) -> State:
        if key is None:
            key = jax.random.PRNGKey(42)
        if depth_cond is None:
            depth_cond = DepthCondition(
                dlogZ=jnp.asarray(1e-2, dtype=mp_policy.measure_dtype),
                max_samples=jnp.asarray(self.max_samples, dtype=mp_policy.count_dtype)
            )
        return _run_single_iteration(state, depth_cond, key)


NestedSampler.register_pytree()


def _compute_allocation_target_fn(self: NestedSampler, state: State) -> AbstractAllocationTargetFn:
    if self.allocation_target == 'uniform':
        return UniformDepth(
            goal_loop_iter=state.goal_loop_iter,
            delta_allocation_degree=self.root_allocation_degree,
            root_allocation_degree=self.root_allocation_degree
        )
    elif self.allocation_target == 'evidence_improving':
        raise NotImplementedError(f"Evidence improving allocation target: {self.allocation_target}")
    elif self.allocation_target == 'posterior_improving':
        raise NotImplementedError(f"Posterior improving allocation target: {self.allocation_target}")
    else:
        raise ValueError(f'Unknown allocation target: {self.allocation_target}')



def _run_until_goal(self: NestedSampler, state: State | None, goal_cond: Callable[[State], bool],
                    depth_cond: DepthCondition,
                    key: PRNGKey) -> State:
    if state is None:
        key, init_key = jax.random.split(key)
        state = _sample_init_state(
            key=init_key,
            root_degree=int(self.root_allocation_degree),
            max_samples=int(self.max_samples),
            model=self.model,
            num_phantom=int(self.sampler.num_phantom()),
            args=self.args,
            params=self.params,
            store_phantom_samples=False,
            batch_size=self.batch_size
        )
    while not goal_cond(state):
        if state.num_samples == self.max_samples:
            # Resize, adding another 100 * shell_size
            self.max_samples += self.shell_size * 100
            state.samples = state.samples.resize(self.max_samples)
            depth_cond.max_samples = self.max_samples

        allocation_target_fn = _compute_allocation_target_fn(self, state)
        state = _run_single_iteration(
            key=key,
            state=state,
            depth_cond=depth_cond,
            allocation_target_fn=allocation_target_fn,
            sampler=self.sampler,
            shell_size=self.shell_size,
            batch_size=self.batch_size,
        )

    return state
