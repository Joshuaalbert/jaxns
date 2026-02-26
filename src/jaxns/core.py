import dataclasses
from functools import partial
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.random
from jaxctx import CtxParams

from jaxns.constrained_sampler import AbstractSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample_indicies
from jaxns.samples import Samples, SeedPoint, PhantomSamples
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition
from jaxns.types import IntArray, BoolArray, PRNGKey


@partial(jax.jit, inline=True, static_argnames=['num_live_points', 'num_phantom', 'max_samples', 'store_phantom_samples', 'batch_size'])
def _sample_init_state(key, num_live_points: int, max_samples: int, model: Model, num_phantom: int = 0, args=(),
                       params: CtxParams | None = None, store_phantom_samples: bool = False, batch_size: int | None = None) -> State:
    def single_sample(key):
        key, subkey = jax.random.split(key)
        U_sample = model.sample_U(subkey)
        log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False)
        num_likelihood_evaluations = jnp.array(1, dtype=mp_policy.count_dtype)
        carry = (key, U_sample, log_L, num_likelihood_evaluations)

        def cond_fn(carry):
            _, _, log_L, _ = carry
            return log_L <= -jnp.inf

        def body_fn(carry):
            key, _, _, num_likelihood_evaluations = carry
            key, subkey = jax.random.split(key)
            U_sample = model.sample_U(subkey)
            log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False)
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
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations,
        phantom_samples=phantom_samples
    )

    sample_atom = Samples(
        log_likelihoods=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        out_degree=jnp.asarray(0, mp_policy.count_dtype),
        num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype),
        U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), samples.U_samples),
        phantom_samples=PhantomSamples(
            U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), samples.phantom_samples.U_samples),
            log_L=jnp.zeros_like(samples.phantom_samples.log_L[0]),
            valid_mask=jnp.zeros_like(samples.phantom_samples.valid_mask[0])
        )
    )
    if not store_phantom_samples:
        sample_atom.phantom_samples.U_samples = None

    samples = _concat(samples, sample_atom).sort()

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


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points', 'shell_size', 'batch_size'])
def _run_ns(key, state: State, target_num_live_points: int, shell_size: int, args=(),
            sampler: AbstractSampler | None = None,
            params=None,
            termination_condition: TerminationCondition | None = None,
            batch_size: int | None = None) -> State:
    """
    Perform a single nested sampling run.

    Args:
        key: PRNG key
        target_num_live_points: the number of live points to use off root
        shell_size: the number of samples to discard and replenish per iteration
        args: arguments to pass to the model
        sampler: the sampler to use to produce i.i.d. samples within likelihood constraints
        params: parameters to pass to the model
        termination_condition: the termination condition to use
        batch_size: how many likelihood evaluations to batch together

    Returns:
        A final state object containing all samples and relevant information for resuming, or evidence calculation.
    """

    # Algorithm
    # repeat until termination condition:
    # choose constraints: compute the recurrence K[i] = K[i-1] - 1 + d[i], and choose indexes where K[i] < num_live_points. I.e. we make each sample have at least a certain number of live points.
    # choose seeds: the any points with likelihoods > the contour, else reparent off root
    # sample (in parallel)

    class OuterCarry(NamedTuple):
        key: jax.Array
        state: State

    def outer_cond_fn(carry: OuterCarry) -> BoolArray:
        register = carry.state.compute_termination_register(target_num_live_points=target_num_live_points)
        done, _ = register.is_done(termination_condition)
        return jnp.logical_not(done)

    def outer_body_fn(outer_carry: OuterCarry) -> OuterCarry:
        # Select likelihood constraints to achieve minimum K[i]>target (randomly without replacement)
        K_per_sample = outer_carry.state.samples.compute_num_live_points_per_sample(
            root_out_degree=outer_carry.state.root_out_degree,
            num_samples=outer_carry.state.num_samples
        )
        # K[i] = K[i-1] - 1 + d(i)
        # so K[i-1] = K[i] + 1 - d(i) is the number of live points just before shrinkage
        K_per_sample_before = K_per_sample + 1 - outer_carry.state.samples.out_degree
        select_weights = jnp.where(
            jnp.logical_and(
                jnp.arange(K_per_sample_before.shape[0]) < outer_carry.state.num_samples, K_per_sample_before < target_num_live_points
            ), 0, -jnp.inf)
        select_contours_key, key = jax.random.split(outer_carry.key, 2)
        parent_idxs = resample_indicies(select_contours_key, log_weights=select_weights, S=shell_size, replace=False)  # [S]
        log_L_constraints = outer_carry.state.samples.log_likelihoods[parent_idxs]  # [S]

        def get_sample(key, log_L_constraint, parent_idx: IntArray):
            seed_key, sample_key = jax.random.split(key)
            # Get seed from samples
            i_start = jax.lax.while_loop(
                lambda i: (i < outer_carry.state.num_samples) & (outer_carry.state.samples.log_likelihoods[i] <= log_L_constraint),
                lambda i: i + 1,
                parent_idx + 1
            )
            no_seeds = i_start == outer_carry.state.num_samples
            log_L_constraint = jnp.where(no_seeds, jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype), log_L_constraint)
            delta_root_out_degree = jnp.where(no_seeds, 1, 0).astype(mp_policy.count_dtype)
            delta_parent_out_degree = jnp.where(no_seeds, 0, 1).astype(mp_policy.count_dtype)
            i_start = jnp.where(no_seeds, 0, i_start).astype(mp_policy.index_dtype)
            seed_select_idx = jax.random.randint(seed_key, (), i_start, outer_carry.state.num_samples)
            seed_point = SeedPoint(
                U0=outer_carry.state.samples.U_samples[seed_select_idx],
                log_L0=outer_carry.state.samples.log_likelihoods[seed_select_idx]
            )
            return sampler.get_sample(
                sample_key, log_L_constraint, seed_point, args=args, params=params,
            ), (delta_root_out_degree, delta_parent_out_degree)

        key, subkey = jax.random.split(outer_carry.key)
        (U_samples, log_likelihoods, num_likelihood_evaluations, phantom_samples), (delta_root_out_degree, delta_parent_out_degree) = jax.lax.map(
            lambda key: get_sample(key, log_L_constraints, parent_idxs),
            jax.random.split(subkey, shell_size),
            batch_size=batch_size
        )

        new_samples = Samples(
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations,
            phantom_samples=phantom_samples
        )

        candidate_supremum_candidate_iid = jnp.argmax(new_samples.log_likelihoods)
        candidate_log_L_supremum = new_samples.log_likelihoods[candidate_supremum_candidate_iid]
        candidate_U_supremum = jax.tree.map(lambda u: u[candidate_supremum_candidate_iid], new_samples.U_samples)

        log_L_supremum = jnp.where(candidate_log_L_supremum > outer_carry.state.log_L_supremum, candidate_log_L_supremum, outer_carry.state.log_L_supremum)
        U_supremum = jax.tree.map(lambda u_new, u_old: jnp.where(candidate_log_L_supremum > outer_carry.state.log_L_supremum, u_new, u_old),
                                  candidate_U_supremum, outer_carry.state.U_supremum)
        if new_samples.phantom_samples.U_samples is not None:
            # flatten, and select only valid ones
            valid_mask = new_samples.phantom_samples.valid_mask.reshape((-1,))
            phantom_log_L = new_samples.phantom_samples.log_L.reshape((-1,))
            phantom_U_samples = jax.tree.map(lambda u: u.reshape((-1,) + u.shape[2:]), new_samples.phantom_samples.U_samples)
            phantom_log_L = jnp.where(valid_mask, phantom_log_L, -jnp.inf)
            candidate_supremum_candidate_phantom = jnp.argmax(phantom_log_L)
            candidate_log_L_supremum_phantom = phantom_log_L[candidate_supremum_candidate_phantom]
            candidate_U_supremum_phantom = jax.tree.map(lambda u: u[candidate_supremum_candidate_phantom], phantom_U_samples)
            log_L_supremum = jnp.where(candidate_log_L_supremum_phantom > log_L_supremum, candidate_log_L_supremum_phantom, log_L_supremum)
            U_supremum = jax.tree.map(lambda u_phantom, u_current: jnp.where(candidate_log_L_supremum_phantom > log_L_supremum, u_phantom, u_current),
                                      candidate_U_supremum_phantom, U_supremum)

        state = State(
            root_out_degree=outer_carry.state.root_out_degree + delta_root_out_degree,
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
    target_num_live_points: int | None = None
    max_samples: int | None = None
    shell_size: int | None = None
    args: tuple = ()
    params: CtxParams | None = None
    sampler: AbstractSampler | None = None
    termination_condition: TerminationCondition | None = None
    store_phantom_samples: bool = False
    batch_size: int | None = None

    def __post_init__(self):
        U_ndims = 0
        if self.target_num_live_points is None or self.max_samples is None or self.shell_size is None:
            U_ndims = int(self.model.U_ndims(self.args, self.params))
        if self.target_num_live_points is None:
            self.target_num_live_points = 100 * U_ndims
        if self.max_samples is None:
            self.max_samples = 10000 * U_ndims
        if self.shell_size is None:
            self.shell_size = max(1, self.target_num_live_points // 2)
        if self.termination_condition is None:
            self.termination_condition = TerminationCondition(dlogZ=1e-2)

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['target_num_live_points', 'max_samples', 'shell_size', 'store_phantom_samples', 'batch_size'])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def run(self, key: PRNGKey | None = None) -> State:
        """
        Creates an initial state, and performs sampling until the termination condition is met, returning the final state.

        Args:
            key: PRNGKey to use for sampling

        Returns:
            the final state after running nested sampling, which can be used for evidence calculation or resuming.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        return _run(self, key)

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
    key, init_key = jax.random.split(key)
    state = _sample_init_state(
        key=init_key,
        num_live_points=int(self.target_num_live_points),
        max_samples=int(self.max_samples),
        model=self.model,
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
