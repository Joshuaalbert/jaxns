import dataclasses
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, NamedTuple, Any

import jax
from jax import numpy as jnp, random

from jaxns.cumulative_ops import cumulative_op_static
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, TreeField
from jaxns.random_utils import sample_uniformly_masked
from jaxns.samples import Samples, SeedPoint, PhantomSamples
from jaxns.types import FloatArray, IntArray, PRNGKey, UType, BoolArray


class AbstractSampler(ABC):
    """
    Performs sampling from the prior within a likelihood constraint, to produce i.i.d. samples for nested sampling.
    The sampler is assumed to be stateless and pure.
    """

    @abstractmethod
    def num_phantom(self) -> int:
        """
        Get the number of phantom samples to produce per real sample. Note that the number of phantom samples may be less than this if the sampler fails to produce enough valid phantom samples, but it will never be more than this.

        Returns:
            the number of phantom samples to produce per real sample.
        """
        ...

    @abstractmethod
    def get_sample(self, key, log_L_constraint: FloatArray, seed_point: SeedPoint, args=(), params=None) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            seed_point: a seed point to begin sampling from

        Returns:
            U_sample: an i.i.d. sample within the constraint
            log_L: the log-likelihood of the sample
            num_likelihood_evaluations: number of likelihood evaluations used to produce the sample
            phantom_samples: samples that satisfy the constraint but were not accepted. Can be used for various things, e.g. estimating evidence uncertainty.
        """
        ...

@partial(jax.jit, inline=True)
def _sample_direction(key: PRNGKey, u0: TreeField[UType], radii: TreeField[UType] | None = None, rotation: UType | None = None) -> TreeField[UType]:
    """
    Choose a direction randomly from S^(D-1).

    Args:
        key: PRNG key
        u0: a point in the sample space, used to determine the shape of the direction.

    Returns:
        direction: [D] direction from S^(D-1)
    """
    ndim = u0.ndim()
    if ndim == 1:
        return u0.ones_like()
    direction = u0.random_normal_like(key)
    if radii is not None:
        direction = radii * direction
    if rotation is not None:
        direction = rotation @ direction
    eps = jnp.asarray(1e-6, direction.norm().dtype)
    norm = jnp.maximum(eps, direction.norm())
    return direction / norm

@partial(jax.jit, inline=True)
def _slice_bounds(point_U0: TreeField[UType], direction: TreeField[UType]) -> tuple[FloatArray, FloatArray]:
    """
    Compute the slice bounds, t, where point_U0 + direction * t intersects uit cube boundary.

    Args:
        point_U0: starting point of slice
        direction: direction of slice

    Returns:
        left_bound: left most point (<= 0).
        right_bound: right most point (>= 0).
    """
    leaf_dtype = jax.tree.leaves(point_U0)[0].dtype
    zero = jnp.zeros((), leaf_dtype)
    one = jnp.ones((), leaf_dtype)
    inf = jnp.full((), jnp.inf, leaf_dtype)
    t1 = (one - point_U0) / direction
    t1_right = jax.tree.reduce(lambda x, y: jnp.minimum(jnp.min(jnp.where(x >= zero, x, inf)), jnp.min(jnp.where(y >= zero, y, inf))), t1,
                               initializer=jnp.inf)
    t1_left = jax.tree.reduce(lambda x, y: jnp.maximum(jnp.max(jnp.where(x <= zero, x, -inf)), jnp.max(jnp.where(y <= zero, y, -inf))), t1,
                              initializer=-jnp.inf)
    t0 = -point_U0 / direction
    t0_right = jax.tree.reduce(lambda x, y: jnp.minimum(jnp.min(jnp.where(x >= zero, x, inf)), jnp.min(jnp.where(y >= zero, y, inf))), t0,
                               initializer=jnp.inf)
    t0_left = jax.tree.reduce(lambda x, y: jnp.maximum(jnp.max(jnp.where(x <= zero, x, -inf)), jnp.max(jnp.where(y <= zero, y, -inf))), t0,
                              initializer=-jnp.inf)
    right_bound = jnp.minimum(t0_right, t1_right)
    left_bound = jnp.maximum(t0_left, t1_left)
    return left_bound, right_bound

@partial(jax.jit, inline=True)
def _pick_point_in_interval(key: PRNGKey, point_U0: TreeField[UType], direction: TreeField[UType], left: FloatArray,
                            right: FloatArray) -> tuple[TreeField[UType], FloatArray]:
    """
    Select a point along slice in [point_U0 + direction * left, point_U0 + direction * right]

    Args:
        key: PRNG key
        point_U0: [D]
        direction: [D]
        left: left most point (<= 0).
        right: right most point (>= 0).

    Returns:
        point_U: [D]
        t: selection point between [left, right]
    """
    leaf_dtype = jax.tree.leaves(point_U0)[0].dtype
    t = left + random.uniform(key, dtype=leaf_dtype) * (right - left)
    point_U = point_U0 + direction * t
    return point_U, t

@partial(jax.jit, inline=True)
def _shrink_interval(t: FloatArray, left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Not successful proposal, so shrink, optionally apply exponential shrinkage.
    """
    zero = jnp.zeros_like(t)
    left = jnp.where(t < zero, t, left)
    right = jnp.where(t > zero, t, right)

    return left, right

@partial(jax.jit, inline=True)
def _new_proposal(
        key: PRNGKey,
        U0: TreeField[UType],
        direction: TreeField[UType],
        slice_width: FloatArray,
        no_step_out: bool,
        gradient_guided: bool,
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
) -> tuple[TreeField[UType], FloatArray, IntArray, TreeField[UType], FloatArray]:
    """
    Sample from a slice about a seed point.

    Args:
        key: PRNG key
        direction: the direction to sample along
        no_step_out: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
        gradient_guided: if true then do householder reflections
        log_L_constraint: the constraint to sample within
        log_likelihood_fn: the log-likelihood function

    Returns:
        point_U: the new sample
        log_L: the log-likelihood of the new sample
        num_likelihood_evaluations: the number of likelihood evaluations performed
    """

    class Carry(NamedTuple):
        key: PRNGKey
        direction: TreeField[UType]
        left: FloatArray
        right: FloatArray
        t: FloatArray
        point_U: TreeField[UType]
        log_L: FloatArray
        num_likelihood_evaluations: IntArray

    def cond(carry: Carry) -> BoolArray:
        satisfaction = carry.log_L > log_L_constraint
        return jnp.bitwise_not(satisfaction)

    def body(carry: Carry) -> Carry:
        key, t_key = random.split(carry.key, 2)
        left, right = _shrink_interval(
            t=carry.t,
            left=carry.left,
            right=carry.right
        )
        point_U, t = _pick_point_in_interval(
            key=t_key,
            point_U0=U0,
            direction=carry.direction,
            left=left,
            right=right
        )
        log_L = log_likelihood_fn(point_U.tree).astype(log_L_constraint.dtype)
        num_likelihood_evaluations = carry.num_likelihood_evaluations + jnp.ones_like(carry.num_likelihood_evaluations)
        return Carry(
            key=key,
            t=t,
            left=left,
            right=right,
            point_U=point_U,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
            direction=carry.direction
        )

    # Chose the direction to go
    num_likelihood_evaluations = jnp.full((), 0, mp_policy.count_dtype)

    run_key, t_key, step_key, after_key = random.split(key, 4)

    (left_bound, right_bound) = _slice_bounds(
        point_U0=U0,
        direction=direction
    )
    slice_width = jnp.asarray(slice_width, left_bound.dtype)

    if no_step_out:
        left, right = left_bound, right_bound
        step_out_num_likelihood_evaluations = jnp.asarray(0, mp_policy.count_dtype)
    else:
        class StepOutCarry(NamedTuple):
            key: PRNGKey
            left: FloatArray
            right: FloatArray
            left_outside_slice: BoolArray
            right_outside_slice: BoolArray
            num_likelihood_evaluations: IntArray

        eps = jnp.asarray(1e-12, left_bound.dtype)
        use_full_slice = jnp.isinf(slice_width)
        effective_slice_width = jnp.maximum(slice_width, eps)
        place_key, step_key = random.split(step_key)
        uniform_origin = random.uniform(place_key, dtype=left_bound.dtype)
        initial_left = -uniform_origin * effective_slice_width
        initial_right = initial_left + effective_slice_width

        left = jnp.where(use_full_slice, left_bound, jnp.maximum(left_bound, initial_left))
        right = jnp.where(use_full_slice, right_bound, jnp.minimum(right_bound, initial_right))

        def _point_at_t(t: FloatArray) -> TreeField[UType]:
            return U0 + direction * t

        def step_out_cond(carry: StepOutCarry) -> BoolArray:
            can_expand_left = carry.left > left_bound
            can_expand_right = carry.right < right_bound
            both_outside_box = jnp.bitwise_not(jnp.bitwise_or(can_expand_left, can_expand_right))
            both_outside_slice = jnp.bitwise_and(carry.left_outside_slice, carry.right_outside_slice)
            return jnp.bitwise_not(jnp.bitwise_or(both_outside_box, both_outside_slice))

        def step_out_body(carry: StepOutCarry) -> StepOutCarry:
            key, choose_key = random.split(carry.key, 2)

            can_expand_left = carry.left > left_bound
            can_expand_right = carry.right < right_bound

            choose_left_random = random.uniform(choose_key, dtype=left_bound.dtype) < jnp.where(
                can_expand_left & can_expand_right, 0.5, jnp.where(can_expand_left, 1., 0.))

            current_width = jnp.maximum(carry.right - carry.left, eps)
            candidate_left = jnp.maximum(left_bound, carry.left - current_width)
            candidate_right = jnp.minimum(right_bound, carry.right + current_width)

            t_eval = jnp.where(choose_left_random, candidate_left, candidate_right)

            next_left = jnp.where(choose_left_random, candidate_left, carry.left)
            next_right = jnp.where(choose_left_random, carry.right, candidate_right)

            log_L_eval = log_likelihood_fn(_point_at_t(t_eval).tree)
            outside_slice_eval = log_L_eval <= log_L_constraint
            next_left_outside_slice = jnp.where(choose_left_random, outside_slice_eval, carry.left_outside_slice)
            next_right_outside_slice = jnp.where(choose_left_random, carry.right_outside_slice, outside_slice_eval)
            num_likelihood_evaluations = carry.num_likelihood_evaluations + jnp.ones_like(carry.num_likelihood_evaluations)

            return StepOutCarry(
                key=key,
                left=next_left,
                right=next_right,
                left_outside_slice=next_left_outside_slice,
                right_outside_slice=next_right_outside_slice,
                num_likelihood_evaluations=num_likelihood_evaluations
            )

        left_outside_slice = log_likelihood_fn(_point_at_t(left).tree) <= log_L_constraint
        right_outside_slice = log_likelihood_fn(_point_at_t(right).tree) <= log_L_constraint
        step_out_init = StepOutCarry(
            key=step_key,
            left=left,
            right=right,
            left_outside_slice=left_outside_slice,
            right_outside_slice=right_outside_slice,
            num_likelihood_evaluations=jnp.asarray(2, mp_policy.count_dtype)
        )

        step_out_carry = jax.lax.while_loop(
            cond_fun=lambda c: jnp.bitwise_and(jnp.bitwise_not(use_full_slice), step_out_cond(c)),
            body_fun=step_out_body,
            init_val=step_out_init
        )
        left, right = step_out_carry.left, step_out_carry.right
        step_out_num_likelihood_evaluations = step_out_carry.num_likelihood_evaluations

    point_U, t = _pick_point_in_interval(
        key=t_key,
        point_U0=U0,
        direction=direction,
        left=left,
        right=right
    )
    log_L = log_likelihood_fn(point_U.tree).astype(log_L_constraint.dtype)
    num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
    num_likelihood_evaluations += step_out_num_likelihood_evaluations
    init_carry = Carry(
        key=run_key,
        direction=direction,
        left=left,
        right=right,
        t=t,
        point_U=point_U,
        log_L=log_L,
        num_likelihood_evaluations=num_likelihood_evaluations
    )

    carry = jax.lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=init_carry
    )

    # Update direction
    direction = carry.direction
    num_likelihood_evaluations = carry.num_likelihood_evaluations
    if gradient_guided:
        # Perform HMC with Householder reflections
        raise NotImplementedError("Gradient guided slice sampler not implemented.")
    else:
        # Randomly choose a new direction
        direction = _sample_direction(after_key, direction)
    next_slice_width = 2 * (carry.right - carry.left)
    return carry.point_U, carry.log_L, num_likelihood_evaluations, direction, next_slice_width


@partial(jax.jit, inline=True)
def get_seed_point(key: PRNGKey, samples: Samples, log_L_constraint: FloatArray) -> SeedPoint:
    """
    Get a seed point from samples that satisfies the likelihood constraint. This is done by masking samples that do not satisfy the constraint, and sampling uniformly from the remaining samples.

    Args:
        key: PRNGKey
        samples: samples to select from.
        log_L_constraint: the constraint to sample within.

    Returns:
        seed point that satisfy the constraint.
    """
    select_mask = samples.log_likelihoods > log_L_constraint
    seed_point = sample_uniformly_masked(
        key=key,
        v=SeedPoint(U0=samples.U_samples, log_L0=samples.log_likelihoods),
        select_mask=select_mask,
        num_samples=1,
        squeeze=True
    )
    return seed_point


@dataclasses.dataclass(slots=True, frozen=True)
class UniDimSliceSampler(AbstractSampler, PureDataclassPytree):
    """
    Slice sampler for a single dimension.

    Args:
        model: AbstractModel
        num_slices: number of slices between acceptance. Note: some other software use units of prior dimension.
        no_step_out: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
            Otherwise, uses a doubling procedure (exponentially finding bracket).
            Note: Perfect is a misnomer, as perfection also depends on the number of slices between acceptance.
        gradient_guided: if true then do HMC with householder reflections.
        collect_phantom_samples: if true, then collect phantom samples
    """

    model: Model
    num_slices: int
    no_step_out: bool = True
    gradient_guided: bool = False
    collect_phantom_samples: bool = False
    phantom_burn_in: int | None = None

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['num_slices', 'no_step_out', 'gradient_guided', 'collect_phantom_samples', 'phantom_burn_in'])


    def _check(self):
        if self.num_slices < 1:
            raise ValueError(f"num_slices should be >= 1, got {self.num_slices}.")
        if self.gradient_guided:
            warnings.warn("Gradient guided slice sampler is experimental and will likely change.")

    def num_phantom(self) -> int:
        if self.collect_phantom_samples:
            if self.phantom_burn_in is None:
                burn_in = int(self.num_slices * 0.1)
            else:
                burn_in = self.phantom_burn_in
            return self.num_slices - 1 - burn_in
        return 0

    def get_sample(self, key, log_L_constraint: FloatArray, seed_point: SeedPoint, args=(), params=None) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:

        class XType(NamedTuple):
            key: jax.Array
            alpha: jax.Array

        log_likelihood_fn = lambda U: self.model.log_likelihood(U, args=args, params=params, allow_nan=False)
        grad_fn = jax.grad(log_likelihood_fn)

        class Carry(NamedTuple):
            U_sample: TreeField[UType]
            log_L_constraint: FloatArray
            log_L: FloatArray
            num_likelihood_evaluations: IntArray
            direction: TreeField[UType]
            slice_width: FloatArray

        def propose_op(carry: Carry, x: XType) -> Carry:
            U_sample, log_L, num_likelihood_evaluations, direction, slice_width = _new_proposal(
                key=x.key,
                U0=carry.U_sample,
                direction=carry.direction,
                slice_width=carry.slice_width,
                no_step_out=self.no_step_out,
                gradient_guided=self.gradient_guided,
                log_L_constraint=carry.log_L_constraint,
                log_likelihood_fn=log_likelihood_fn,
            )

            carry = Carry(
                U_sample=U_sample,
                log_L_constraint=carry.log_L_constraint,
                log_L=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations + carry.num_likelihood_evaluations,
                direction=direction,
                slice_width=slice_width
            )
            return carry

        direction_key, sample_key = jax.random.split(key, 2)

        init_direction = _sample_direction(direction_key, TreeField(seed_point.U0))
        slice_width_dtype = jax.tree.leaves(seed_point.U0)[0].dtype

        #### initial proposal to get slice width for cumulative op with perfect stepout
        sample_key, init_sample_key = random.split(sample_key, 2)

        U_sample, log_L, num_likelihood_evaluations, init_direction, slice_width = _new_proposal(
            key=init_sample_key,
            U0=TreeField(seed_point.U0),
            direction=init_direction,
            slice_width=jnp.asarray(jnp.inf, slice_width_dtype),
            no_step_out=True,
            gradient_guided=self.gradient_guided,
            log_L_constraint=log_L_constraint,
            log_likelihood_fn=log_likelihood_fn,
        )

        init_carry = Carry(
            U_sample=U_sample,
            log_L_constraint=log_L_constraint,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
            direction=init_direction,
            slice_width=slice_width
        )

        xs = XType(
            key=random.split(sample_key, self.num_slices - 1),
            alpha=jnp.linspace(0.5, 1., self.num_slices - 1)
        )
        final_carry, cumulative_samples = cumulative_op_static(
            op=propose_op,
            init=init_carry,
            xs=xs
        )

        # concat initial sample to cumulative samples
        cumulative_samples = jax.tree.map(
            lambda x, y: jnp.concatenate([x[None], y], axis=0),
            init_carry,
            cumulative_samples
        )

        # Last sample is the final sample, the rest are potential phantom samples
        # Take only the last num_phantom_save phantom samples
        assert self.num_phantom() <= self.num_slices - 1, "num_phantom() should be in [0, num_slices - 1]"

        phantom_fraction = jax.tree.map(lambda x: x[self.num_slices - 1 - self.num_phantom():-1], cumulative_samples)
        phantom_samples = PhantomSamples(
            U_samples=phantom_fraction.U_sample.tree,
            log_L=phantom_fraction.log_L,
            valid_mask=jnp.ones(phantom_fraction.log_L.shape, mp_policy.bool_dtype)
        )

        U_sample = final_carry.U_sample.tree
        log_L_sample = final_carry.log_L
        num_likelihood_evaluations = final_carry.num_likelihood_evaluations

        return U_sample, log_L_sample, num_likelihood_evaluations, phantom_samples


UniDimSliceSampler.register_pytree()
