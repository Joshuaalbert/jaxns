import dataclasses
from typing import NamedTuple, Tuple, Any, Callable

import jax
from jax import numpy as jnp, random

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.cumulative_ops import cumulative_op_static
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.pytree_utils import tree_dot
from jaxns.internals.random import sample_uniformly_masked
from jaxns.internals.types import PRNGKey, FloatArray, BoolArray, IntArray, UType
from jaxns.nested_samplers.common.types import Sample, SampleCollection, LivePointCollection
from jaxns.samplers.abc import EphemeralState
from jaxns.samplers.bases import SeedPoint, BaseAbstractMarkovSampler

__all__ = [
    'UniDimSliceSampler'
]


def _sample_direction(n_key: PRNGKey, ndim: int) -> FloatArray:
    """
    Choose a direction randomly from S^(D-1).

    Args:
        n_key: PRNG key
        ndim: int, number of dimentions

    Returns:
        direction: [D] direction from S^(D-1)
    """
    if ndim == 1:
        return jnp.ones((), mp_policy.measure_dtype)
    direction = random.normal(n_key, shape=(ndim,), dtype=mp_policy.measure_dtype)
    direction /= jnp.linalg.norm(direction)
    return direction


def _slice_bounds(point_U0: FloatArray, direction: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """
    Compute the slice bounds, t, where point_U0 + direction * t intersects uit cube boundary.

    Args:
        point_U0: [D]
        direction: [D]

    Returns:
        left_bound: left most point (<= 0).
        right_bound: right most point (>= 0).
    """
    zero = jnp.zeros((), mp_policy.measure_dtype)
    one = jnp.ones((), mp_policy.measure_dtype)
    inf = jnp.full((), jnp.inf, mp_policy.measure_dtype)
    t1 = (one - point_U0) / direction
    t1_right = jnp.min(jnp.where(t1 >= zero, t1, inf))
    t1_left = jnp.max(jnp.where(t1 <= zero, t1, -inf))
    t0 = -point_U0 / direction
    t0_right = jnp.min(jnp.where(t0 >= zero, t0, inf))
    t0_left = jnp.max(jnp.where(t0 <= zero, t0, -inf))
    right_bound = jnp.minimum(t0_right, t1_right)
    left_bound = jnp.maximum(t0_left, t1_left)
    return left_bound, right_bound


def _pick_point_in_interval(key: PRNGKey, point_U0: FloatArray, direction: FloatArray, left: FloatArray,
                            right: FloatArray) -> Tuple[FloatArray, FloatArray]:
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
    u = random.uniform(key, dtype=mp_policy.measure_dtype)
    t = left + u * (right - left)
    point_U = point_U0 + t * direction
    # close_to_zero = (left >= -10*jnp.finfo(left.dtype).eps) & (right <= 10*jnp.finfo(right.dtype).eps)
    # point_U = jnp.where(close_to_zero, point_U0, point_U)
    # t = jnp.where(close_to_zero, jnp.zeros_like(t), t)
    return point_U, t


def _shrink_interval(key: PRNGKey, t: FloatArray, left: FloatArray, right: FloatArray,
                     midpoint_shrink: bool, alpha: jax.Array) -> Tuple[FloatArray, FloatArray]:
    """
    Not successful proposal, so shrink, optionally apply exponential shrinkage.
    """
    zero = jnp.zeros_like(t)
    # witout exponential shrinkage, we shrink to failed proposal point, which is 100% correct.
    left = jnp.where(t < zero, t, left)
    right = jnp.where(t > zero, t, right)

    if midpoint_shrink:
        # For this to be correct it must be invariant to monotonic rescaling of the likelihood.
        # Therefore, it must only use the knowledge of ordering of the likelihoods.
        # Basic version: shrink to midpoint of interval, i.e. alpha = 0.5.
        # Extended version: shrink to random point in interval.
        # do_midpoint_shrink = random.uniform(key) < 0.5
        # alpha = 1  # 0.8  # random.uniform(key)
        left = jnp.where((t < zero), alpha * left, left)
        right = jnp.where((t > zero), alpha * right, right)
    return left, right


def _new_proposal(
        key: PRNGKey,
        seed_point: SeedPoint,
        direction: UType,
        midpoint_shrink: bool,
        alpha: jax.Array,
        perfect: bool,
        gradient_slice: bool,
        gradient_guided: bool,
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable,
        grad_fn: Callable
) -> Tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """
    Sample from a slice about a seed point.

    Args:
        key: PRNG key
        seed_point: the seed point to sample from
        direction: the direction to sample along
        midpoint_shrink: if true then contract to the midpoint of interval on rejection. Otherwise, normal contract
        alpha: exponential shrinkage factor
        perfect: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
        gradient_slice: if true the slice along gradient direction
        gradient_guided: if true then do householder reflections
        log_L_constraint: the constraint to sample within
        log_likelihood_fn: the log-likelihood function
        grad_fn: the gradient function

    Returns:
        point_U: the new sample
        momentum: the new momentum
        log_L: the log-likelihood of the new sample
        num_likelihood_evaluations: the number of likelihood evaluations performed
    """

    class Carry(NamedTuple):
        key: PRNGKey
        direction: FloatArray
        left: FloatArray
        right: FloatArray
        t: FloatArray
        point_U: UType
        log_L: FloatArray
        num_likelihood_evaluations: IntArray

    def cond(carry: Carry) -> BoolArray:
        satisfaction = carry.log_L > log_L_constraint
        # Allow if on plateau to fly around the plateau for a while
        lesser_satisfaction = jnp.bitwise_and(seed_point.log_L0 == log_L_constraint, carry.log_L == log_L_constraint)
        # done = jnp.bitwise_or(jnp.bitwise_or(close_to_zero_interval, satisfaction), lesser_satisfaction)
        done = jnp.bitwise_or(satisfaction, lesser_satisfaction)
        return jnp.bitwise_not(done)

    def body(carry: Carry) -> Carry:
        key, t_key, shrink_key = random.split(carry.key, 3)
        left, right = _shrink_interval(
            key=shrink_key,
            t=carry.t,
            left=carry.left,
            right=carry.right,
            midpoint_shrink=midpoint_shrink,
            alpha=alpha
        )
        point_U, t = _pick_point_in_interval(
            key=t_key,
            point_U0=seed_point.U0,
            direction=carry.direction,
            left=left,
            right=right
        )
        log_L = log_likelihood_fn(point_U)
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

    run_key, n_key, t_key, after_key = random.split(key, 4)
    if gradient_slice:
        # climb the gradient
        grad = grad_fn(seed_point.U0)
        num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
        grad_norm = jnp.linalg.norm(grad)
        grad_mask = jnp.bitwise_or(jnp.equal(grad_norm, jnp.zeros_like(grad_norm)), ~jnp.isfinite(grad_norm))
        grad_dir = grad / grad_norm
        direction = jnp.where(grad_mask, direction, grad_dir)
        (left, right) = _slice_bounds(
            point_U0=seed_point.U0,
            direction=direction
        )
        left = jnp.where(grad_mask, left, jnp.zeros_like(left))
    else:

        if perfect:
            (left, right) = _slice_bounds(
                point_U0=seed_point.U0,
                direction=direction
            )
        else:
            # TODO: implement doubling step out
            raise NotImplementedError("Step out not implemented.")

    point_U, t = _pick_point_in_interval(
        key=t_key,
        point_U0=seed_point.U0,
        direction=direction,
        left=left,
        right=right
    )
    log_L = log_likelihood_fn(point_U)
    num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
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
        # Perform a Householder reflection at the accepted point
        after_key1, after_key2 = jax.random.split(after_key, 2)
        grad = grad_fn(carry.point_U)
        num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
        grad_norm = jnp.linalg.norm(grad)
        grad_mask = jnp.bitwise_or(jnp.equal(grad_norm, jnp.zeros_like(grad_norm)), ~jnp.isfinite(grad_norm))
        grad = grad / grad_norm

        reflect_direction = direction - 2 * tree_dot(direction, grad) * grad
        reflect_direction /= jnp.linalg.norm(reflect_direction)

        random_direction = _sample_direction(after_key1, direction.size)

        choose_dir = jax.random.randint(after_key2, shape=(), minval=0, maxval=2)
        direction = jnp.where(
            choose_dir == 0,
            reflect_direction,
            random_direction
        )
        direction = jnp.where(grad_mask, random_direction, direction)
    else:
        # Randomly choose a new direction
        direction = _sample_direction(after_key, direction.size)
    return carry.point_U, direction, carry.log_L, num_likelihood_evaluations


@dataclasses.dataclass(eq=False)
class UniDimSliceSampler(BaseAbstractMarkovSampler[SampleCollection]):
    """
    Slice sampler for a single dimension.

    Args:
        model: AbstractModel
        num_slices: number of slices between acceptance. Note: some other software use units of prior dimension.
        midpoint_shrink: if true then contract to the midpoint of interval on rejection. Otherwise, contract to
            rejection point. Speeds up convergence, but introduces minor auto-correlation.
        num_phantom_save: number of phantom samples to save. Phantom samples are samples that meeting the constraint
            but are not accepted. They can be used for numerous things, e.g. to estimate the evidence uncertainty.
        perfect: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
            Otherwise, uses a doubling procedure (exponentially finding bracket).
            Note: Perfect is a misnomer, as perfection also depends on the number of slices between acceptance.
        gradient_slice: if true then always slice along increasing gradient direction.
        adaptive_shrink: if true then shrink interval to random point in interval, rather than midpoint.
    """

    model: BaseAbstractModel
    num_slices: int
    num_phantom_save: int
    midpoint_shrink: bool
    perfect: bool
    gradient_slice: bool = False
    adaptive_shrink: bool = False
    gradient_guided: bool = False

    def __post_init__(self):
        if self.num_slices < 1:
            raise ValueError(f"num_slices should be >= 1, got {self.num_slices}.")
        if self.num_phantom_save < 0:
            raise ValueError(f"num_phantom_save should be >= 0, got {self.num_phantom_save}.")
        if self.num_phantom_save >= self.num_slices:
            raise ValueError(
                f"num_phantom_save should be < num_slices, got {self.num_phantom_save} >= {self.num_slices}.")
        self.num_slices = int(self.num_slices)
        self.num_phantom_save = int(self.num_phantom_save)
        self.midpoint_shrink = bool(self.midpoint_shrink)
        self.perfect = bool(self.perfect)
        self.gradient_slice = bool(self.gradient_slice)
        self.adaptive_shrink = bool(self.adaptive_shrink)
        self.gradient_guided = bool(self.gradient_guided)
        if self.adaptive_shrink:
            raise NotImplementedError("Adaptive shrinkage not implemented.")
        if not self.perfect:
            raise ValueError("Only perfect slice sampler is implemented.")

    def num_phantom(self) -> int:
        return self.num_phantom_save

    def _pre_process(self, ephemeral_state: EphemeralState) -> Any:
        if self.perfect:  # nothing needed
            return ephemeral_state.live_points_collection
        else:  # TODO: step out with doubling
            return ephemeral_state.live_points_collection

    def _post_process(self, ephemeral_state: EphemeralState,
                      sampler_state: Any) -> Any:
        if self.perfect:  # nothing needed
            return ephemeral_state.live_points_collection
        else:  # TODO: step out with doubling
            return ephemeral_state.live_points_collection

    def get_seed_point(self, key: PRNGKey, sampler_state: LivePointCollection,
                       log_L_constraint: FloatArray) -> SeedPoint:

        sample_collection = sampler_state

        select_mask = sample_collection.log_L > log_L_constraint

        seed_point = sample_uniformly_masked(
            key=key,
            v=SeedPoint(U0=sample_collection.U_sample, log_L0=sample_collection.log_L),
            select_mask=select_mask,
            num_samples=1,
            squeeze=True
        )

        return seed_point

    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             sampler_state: SampleCollection) -> Tuple[Sample, Sample]:

        class XType(NamedTuple):
            key: jax.Array
            alpha: jax.Array

        log_likelihood_fn = self.model.forward
        grad_fn = jax.grad(log_likelihood_fn)

        class Carry(NamedTuple):
            sample: Sample
            direction: UType

        def propose_op(carry: Carry, x: XType) -> Carry:
            U_sample, direction, log_L, num_likelihood_evaluations = _new_proposal(
                key=x.key,
                seed_point=SeedPoint(
                    U0=carry.sample.U_sample,
                    log_L0=carry.sample.log_L
                ),
                direction=carry.direction,
                midpoint_shrink=self.midpoint_shrink,
                alpha=x.alpha,
                perfect=self.perfect,
                gradient_slice=self.gradient_slice,
                gradient_guided=self.gradient_guided,
                log_L_constraint=carry.sample.log_L_constraint,
                log_likelihood_fn=log_likelihood_fn,
                grad_fn=grad_fn
            )

            carry = Carry(
                sample=Sample(
                    U_sample=U_sample,
                    log_L_constraint=carry.sample.log_L_constraint,
                    log_L=log_L,
                    num_likelihood_evaluations=num_likelihood_evaluations + carry.sample.num_likelihood_evaluations
                ),
                direction=direction
            )
            return carry

        init_sample = Sample(
            U_sample=seed_point.U0,
            log_L_constraint=log_L_constraint,
            log_L=seed_point.log_L0,
            num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype)
        )

        direction_key, sample_key = jax.random.split(key, 2)

        n = init_sample.U_sample.size
        direction = _sample_direction(direction_key, n)

        init_carry = Carry(
            sample=init_sample,
            direction=direction
        )

        xs = XType(
            key=random.split(sample_key, self.num_slices),
            alpha=jnp.linspace(0.5, 1., self.num_slices)
        )
        final_carry, cumulative_samples = cumulative_op_static(
            op=propose_op,
            init=init_carry,
            xs=xs
        )

        # Last sample is the final sample, the rest are potential phantom samples
        # Take only the last num_phantom_save phantom samples
        phantom_samples: Sample = jax.tree.map(lambda x: x[-(self.num_phantom_save + 1):-1], cumulative_samples.sample)

        phantom_samples = phantom_samples._replace(
            num_likelihood_evaluations=jnp.full(
                phantom_samples.num_likelihood_evaluations.shape,
                0,
                mp_policy.count_dtype
            )
        )
        return final_carry.sample, phantom_samples
