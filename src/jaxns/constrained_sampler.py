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


def _sample_direction(key: PRNGKey, u0: TreeField[UType]) -> TreeField[UType]:
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
        return TreeField(u0.ones_like())
    direction = TreeField(u0.random_normal_like(key))
    eps = jnp.asarray(1e-6, direction.norm().dtype)
    norm = jnp.maximum(eps, direction.norm())
    direction = TreeField(jax.tree.map(lambda x: x / norm, direction.tree))
    return direction


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
    leaf_dtype = jax.tree.leaves(point_U0.tree)[0].dtype
    zero = jnp.zeros((), leaf_dtype)
    one = jnp.ones((), leaf_dtype)
    inf = jnp.full((), jnp.inf, leaf_dtype)
    t1 = jax.tree.map(lambda p, d: (one - p) / d, point_U0.tree, direction.tree)
    t1_right = jax.tree.reduce(lambda x, y: jnp.minimum(jnp.min(jnp.where(x >= zero, x, inf)), jnp.min(jnp.where(y >= zero, y, inf))), t1,
                               initializer=jnp.inf)
    t1_left = jax.tree.reduce(lambda x, y: jnp.maximum(jnp.max(jnp.where(x <= zero, x, -inf)), jnp.max(jnp.where(y <= zero, y, -inf))), t1,
                              initializer=-jnp.inf)
    t0 = jax.tree.map(lambda p, d: -p / d, point_U0.tree, direction.tree)
    t0_right = jax.tree.reduce(lambda x, y: jnp.minimum(jnp.min(jnp.where(x >= zero, x, inf)), jnp.min(jnp.where(y >= zero, y, inf))), t0,
                               initializer=jnp.inf)
    t0_left = jax.tree.reduce(lambda x, y: jnp.maximum(jnp.max(jnp.where(x <= zero, x, -inf)), jnp.max(jnp.where(y <= zero, y, -inf))), t0,
                              initializer=-jnp.inf)
    right_bound = jnp.minimum(t0_right, t1_right)
    left_bound = jnp.maximum(t0_left, t1_left)
    return left_bound, right_bound


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
    leaf_dtype = jax.tree.leaves(point_U0.tree)[0].dtype
    t = left + random.uniform(key, dtype=leaf_dtype) * (right - left)
    point_U = TreeField(jax.tree.map(lambda p, d: p + t * d, point_U0.tree, direction.tree))
    return point_U, t


def _shrink_interval(t: FloatArray, left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Not successful proposal, so shrink, optionally apply exponential shrinkage.
    """
    zero = jnp.zeros_like(t)
    left = jnp.where(t < zero, t, left)
    right = jnp.where(t > zero, t, right)

    return left, right


def _new_proposal(
        key: PRNGKey,
        U0: TreeField[UType],
        direction: TreeField[UType],
        perfect: bool,
        gradient_guided: bool,
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
) -> tuple[TreeField[UType], FloatArray, IntArray, TreeField[UType]]:
    """
    Sample from a slice about a seed point.

    Args:
        key: PRNG key
        direction: the direction to sample along
        perfect: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
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
        key, t_key, shrink_key = random.split(carry.key, 3)
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

    run_key, n_key, t_key, after_key = random.split(key, 4)

    if perfect:
        (left, right) = _slice_bounds(
            point_U0=U0,
            direction=direction
        )
    else:
        # TODO: implement doubling step out
        raise NotImplementedError("Step out not implemented.")

    point_U, t = _pick_point_in_interval(
        key=t_key,
        point_U0=U0,
        direction=direction,
        left=left,
        right=right
    )
    log_L = log_likelihood_fn(point_U.tree).astype(log_L_constraint.dtype)
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
        raise NotImplementedError("Gradient guided slice sampler not implemented.")
    else:
        # Randomly choose a new direction
        direction = _sample_direction(after_key, direction)
    return carry.point_U, carry.log_L, num_likelihood_evaluations, direction


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
        perfect: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
            Otherwise, uses a doubling procedure (exponentially finding bracket).
            Note: Perfect is a misnomer, as perfection also depends on the number of slices between acceptance.
        gradient_guided: if true then do householder reflections at between proposals with a 50% probability.
    """

    model: Model
    num_slices: int
    perfect: bool = True
    gradient_guided: bool = False

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['num_slices', 'perfect', 'gradient_guided'])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)

    def _check(self):
        if self.num_slices < 1:
            raise ValueError(f"num_slices should be >= 1, got {self.num_slices}.")
        if not self.perfect:
            raise ValueError("Only perfect slice sampler is implemented.")
        if self.gradient_guided:
            warnings.warn("Gradient guided slice sampler is experimental and will likely change.")

    def num_phantom(self) -> int:
        return self.num_slices - 1

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

        def propose_op(carry: Carry, x: XType) -> Carry:
            U_sample, log_L, num_likelihood_evaluations, direction = _new_proposal(
                key=x.key,
                U0=carry.U_sample,
                direction=carry.direction,
                perfect=self.perfect,
                gradient_guided=self.gradient_guided,
                log_L_constraint=carry.log_L_constraint,
                log_likelihood_fn=log_likelihood_fn,
            )

            carry = Carry(
                U_sample=U_sample,
                log_L_constraint=carry.log_L_constraint,
                log_L=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations + carry.num_likelihood_evaluations,
                direction=direction
            )
            return carry

        direction_key, sample_key = jax.random.split(key, 2)

        direction = _sample_direction(direction_key, TreeField(seed_point.U0))

        init_carry = Carry(
            U_sample=TreeField(seed_point.U0),
            log_L_constraint=log_L_constraint,
            log_L=seed_point.log_L0,
            num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype),
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
        phantom_fraction = jax.tree.map(lambda x: x[:-1], cumulative_samples)
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
