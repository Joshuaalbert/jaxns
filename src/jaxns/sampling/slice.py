"""One-dimensional slice transitions and periodic chart geometry."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax import random

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import TreeField, pytree_ravel
from jaxns.sampling.ellipsoid import SamplerData
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType


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


def _periodic_tree(
        point: TreeField[UType],
        periodic: tuple[bool, ...],
) -> TreeField[UType]:
    """Expand static scalar flags to the structure of one U-space point."""
    leaves, structure = jax.tree.flatten(point.tree)
    masks = []
    offset = 0
    for leaf in leaves:
        size = int(np.prod(leaf.shape))
        # [...] mask is constant in the compiled program and aligned to leaf.
        mask = jnp.asarray(
            periodic[offset:offset + size],
            dtype=mp_policy.bool_dtype,
        ).reshape(leaf.shape)
        masks.append(mask)
        offset += size
    if offset != len(periodic):
        raise ValueError(
            "Periodic U-space topology does not match the sampled U shape."
        )
    return TreeField(jax.tree.unflatten(structure, masks))


def _open_chart(
        key: PRNGKey,
        point: TreeField[UType],
        periodic: tuple[bool, ...],
) -> tuple[TreeField[UType], TreeField[UType]]:
    """Translate periodic coordinates into an independent random chart."""
    offset = point.random_uniform_like(key)
    # The offset is state-independent. For a fixed offset this is a
    # measure-preserving torus translation, so mixing over offsets preserves
    # reversibility of the existing cube-bracket transition.
    return _to_chart(point, offset, periodic), offset


def _to_chart(
        point: TreeField[UType],
        offset: TreeField[UType],
        periodic: tuple[bool, ...],
) -> TreeField[UType]:
    """Map a canonical point into an already selected random chart."""
    mask = _periodic_tree(point, periodic)
    return TreeField(jax.tree.map(
        lambda value, shift, select: jnp.where(
            select,
            jnp.mod(value + shift, 1.0),
            value,
        ),
        point.tree,
        offset.tree,
        mask.tree,
    ))


def _close_chart(
        point: TreeField[UType],
        offset: TreeField[UType],
        periodic: tuple[bool, ...],
) -> TreeField[UType]:
    """Return a chart point to canonical U in the half-open unit cube."""
    mask = _periodic_tree(point, periodic)
    return TreeField(jax.tree.map(
        lambda value, shift, select: jnp.where(
            select,
            jnp.mod(value - shift, 1.0),
            value,
        ),
        point.tree,
        offset.tree,
        mask.tree,
    ))


def _slice_keys(
        key: PRNGKey,
        periodic: tuple[bool, ...],
) -> tuple[PRNGKey, PRNGKey, PRNGKey, PRNGKey, PRNGKey]:
    """Assign one independent chart key without perturbing the ordinary path."""
    chart_key = key
    if periodic:
        chart_key, key = random.split(key, 2)
    run_key, t_key, step_key, after_key = random.split(key, 4)
    return chart_key, run_key, t_key, step_key, after_key


def _sample_ellipsoidal_direction(
        key: PRNGKey,
        u0: TreeField[UType],
        log_L_constraint: FloatArray,
        data: SamplerData,
        prob_isotropic: float,
) -> TreeField[UType]:
    """Draw from fixed contour-eligible geometry or its isotropic fallback."""
    direction, _ = _draw_ellipsoidal_direction(
        key,
        u0,
        log_L_constraint,
        data,
        prob_isotropic,
    )
    return direction


def _draw_ellipsoidal_direction(
        key: PRNGKey,
        u0: TreeField[UType],
        log_L_constraint: FloatArray,
        data: SamplerData,
        prob_isotropic: float,
) -> tuple[TreeField[UType], BoolArray]:
    """Draw a direction and report whether the isotropic kernel was used."""
    isotropic_key, choice_key, component_key, normal_key = random.split(key, 4)
    eligible = data.valid & (data.log_L_max > log_L_constraint)
    use_isotropic = (
        jnp.logical_not(jnp.any(eligible))
        | (
            random.uniform(choice_key, dtype=mp_policy.measure_dtype)
            < jnp.asarray(prob_isotropic, mp_policy.measure_dtype)
        )
    )

    def draw_ellipsoidal(_):
        # Selection mass is geometric volume, not the EM mixture mass. The
        # latter describes the weighted sample population and would bias this
        # direction law towards densely sampled rather than spatially broad
        # components.
        logits = jnp.where(eligible, data.log_volumes, -jnp.inf)
        component = random.categorical(component_key, logits).astype(
            mp_policy.index_dtype
        )
        # The current values are intentionally ignored. Only their flattened
        # structure and dtype define the direction space, keeping this law
        # independent of the chain's current point.
        flat, unravel = pytree_ravel(u0.tree)
        normal = random.normal(normal_key, flat.shape, dtype=flat.dtype)
        rotation = data.rotations[component].astype(flat.dtype)
        radii = data.radii[component].astype(flat.dtype)
        transformed = rotation @ (
            radii * normal
        )
        norm = jnp.maximum(
            jnp.linalg.norm(transformed),
            jnp.asarray(1e-6, transformed.dtype),
        )
        return TreeField(unravel(transformed / norm))

    direction = jax.lax.cond(
        use_isotropic,
        lambda unused: _sample_direction(isotropic_key, u0),
        draw_ellipsoidal,
        operand=None,
    )
    return direction, use_isotropic


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


def _new_proposal(
        key: PRNGKey,
        U0: TreeField[UType],
        direction: TreeField[UType],
        slice_width: FloatArray,
        no_step_out: bool,
        gradient_guided: bool,
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        periodic: tuple[bool, ...] = (),
        sampler_data: SamplerData | None = None,
        prob_isotropic: float = 1.0,
) -> tuple[
    TreeField[UType],
    FloatArray,
    IntArray,
    TreeField[UType],
    FloatArray,
    BoolArray,
]:
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
            right=carry.right,
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

    chart_key, run_key, t_key, step_key, after_key = _slice_keys(
        key,
        periodic,
    )
    chart_offset = None
    if periodic:
        # Perfect bracketing remains finite because it runs in a randomly cut
        # cube chart. Likelihoods and retained samples always use canonical U.
        U0, chart_offset = _open_chart(chart_key, U0, periodic)
        canonical_log_likelihood_fn = log_likelihood_fn

        def log_likelihood_fn(value):
            canonical = _close_chart(
                TreeField(value),
                chart_offset,
                periodic,
            )
            return canonical_log_likelihood_fn(canonical.tree)

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
        if sampler_data is None:
            direction = _sample_direction(after_key, direction)
            direction_isotropic = jnp.asarray(True, mp_policy.bool_dtype)
        else:
            direction, direction_isotropic = _draw_ellipsoidal_direction(
                after_key,
                direction,
                log_L_constraint,
                sampler_data,
                prob_isotropic,
            )
    next_slice_width = 2 * (carry.right - carry.left)
    point_U = carry.point_U
    if periodic:
        point_U = _close_chart(point_U, chart_offset, periodic)
    return (
        point_U,
        carry.log_L,
        num_likelihood_evaluations,
        direction,
        next_slice_width,
        direction_isotropic,
    )

