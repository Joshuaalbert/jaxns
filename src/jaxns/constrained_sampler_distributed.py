import dataclasses
import warnings
from typing import Any

import jax
from jax import numpy as jnp, random

from jaxns.constrained_sampler import (
    AbstractSampler,
    _pick_point_in_interval,
    _sample_direction,
    _shrink_interval,
    _slice_bounds,
)
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import TreeField, PureDataclassPytree
from jaxns.samples import PhantomSamples, SeedPoint


def _evaluate_log_likelihood(evaluator, u, dtype):
    return jnp.asarray(evaluator.evaluate(u), dtype=dtype)


def _distributed_new_proposal(
        key,
        evaluator,
        U0: TreeField,
        direction: TreeField,
        slice_width,
        no_step_out: bool,
        gradient_guided: bool,
        log_L_constraint,
):
    if gradient_guided:
        raise NotImplementedError("Gradient guided distributed slice sampler not implemented.")

    num_likelihood_evaluations = jnp.asarray(0, mp_policy.count_dtype)
    run_key, t_key, step_key, after_key = random.split(key, 4)
    left_bound, right_bound = _slice_bounds(point_U0=U0, direction=direction)
    slice_width = jnp.asarray(slice_width, dtype=left_bound.dtype)

    if no_step_out:
        left, right = left_bound, right_bound
        step_out_num_likelihood_evaluations = jnp.asarray(0, mp_policy.count_dtype)
    else:
        eps = jnp.asarray(1e-12, left_bound.dtype)
        use_full_slice = bool(jnp.isinf(slice_width))
        effective_slice_width = jnp.maximum(slice_width, eps)
        place_key, step_key = random.split(step_key)
        uniform_origin = random.uniform(place_key, dtype=left_bound.dtype)
        initial_left = -uniform_origin * effective_slice_width
        initial_right = initial_left + effective_slice_width

        left = jnp.where(use_full_slice, left_bound, jnp.maximum(left_bound, initial_left))
        right = jnp.where(use_full_slice, right_bound, jnp.minimum(right_bound, initial_right))

        def _point_at_t(t):
            return U0 + direction * t

        left_outside_slice = bool(
            _evaluate_log_likelihood(evaluator, _point_at_t(left).tree, log_L_constraint.dtype) <= log_L_constraint,
        )
        right_outside_slice = bool(
            _evaluate_log_likelihood(evaluator, _point_at_t(right).tree, log_L_constraint.dtype) <= log_L_constraint,
        )
        step_out_num_likelihood_evaluations = jnp.asarray(2, mp_policy.count_dtype)

        while not use_full_slice:
            can_expand_left = bool(left > left_bound)
            can_expand_right = bool(right < right_bound)
            both_outside_box = not (can_expand_left or can_expand_right)
            both_outside_slice = left_outside_slice and right_outside_slice
            if both_outside_box or both_outside_slice:
                break

            step_key, choose_key = random.split(step_key)
            if can_expand_left and can_expand_right:
                choose_left_random = bool(random.uniform(choose_key, dtype=left_bound.dtype) < 0.5)
            else:
                choose_left_random = can_expand_left

            current_width = jnp.maximum(right - left, eps)
            candidate_left = jnp.maximum(left_bound, left - current_width)
            candidate_right = jnp.minimum(right_bound, right + current_width)
            t_eval = candidate_left if choose_left_random else candidate_right
            outside_slice_eval = bool(
                _evaluate_log_likelihood(evaluator, _point_at_t(t_eval).tree, log_L_constraint.dtype)
                <= log_L_constraint,
            )
            if choose_left_random:
                left = candidate_left
                left_outside_slice = outside_slice_eval
            else:
                right = candidate_right
                right_outside_slice = outside_slice_eval
            step_out_num_likelihood_evaluations += jnp.asarray(1, mp_policy.count_dtype)

    point_U, t = _pick_point_in_interval(
        key=t_key,
        point_U0=U0,
        direction=direction,
        left=left,
        right=right,
    )
    log_L = _evaluate_log_likelihood(evaluator, point_U.tree, log_L_constraint.dtype)
    num_likelihood_evaluations += jnp.asarray(1, mp_policy.count_dtype)
    num_likelihood_evaluations += step_out_num_likelihood_evaluations

    while bool(log_L <= log_L_constraint):
        run_key, t_key = random.split(run_key)
        left, right = _shrink_interval(t=t, left=left, right=right)
        point_U, t = _pick_point_in_interval(
            key=t_key,
            point_U0=U0,
            direction=direction,
            left=left,
            right=right,
        )
        log_L = _evaluate_log_likelihood(evaluator, point_U.tree, log_L_constraint.dtype)
        num_likelihood_evaluations += jnp.asarray(1, mp_policy.count_dtype)

    direction = _sample_direction(after_key, direction)
    next_slice_width = jnp.asarray(2.0, left.dtype) * (right - left)
    return point_U.tree, log_L, num_likelihood_evaluations, direction, next_slice_width


@dataclasses.dataclass(slots=True)
class DistributedUniDimSliceSampler(AbstractSampler, PureDataclassPytree):
    model: Model
    evaluator: Any
    num_slices: int
    no_step_out: bool = True
    gradient_guided: bool = False
    collect_phantom_samples: bool = False
    phantom_burn_in: int | None = None

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, ['num_slices', 'no_step_out', 'gradient_guided', 'collect_phantom_samples', 'phantom_burn_in'])


    def __post_init__(self):
        if self.num_slices < 1:
            raise ValueError(f"num_slices should be >= 1, got {self.num_slices}.")
        if self.gradient_guided:
            warnings.warn("Gradient guided slice sampler is experimental and will likely change.")

    def num_phantom(self) -> int:
        if self.collect_phantom_samples:
            burn_in = int(self.num_slices * 0.1) if self.phantom_burn_in is None else self.phantom_burn_in
            return self.num_slices - 1 - burn_in
        return 0

    def get_sample(self, key, log_L_constraint, seed_point: SeedPoint, args=(), params=None):
        del args, params
        direction_key, sample_key = jax.random.split(key, 2)
        init_direction = _sample_direction(direction_key, TreeField(seed_point.U0))
        slice_width_dtype = jax.tree.leaves(seed_point.U0)[0].dtype

        sample_key, init_sample_key = random.split(sample_key, 2)
        U_sample, log_L, num_likelihood_evaluations, direction, slice_width = _distributed_new_proposal(
            key=init_sample_key,
            evaluator=self.evaluator,
            U0=TreeField(seed_point.U0),
            direction=init_direction,
            slice_width=jnp.asarray(jnp.inf, dtype=slice_width_dtype),
            no_step_out=True,
            gradient_guided=self.gradient_guided,
            log_L_constraint=log_L_constraint,
        )

        all_u_samples = [U_sample]
        all_log_l = [log_L]
        num_likelihood_evaluations = jnp.asarray(num_likelihood_evaluations, mp_policy.count_dtype)

        proposal_keys = random.split(sample_key, max(0, self.num_slices - 1))
        current_u = TreeField(U_sample)
        current_log_l = log_L
        current_direction = direction
        current_slice_width = slice_width
        for proposal_key in proposal_keys:
            current_u_tree, current_log_l, delta_num_likelihood_evaluations, current_direction, current_slice_width = (
                _distributed_new_proposal(
                    key=proposal_key,
                    evaluator=self.evaluator,
                    U0=current_u,
                    direction=current_direction,
                    slice_width=current_slice_width,
                    no_step_out=self.no_step_out,
                    gradient_guided=self.gradient_guided,
                    log_L_constraint=log_L_constraint,
                )
            )
            current_u = TreeField(current_u_tree)
            all_u_samples.append(current_u_tree)
            all_log_l.append(current_log_l)
            num_likelihood_evaluations += delta_num_likelihood_evaluations

        num_phantom = self.num_phantom()
        if num_phantom == 0:
            phantom_u_samples = jax.tree.map(
                lambda x: jnp.zeros((0,) + x.shape, dtype=x.dtype),
                all_u_samples[-1],
            )
            phantom_log_l = jnp.zeros((0,), dtype=log_L_constraint.dtype)
            phantom_valid_mask = jnp.zeros((0,), dtype=mp_policy.bool_dtype)
        else:
            phantom_start = self.num_slices - 1 - num_phantom
            phantom_u_samples = jax.tree.map(
                lambda *xs: jnp.stack(xs, axis=0),
                *all_u_samples[phantom_start:-1],
            )
            phantom_log_l = jnp.stack(all_log_l[phantom_start:-1], axis=0)
            phantom_valid_mask = jnp.ones((num_phantom,), dtype=mp_policy.bool_dtype)

        phantom_samples = PhantomSamples(
            U_samples=phantom_u_samples,
            log_L=phantom_log_l,
            valid_mask=phantom_valid_mask,
        )
        return all_u_samples[-1], all_log_l[-1], num_likelihood_evaluations, phantom_samples
