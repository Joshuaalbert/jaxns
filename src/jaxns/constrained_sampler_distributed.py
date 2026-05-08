import dataclasses
from typing import Any

import jax
from jax import numpy as jnp, random

from jaxns.constrained_sampler import (
    AbstractSampler,
    GALILEAN_TRAJECTORIES,
    _as_mode_name,
    _pick_point_in_interval,
    _resolve_max_shrinkage_steps,
    _resolve_num_slices,
    _resolve_phantom_burn_in,
    _resolve_positive_limit,
    _sample_direction_from_kernel,
    _sample_galilean_markov_transition,
    _sample_direction,
    _shrink_interval,
    _slice_bounds,
    _validate_trajectory_mode,
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
        log_L0=None,
        max_shrinkage_steps: int = 32,
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

    found_proposal = bool(log_L > log_L_constraint)
    for _ in range(max_shrinkage_steps):
        if found_proposal:
            break
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
        found_proposal = bool(log_L > log_L_constraint)

    if not found_proposal:
        point_U = U0
        if log_L0 is None:
            log_L = _evaluate_log_likelihood(
                evaluator,
                U0.tree,
                log_L_constraint.dtype,
            )
            num_likelihood_evaluations += jnp.asarray(1, mp_policy.count_dtype)
        else:
            log_L = jnp.asarray(log_L0, dtype=log_L_constraint.dtype)

    direction = _sample_direction(after_key, direction)
    next_slice_width = jnp.asarray(2.0, left.dtype) * (right - left)
    return point_U.tree, log_L, num_likelihood_evaluations, direction, next_slice_width


@dataclasses.dataclass(slots=True)
class DistributedUniDimSliceSampler(AbstractSampler, PureDataclassPytree):
    """
    Worker-backed one-dimensional slice sampler for strict constrained priors.

    The sampler requires ``seed_point.log_L0 > log_L_constraint`` and rejects
    equality with the parent contour. Execution code must supply a valid seed or
    use its sentinel fallback before invoking the sampler.

    Phantom collection follows the v3 likelihood-only contract: retained
    phantoms are post-burn-in chain-state likelihood diagnostics with validity
    structure. Phantom coordinates are not stored, and phantoms do not feed
    posterior samples or classic race-sample counts.

    Args:
        model: model defining the constrained prior.
        evaluator: worker-backed likelihood evaluator.
        num_slices: number of slice transitions used to produce one classic
            sample.
        no_step_out: if true, use maximal slice bounds and shrinkage without
            the doubling step-out procedure.
        gradient_guided: if true, request gradient-guided transitions. This is
            currently unsupported for the distributed sampler.
        collect_phantom_samples: if true, retain post-burn-in phantom
            likelihood diagnostics.
        phantom_burn_in: number of initial chain states to exclude before
            retaining phantom likelihood diagnostics. Must satisfy
            ``0 <= burn_in <= num_slices - 1`` so ``num_phantom()`` is never
            negative.
    """

    model: Model
    evaluator: Any
    num_slices: int
    no_step_out: bool = True
    gradient_guided: bool = False
    collect_phantom_samples: bool = False
    phantom_burn_in: int | None = None
    trajectory: object = "straight_line"
    max_shrinkage_steps: int = 32
    galilean_initial_step_size: float = 0.05
    max_galilean_reflections: int = 64
    max_galilean_step_halvings: int = 32
    max_galilean_step_doublings: int = 32

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(
            this,
            [
                'num_slices',
                'no_step_out',
                'gradient_guided',
                'collect_phantom_samples',
                'phantom_burn_in',
                'trajectory',
                'max_shrinkage_steps',
                'galilean_initial_step_size',
                'max_galilean_reflections',
                'max_galilean_step_halvings',
                'max_galilean_step_doublings',
            ],
        )


    def __post_init__(self):
        _resolve_phantom_burn_in(self.num_slices, self.phantom_burn_in)
        _resolve_max_shrinkage_steps(self.max_shrinkage_steps)
        _resolve_positive_limit(
            self.max_galilean_reflections,
            "max_galilean_reflections",
        )
        _resolve_positive_limit(
            self.max_galilean_step_halvings,
            "max_galilean_step_halvings",
        )
        _resolve_positive_limit(
            self.max_galilean_step_doublings,
            "max_galilean_step_doublings",
        )
        _validate_trajectory_mode(self.trajectory)
        if self.gradient_guided:
            raise NotImplementedError(
                "The legacy gradient_guided flag is ambiguous with explicit "
                "trajectory modes. Use the Ticket 0012 Galilean trajectory "
                "once it is implemented."
            )

    def num_phantom(self) -> int:
        if self.collect_phantom_samples:
            burn_in = _resolve_phantom_burn_in(
                self.num_slices,
                self.phantom_burn_in,
            )
            return _resolve_num_slices(self.num_slices) - 1 - burn_in
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            args=(),
            params=None,
            adaptation_context=None,
    ):
        del args, params, adaptation_context
        if not bool(jnp.asarray(seed_point.log_L0 > log_L_constraint)):
            raise ValueError("Seed point must satisfy the strict likelihood constraint.")
        num_slices = _resolve_num_slices(self.num_slices)
        trajectory_mode = _as_mode_name(self.trajectory)
        if trajectory_mode in GALILEAN_TRAJECTORIES:
            if hasattr(self.evaluator, "grad_log_likelihood"):
                grad_log_likelihood_fn = self.evaluator.grad_log_likelihood
            elif hasattr(self.evaluator, "gradient"):
                grad_log_likelihood_fn = self.evaluator.gradient
            else:
                raise NotImplementedError(
                    "Distributed Galilean sampling requires evaluator."
                    "grad_log_likelihood or evaluator.gradient."
                )

            def log_likelihood_fn(u):
                return _evaluate_log_likelihood(
                    self.evaluator,
                    u,
                    log_L_constraint.dtype,
                )

            all_u_samples = []
            all_log_l = []
            num_likelihood_evaluations = jnp.asarray(
                0,
                mp_policy.count_dtype,
            )
            direction_template = TreeField(seed_point.U0)
            current_u = direction_template
            proposal_keys = random.split(key, num_slices)
            direction_keys = random.split(random.fold_in(key, 1), num_slices)

            for proposal_key, direction_key in zip(
                    proposal_keys,
                    direction_keys,
            ):
                direction = _sample_direction_from_kernel(
                    key=direction_key,
                    direction_kernel="isotropic",
                    current_point=direction_template,
                )
                current_u, current_log_l, delta_num_evaluations = (
                    _sample_galilean_markov_transition(
                        key=proposal_key,
                        U0=current_u,
                        direction=direction,
                        log_L_constraint=log_L_constraint,
                        log_likelihood_fn=log_likelihood_fn,
                        grad_log_likelihood_fn=grad_log_likelihood_fn,
                        initial_step_size=jnp.asarray(
                            self.galilean_initial_step_size,
                            dtype=log_L_constraint.dtype,
                        ),
                        max_reflections=self.max_galilean_reflections,
                        max_step_halvings=self.max_galilean_step_halvings,
                        max_step_doublings=self.max_galilean_step_doublings,
                    )
                )
                all_u_samples.append(current_u.tree)
                all_log_l.append(current_log_l)
                num_likelihood_evaluations += delta_num_evaluations

            num_phantom = self.num_phantom()
            if num_phantom == 0:
                phantom_log_l = jnp.zeros((0,), dtype=log_L_constraint.dtype)
                phantom_valid_mask = jnp.zeros((0,), dtype=mp_policy.bool_dtype)
            else:
                phantom_start = num_slices - 1 - num_phantom
                phantom_log_l = jnp.stack(all_log_l[phantom_start:-1], axis=0)
                phantom_valid_mask = jnp.ones((num_phantom,), dtype=mp_policy.bool_dtype)

            phantom_samples = PhantomSamples(
                U_samples=None,
                log_L=phantom_log_l,
                valid_mask=phantom_valid_mask,
            )
            return all_u_samples[-1], all_log_l[-1], num_likelihood_evaluations, phantom_samples

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
            log_L0=seed_point.log_L0,
            max_shrinkage_steps=self.max_shrinkage_steps,
        )

        all_u_samples = [U_sample]
        all_log_l = [log_L]
        num_likelihood_evaluations = jnp.asarray(num_likelihood_evaluations, mp_policy.count_dtype)

        proposal_keys = random.split(sample_key, max(0, num_slices - 1))
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
                    log_L0=current_log_l,
                    max_shrinkage_steps=self.max_shrinkage_steps,
                )
            )
            current_u = TreeField(current_u_tree)
            all_u_samples.append(current_u_tree)
            all_log_l.append(current_log_l)
            num_likelihood_evaluations += delta_num_likelihood_evaluations

        num_phantom = self.num_phantom()
        if num_phantom == 0:
            phantom_log_l = jnp.zeros((0,), dtype=log_L_constraint.dtype)
            phantom_valid_mask = jnp.zeros((0,), dtype=mp_policy.bool_dtype)
        else:
            # Retain only post-burn-in likelihood diagnostics. The distributed
            # sampler intentionally discards phantom coordinates so they cannot
            # feed posterior or classic race samples.
            phantom_start = num_slices - 1 - num_phantom
            phantom_log_l = jnp.stack(all_log_l[phantom_start:-1], axis=0)
            phantom_valid_mask = jnp.ones((num_phantom,), dtype=mp_policy.bool_dtype)

        phantom_samples = PhantomSamples(
            U_samples=None,
            log_L=phantom_log_l,
            valid_mask=phantom_valid_mask,
        )
        return all_u_samples[-1], all_log_l[-1], num_likelihood_evaluations, phantom_samples
