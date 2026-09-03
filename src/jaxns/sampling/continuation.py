"""Barrier-free continuation of fixed-width slice-sampling batches."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp
from jax import random

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree, TreeField
from jaxns.samples import PhantomSamples
from jaxns.sampling.protocol import (
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
)
from jaxns.sampling.slice import (
    _close_chart,
    _draw_ellipsoidal_direction,
    _open_chart,
    _pick_point_in_interval,
    _sample_direction,
    _shrink_interval,
    _slice_bounds,
    _slice_keys,
    _to_chart,
)
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType

if TYPE_CHECKING:
    from jaxns.constrained_sampler import UniDimSliceSampler


@dataclasses.dataclass(slots=True, frozen=True)
class SliceBatchState(PureDataclassPytree):
    """Logical slice chains continued between batched likelihood calls."""

    anchor: UType  # [S, ...] last accepted point in each chain
    chart_offset: UType | None  # [S, ...] periodic chart shift; otherwise None
    log_likelihood: FloatArray  # [S]
    directions: UType  # [S, T, ...] direction for each transition
    direction_is_isotropic: BoolArray  # [S, T]
    left: FloatArray  # [S]
    right: FloatArray  # [S]
    t: FloatArray  # [S]
    proposal: UType  # [S, ...] point awaiting likelihood evaluation
    run_key: PRNGKey  # [S, 2]
    transition_keys: PRNGKey  # [S, T, 2]
    transition_index: IntArray  # [S]
    done: BoolArray  # [S]
    num_likelihood_evaluations: IntArray  # [S] logical calls only
    phantom_samples: UType  # [S, P, ...] retained start prefix
    phantom_log_likelihoods: FloatArray  # [S, P]


SliceBatchState.register_pytree()


def _initialise_slice_chains(
        sampler: UniDimSliceSampler,
        request: ConstrainedSampleRequest,
) -> SliceBatchState:
    """Prepare one unevaluated proposal per logical slice chain."""
    num_slices = sampler.num_slices
    num_phantom = sampler.num_phantom()

    def draw_direction_stream(
            key,
            log_L_constraint,
            seed_u,
            direction_data,
    ):
        def draw_direction(direction_key):
            if direction_data is None:
                return (
                    _sample_direction(direction_key, TreeField(seed_u)),
                    jnp.asarray(True, mp_policy.bool_dtype),
                )
            return _draw_ellipsoidal_direction(
                direction_key,
                TreeField(seed_u),
                log_L_constraint,
                direction_data,
            )

        # Preserve the scalar sampler's random-key schedule exactly. Pool
        # execution order must never change a logical chain's random stream.
        direction_key, sample_key = random.split(key, 2)
        initial_direction, initial_is_isotropic = draw_direction(
            direction_key
        )
        sample_key, initial_key = random.split(sample_key, 2)
        later_keys = random.split(sample_key, num_slices - 1)
        transition_keys = jnp.concatenate(
            [initial_key[None], later_keys],
            axis=0,
        )

        # Direction proposals depend on the fixed parent contour and unit-cube
        # structure, not on accepted chain coordinates. Hoisting this stream
        # avoids evaluating both direction branches inside a vmapped condition
        # for every rejected proposal, while preserving the scalar direction
        # law and the requirement that every transition uses the same law.
        # The compile, memory, and execution trade-off is recorded in
        # benchmarks/issue_244/REPORT.md.
        if num_slices == 1:
            directions = jax.tree.map(
                lambda value: value[None],
                initial_direction.tree,
            )
            directions_are_isotropic = initial_is_isotropic[None]
        else:
            def draw_later_direction(transition_key):
                _, _, _, _, after_key = _slice_keys(
                    transition_key,
                    sampler._periodic,
                )
                return draw_direction(after_key)

            # The scalar reference draws one direction per scan step. Keep
            # that transition ordering so XLA performs the same matrix-vector
            # reductions for ellipsoidal directions; batching the transition
            # axis itself can change floating-point reduction order enough to
            # send a difficult chain down a different slice path.
            def scan_direction(unused, transition_key):
                return unused, draw_later_direction(transition_key)

            _, (
                later_directions,
                later_is_isotropic,
            ) = jax.lax.scan(
                scan_direction,
                None,
                transition_keys[:-1],
            )
            directions = jax.tree.map(
                lambda initial, later: jnp.concatenate(
                    [initial[None], later],
                    axis=0,
                ),
                initial_direction.tree,
                later_directions.tree,
            )
            directions_are_isotropic = jnp.concatenate(
                [initial_is_isotropic[None], later_is_isotropic],
                axis=0,
            )

        # Per-chain shapes are [T, ...], [T], and [T, 2], respectively.
        return directions, directions_are_isotropic, transition_keys

    def draw_direction_streams(direction_data):
        return jax.vmap(
            lambda key, constraint, seed_u: draw_direction_stream(
                key,
                constraint,
                seed_u,
                direction_data,
            )
        )(
            request.keys,
            request.log_L_constraints,
            request.seed_points.U0,
        )

    sampler_data = request.sampler_data
    if sampler_data is None:
        # [S, T, ...], [S, T], [S, T, 2]
        directions, directions_are_isotropic, transition_keys = (
            draw_direction_streams(None)
        )
    else:
        # Enabled is shared by the whole request. Keep this condition outside
        # both lane batching and transition scans so a retained disabled fit
        # runs the exact plain-isotropic key stream without evaluating any GMM
        # selection or matrix operations. XLA still compiles both branches,
        # because users may toggle the same state without discarding its fit.
        # Outputs are [S, T, ...], [S, T], and [S, T, 2], respectively.
        directions, directions_are_isotropic, transition_keys = jax.lax.cond(
            sampler_data.enabled,
            lambda unused: draw_direction_streams(sampler_data),
            lambda unused: draw_direction_streams(None),
            operand=None,
        )

    def initialise_one(
            valid,
            log_L_constraint,
            seed_u,
            seed_log_likelihood,
            chain_directions,
            chain_directions_are_isotropic,
            chain_transition_keys,
    ):

        # Perfect bracketing always has exactly one proposal ready. A rejected
        # likelihood shrinks this interval; an accepted likelihood advances
        # this logical chain to its next slice transition without a barrier.
        chart_key, run_key, t_key, _, _ = _slice_keys(
            chain_transition_keys[0],
            sampler._periodic,
        )
        chart_offset = None
        chart_seed = TreeField(seed_u)
        if sampler._periodic:
            chart_seed, chart_offset = _open_chart(
                chart_key,
                chart_seed,
                sampler._periodic,
            )
        initial_direction = TreeField(  # [...]
            jax.tree.map(lambda values: values[0], chain_directions)
        )
        left, right = _slice_bounds(
            chart_seed,
            initial_direction,
        )
        proposal, t = _pick_point_in_interval(
            t_key,
            chart_seed,
            initial_direction,
            left,
            right,
        )
        if sampler._periodic:
            proposal = _close_chart(
                proposal,
                chart_offset,
                sampler._periodic,
            )
        phantom_samples = jax.tree.map(
            lambda value: jnp.zeros(
                (num_phantom,) + value.shape,
                value.dtype,
            ),
            seed_u,
        )
        return SliceBatchState(
            anchor=seed_u,
            chart_offset=(
                None if chart_offset is None else chart_offset.tree
            ),
            log_likelihood=jnp.asarray(
                seed_log_likelihood,
                log_L_constraint.dtype,
            ),
            directions=chain_directions,
            direction_is_isotropic=chain_directions_are_isotropic,
            left=left,
            right=right,
            t=t,
            proposal=proposal.tree,
            run_key=run_key,
            transition_keys=chain_transition_keys,
            transition_index=jnp.asarray(0, mp_policy.index_dtype),
            # Static scheduler padding is not a logical chain. Mark it done
            # immediately so an unused lane cannot extend the physical loop;
            # it still occupies the fixed likelihood width as neutral filler
            # while any valid chain remains active.
            done=jnp.bitwise_not(valid),
            num_likelihood_evaluations=jnp.asarray(
                0,
                mp_policy.count_dtype,
            ),
            phantom_samples=phantom_samples,
            phantom_log_likelihoods=jnp.full(
                (num_phantom,),
                -jnp.inf,
                log_L_constraint.dtype,
            ),
        )

    return jax.vmap(initialise_one)(
        request.valid,
        request.log_L_constraints,
        request.seed_points.U0,
        request.seed_points.log_L0,
        directions,
        directions_are_isotropic,
        transition_keys,
    )


def _continue_slice_chains(
        sampler: UniDimSliceSampler,
        request: ConstrainedSampleRequest,
        *,
        args=(),
        params=None,
) -> ConstrainedSampleBatch:
    """Batch ready likelihoods while each logical chain advances freely.

    The physical likelihood width equals the logical request width. A finished
    lane evaluates the neutral unit-cube point ``U=0.5`` until the slowest
    complete chain finishes. These filler calls are substantially fewer than
    the masked calls introduced by a barrier after every slice transition,
    while the reported counter remains the exact logical count.

    As with the complete-chain reference, the enclosing local or worker
    program owns JIT compilation so registered non-array session objects remain
    captured Python closure values.
    """
    num_chains = request.log_L_constraints.shape[0]
    num_slices = sampler.num_slices
    num_phantom = sampler.num_phantom()
    assert 0 <= num_phantom <= num_slices - 1, (
        "num_phantom() should be in [0, num_slices - 1]"
    )

    def evaluate_one(u_sample):
        return sampler.model.log_likelihood(
            u_sample,
            args=args,
            params=params,
            allow_nan=False,
        ).astype(request.log_L_constraints.dtype)

    def prepare_transition(
            chain,
            anchor,
            direction,
            transition_index,
    ):
        transition_key = chain.transition_keys[transition_index]
        chart_key, run_key, t_key, _, _ = _slice_keys(
            transition_key,
            sampler._periodic,
        )
        chart_offset = None
        chart_anchor = TreeField(anchor)
        if sampler._periodic:
            chart_anchor, chart_offset = _open_chart(
                chart_key,
                chart_anchor,
                sampler._periodic,
            )
        left, right = _slice_bounds(
            chart_anchor,
            direction,
        )
        proposal, t = _pick_point_in_interval(
            t_key,
            chart_anchor,
            direction,
            left,
            right,
        )
        if sampler._periodic:
            proposal = _close_chart(
                proposal,
                chart_offset,
                sampler._periodic,
            )
        return dataclasses.replace(
            chain,
            anchor=anchor,
            chart_offset=(
                None if chart_offset is None else chart_offset.tree
            ),
            left=left,
            right=right,
            t=t,
            proposal=proposal.tree,
            run_key=run_key,
        )

    def consume_one(chain, log_likelihood, active, constraint):
        def consume(active_chain):
            transition_index = active_chain.transition_index
            active_chain = dataclasses.replace(
                active_chain,
                log_likelihood=log_likelihood,
                num_likelihood_evaluations=(
                    active_chain.num_likelihood_evaluations
                    + jnp.asarray(1, mp_policy.count_dtype)
                ),
            )

            def accept(accepted_chain):
                if num_phantom == 0:
                    phantom_samples = accepted_chain.phantom_samples
                    phantom_log_likelihoods = (
                        accepted_chain.phantom_log_likelihoods
                    )
                else:
                    phantom_index = jnp.minimum(
                        transition_index,
                        num_phantom - 1,
                    )
                    retain = transition_index < num_phantom
                    phantom_samples = jax.tree.map(
                        lambda history, value: history.at[
                            phantom_index
                        ].set(
                            jnp.where(
                                retain,
                                value,
                                history[phantom_index],
                            )
                        ),
                        accepted_chain.phantom_samples,
                        accepted_chain.proposal,
                    )
                    phantom_log_likelihoods = (
                        accepted_chain.phantom_log_likelihoods.at[
                            phantom_index
                        ].set(
                            jnp.where(
                                retain,
                                log_likelihood,
                                accepted_chain.phantom_log_likelihoods[
                                    phantom_index
                                ],
                            )
                        )
                    )
                next_index = transition_index + jnp.asarray(
                    1,
                    mp_policy.index_dtype,
                )
                finished = next_index == num_slices
                accepted_chain = dataclasses.replace(
                    accepted_chain,
                    anchor=accepted_chain.proposal,
                    phantom_samples=phantom_samples,
                    phantom_log_likelihoods=phantom_log_likelihoods,
                    transition_index=next_index,
                    done=finished,
                )

                def prepare_next(unfinished_chain):
                    # A batched ``cond`` traces this branch for a finished lane
                    # too. The safe index makes that unused value explicit and
                    # avoids relying on out-of-range gather clamping.
                    safe_index = jnp.minimum(next_index, num_slices - 1)
                    direction = TreeField(
                        jax.tree.map(
                            lambda values: values[safe_index],
                            unfinished_chain.directions,
                        )
                    )
                    return prepare_transition(
                        unfinished_chain,
                        unfinished_chain.anchor,
                        direction,
                        safe_index,
                    )

                return jax.lax.cond(
                    finished,
                    lambda value: value,
                    prepare_next,
                    accepted_chain,
                )

            def reject(rejected_chain):
                run_key, t_key = random.split(
                    rejected_chain.run_key,
                    2,
                )
                left, right = _shrink_interval(
                    rejected_chain.t,
                    rejected_chain.left,
                    rejected_chain.right,
                )
                direction = TreeField(
                    jax.tree.map(
                        lambda values: values[transition_index],
                        rejected_chain.directions,
                    )
                )
                chart_anchor = TreeField(rejected_chain.anchor)
                if sampler._periodic:
                    chart_offset = TreeField(
                        rejected_chain.chart_offset
                    )
                    # Retrying a rejected proposal must retain the transition's
                    # existing chart; drawing another seam here would change
                    # the slice interval and invalidate shrinkage.
                    chart_anchor = _to_chart(
                        chart_anchor,
                        chart_offset,
                        sampler._periodic,
                    )
                proposal, t = _pick_point_in_interval(
                    t_key,
                    chart_anchor,
                    direction,
                    left,
                    right,
                )
                if sampler._periodic:
                    proposal = _close_chart(
                        proposal,
                        chart_offset,
                        sampler._periodic,
                    )
                return dataclasses.replace(
                    rejected_chain,
                    left=left,
                    right=right,
                    t=t,
                    proposal=proposal.tree,
                    run_key=run_key,
                )

            return jax.lax.cond(
                log_likelihood > constraint,
                accept,
                reject,
                active_chain,
            )

        return jax.lax.cond(
            active,
            consume,
            lambda value: value,
            chain,
        )

    def continue_condition(state):
        return jnp.any(jnp.bitwise_not(state.done))

    def continue_body(state):
        active = jnp.bitwise_not(state.done)
        # Static-width filler keeps one likelihood program and lets scientific
        # likelihoods retain ordinary vmap semantics. Filler results never
        # enter chain state or the user-visible logical evaluation count.
        proposals = jax.tree.map(
            lambda values: jnp.where(
                active.reshape(
                    (num_chains,) + (1,) * (values.ndim - 1)
                ),
                values,
                jnp.full_like(values, 0.5),
            ),
            state.proposal,
        )
        if num_chains == 1:
            log_likelihoods = evaluate_one(
                jax.tree.map(lambda values: values[0], proposals)
            )[None]
        else:
            log_likelihoods = jax.vmap(evaluate_one)(proposals)
        return jax.vmap(consume_one)(
            state,
            log_likelihoods,
            active,
            request.log_L_constraints,
        )

    state = jax.lax.while_loop(
        continue_condition,
        continue_body,
        _initialise_slice_chains(sampler, request),
    )
    phantom_samples = PhantomSamples(
        U_samples=state.phantom_samples,
        log_L=state.phantom_log_likelihoods,
        valid_mask=jnp.broadcast_to(
            request.valid[:, None],
            (num_chains, num_phantom),
        ),
    )
    if request.sampler_data is None:
        num_directions = jnp.zeros(
            (num_chains,),
            mp_policy.count_dtype,
        )
        num_isotropic = jnp.zeros(
            (num_chains,),
            mp_policy.count_dtype,
        )
    else:
        num_directions = jnp.full(
            (num_chains,),
            num_slices,
            mp_policy.count_dtype,
        )
        num_isotropic = jnp.sum(
            state.direction_is_isotropic.astype(mp_policy.count_dtype),
            axis=1,
        )
    return ConstrainedSampleBatch(
        U_samples=state.anchor,
        log_likelihoods=state.log_likelihood,
        num_likelihood_evaluations=state.num_likelihood_evaluations,
        phantom_samples=phantom_samples,
        num_directions=num_directions,
        num_isotropic=num_isotropic,
    )
