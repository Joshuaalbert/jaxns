"""Frozen A selector for exact identity and controlled performance comparisons.

Copied from aa79a0dc395c0d64b85cbb88d0e3e732decd7361. Only the function name
changes. Production helpers for the unchanged classic index and reservations
are shared; tests also compare every phantom rank with direct enumeration.
"""
import jax
from jax import numpy as jnp

from jaxns.algorithm.depth import (
    _frozen_seed_count_at_constraint,
    _sample_frozen_seed_rank,
    _select_reservoir_slots,
)
from jaxns.algorithm.scheduler import ThreadSchedule, _seed_reservation_contains
from jaxns.mixed_precision import mp_policy
from jaxns.sampling.phantom_seeds import (
    phantom_rank_to_identity,
    phantom_seed_cumulative_batch,
    seed_stride,
)
from jaxns.state import State
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey


# Original function SHA256: fb2ea2499c00cf5e1d4509f99aab84321bb5e9c8ce8c9fce9b82ff3c2e70b902
def reference_stationary_seeds(
        key: PRNGKey,
        state: State,
        schedule: ThreadSchedule,
        log_L_constraint: FloatArray,
        from_root: BoolArray,
        valid: BoolArray,
        reserved_seed_idx: IntArray,
        reserved_log_L_constraint: FloatArray,
        reserved_valid: BoolArray,
) -> IntArray:
    """Draw same-contour seeds without replacement whenever possible.

    Classic publication and its recent-row reservoir follow R. Every stored
    phantom is an additional immediately available identity. The original
    permuted rotated strata and same-contour collision retries operate on
    this augmented population. Likelihood is used only to test eligibility.
    """
    shell_size = valid.shape[0]
    stride = seed_stride(state.samples)
    if stride > 1:
        phantom_cumulative_batch = phantom_seed_cumulative_batch(
            state.samples, state.num_samples, log_L_constraint,
        )  # [S, N], shared eligibility lookup for every lane
    rows = jnp.arange(shell_size, dtype=mp_policy.index_dtype)  # [S]
    assignment_key, strata_key, rejection_key = jax.random.split(key, 3)
    strata = jax.random.permutation(assignment_key, rows)  # [S]
    rotation = jax.random.uniform(
        strata_key,
        (),
        dtype=mp_policy.measure_dtype,
    )
    stratified_fraction = jnp.mod(
        rotation
        + strata.astype(mp_policy.measure_dtype)
        / jnp.asarray(shell_size, mp_policy.measure_dtype),
        1.0,
    )  # [S]
    same_group = (
        valid[:, None]
        & valid[None, :]
        & (log_L_constraint[:, None] == log_L_constraint[None, :])
    )  # [S, S]
    group_position = jnp.sum(
        same_group & (rows[None, :] < rows[:, None]),
        axis=1,
        dtype=mp_policy.index_dtype,
    )  # [S]
    reserved_same_group = (
        valid[:, None]
        & reserved_valid[None, :]
        & (
            log_L_constraint[:, None]
            == reserved_log_L_constraint[None, :]
        )
    )  # [S, S]
    reserved_count = jnp.sum(
        reserved_same_group,
        axis=1,
        dtype=mp_policy.index_dtype,
    )  # [S]
    retained_start_group = (
        valid
        & schedule.new_start
        & (
            log_L_constraint
            == schedule.start_seed_log_L_constraint
        )
    )  # [S]

    frozen_count = jax.vmap(
        lambda constraint: _frozen_seed_count_at_constraint(
            schedule,
            constraint,
        )
    )(log_L_constraint)  # [S]

    # The frozen rank index handles root and non-root contours uniformly. The
    # recent-row reservoir still distinguishes sentinel births from ordinary
    # birth contours when checking whether a retained row crosses the request.
    use_root = valid & from_root  # [S]
    reservoir_sample_idx = schedule.seed_reservoir_idx  # [R]
    reservoir_safe_idx = jnp.maximum(reservoir_sample_idx, 0)  # [R]
    reservoir_birth = state.samples.log_L_constraints[
        reservoir_safe_idx
    ]  # [R]
    reservoir_log_L = state.samples.log_likelihoods[
        reservoir_safe_idx
    ]  # [R]
    # The frozen index is exact. A value-independent bounded reservoir adds
    # recent stationary rows while this schedule is active. A geometric refresh
    # may promote them sooner; a drained projected schedule always publishes
    # every newly stationary classic before continuing the allocation target.
    reservoir_eligible = (
        valid[:, None]
        & schedule.seed_reservoir_valid[None, :]
        & jnp.where(
            use_root[:, None],
            jnp.isneginf(reservoir_birth)[None, :],
            (
                reservoir_birth[None, :] <= log_L_constraint[:, None]
            )
            & (
                reservoir_log_L[None, :] > log_L_constraint[:, None]
            ),
        )
    )  # [S, R]
    reservoir_count = jnp.sum(
        reservoir_eligible,
        axis=1,
        dtype=mp_policy.index_dtype,
    )  # [S]
    # The recent reservoir evolves while a wide start group spans batches.
    # Count only retained identities that remain in the current eligible
    # frozen+reservoir population; evicted rows must not make an unseen recent
    # row look exhausted. Publication makes every older identity frozen, so
    # the same compact calculation remains exact across source generations.
    published_reserved = schedule.num_published_start_seeds  # []
    reservoir_was_reserved = jax.lax.cond(
        jnp.any(retained_start_group),
        lambda unused: jax.vmap(
            lambda one_seed: _seed_reservation_contains(
                schedule.start_seed_reservation_idx,
                schedule.start_seed_reservation_group,
                schedule.current_start_group,
                one_seed,
            )
        )(reservoir_safe_idx * stride),
        lambda unused: jnp.zeros(
            reservoir_safe_idx.shape,
            dtype=mp_policy.bool_dtype,
        ),
        operand=None,
    )  # [R]
    retained_reserved = (
        published_reserved
        + jnp.sum(
            reservoir_eligible & reservoir_was_reserved[None, :],
            axis=1,
            dtype=mp_policy.index_dtype,
        )
    )  # [S]
    reserved_count = reserved_count + jnp.where(
        retained_start_group,
        retained_reserved,
        jnp.asarray(0, mp_policy.index_dtype),
    )
    proposal_width = 64
    selected_seed_idx = jnp.full(
        (shell_size,),
        -1,
        dtype=mp_policy.index_dtype,
    )  # [S]

    def select_lane(lane_idx, selected):
        constraint = log_L_constraint[lane_idx]
        lane_valid = valid[lane_idx]
        lane_retains_start_group = retained_start_group[lane_idx]
        if stride > 1:
            phantom_cumulative = phantom_cumulative_batch[lane_idx]  # [N]
            phantom_count = phantom_cumulative[-1]
        else:
            phantom_count = jnp.asarray(0, mp_policy.index_dtype)

        # Earlier local lanes and in-flight distributed tasks are the complete
        # set of simultaneous reservations. Only equal-contour identities are
        # excluded; different constrained priors do not compete for seeds.
        prior_valid = valid & (rows < lane_idx)  # [S]
        forbidden_idx = jnp.concatenate(
            (reserved_seed_idx, selected),
            axis=0,
        )  # [2S]
        forbidden_constraint = jnp.concatenate(
            (reserved_log_L_constraint, log_L_constraint),
            axis=0,
        )  # [2S]
        forbidden_valid = jnp.concatenate(
            (reserved_valid, prior_valid),
            axis=0,
        )  # [2S]
        forbidden_same_contour = (
            forbidden_valid
            & (forbidden_idx >= 0)
            & (forbidden_constraint == constraint)
        )  # [2S]
        distinct_count = frozen_count[lane_idx].astype(
            mp_policy.index_dtype
        ) + reservoir_count[lane_idx] + phantom_count
        remaining_distinct = jnp.maximum(
            distinct_count - reserved_count[lane_idx],
            0,
        )
        require_unused = group_position[lane_idx] < remaining_distinct

        frozen_seed_count = frozen_count[lane_idx].astype(
            mp_policy.index_dtype
        )
        classic_count = frozen_seed_count + reservoir_count[lane_idx]
        total_count = classic_count + phantom_count
        reservoir_cumulative = jnp.cumsum(
            reservoir_eligible[lane_idx].astype(mp_policy.index_dtype)
        )  # [R]

        def rejection_cond(carry):
            _, _, accepted = carry
            return lane_valid & jnp.logical_not(accepted)

        def rejection_body(carry):
            attempt, current_seed, _ = carry
            proposal_fraction = jax.random.uniform(
                jax.random.fold_in(
                    jax.random.fold_in(rejection_key, lane_idx),
                    attempt,
                ),
                (proposal_width,),
                dtype=mp_policy.measure_dtype,
            )  # [C]
            # The first proposal gives the batch systematic coverage of the
            # stationary lineage population. Random rotation and assignment
            # keep every lane marginally uniform without coupling seed rank to
            # the shallow-to-deep ordering of heterogeneous thread contours.
            # Independent proposals handle any no-replacement collision.
            proposal_fraction = proposal_fraction.at[0].set(jnp.where(
                attempt == 0,
                stratified_fraction[lane_idx],
                proposal_fraction[0],
            ))
            proposal_rank = (
                proposal_fraction * jnp.maximum(total_count, 1)
            ).astype(mp_policy.index_dtype)  # [C]
            frozen_offset = jnp.minimum(
                proposal_rank,
                jnp.maximum(frozen_seed_count - 1, 0),
            )
            frozen_fraction = (
                frozen_offset.astype(mp_policy.measure_dtype) + 0.5
            ) / jnp.maximum(
                frozen_seed_count,
                1,
            ).astype(mp_policy.measure_dtype)
            frozen_candidate = jax.vmap(
                lambda fraction: _sample_frozen_seed_rank(
                    schedule,
                    constraint,
                    fraction,
                )
            )(frozen_fraction)  # [C]
            reservoir_rank = jnp.maximum(
                proposal_rank - frozen_seed_count,
                0,
            )
            reservoir_slot = _select_reservoir_slots(
                reservoir_cumulative,
                reservoir_rank,
            )  # [C]
            reservoir_candidate = reservoir_sample_idx[
                reservoir_slot
            ]  # [C]
            candidate = jnp.where(
                proposal_rank < frozen_seed_count,
                frozen_candidate,
                reservoir_candidate,
            ) * stride  # [C], classic slot is zero
            if stride > 1:
                phantom_candidate = phantom_rank_to_identity(
                    state.samples, phantom_cumulative, constraint,
                    jnp.clip(
                        proposal_rank - classic_count, 0,
                        jnp.maximum(phantom_count - 1, 0),
                    ),
                )  # [C]
                candidate = jnp.where(
                    proposal_rank < classic_count, candidate,
                    phantom_candidate,
                )
            safe_candidate = jnp.maximum(candidate, 0)
            candidate_forbidden = jnp.any(
                (candidate[:, None] == forbidden_idx[None, :])
                & forbidden_same_contour[None, :],
                axis=1,
            )  # [C]
            candidate_was_start_seed = jax.lax.cond(
                lane_retains_start_group,
                lambda unused: jax.vmap(
                    lambda one_seed: _seed_reservation_contains(
                        schedule.start_seed_reservation_idx,
                        schedule.start_seed_reservation_group,
                        schedule.current_start_group,
                        one_seed,
                    )
                )(safe_candidate),
                lambda unused: jnp.zeros(
                    candidate.shape,
                    dtype=mp_policy.bool_dtype,
                ),
                operand=None,
            )  # [C]
            eligible = jnp.logical_not(
                require_unused
                & (candidate_forbidden | candidate_was_start_seed)
            )
            found = jnp.any(eligible)
            first = jnp.argmax(eligible).astype(mp_policy.index_dtype)
            return (
                attempt + jnp.asarray(1, attempt.dtype),
                jnp.where(found, candidate[first], current_seed),
                found,
            )

        _, seed, _ = jax.lax.while_loop(
            rejection_cond,
            rejection_body,
            (
                jnp.asarray(0, mp_policy.index_dtype),
                jnp.asarray(0, mp_policy.index_dtype),
                jnp.logical_not(lane_valid),
            ),
        )
        return selected.at[lane_idx].set(jnp.where(lane_valid, seed, -1))

    return jax.lax.fori_loop(
        0,
        shell_size,
        select_lane,
        selected_seed_idx,
    )
