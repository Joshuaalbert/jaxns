from typing import Tuple, Callable

from jax import numpy as jnp, random, tree_map, value_and_grad
from jax.lax import scan
from jax.lax import dynamic_update_slice
from jax.lax import while_loop
from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.maps import replace_index
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size
from jaxns.likelihood_samplers.parallel_slice_sampling import ProposalState, change_direction, shrink_interval, \
    sample_direction, \
    slice_bounds, pick_point_in_interval
from jaxns.nested_sampler.live_points import compute_num_live_points_from_unit_threads, infimum_constraint
from jaxns.prior_transforms import PriorChain
from jaxns.internals.types import NestedSamplerState, EvidenceCalculation, Reservoir, ThreadStats, int_type, \
    SampleCollection, float_type


def build_get_sample(prior_chain: PriorChain, loglikelihood_from_U,
                     midpoint_shrink: bool, destructive_shrink: bool, gradient_boost: bool) \
        -> Callable[..., Reservoir]:
    """
    Builds slice sampler that performs sampling from a given seed point.
    """

    def get_sample(key: jnp.ndarray, point_U0: jnp.ndarray, log_L0: jnp.ndarray, log_L_constraint: jnp.ndarray,
                   num_slices: jnp.ndarray) -> Reservoir:
        """
        Performs slice sampling from a seed location, within a given likelihood constraint.

        Args:
            key: PRNG key
            points_U0: [D] seed location in U-domain.
            log_L0: log-likelihood at the seed location.
            log_L_constraint: the log-likelihood constraint to sample within.

        Returns:
            Reservoir
        """

        slice_sampler_key, select_key, proposal_key, n_key, t_key = random.split(key, 5)

        direction = sample_direction(n_key, point_U0.size)
        num_likelihood_evaluations = jnp.full((), 0, int_type)
        if gradient_boost:
            # sets the initial direction of search up-hill (biasing away from uniform sampling, but pulling toward peak)
            _, grad_direction = value_and_grad(loglikelihood_from_U)(point_U0)
            grad_direction /= jnp.linalg.norm(grad_direction)
            direction = jnp.where(jnp.isnan(grad_direction), direction, grad_direction)
            num_likelihood_evaluations += jnp.full((), 1, int_type)
        (left, right) = slice_bounds(point_U0, direction)
        point_U, t = pick_point_in_interval(t_key, point_U0, direction, left, right)
        init_proposal_state = ProposalState(key=proposal_key,
                                            process_step=jnp.full((), 3, int_type),
                                            proposal_count=jnp.zeros((), int_type),
                                            num_likelihood_evaluations=num_likelihood_evaluations,
                                            point_U0=point_U0,
                                            log_L0=log_L0,
                                            direction=direction,
                                            left=left,
                                            right=right,
                                            point_U=point_U,
                                            t=t)

        def slice_sampler_body(body_state: Tuple[ProposalState, jnp.ndarray]) -> Tuple[ProposalState, jnp.ndarray]:
            """
            Iteratively samples num_parallel_samplers new points using a sampler, then replaces that number of points in
            the reservoir.
            """
            (proposal_state, _) = body_state
            log_L_point_U = loglikelihood_from_U(proposal_state.point_U)
            num_likelihood_evaluations = proposal_state.num_likelihood_evaluations + jnp.ones_like(
                proposal_state.num_likelihood_evaluations)
            # assumes that log_L0 > log_L_constraint
            good_proposal = jnp.greater(log_L_point_U, log_L_constraint)
            proposal_count = jnp.where(good_proposal,
                                       proposal_state.proposal_count + jnp.ones_like(proposal_state.proposal_count),
                                       proposal_state.proposal_count)

            enough_proposals = proposal_count >= num_slices

            # 0: successful proposal & enough proposals -> done
            # 1: successful proposal & not enough proposals -> change direction
            # 2: unsuccessful proposal -> shrink interval

            process_step = jnp.where(good_proposal & enough_proposals,
                                     jnp.full(good_proposal.shape, 0, proposal_state.process_step.dtype),
                                     jnp.where(good_proposal & ~enough_proposals,
                                               jnp.full(good_proposal.shape, 1, proposal_state.process_step.dtype),
                                               jnp.full(good_proposal.shape, 2, proposal_state.process_step.dtype)
                                               )
                                     )

            def _map_where(cond, a_tree, b_tree):
                return tree_map(lambda a, b: jnp.where(cond, a, b), a_tree, b_tree)

            proposal_state_from_1 = change_direction(proposal_state, log_L_point_U)
            proposal_state_from_2 = shrink_interval(proposal_state, log_L_point_U,
                                                    log_L_contour=log_L_constraint,
                                                    midpoint_shrink=midpoint_shrink,
                                                    destructive_shrink=destructive_shrink)

            # replace with the proposal state as appropriate

            proposal_state = _map_where(process_step == 1,
                                        proposal_state_from_1, proposal_state)
            proposal_state = _map_where(process_step == 2,
                                        proposal_state_from_2, proposal_state)

            proposal_state = proposal_state._replace(process_step=process_step,
                                                     proposal_count=proposal_count,
                                                     num_likelihood_evaluations=num_likelihood_evaluations)
            return (proposal_state, log_L_point_U)

        def slice_sampler_cond(body_state: Tuple[ProposalState, jnp.ndarray]) -> bool:
            """
            Stops when there have been enough proposals.
            """
            (proposal_state, _) = body_state
            return jnp.bitwise_not(proposal_state.proposal_count >= num_slices)

        (proposal_state, log_L_sample) = while_loop(slice_sampler_cond,
                                                    slice_sampler_body,
                                                    (init_proposal_state, -jnp.inf))
        # passthrough when num_slices==0
        pass_through = num_slices == jnp.zeros_like(num_slices)
        log_L_sample = jnp.where(pass_through, log_L0, log_L_sample)
        point_U = jnp.where(pass_through, point_U0, proposal_state.point_U)
        reservoir = Reservoir(points_U=point_U,
                              points_X=prior_chain(point_U),
                              log_L_constraint=log_L_constraint,
                              log_L_samples=log_L_sample,
                              num_likelihood_evaluations=proposal_state.num_likelihood_evaluations,
                              num_slices=proposal_state.proposal_count)
        return reservoir

    return get_sample


def normalise_log_space(x: LogSpace) -> LogSpace:
    """
    Safely normalise a LogSpace, accounting for zero-sum.
    """
    norm = x.sum()
    x /= norm
    x = LogSpace(jnp.where(jnp.isneginf(norm.log_abs_val), -jnp.inf, x.log_abs_val))
    return x


def sample_goal_distribution(key, log_goal_weights, S: int, *, replace: bool = True):
    """
    Sample indices that match unnormalised log_probabilities.

    Args:
        key: PRNG key
        log_goal_weights: unnormalised log probabilities
        S: number of samples
        replace: bool, whether to sample with replacement

    Returns:
        indices that draw from target density
    """
    if replace:
        p_cuml = LogSpace(log_goal_weights).cumsum()
        # 1 - U in (0,1] instead of [0,1)
        r = p_cuml[-1] * LogSpace(jnp.log(1 - random.uniform(key, (S,))))
        idx = jnp.searchsorted(p_cuml.log_abs_val, r.log_abs_val)
    else:
        assert S <= log_goal_weights.size
        g = -random.gumbel(key, shape=log_goal_weights.shape) - log_goal_weights
        idx = jnp.argsort(g)[:S]
    return idx


def compute_remaining_evidence(sample_idx, log_dZ_mean):
    # Z_remaining = dZ_mean.cumsum(reverse=True)

    def logsumexp_cumsum_body(_state):
        (log_abs_val, idx) = _state
        next_idx = idx - jnp.ones_like(idx)
        next_val = LogSpace(log_abs_val[idx]) + LogSpace(log_abs_val[next_idx])
        next_log_abs_val = dynamic_update_slice(log_abs_val, next_val.log_abs_val[None], [next_idx])
        return (next_log_abs_val, next_idx)

    # Calculate remaining evidence, doing only the minimal amount of work necessary.

    (log_Z_remaining, _) = while_loop(lambda _state: _state[1] > 0,
                                      logsumexp_cumsum_body,
                                      (log_dZ_mean,
                                       sample_idx - jnp.ones_like(sample_idx)))
    return log_Z_remaining


def _get_dynamic_goal(state: NestedSamplerState, G: jnp.ndarray):
    """
    Get contours that we'd like to reinforce.
    """

    n_i = state.sample_collection.num_live_points
    dZ_mean = LogSpace(state.sample_collection.log_dZ_mean)
    # Calculate remaining evidence, doing only the amount of work necessary.
    Z_remaining = LogSpace(compute_remaining_evidence(state.sample_idx, state.sample_collection.log_dZ_mean))
    # TODO: numerically compute goal using custome norm
    I_evidence = ((LogSpace(jnp.log(n_i + 1.)) * Z_remaining + LogSpace(jnp.log(n_i)) * dZ_mean) / (
            LogSpace(jnp.log(n_i + 1.)).sqrt() * LogSpace(jnp.log(n_i + 2.)) ** (1.5)))
    I_evidence = normalise_log_space(I_evidence)

    I_posterior = dZ_mean
    I_posterior = normalise_log_space(I_posterior)

    I_goal = LogSpace(jnp.log(1. - G)) * I_evidence + LogSpace(jnp.log(G)) * I_posterior
    # I_goal = normalise_log_space(I_goal) # unnecessary for sampling

    mask = jnp.arange(I_goal.size) >= state.sample_idx
    I_goal = LogSpace(jnp.where(mask, -jnp.inf, I_goal.log_abs_val))

    return I_goal.log_abs_val



def get_dynamic_goal(key, state: NestedSamplerState, num_samples: int, G:jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Determines what seed points to sample above.
    """
    contours = jnp.concatenate([state.sample_collection.log_L_constraint[0:1],
                                state.sample_collection.log_L_samples])
    if G is None:
        raise ValueError(f"G should be a float in [0,1].")
    log_goal_weights = _get_dynamic_goal(state, G)
    # Probabilistically sample the contours according to goal distribution
    indices_constraint_reinforce = sample_goal_distribution(key, log_goal_weights, num_samples, replace=True)
    start_idx = indices_constraint_reinforce.min()
    end_idx = indices_constraint_reinforce.max()
    log_L_constraint_start = contours[start_idx]
    log_L_constraint_end = contours[end_idx]

    return log_L_constraint_start, log_L_constraint_end


def compute_evidence(state: NestedSamplerState):
    """
    Computes the evidence and statistics.
    """
    # Sample collection has unsorted samples, and incorrect num_live_points.
    sample_collection = state.sample_collection
    num_live_points, sort_idx = compute_num_live_points_from_unit_threads(
        log_L_constraints=sample_collection.log_L_constraint,
        log_L_samples=sample_collection.log_L_samples,
        num_samples=state.sample_idx,
        return_sort_idx=True)

    sample_collection = sample_collection._replace(num_live_points=num_live_points,
                                                   log_L_samples=sample_collection.log_L_samples[sort_idx],
                                                   log_L_constraint=sample_collection.log_L_constraint[sort_idx],
                                                   points_X=tree_map(lambda x: x[sort_idx], sample_collection.points_X),
                                                   points_U=sample_collection.points_U[sort_idx],
                                                   num_likelihood_evaluations=
                                                   sample_collection.num_likelihood_evaluations[sort_idx],
                                                   num_slices=sample_collection.num_slices[sort_idx]
                                                   )
    # The initial log_L_contour is L_min after lexigraphic sort, which is usually -inf, but not always.
    init_log_L_contour = sample_collection.log_L_constraint[0]
    # This contains the required information to compute Z and ZH

    initial_evidence_calculation = _init_evidence_calculation()

    def thread_cond(body_state: Tuple[EvidenceCalculation, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        (evidence_calculation, idx, log_L_contour, log_dZ_mean, log_X_mean) = body_state
        return idx < state.sample_idx

    def thread_body(body_state: Tuple[EvidenceCalculation, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) \
            -> Tuple[EvidenceCalculation, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        (evidence_calculation, idx, log_L_contour, log_dZ_mean, log_X_mean) = body_state
        next_num_live_points = sample_collection.num_live_points[idx]
        next_log_L = sample_collection.log_L_samples[idx]
        next_log_L_contour = next_log_L
        # Get log_dZ_mean, and log_X_mean
        next_evidence_calculation = _update_evidence_calculation(num_live_points=next_num_live_points,
                                                                 log_L=log_L_contour,
                                                                 next_log_L_contour=next_log_L,
                                                                 evidence_calculation=evidence_calculation)

        next_dZ_mean = (LogSpace(next_evidence_calculation.log_Z_mean)
                        - LogSpace(evidence_calculation.log_Z_mean)).abs()

        next_log_dZ_mean = dynamic_update_slice(log_dZ_mean, next_dZ_mean.log_abs_val[None], [idx])
        next_log_X_mean = dynamic_update_slice(log_X_mean, next_evidence_calculation.log_X_mean[None], [idx])
        next_idx = idx + jnp.ones_like(idx)
        return (next_evidence_calculation, next_idx, next_log_L_contour, next_log_dZ_mean, next_log_X_mean)

    (final_evidence_calculation, final_idx, final_log_L_contour, final_log_dZ_mean, final_log_X_mean) = \
        while_loop(thread_cond,
                   thread_body,
                   (initial_evidence_calculation, jnp.asarray(0, int_type), init_log_L_contour,
                    sample_collection.log_dZ_mean, sample_collection.log_X_mean))

    final_sample_collection = sample_collection._replace(log_dZ_mean=final_log_dZ_mean, log_X_mean=final_log_X_mean)
    state = state._replace(evidence_calculation=final_evidence_calculation,
                           sample_collection=final_sample_collection,
                           log_L_contour=final_log_L_contour)
    return state


def collect_samples(state: NestedSamplerState, new_reservoir: Reservoir) -> NestedSamplerState:
    """
    Merge samples from new reservoir into sample collection, then recalculate statistics.

    Args:
        state: NestedSamplerState
        new_reservoir: Reservoir

    Returns:
        NestedSamplerState
    """
    old_reservoir = _sample_collection_to_reservoir(state.sample_collection)
    # Insert the new samples with a slice update
    reservoir = tree_map(lambda old, new: replace_index(old, new, state.sample_idx), old_reservoir, new_reservoir)
    # Update the sample collection
    sample_collection = state.sample_collection._replace(**reservoir._asdict())
    # Update the number of samples, and sample collection
    state = state._replace(sample_idx=state.sample_idx + new_reservoir.log_L_samples.size,
                           sample_collection=sample_collection,
                           num_likelihood_evaluations=state.num_likelihood_evaluations + jnp.sum(
                               new_reservoir.num_likelihood_evaluations))
    # Recompute statistics
    state = compute_evidence(state)
    return state


def update_samples(state: NestedSamplerState, new_reservoir: Reservoir, update_indices) -> NestedSamplerState:
    """
    Merge samples from new reservoir into sample collection, then recalculate statistics.

    Args:
        state: NestedSamplerState
        new_reservoir: Reservoir

    Returns:
        NestedSamplerState
    """
    old_reservoir = _sample_collection_to_reservoir(state.sample_collection)
    # Insert the new samples with a slice update
    reservoir = tree_map(lambda old, new: old.at[update_indices].set(new), old_reservoir, new_reservoir)
    # Update the sample collection
    sample_collection = state.sample_collection._replace(**reservoir._asdict())
    # Update the number of samples, and sample collection
    state = state._replace(sample_collection=sample_collection)
    # Recompute statistics
    state = compute_evidence(state)
    return state


def _update_evidence_calculation(num_live_points: jnp.ndarray,
                                 log_L: jnp.ndarray,
                                 next_log_L_contour: jnp.ndarray,
                                 evidence_calculation: EvidenceCalculation) -> EvidenceCalculation:
    num_live_points = num_live_points.astype(log_L.dtype)
    next_L = LogSpace(next_log_L_contour)
    L_contour = LogSpace(log_L)
    midL = LogSpace(jnp.log(0.5)) * (next_L + L_contour)
    X_mean = LogSpace(evidence_calculation.log_X_mean)
    X2_mean = LogSpace(evidence_calculation.log_X2_mean)
    Z_mean = LogSpace(evidence_calculation.log_Z_mean)
    ZX_mean = LogSpace(evidence_calculation.log_ZX_mean)
    Z2_mean = LogSpace(evidence_calculation.log_Z2_mean)
    dZ2_mean = LogSpace(evidence_calculation.log_dZ2_mean)

    # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
    # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
    T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
    # T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
    t_mean = LogSpace(- jnp.log(num_live_points + 1.))
    # T2_mean = LogSpace(jnp.log(num_live_points) - jnp.log( num_live_points + 2.))
    # T2_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 2./num_live_points))
    T2_mean = LogSpace(- jnp.logaddexp((0.), jnp.log(2.) - jnp.log(num_live_points)))
    # T2_mean = LogSpace(- jnp.logaddexp(jnp.log(2.), -jnp.log(num_live_points)))
    t2_mean = LogSpace(jnp.log(2.) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points) - jnp.log(num_live_points + 2.))
    tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))
    # tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))

    next_X_mean = X_mean * T_mean
    next_X2_mean = X2_mean * T2_mean
    next_Z_mean = Z_mean + X_mean * t_mean * midL
    next_ZX_mean = ZX_mean * T_mean + X2_mean * tT_mean * midL
    next_Z2_mean = Z2_mean + LogSpace(jnp.log(2.)) * ZX_mean * t_mean * midL + (X2_mean * t2_mean * midL ** 2)
    next_dZ2_mean = dZ2_mean + (X2_mean * t2_mean * midL ** 2)

    next_evidence_calculation = evidence_calculation._replace(
        log_X_mean=next_X_mean.log_abs_val.astype(evidence_calculation.log_X_mean.dtype),
        log_X2_mean=next_X2_mean.log_abs_val.astype(evidence_calculation.log_X2_mean.dtype),
        log_Z_mean=next_Z_mean.log_abs_val.astype(evidence_calculation.log_Z_mean.dtype),
        log_Z2_mean=next_Z2_mean.log_abs_val.astype(evidence_calculation.log_Z2_mean.dtype),
        log_ZX_mean=next_ZX_mean.log_abs_val.astype(evidence_calculation.log_ZX_mean.dtype),
        log_dZ2_mean=next_dZ2_mean.log_abs_val.astype(evidence_calculation.log_dZ2_mean.dtype)
    )
    # log_Z_mean, log_Z_var = linear_to_log_stats(next_evidence_calculation.log_Z_mean, log_f2_mean=next_evidence_calculation.log_Z2_mean)

    # next_evidence_calculation = tree_map(lambda old, new: jnp.where(jnp.isnan(new), old, new),
    #                                      evidence_calculation, next_evidence_calculation)

    return next_evidence_calculation


def _single_sample_constraint_for_contour_and_idx(indices_contour_reinforce,
                                                  log_L_constraint,
                                                  log_L_samples,
                                                  log_X_mean):
    # Get the mean enclosed prior volume at the single sample constraint.
    log_X_mean_contour_reinforce = log_X_mean[indices_contour_reinforce]
    X_mean_contour_reinforce = LogSpace(log_X_mean_contour_reinforce)
    # X_constraint_mean = min(1, X_contour_mean * 2)
    X_mean_constraint_reinforce = (X_mean_contour_reinforce * LogSpace(jnp.log(2.))
                                   ).minimum(LogSpace(jnp.asarray(0.)))
    # Now, figure out what likelihood this constraint corresponds to.
    # The maximum likelihood would be the infimum on the contour
    # Note: sample_collection already sorted so sort_idx not needed
    _, log_L_contraint_reinforce_max = infimum_constraint(
        log_L_constraints=log_L_constraint,
        log_L_samples=log_L_samples,
        sort_idx=None,
        return_contours=True)
    log_L_contraint_reinforce_max = log_L_contraint_reinforce_max[indices_contour_reinforce]
    # The mimimum would be the zero likelihood
    x = -X_mean_constraint_reinforce.log_abs_val
    xp = -log_X_mean
    fp = log_L_samples
    left = -jnp.inf
    right = log_L_contraint_reinforce_max
    constraint_supremum_idx = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, len(xp) - 1)
    df = fp[constraint_supremum_idx] - fp[constraint_supremum_idx - 1]
    dx = xp[constraint_supremum_idx] - xp[constraint_supremum_idx - 1]
    delta = x - xp[constraint_supremum_idx - 1]
    f = jnp.where((dx == 0), fp[constraint_supremum_idx], fp[constraint_supremum_idx - 1] + (delta / dx) * df)
    f = jnp.where(x < xp[0], left, f)
    log_L_contraint_reinforce = jnp.where(x > xp[-1], right, f)
    return log_L_contraint_reinforce, constraint_supremum_idx


def _update_thread_stats(state: NestedSamplerState):
    log_Z_mean, log_Z_var = linear_to_log_stats(
        log_f_mean=state.evidence_calculation.log_Z_mean,
        log_f2_mean=state.evidence_calculation.log_Z2_mean)

    ess = effective_sample_size(state.evidence_calculation.log_Z_mean,
                                state.evidence_calculation.log_dZ2_mean)

    log_L_contour_max = state.sample_collection.log_L_samples[state.sample_idx - 1]

    thread_stats_update = \
        ThreadStats(evidence_uncert=jnp.sqrt(log_Z_var),
                    ess=ess,
                    evidence=log_Z_mean,
                    log_L_max=log_L_contour_max,
                    num_likelihood_evaluations=jnp.sum(state.sample_collection.num_likelihood_evaluations)
                    )
    thread_stats = tree_map(lambda operand, update: replace_index(operand, update, state.step_idx),
                            state.thread_stats, thread_stats_update)
    return thread_stats


def _sample_collection_to_reservoir(sample_collection: SampleCollection) -> Reservoir:
    reservoir = Reservoir(points_U=sample_collection.points_U,
                          points_X=sample_collection.points_X,
                          log_L_constraint=sample_collection.log_L_constraint,
                          log_L_samples=sample_collection.log_L_samples,
                          num_likelihood_evaluations=sample_collection.num_likelihood_evaluations,
                          num_slices=sample_collection.num_slices)
    return reservoir


def _init_evidence_calculation():
    evidence_calculation = EvidenceCalculation(
        log_X_mean=jnp.asarray(0., float_type),
        log_X2_mean=jnp.asarray(0., float_type),
        log_Z_mean=jnp.asarray(-jnp.inf, float_type),
        log_ZX_mean=jnp.asarray(-jnp.inf, float_type),
        log_Z2_mean=jnp.asarray(-jnp.inf, float_type),
        log_dZ2_mean=jnp.asarray(-jnp.inf, float_type)
    )
    return evidence_calculation
