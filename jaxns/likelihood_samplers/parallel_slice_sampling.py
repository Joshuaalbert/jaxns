import jax.numpy as jnp
from jax.lax import while_loop, dynamic_update_slice, scan
from jax import random, vmap, tree_map
from typing import NamedTuple, Tuple

from jaxns.types import Reservoir
from jaxns.prior_transforms.prior_chain import PriorChain
import logging

logger = logging.getLogger(__name__)


class ProposalState(NamedTuple):
    key: jnp.ndarray  # PRNG key
    process_step: jnp.ndarray  # what stage of the sampling each independent point is at
    proposal_count: jnp.ndarray  # how many successful proposals have happened for each independent point
    num_likelihood_evaluations: jnp.ndarray  # how many  likelihood evaluations have been made for each independent point
    point_U0: jnp.ndarray  # the origin of current slice of each independent point
    log_L0: jnp.ndarray  # the value of log-likelihood at each indpendent point origin
    direction: jnp.ndarray  # the direction of slicing
    left: jnp.ndarray  # the left bound of slice
    right: jnp.ndarray  # the right bound of slice
    point_U: jnp.ndarray  # the point up for likelihood computation
    t: jnp.ndarray  # the parameter measuring between left and right bounds.


class SliceSamplerState(NamedTuple):
    key: jnp.ndarray
    done: jnp.ndarray
    reservoir: Reservoir


def pick_point_in_interval(t_key, point_U0, direction, left, right):
    """
    Select a point along slice in [point_U0 + direction * left, point_U0 + direction * right]

    Args:
        t_key: PRNG key
        point_U0: [D]
        direction: [D]
        left: left most point (<= 0).
        right: right most point (>= 0).

    Returns:
        point_U: [D]
        t: selection point between [left, right]
    """
    t = random.uniform(t_key, minval=left, maxval=right)
    point_U = point_U0 + t * direction
    return point_U, t


def slice_bounds(point_U0, direction):
    """
    Compute the slice bounds, t, where point_U0 + direction * t intersects uit cube boundary.

    Args:
        point_U0: [D]
        direction: [D]

    Returns:
        left_bound: left most point (<= 0).
        right_bound: right most point (>= 0).
    """
    t1 = (1. - point_U0) / direction
    t1_right = jnp.min(jnp.where(t1 >= 0., t1, jnp.inf))
    t1_left = jnp.max(jnp.where(t1 <= 0., t1, -jnp.inf))
    t0 = -point_U0 / direction
    t0_right = jnp.min(jnp.where(t0 >= 0., t0, jnp.inf))
    t0_left = jnp.max(jnp.where(t0 <= 0., t0, -jnp.inf))
    right_bound = jnp.minimum(t0_right, t1_right)
    left_bound = jnp.maximum(t0_left, t1_left)
    return left_bound, right_bound


def sample_direction(n_key, ndim: int):
    """
    Choose a direction randomly from S^(D-1).

    Args:
        n_key: PRNG key
        ndim: int, number of dimentions

    Returns:
        direction: [D] direction from S^(D-1)
    """
    direction = random.normal(n_key, shape=(ndim,))
    direction /= jnp.linalg.norm(direction)
    return direction


def select_from_reservoir(key, reservoir_log_L, reservoir_contraint_satisfied, reservoir_points_U):
    """
    Selection a point in the reservoir where constraint is satisfied.

    Args:
        key: PRNG key
        reservoir_log_L: [N] reservoir log-likelihood
        log_L_contour: contour above which reservoir can be picked from
        reservoir_points_U: [N, D] reservoir points in U domain.

    Returns:
        point_U0: [D] point where constraint is satisfied for sure.
        log_L0: scalar, log-likelihood at point_U0.
    """
    g = jnp.where(reservoir_contraint_satisfied,
                  random.gumbel(key, reservoir_log_L.shape),
                  -jnp.inf)
    select_idx = jnp.argmax(g)
    point_U0 = reservoir_points_U[select_idx]
    log_L0 = reservoir_log_L[select_idx]
    return point_U0, log_L0


def _parallel_sampling(loglikelihood_from_U, prior_chain: PriorChain, key, log_L_contour, num_samples: jnp.ndarray,
                       reservoir_state: Reservoir, num_slices: int, midpoint_shrink: bool,
                       num_parallel_samplers: int, strict_contour: jnp.ndarray) -> Reservoir:
    """
    Perform parallel slice sampling, applying a state transition strategy in the algorithm such that we can always
    apply vmap to the log-likelihood, without wasting those computations.

    Args:
        loglikelihood_from_U: callable(U_flat)
        key: PRNG key
        log_L_contour: scalar
        num_samples: int, how many samples to draw
        reservoir_state: Reservoir
        num_slice: int, how many proposals per accepted sample
        midpoint_shrink: bool, whether to apply additional shrinkage, slightly invalidates the exactness of sampling.
        num_parallel_samplers: int, how many likelihood calculations to do at the same time, with vmap.

    Returns:
        next_reservoir_state: Reservoir
    """
    init_num_available = jnp.sum(reservoir_state.available)

    def propose_new_point_from_reservoir(slice_sampler_state: SliceSamplerState) -> Tuple[
        SliceSamplerState, ProposalState]:
        """
        Select a point from satisfying points in reservoir.
        Pick direction.
        Find full interval.
        Pick point on interval.
        Ensures key in slice_sampler_state is new.
        """
        slice_sampler_key, select_key, proposal_key, n_key, t_key = random.split(slice_sampler_state.key, 5)
        point_U0, log_L0 = select_from_reservoir(select_key,
                                                 slice_sampler_state.reservoir.log_L,
                                                 slice_sampler_state.reservoir.log_L >= log_L_contour,
                                                 # for the case available doesn't map to constraint satisfaction
                                                 slice_sampler_state.reservoir.points_U)
        direction = sample_direction(n_key, point_U0.size)
        (left, right) = slice_bounds(point_U0, direction)
        point_U, t = pick_point_in_interval(t_key, point_U0, direction, left, right)
        return (slice_sampler_state._replace(key=slice_sampler_key),
                ProposalState(key=proposal_key,
                              process_step=jnp.full((), 3, jnp.int_),
                              proposal_count=jnp.zeros((), jnp.int_),
                              num_likelihood_evaluations=jnp.zeros((), jnp.int_),
                              point_U0=point_U0,
                              log_L0=log_L0,
                              direction=direction,
                              left=left,
                              right=right,
                              point_U=point_U,
                              t=t))

    def replace_point_in_reservoir(slice_sampler_state: SliceSamplerState, from_proposal_state: ProposalState,
                                   log_L_proposal: jnp.ndarray) -> Tuple[SliceSamplerState, ProposalState]:
        """
        Successful proposal and proposal_count==num_slices, so
        Replace reservoir at idx_contour.
        Return new reservoir sample.
        """
        # replace the first occurance of non-satisfied contour
        # Cannot use jnp.argmin(slice_sampler_state.reservoir.available) because unavailable doesn't mean unsatisfied constraint
        # but unsatisfied constraint mean unavailable.
        # replace_idx = jnp.argmin(slice_sampler_state.reservoir.log_L > log_L_contour)
        replace_idx = jnp.argmin(slice_sampler_state.reservoir.available)

        next_point_X = prior_chain(from_proposal_state.point_U)
        new_reservoir_point = Reservoir(points_U=from_proposal_state.point_U,
                                        points_X=dict((name, next_point_X[name])
                                                      for name in slice_sampler_state.reservoir.points_X),
                                        log_L=log_L_proposal,
                                        num_likelihood_evaluations=from_proposal_state.num_likelihood_evaluations,
                                        available=jnp.asarray(True, jnp.bool_))

        next_reservoir = tree_map(lambda operand, update: dynamic_update_slice(operand,
                                                                               update[None],
                                                                               [replace_idx] + [0] * len(update.shape)),
                                  slice_sampler_state.reservoir,
                                  new_reservoir_point)

        slice_sampler_state = slice_sampler_state._replace(reservoir=next_reservoir)

        return propose_new_point_from_reservoir(slice_sampler_state)

    def shrink_interval(from_proposal_state: ProposalState, log_L_proposal: jnp.ndarray,
                        midpoint_shrink: bool) -> ProposalState:
        """
        Not successful proposal, so shrink.
        """
        left = jnp.where(from_proposal_state.t < 0., from_proposal_state.t, from_proposal_state.left)
        right = jnp.where(from_proposal_state.t > 0., from_proposal_state.t, from_proposal_state.right)
        key, t_key, midpoint_key = random.split(from_proposal_state.key, 3)
        if midpoint_shrink:
            # y(t) = m * t + b
            # y(0) = b
            # (y(t_R) - y(o))/t_R
            # y(t_R*alpha) = (y(t_R) - y(0))*alpha + y(0)
            alpha_key, beta_key = random.split(midpoint_key, 2)
            alpha = random.uniform(alpha_key)
            beta = random.uniform(beta_key)
            logL_alpha = from_proposal_state.log_L0 + alpha * (log_L_proposal - from_proposal_state.log_L0)
            logL_beta = log_L_proposal + beta * (log_L_contour - log_L_proposal)
            do_mid_point_shrink = logL_alpha < logL_beta
            left = jnp.where((from_proposal_state.t < 0.) & do_mid_point_shrink, alpha * left, left)
            right = jnp.where((from_proposal_state.t > 0.) & do_mid_point_shrink, alpha * right, right)
        point_U, t = pick_point_in_interval(t_key, from_proposal_state.point_U0, from_proposal_state.direction,
                                            left, right)
        return from_proposal_state._replace(key=key,
                                            process_step=jnp.full((), 3, jnp.int_),
                                            left=left,
                                            right=right,
                                            point_U=point_U,
                                            t=t)

    def change_direction(from_proposal_state: ProposalState, log_L_proposal: jnp.ndarray) -> ProposalState:
        """
        Successful proposal, but not enough proposals to de-correlate.
        Pick a new direction and propose from current point.
        """
        proposal_key, n_key, t_key = random.split(from_proposal_state.key, 3)
        point_U0 = from_proposal_state.point_U
        log_L0 = log_L_proposal
        direction = sample_direction(n_key, point_U0.size)
        (left, right) = slice_bounds(point_U0, direction)
        point_U, t = pick_point_in_interval(t_key, point_U0, direction, left, right)

        return from_proposal_state._replace(key=proposal_key,
                                            process_step=jnp.full((), 3, jnp.int_),
                                            point_U0=point_U0,
                                            log_L0=log_L0,
                                            direction=direction,
                                            left=left,
                                            right=right,
                                            point_U=point_U,
                                            t=t)

    def slice_sampler_body(state: Tuple[SliceSamplerState, ProposalState]) \
            -> Tuple[SliceSamplerState, ProposalState]:
        """
        Iteratively samples num_parallel_samplers new points using a sampler, then replaces that number of points in
        the reservoir.
        """
        # 0: successful proposal & enough proposals
        # 1: successful proposal & not enough proposals
        # 2: unsuccessful proposal

        (slice_sampler_state, proposals_state) = state
        # Parallel computation of log-likelihood
        # TODO: consider how xmap could be used to facilitate a multi-device solution.
        log_L_points_U = vmap(loglikelihood_from_U)(proposals_state.point_U)
        # break plateau degeneracies with random perturbation that has no effect on accuracy
        key, break_plateau_key = random.split(slice_sampler_state.key, 2)
        slice_sampler_state = slice_sampler_state._replace(key=key)

        proposals_state = proposals_state._replace(
            num_likelihood_evaluations=proposals_state.num_likelihood_evaluations + jnp.ones_like(
                proposals_state.num_likelihood_evaluations))
        # TODO: analyse whether this should be > or >= in case of plateau. In case of shrinkage, there should be
        good_proposal_strict = jnp.greater(log_L_points_U, log_L_contour)
        good_proposal_lax = jnp.greater_equal(log_L_points_U, log_L_contour)
        good_proposal = jnp.where(strict_contour,
                                  good_proposal_strict,
                                  good_proposal_lax)

        proposal_count = jnp.where(good_proposal,
                                   proposals_state.proposal_count + jnp.ones_like(proposals_state.proposal_count),
                                   proposals_state.proposal_count)

        enough_proposals = proposal_count == num_slices

        process_step = jnp.where(good_proposal & enough_proposals,
                                 jnp.full(good_proposal.shape, 0, proposals_state.process_step.dtype),
                                 jnp.where(good_proposal & ~enough_proposals,
                                           jnp.full(good_proposal.shape, 1, proposals_state.process_step.dtype),
                                           jnp.full(good_proposal.shape, 2, proposals_state.process_step.dtype)
                                           )
                                 )
        proposals_state = proposals_state._replace(process_step=process_step,
                                                   proposal_count=proposal_count)

        def update_body(slice_sampler_state: SliceSamplerState,
                        X: Tuple[ProposalState, jnp.ndarray]) -> Tuple[SliceSamplerState, ProposalState]:
            """
            For each parallel sampler we update sequentially to account for ordered updating of slice_sampler_state.
            """
            (proposal_state, log_L_point_U) = X

            def _map_where(cond, a_tree, b_tree):
                return tree_map(lambda a, b: jnp.where(cond, a, b), a_tree, b_tree)

            slice_sampler_state_from_0, proposal_state_from_0 = replace_point_in_reservoir(slice_sampler_state,
                                                                                           proposal_state,
                                                                                           log_L_point_U)
            proposal_state_from_1 = change_direction(proposal_state, log_L_point_U)
            proposal_state_from_2 = shrink_interval(proposal_state, log_L_point_U, midpoint_shrink=midpoint_shrink)

            # replace with the proposal state as appropriate
            in_process_step = proposal_state.process_step

            slice_sampler_state = _map_where(in_process_step == 0,
                                             slice_sampler_state_from_0,
                                             slice_sampler_state)

            proposal_state = _map_where(in_process_step == 0,
                                        proposal_state_from_0, proposal_state)
            proposal_state = _map_where(in_process_step == 1,
                                        proposal_state_from_1, proposal_state)
            proposal_state = _map_where(in_process_step == 2,
                                        proposal_state_from_2, proposal_state)
            return (slice_sampler_state, proposal_state)

        # Prepare proposal points for next batch of likelihood evaluations.
        slice_sampler_state, proposals_state = scan(update_body,
                                                    slice_sampler_state,
                                                    (proposals_state, log_L_points_U))

        done = jnp.greater_equal(
            jnp.sum(slice_sampler_state.reservoir.available),
            jnp.minimum(slice_sampler_state.reservoir.available.size, init_num_available + num_samples)
        )

        slice_sampler_state = slice_sampler_state._replace(done=done)

        return (slice_sampler_state, proposals_state)

    def slice_sampler_cond(state: Tuple[SliceSamplerState, ProposalState]) -> bool:
        """
        Stops when all points in the reservoir are above the contour.
        """
        (slice_sampler_state, proposals_state) = state
        return jnp.bitwise_not(slice_sampler_state.done)

    # Initialise state

    # TODO: replace with while and dynamic loop to only do work if num_samples > 0.

    init_slice_sampler_state = SliceSamplerState(key=key,
                                                 done=(num_samples == jnp.asarray(0, num_samples.dtype)) | (
                                                             init_num_available == reservoir_state.available.size),
                                                 reservoir=Reservoir(points_U=reservoir_state.points_U,
                                                                     points_X=reservoir_state.points_X,
                                                                     log_L=reservoir_state.log_L,
                                                                     num_likelihood_evaluations=reservoir_state.num_likelihood_evaluations,
                                                                     available=reservoir_state.available)
                                                 )

    def _get_init_point(key) -> ProposalState:
        """
        Propose a new point from reservoir, changing the PRNG key between each generation.
        """
        slice_sampler_state = init_slice_sampler_state._replace(key=key)
        _, proposal_state = propose_new_point_from_reservoir(slice_sampler_state)
        return proposal_state

    key, _key = random.split(init_slice_sampler_state.key, 2)
    init_slice_sampler_state = init_slice_sampler_state._replace(key=key)
    keys = random.split(_key, num_parallel_samplers)
    init_proposals_state = vmap(_get_init_point)(keys)

    # Run until all points satisfy the contour constraint. This is equivalent to filling all indicies larger than
    # init_idx_contour with new points.
    # TODO: handle the case of discrete bounded likelihood, and seed point is at upper bound.
    # likely need to use >= acceptance instead of >
    (slice_sampler_state, proposals_state) = while_loop(slice_sampler_cond,
                                                        slice_sampler_body,
                                                        (init_slice_sampler_state, init_proposals_state))

    return slice_sampler_state.reservoir
