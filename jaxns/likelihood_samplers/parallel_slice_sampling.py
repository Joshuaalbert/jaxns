from jax import random, numpy as jnp
from typing import NamedTuple
import logging

from jaxns.internals.types import int_type

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
    if ndim == 1:
        return jnp.ones(())
    direction = random.normal(n_key, shape=(ndim,))
    direction /= jnp.linalg.norm(direction)
    return direction


def shrink_interval(from_proposal_state: ProposalState, log_L_proposal: jnp.ndarray, log_L_contour: jnp.ndarray,
                    midpoint_shrink: bool, destructive_shrink:bool) -> ProposalState:
    """
    Not successful proposal, so shrink, optionally apply exponential shrinkage.
    """
    # witout exponential shrinkage, we shrink to failed proposal point, which is 100% correct.
    left = jnp.where(from_proposal_state.t < 0., from_proposal_state.t, from_proposal_state.left)
    right = jnp.where(from_proposal_state.t > 0., from_proposal_state.t, from_proposal_state.right)
    key, t_key, midpoint_key = random.split(from_proposal_state.key, 3)
    if destructive_shrink:
        # shrink to constraint using linear approximation
        # Line logic:
        # logL(t) = m * t + b
        # logL(0) = b
        # (logL(t_R) - logL(0))/t_R
        # logL(t_R*alpha) = (logL(t_R) - logL(0))*alpha + logL(0)
        # alpha = (logL_contour - logL(0))/(logL(t_R) - logL(0))
        alpha = (log_L_contour - from_proposal_state.log_L0) / (log_L_proposal - from_proposal_state.log_L0)
        left = jnp.where((from_proposal_state.t < 0.), alpha * left, left)
        right = jnp.where((from_proposal_state.t > 0.), alpha * right, right)
    if midpoint_shrink:
        # we take two points along lines from origin:
        #  - alpha: one from a satisfying point (t=0) to non-satisfying proposal,
        #  - beta: and, one from non-satisfying proposal to the constraint.
        # We shrink to alpha point if beta point is above alpha point.
        # Intuitively, the tangent from constraint is more accurate than proposal and should be a supremum of heights
        # in reasonable cases.
        # An extension is to make alpha shrink to constraint line, which would shrink very fast, but introduce
        # auto-correlation which must be later refined away.
        # Line logic:
        # logL(t) = m * t + b
        # logL(0) = b
        # (logL(t_R) - logL(0))/t_R
        # logL(t_R*alpha) = (logL(t_R) - logL(0))*alpha + logL(0)
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
                                        process_step=jnp.full((), 3, int_type),
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
    # project out the previous direction to sample in orthogonal slice
    if point_U0.size > 1:
        direction = direction - direction * (direction @ from_proposal_state.direction)
    (left, right) = slice_bounds(point_U0, direction)
    point_U, t = pick_point_in_interval(t_key, point_U0, direction, left, right)

    return from_proposal_state._replace(key=proposal_key,
                                        process_step=jnp.full((), 3, int_type),
                                        point_U0=point_U0,
                                        log_L0=log_L0,
                                        direction=direction,
                                        left=left,
                                        right=right,
                                        point_U=point_U,
                                        t=t)
