from jax import numpy as jnp


def supremum_contour_idx(log_L_contours, log_L_samples, sort_idx=None):
    """
    For a contour l* find `i` such that log_L_samples[i] is a lowest strict upper bound of L*.

    E.g. if log_L_samples = [0,1,1,2,2,3] and L* = 1 this would return 3 since log_L_samples[3] = 2 is a strict supremum.
    By convention, we return the first supremum idx.

    Note: min(log_L_contours) < min(log_L_samples) and max(log_L_contours) < max(log_L_samples)

    Args:
        log_L_constraints:
        log_L_samples:
        sort_idx:
        return_contours:

    Returns:

    """
    if sort_idx is not None:
        log_L_samples = log_L_samples[sort_idx]
    contour_idx = jnp.searchsorted(log_L_samples, log_L_contours, side='right')
    return contour_idx


def infimum_constraint(log_L_constraints, log_L_samples, sort_idx=None, return_contours: bool = False):
    """
    For a single sample find `i` such that log_L_contours[i] is the greatest strict lower bound of log_L_sample.
    E.g.

    Args:
        log_L_constraints:
        log_L_samples:
        sort_idx:
        return_contours: if true also return the value of the contour at `i`, i.e. the constraint.
    """
    if sort_idx is not None:
        log_L_constraints = log_L_constraints[sort_idx]
        log_L_samples = log_L_samples[sort_idx]
    # mask the non-samples, already done since they should be inf.
    # if num_samples is None:
    #       num_samples = log_L_samples.size
    # empty_mask = jnp.arange(log_L_samples.size) >= num_samples
    # log_L_constraints = jnp.where(empty_mask, jnp.inf, log_L_constraints)
    # log_L_samples = jnp.where(empty_mask, jnp.inf, log_L_samples)
    log_L_contours = jnp.concatenate([log_L_constraints[0:1], log_L_samples[:-1]], axis=0)
    contour_idx = jnp.searchsorted(log_L_contours, log_L_samples, side='left') - 1
    if return_contours:
        # todo: consider clamping to (0, n-1) and avoid the where op
        constraints = jnp.where(contour_idx < 0,
                                -jnp.inf,
                                log_L_contours[contour_idx])
        return contour_idx, constraints
    return contour_idx


def infimum_contour(log_L_constraints, log_L_samples, sort_idx=None, return_contours: bool = False):
    """
    For a single sample find `i` such that log_L_contours[i] is the greatest strict lower bound of log_L_sample.
    E.g.

    Args:
        log_L_constraints:
        log_L_samples:
        sort_idx:
        return_contours: if true also return the value of the contour at `i`, i.e. the constraint.
    """
    if sort_idx is not None:
        log_L_constraints = log_L_constraints[sort_idx]
        log_L_samples = log_L_samples[sort_idx]
    # mask the non-samples, already done since they should be inf.
    # if num_samples is None:
    #       num_samples = log_L_samples.size
    # empty_mask = jnp.arange(log_L_samples.size) >= num_samples
    # log_L_constraints = jnp.where(empty_mask, jnp.inf, log_L_constraints)
    # log_L_samples = jnp.where(empty_mask, jnp.inf, log_L_samples)
    log_L_contours = jnp.concatenate([log_L_constraints[0:1], log_L_samples[:-1]], axis=0)
    # todo: use sort_idx as sort_key to avoid gather op
    contour_idx = jnp.searchsorted(log_L_contours, log_L_samples, side='left') - 1
    if return_contours:
        # todo: consider clamping to (0, n-1) and avoid the where op
        constraints = jnp.where(contour_idx < 0,
                                -jnp.inf,
                                log_L_contours[contour_idx])
        return contour_idx, constraints
    return contour_idx
