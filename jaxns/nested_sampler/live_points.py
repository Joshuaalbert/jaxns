from jax import numpy as jnp


def compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples, num_samples=None, return_sort_idx:bool=False):
    """
    Compute the number of live points of shrinkage distribution, from an arbitrary list of samples with
    corresponding sampling constraints.

    Args:
        log_L_constraints: [N] likelihood constraint that sample was uniformly sampled within
        log_L_samples: [N] likelihood of the sample
        return_sort_idx: bool, optional, whether to return the sort indices for sorting samples.

    Returns:
        if return_sort_idx is true:
            tuple where first element is num_live_points for shrinkage distribution, and second element is the indices
            to sort samples
        otherwise:
            num_live_points for shrinkage distribution
    """
    # mask the samples that are not yet taken
    if num_samples is None:
        num_samples = log_L_samples.size
    empty_mask = jnp.arange(log_L_samples.size) >= num_samples
    # masking samples is already done, since they are inf by default.
    # log_L_samples = jnp.where(empty_mask, jnp.inf, log_L_samples)
    # log_L_constraints = jnp.where(empty_mask, jnp.inf, log_L_constraints)
    # sort log_L_samples, breaking degeneracies on log_L_constraints
    idx_sort = jnp.lexsort((log_L_constraints, log_L_samples))
    log_L_samples = log_L_samples[idx_sort]
    log_L_constraints = log_L_constraints[idx_sort]
    # n = jnp.sum(available & (log_L_samples >= contour) & (log_L_constraints <= contour))
    # n = jnp.sum(available & (log_L_constraints <= contour))
    # n = available.size - jnp.sum(jnp.bitwise_not(available) | jnp.bitwise_not(log_L_constraints <= contour))
    # n = available.size - jnp.sum(jnp.bitwise_not(available)) - jnp.sum(jnp.bitwise_not(log_L_constraints <= contour)) + jnp.sum(jnp.bitwise_not(available) & jnp.bitwise_not(log_L_constraints <= contour))
    # n = jnp.sum(available) - jnp.sum(log_L_constraints > contour) + jnp.sum(jnp.bitwise_not(available) & (log_L_constraints > contour))
    # Since jnp.sum(jnp.bitwise_not(available) & (log_L_constraints > contour)) can be shown to be zero
    # n = jnp.sum(available) - jnp.sum(log_L_constraints > contour)

    _log_L_samples = jnp.concatenate([log_L_constraints[0:1], log_L_samples])
    n_base = jnp.arange(1, log_L_samples.size + 1)[::-1]
    # jnp.sum(log_L_samples[idx-1] < log_L_constraints)
    u = jnp.searchsorted(_log_L_samples, log_L_constraints, side='left')
    b = u.size - jnp.cumsum(jnp.bincount(u, length=u.size))
    n = n_base - b
    n = jnp.where(empty_mask[idx_sort], 0, n)
    n = jnp.asarray(n, log_L_samples.dtype)
    if return_sort_idx:
        return (n, idx_sort)
    return n

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

def infimum_constraint(log_L_constraints, log_L_samples, sort_idx=None, return_contours:bool=False):
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
    log_L_contours = jnp.concatenate([log_L_constraints[0:1], log_L_samples[:-1]],axis=0)
    contour_idx = jnp.searchsorted(log_L_contours, log_L_samples, side='left') - 1
    if return_contours:
        # todo: consider clamping to (0, n-1) and avoid the where op
        constraints = jnp.where(contour_idx < 0,
                            -jnp.inf,
                            log_L_contours[contour_idx])
        return contour_idx, constraints
    return contour_idx


def infimum_contour(log_L_constraints, log_L_samples, sort_idx=None, return_contours:bool=False):
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
    log_L_contours = jnp.concatenate([log_L_constraints[0:1], log_L_samples[:-1]],axis=0)
    # todo: use sort_idx as sort_key to avoid gather op
    contour_idx = jnp.searchsorted(log_L_contours, log_L_samples, side='left') - 1
    if return_contours:
        # todo: consider clamping to (0, n-1) and avoid the where op
        constraints = jnp.where(contour_idx < 0,
                            -jnp.inf,
                            log_L_contours[contour_idx])
        return contour_idx, constraints
    return contour_idx

