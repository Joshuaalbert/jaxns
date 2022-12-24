from typing import Tuple, Union

from etils.array_types import FloatArray, ui64
from jax import numpy as jnp, tree_map
from jax._src.lax.control_flow import while_loop

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.types import float_type, int_type
from jaxns.new_code.types import EvidenceCalculation, SampleCollection, SampleStatistics


def _init_evidence_calculation() -> EvidenceCalculation:
    """
    Returns an initial evidence calculation object.

    Returns:
        EvidenceCalculation
    """
    evidence_calculation = EvidenceCalculation(
        log_X_mean=jnp.asarray(0., float_type),
        log_X2_mean=jnp.asarray(0., float_type),
        log_Z_mean=jnp.asarray(-jnp.inf, float_type),
        log_ZX_mean=jnp.asarray(-jnp.inf, float_type),
        log_Z2_mean=jnp.asarray(-jnp.inf, float_type),
        log_dZ2_mean=jnp.asarray(-jnp.inf, float_type)
    )
    return evidence_calculation


def _update_evidence_calculation(num_live_points: FloatArray, log_L: FloatArray, next_log_L_contour: FloatArray,
                                 evidence_calculation: EvidenceCalculation) -> EvidenceCalculation:
    """
    Update an evidence calculation with the next sample from the shrinkage distribution.

    Args:
        num_live_points: number of live points at the current shrinkage step
        log_L: the log likelihood of the current contour
        next_log_L_contour: the log likelihood of the next contour after shrinkage
        evidence_calculation: the evidence calculation to update

    Returns:
        an updated evidence calculation
    """
    # num_live_points = num_live_points.astype(float_type)
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
        log_X_mean=next_X_mean.log_abs_val.astype(float_type),
        log_X2_mean=next_X2_mean.log_abs_val.astype(float_type),
        log_Z_mean=next_Z_mean.log_abs_val.astype(float_type),
        log_Z2_mean=next_Z2_mean.log_abs_val.astype(float_type),
        log_ZX_mean=next_ZX_mean.log_abs_val.astype(float_type),
        log_dZ2_mean=next_dZ2_mean.log_abs_val.astype(float_type)
    )
    # log_Z_mean, log_Z_var = linear_to_log_stats(next_evidence_calculation.log_Z_mean, log_f2_mean=next_evidence_calculation.log_Z2_mean)

    # next_evidence_calculation = tree_map(lambda old, new: jnp.where(jnp.isnan(new), old, new),
    #                                      evidence_calculation, next_evidence_calculation)

    return next_evidence_calculation


def compute_evidence(sample_collection: SampleCollection, num_live_points: FloatArray) \
        -> Tuple[EvidenceCalculation, SampleStatistics]:
    """
    Compute the evidence by traversing the sample collection.

    Args:
        sample_collection: the sorted sample collection to use to calculate the evidence
        num_live_points: the number of live points at each sample (see compute_num_live_points_from_unit_threads)

    Returns:
        evidence calculation and the sample statistics
    """
    num_samples = sample_collection.sample_idx

    CarryType = Tuple[EvidenceCalculation, ui64, FloatArray, FloatArray, FloatArray]

    def thread_cond(body_state: CarryType):
        (evidence_calculation, idx, log_L_contour, log_dZ_mean, log_X_mean) = body_state
        return idx < num_samples

    def thread_body(body_state: CarryType) -> CarryType:
        (evidence_calculation, idx, log_L_contour, log_dZ_mean, log_X_mean) = body_state
        next_num_live_points = num_live_points[idx]
        next_log_L = sample_collection.reservoir.log_L[idx]
        next_log_L_contour = next_log_L
        # Get log_dZ_mean, and log_X_mean
        next_evidence_calculation = _update_evidence_calculation(
            num_live_points=next_num_live_points,
            log_L=log_L_contour,
            next_log_L_contour=next_log_L,
            evidence_calculation=evidence_calculation
        )

        next_dZ_mean = (LogSpace(next_evidence_calculation.log_Z_mean)
                        - LogSpace(evidence_calculation.log_Z_mean)).abs()

        next_log_dZ_mean = log_dZ_mean.at[idx].set(next_dZ_mean.log_abs_val)
        next_log_X_mean = log_X_mean.at[idx].set(next_evidence_calculation.log_X_mean)
        next_idx = idx + jnp.ones_like(idx)
        return (next_evidence_calculation, next_idx, next_log_L_contour, next_log_dZ_mean, next_log_X_mean)

    initial_evidence_calculation = _init_evidence_calculation()
    init_log_L_contour = sample_collection.reservoir.log_L_constraint[0]
    init_log_dZ_mean = jnp.full(num_live_points.shape, -jnp.inf, float_type)
    init_log_X_mean = jnp.full(num_live_points.shape, -jnp.inf, float_type)

    (final_evidence_calculation, final_idx, final_log_L_contour, final_log_dZ_mean, final_log_X_mean) = \
        while_loop(thread_cond,
                   thread_body,
                   (initial_evidence_calculation, jnp.asarray(0, int_type), init_log_L_contour,
                    init_log_dZ_mean, init_log_X_mean))

    sample_stats = SampleStatistics(
        num_live_points=num_live_points,
        log_X_mean=final_log_X_mean,
        log_dZ_mean=final_log_dZ_mean,
    )
    return final_evidence_calculation, sample_stats


def analyse_sample_collection(sample_collection: SampleCollection, sorted_collection: bool = True) \
        -> Tuple[EvidenceCalculation, SampleStatistics]:
    """
    Computes the evidence and statistics.

    Args:
        sample_collection: the sample collection
        sorted_collection: whether the sample collection is sorted

    Returns:
        evidence calculation and sample stats
    """
    # Sample collection has unsorted samples, and incorrect num_live_points.
    if sorted_collection:
        num_live_points = compute_num_live_points_from_unit_threads(
            log_L_constraints=sample_collection.reservoir.log_L_constraint,
            log_L_samples=sample_collection.reservoir.log_L,
            num_samples=sample_collection.sample_idx,
            sorted_collection=True
        )
    else:
        num_live_points, sort_idx = compute_num_live_points_from_unit_threads(
            log_L_constraints=sample_collection.reservoir.log_L_constraint,
            log_L_samples=sample_collection.reservoir.log_L,
            num_samples=sample_collection.sample_idx,
            sorted_collection=False
        )

        sample_collection = sample_collection._replace(
            reservoir=tree_map(lambda x: x[sort_idx], sample_collection.reservoir)
        )

    evidence_calculation, sample_stats = compute_evidence(
        sample_collection=sample_collection,
        num_live_points=num_live_points
    )

    return evidence_calculation, sample_stats


def compute_num_live_points_from_unit_threads(log_L_constraints: FloatArray, log_L_samples: FloatArray,
                                              num_samples: ui64 = None, sorted_collection: bool = True) \
        -> Union[FloatArray, Tuple[FloatArray, ui64]]:
    """
    Compute the number of live points of shrinkage distribution, from an arbitrary list of samples with
    corresponding sampling constraints.

    Args:
        log_L_constraints: [N] likelihood constraint that sample was uniformly sampled within
        log_L_samples: [N] likelihood of the sample
        sorted_collection: bool, whether the sample collection was already sorted.

    Returns:
        if sorted_collection is true:
            num_live_points for shrinkage distribution
        otherwise:
            num_live_points for shrinkage distribution, and sort indicies
    """
    # mask the samples that are not yet taken

    if num_samples is None:
        num_samples = log_L_samples.size
    empty_mask = jnp.greater_equal(jnp.arange(log_L_samples.size), num_samples)

    # masking samples is already done, since they are inf by default.
    # log_L_samples = jnp.where(empty_mask, jnp.inf, log_L_samples)
    # log_L_constraints = jnp.where(empty_mask, jnp.inf, log_L_constraints)
    # sort log_L_samples, breaking degeneracies on log_L_constraints

    if not sorted_collection:
        idx_sort = jnp.lexsort((log_L_constraints, log_L_samples))
        log_L_samples = log_L_samples[idx_sort]
        log_L_constraints = log_L_constraints[idx_sort]
        empty_mask = empty_mask[idx_sort]

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
    n = jnp.where(empty_mask, 0, n)
    n = jnp.asarray(n, log_L_samples.dtype)
    if not sorted_collection:
        return (n, idx_sort)
    return n
