from typing import Tuple, Union, List

from jax import numpy as jnp, tree_map
from jax._src.lax.control_flow import while_loop, scan
from jax._src.lax.slicing import dynamic_update_slice

from jaxns.internals.log_semiring import LogSpace
from jaxns.types import EvidenceCalculation, SampleCollection, SampleStatistics, float_type, int_type, Reservoir
from jaxns.types import FloatArray, IntArray

__all__ = [
    'compute_evidence',
    'analyse_sample_collection',
    'compute_num_live_points_from_unit_threads'
]


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


def _update_evidence_calculation_dual(num_live_points: FloatArray, log_L: FloatArray, next_log_L_contour: FloatArray,
                                      evidence_calculation: EvidenceCalculation) -> EvidenceCalculation:
    """
    Update an evidence calculation with the next sample from the shrinkage distribution.

    Dual representation:

        Z_i = Z_i-1 + X_i * dL_i

    Whereas the prime representation is Z_i = Z_i-1 + L_i * dX_i

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

    X_mean = LogSpace(evidence_calculation.log_X_mean)
    X2_mean = LogSpace(evidence_calculation.log_X2_mean)
    Z_mean = LogSpace(evidence_calculation.log_Z_mean)
    ZX_mean = LogSpace(evidence_calculation.log_ZX_mean)
    Z2_mean = LogSpace(evidence_calculation.log_Z2_mean)
    dZ2_mean = LogSpace(evidence_calculation.log_dZ2_mean)

    dL = (next_L - L_contour).abs()
    one = LogSpace(0.)
    two = LogSpace(jnp.log(2.))

    # T ~ Beta[n, 1]
    # T_mean = n / (n + 1) = 1 / (1 + 1/n)
    # T2_mean = n / (n + 2) = 1 / (1 + 2/n)

    # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
    T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
    # T2_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 2./num_live_points))
    T2_mean = LogSpace(- jnp.logaddexp(0., jnp.log(2.) - jnp.log(num_live_points)))

    next_X_mean = X_mean * T_mean
    next_X2_mean = X2_mean * T2_mean
    next_Z_mean = Z_mean + (T_mean * X_mean) * dL
    next_ZX_mean = ZX_mean * T_mean + T2_mean * X2_mean * dL
    next_Z2_mean = Z2_mean + two * (ZX_mean * T_mean) * dL + (T2_mean * X2_mean) * dL ** 2
    next_dZ2_mean = dZ2_mean + (T2_mean * X2_mean) * dL ** 2

    next_evidence_calculation = evidence_calculation._replace(
        log_X_mean=next_X_mean.log_abs_val.astype(float_type),
        log_X2_mean=next_X2_mean.log_abs_val.astype(float_type),
        log_Z_mean=next_Z_mean.log_abs_val.astype(float_type),
        log_Z2_mean=next_Z2_mean.log_abs_val.astype(float_type),
        log_ZX_mean=next_ZX_mean.log_abs_val.astype(float_type),
        log_dZ2_mean=next_dZ2_mean.log_abs_val.astype(float_type)
    )

    return next_evidence_calculation


def compute_evidence_no_stats(sample_collection: SampleCollection, num_live_points: FloatArray) \
        -> EvidenceCalculation:
    """
    Compute the evidence by traversing the sample collection.

    Args:
        sample_collection: the sorted sample collection to use to calculate the evidence
        num_live_points: the number of live points at each sample (see compute_num_live_points_from_unit_threads)

    Returns:
        evidence calculation
    """
    num_samples = sample_collection.sample_idx

    CarryType = Tuple[EvidenceCalculation, IntArray, FloatArray]

    def thread_cond(body_state: CarryType):
        (evidence_calculation, idx, log_L_contour) = body_state
        return idx < num_samples

    def thread_body(body_state: CarryType) -> CarryType:
        (evidence_calculation, idx, log_L_contour) = body_state
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

        next_idx = idx + jnp.ones_like(idx)
        return (next_evidence_calculation, next_idx, next_log_L_contour)

    initial_evidence_calculation = _init_evidence_calculation()
    init_log_L_contour = sample_collection.reservoir.log_L_constraint[0]

    (final_evidence_calculation, final_idx, final_log_L_contour) = while_loop(thread_cond,
                                                                              thread_body,
                                                                              (initial_evidence_calculation,
                                                                               jnp.asarray(0, int_type),
                                                                               init_log_L_contour))

    return final_evidence_calculation


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

    CarryType = Tuple[EvidenceCalculation, IntArray, FloatArray, FloatArray, FloatArray]

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


def compute_evidence_dual(sample_collection: SampleCollection, num_live_points: FloatArray) \
        -> Tuple[EvidenceCalculation, SampleStatistics]:
    """
    Compute the evidence by traversing the sample collection, using dual representation.

    Z = int_0^L_max (1 - X(L)) dL

    Args:
        sample_collection: the sorted sample collection to use to calculate the evidence
        num_live_points: the number of live points at each sample (see compute_num_live_points_from_unit_threads)

    Returns:
        evidence calculation and the sample statistics
    """
    num_samples = sample_collection.sample_idx

    CarryType = Tuple[EvidenceCalculation, IntArray, FloatArray, FloatArray, FloatArray]

    def thread_cond(body_state: CarryType):
        (evidence_calculation, idx, log_L_contour, log_dZ_mean, log_X_mean) = body_state
        return idx < num_samples

    def thread_body(body_state: CarryType) -> CarryType:
        (evidence_calculation, idx, log_L_contour, log_dZ_mean, log_X_mean) = body_state
        next_num_live_points = num_live_points[idx]
        next_log_L = sample_collection.reservoir.log_L[idx]
        next_log_L_contour = next_log_L
        # Get log_dZ_mean, and log_X_mean
        next_evidence_calculation = _update_evidence_calculation_dual(
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


def analyse_sample_collection(sample_collection: SampleCollection, sorted_collection: bool = True, dual: bool = False) \
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

    if dual:
        evidence_calculation, sample_stats = compute_evidence_dual(
            sample_collection=sample_collection,
            num_live_points=num_live_points
        )
    else:
        evidence_calculation, sample_stats = compute_evidence(
            sample_collection=sample_collection,
            num_live_points=num_live_points
        )

    return evidence_calculation, sample_stats


def perfect_live_point_computation_jax(log_L_constraints: jnp.ndarray, log_L_samples: jnp.ndarray,
                                       num_samples: Union[jnp.ndarray, None] = None):
    # log_L_constraints has shape [N]
    # log_L_samples has shape [N]
    sort_idx = jnp.lexsort((log_L_constraints, log_L_samples))
    log_L_samples = log_L_samples[sort_idx]
    log_L_constraints = log_L_constraints[sort_idx]
    log_L_contour = log_L_constraints[0]

    def outer_cond(state):
        (i, log_L_contour, log_L_samples, log_L_constraints, num_live_points) = state
        return i < log_L_samples.size

    def outer_loop_body(state):
        (i, log_L_contour, log_L_samples, log_L_constraints, num_live_points) = state
        log_L_dead = log_L_samples[i]

        def inner_cond(state):
            (i, log_L_samples, log_L_constraints, num_live_points) = state
            return jnp.bitwise_and(log_L_samples[i] == log_L_dead, i < log_L_constraints.size)

        def inner_loop_body(state):
            (i, log_L_samples, log_L_constraints, num_live_points) = state
            count = jnp.sum(jnp.bitwise_and(log_L_contour < log_L_samples, log_L_constraints <= log_L_contour))
            log_L_samples = log_L_samples.at[i].set(-jnp.inf)
            log_L_constraints = log_L_constraints.at[i].set(jnp.inf)
            num_live_points = num_live_points.at[i].set(count.astype(log_L_samples.dtype))
            return (i + 1, log_L_samples, log_L_constraints, num_live_points)

        (i, log_L_samples, log_L_constraints, num_live_points) = while_loop(
            inner_cond,
            inner_loop_body,
            (i, log_L_samples, log_L_constraints, num_live_points)
        )
        log_L_contour = log_L_dead
        return (i, log_L_contour, log_L_samples, log_L_constraints, num_live_points)

    (i, log_L_contour, log_L_samples, log_L_constraints, num_live_points) = while_loop(
        outer_cond,
        outer_loop_body,
        (jnp.asarray(0, int_type), log_L_contour, log_L_samples, log_L_constraints, jnp.zeros_like(log_L_samples))
    )
    if num_samples is not None:
        empty_mask = jnp.greater_equal(jnp.arange(log_L_samples.size), num_samples)
        num_live_points = jnp.where(empty_mask, 0, num_live_points)
    return num_live_points, sort_idx


def fast_triu_rowsum(a, b):
    """
    Computes a fast row sum of the upper triangular part of the outer product of a and b.

    Args:
        a: [N] array
        b: [N] array

    Returns:
        [N] of the row sum of upper trianguarl part of outer product of a and b.
    """
    b_cumsum_rev = jnp.cumsum(b[::-1])[::-1]
    return a * b_cumsum_rev


def fast_perfect_live_point_computation_jax(log_L_constraints: jnp.ndarray, log_L_samples: jnp.ndarray,
                                            num_samples: Union[jnp.ndarray, None] = None):
    # log_L_constraints has shape [N]
    # log_L_samples has shape [N]
    sort_idx = jnp.lexsort((log_L_constraints, log_L_samples))
    log_L_samples = log_L_samples[sort_idx]
    log_L_constraints = log_L_constraints[sort_idx]
    log_L_contour = log_L_constraints[0]
    search_contours = jnp.concatenate([log_L_contour[None], log_L_samples], axis=0)

    contour_map_idx = jnp.searchsorted(search_contours, log_L_samples, side='left') - 1
    log_L_contours = search_contours[contour_map_idx]
    diag_i = jnp.arange(log_L_samples.size)
    right_most_idx = jnp.searchsorted(jnp.sort(log_L_constraints), log_L_contours, side='right') - 1
    left_most_idx = jnp.maximum(diag_i, jnp.searchsorted(log_L_samples, log_L_contours, side='right') - 1)
    num_live_points = jnp.maximum(0, right_most_idx - left_most_idx + 1)

    if num_samples is not None:
        empty_mask = jnp.greater_equal(jnp.arange(log_L_samples.size), num_samples)
        num_live_points = jnp.where(empty_mask, jnp.asarray(0., log_L_samples.dtype), num_live_points)

    return num_live_points, sort_idx


def compute_num_live_points_from_unit_threads(log_L_constraints: FloatArray, log_L_samples: FloatArray,
                                              num_samples: IntArray = None, sorted_collection: bool = True) \
        -> Union[FloatArray, Tuple[FloatArray, IntArray]]:
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
    num_live_points, sort_idx = fast_perfect_live_point_computation_jax(log_L_constraints=log_L_constraints,
                                                                        log_L_samples=log_L_samples,
                                                                        num_samples=num_samples)

    if not sorted_collection:
        return num_live_points, sort_idx

    return num_live_points


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


def combine_reservoirs(*reservoirs: Reservoir) -> Tuple[Reservoir, jnp.ndarray]:
    """
    Combine two reservoirs sorting, and updating the number of live points properly.

    Args:
        *reservoirs: Reservoirs to combine.

    Returns:
        a sorted, updated reservoir
    """
    if len(reservoirs) == 0:
        raise ValueError(f"Need at least one reservoir to combine.")
    if len(reservoirs) == 1:
        return reservoirs[0]

    def sort_reservoir(reservoir: Reservoir) -> Reservoir:
        sort_idx = jnp.lexsort((reservoir.log_L_constraint, reservoir.log_L))
        return tree_map(lambda x: x[sort_idx], reservoir)

    reservoirs: List[Reservoir] = list(map(sort_reservoir, reservoirs))

    # point_U -> concatenate and sort
    # log_L_constraint -> concatenate and sort
    # log_L -> concatenate and sort
    # num_likelihood_evaluations -> concatenate and sort
    # num_live_points -> concatenate, sort, and sum
    # num_slices -> concatenate and sort
    # iid -> concatenate and sort

    output_reservoir: Reservoir = tree_map(lambda *x: jnp.concatenate(x, axis=0), *reservoirs)
    sort_idx = jnp.lexsort((output_reservoir.log_L_constraint, output_reservoir.log_L))
    output_reservoir: Reservoir = tree_map(lambda x: x[sort_idx], output_reservoir)

    # combine each reservoir iteratively
    combined_num_live_points = jnp.zeros_like(output_reservoir.log_L)

    for r in reservoirs:
        num_live_points = compute_num_live_points_from_unit_threads(r.log_L_constraint, r.log_L,
                                                                    sorted_collection=True)
        print(num_live_points)
        j = 0
        for i, log_L in enumerate(r.log_L):
            while log_L > output_reservoir.log_L[j]:
                j += 1
            combined_num_live_points = combined_num_live_points.at[j].add(num_live_points[i])

    return output_reservoir, combined_num_live_points


def compute_shrinkage_stats(num_live_points):
    def _update_stats(state, num_live_points):
        X_mean = LogSpace(state['log_X_mean'])
        X2_mean = LogSpace(state['log_X2_mean'])

        T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
        T2_mean = LogSpace(- jnp.logaddexp((0.), jnp.log(2.) - jnp.log(num_live_points)))

        next_X_mean = X_mean * T_mean
        next_X2_mean = X2_mean * T2_mean

        return dict(log_X_mean=next_X_mean.log_abs_val, log_X2_mean=next_X2_mean.log_abs_val), dict(
            log_X_mean=next_X_mean.log_abs_val, log_X2_mean=next_X2_mean.log_abs_val)

    state = dict(
        log_X_mean=jnp.asarray(0., float_type),
        log_X2_mean=jnp.asarray(0., float_type),
    )
    _, stats = scan(_update_stats,
                    state,
                    num_live_points)
    log_X_uncert = (LogSpace(stats['log_X2_mean']) - LogSpace(stats['log_X_mean']).square()).sqrt().log_abs_val
    return stats['log_X_mean'], log_X_uncert
