from typing import Tuple, Union

from etils.array_types import FloatArray, IntArray
from jax import numpy as jnp, tree_map, random
from jax._src.lax.control_flow import while_loop, fori_loop
from jax._src.lax.slicing import dynamic_update_slice

from jaxns.internals.log_semiring import LogSpace, normalise_log_space
from jaxns.internals.maps import replace_index
from jaxns.internals.stats import linear_to_log_stats
from jaxns.types import EvidenceCalculation, SampleCollection, SampleStatistics, float_type, int_type
from jaxns.types import NestedSamplerState

__all__ = ['compute_evidence',
           'analyse_sample_collection',
           'compute_num_live_points_from_unit_threads']


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


def perfect_live_point_computation_jax(log_L_constraints: jnp.ndarray, log_L_samples: jnp.ndarray,
                                       num_samples: jnp.ndarray | None = None):
    # log_L_constraints has shape [N]
    # log_L_samples has shape [N]
    sort_idx = jnp.lexsort((log_L_constraints, log_L_samples))

    log_L_contour = log_L_constraints[sort_idx[0]]

    def loop_body(j, carry):
        log_L_dead, log_L_contour, log_L_samples, log_L_constraints, num_live_points = carry
        i = sort_idx[j]
        log_L_dead = log_L_samples[i]
        count = jnp.sum(jnp.bitwise_and(log_L_contour <= log_L_samples, log_L_constraints <= log_L_contour))
        # log_L_contour = jnp.where(log_L_dead > log_L_contour, log_L_dead, log_L_contour)
        log_L_contour = log_L_dead
        log_L_samples = log_L_samples.at[i].set(-jnp.inf)
        log_L_constraints = log_L_constraints.at[i].set(jnp.inf)
        num_live_points = num_live_points.at[j].set(count)
        return log_L_dead, log_L_contour, log_L_samples, log_L_constraints, num_live_points

    carry = (0, log_L_contour, log_L_samples, log_L_constraints, jnp.zeros(len(log_L_samples), dtype=jnp.int32))
    _, _, _, _, num_live_points = fori_loop(0, len(sort_idx), loop_body, carry)

    if num_samples is not None:
        empty_mask = jnp.greater_equal(jnp.arange(log_L_samples.size), num_samples)
        num_live_points = jnp.where(empty_mask, jnp.asarray(0., log_L_samples.dtype), num_live_points)

    return num_live_points, sort_idx


def fast_perfect_live_point_computation_jax(log_L_constraints: jnp.ndarray, log_L_samples: jnp.ndarray,
                                            num_samples: jnp.ndarray | None = None):
    # log_L_constraints has shape [N]
    # log_L_samples has shape [N]
    sort_idx = jnp.lexsort((log_L_constraints, log_L_samples))

    log_L_contour = log_L_constraints[sort_idx[0]]

    # masking samples is already done, since they are inf by default.
    # log_L_samples = jnp.where(empty_mask, jnp.inf, log_L_samples)
    # log_L_constraints = jnp.where(empty_mask, jnp.inf, log_L_constraints)
    # sort log_L_samples, breaking degeneracies on log_L_constraints

    # n = jnp.sum(available & (log_L_samples >= contour) & (log_L_constraints <= contour))
    # n = jnp.sum(available & (log_L_constraints <= contour))
    # n = available.size - jnp.sum(jnp.bitwise_not(available) | jnp.bitwise_not(log_L_constraints <= contour))
    # n = available.size - jnp.sum(jnp.bitwise_not(available)) - jnp.sum(jnp.bitwise_not(log_L_constraints <= contour)) + jnp.sum(jnp.bitwise_not(available) & jnp.bitwise_not(log_L_constraints <= contour))
    # n = jnp.sum(available) - jnp.sum(log_L_constraints > contour) + jnp.sum(jnp.bitwise_not(available) & (log_L_constraints > contour))
    # Since jnp.sum(jnp.bitwise_not(available) & (log_L_constraints > contour)) can be shown to be zero
    # n = jnp.sum(available) - jnp.sum(log_L_constraints > contour)

    _log_L_samples = jnp.concatenate([log_L_contour[None], log_L_samples[sort_idx]])
    n_base = jnp.arange(1, log_L_samples.size + 1)[::-1]

    u = jnp.searchsorted(_log_L_samples, log_L_constraints[sort_idx], side='left')
    b = u.size - jnp.cumsum(jnp.bincount(u, length=u.size))
    num_live_points = n_base - b
    num_live_points = num_live_points.astype(log_L_samples.dtype)

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
    # mask the samples that are not yet taken

    num_live_points, sort_idx = fast_perfect_live_point_computation_jax(log_L_constraints=log_L_constraints,
                                                                        log_L_samples=log_L_samples,
                                                                        num_samples=num_samples)

    if not sorted_collection:
        return num_live_points, sort_idx

    return num_live_points


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


def evidence_goal(state: NestedSamplerState):
    """
    Estimates the impact of adding a sample at a certain likelihood contour by computing the impact of removing a point.

    Args:
        state:

    Returns:

    """
    # evidence uncertainty minimising goal.
    # remove points and see what increases uncertainty the most.

    _, log_Z_var0 = linear_to_log_stats(log_f_mean=state.evidence_calculation.log_Z_mean,
                                        log_f2_mean=state.evidence_calculation.log_Z2_mean)
    num_shrinkages = -state.evidence_calculation.log_X_mean
    delta_idx = state.sample_idx / (2. * num_shrinkages)

    def body(body_state):
        (remove_idx, inf_max_dvar, inf_max_dvar_idx, sup_max_dvar, sup_max_dvar_idx) = body_state
        # removing a sample is equivalent to setting that n=inf at that point
        perturbed_num_live_points = replace_index(state.sample_collection.num_live_points, jnp.inf,
                                                  remove_idx.astype(int_type))
        perturbed_sample_collection = state.sample_collection._replace(num_live_points=perturbed_num_live_points)
        (evidence_calculation, _, _, _, _) = \
            compute_evidence(num_samples=state.sample_idx, sample_collection=perturbed_sample_collection)
        _, log_Z_var = linear_to_log_stats(log_f_mean=evidence_calculation.log_Z_mean,
                                           log_f2_mean=evidence_calculation.log_Z2_mean)
        dvar = log_Z_var - log_Z_var0

        return remove_idx + delta_idx, dvar


def _get_dynamic_goal(state: NestedSamplerState, G: jnp.ndarray):
    """
    Get contiguous contours that we'd like to reinforce.

    We have two objectives, which can be mixed by setting `G`.
    G=0: choose contours that decrease evidence uncertainty the most.
    G=1: choose contours that increase ESS the most.

    Note: This slightly departs from the Dynamic Nested Sampling paper.
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


def get_dynamic_goal(key, state: NestedSamplerState, num_samples: int, G: jnp.ndarray) -> Tuple[
    jnp.ndarray, jnp.ndarray]:
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
