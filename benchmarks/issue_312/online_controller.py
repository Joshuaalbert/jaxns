"""Chronological host learning from ordinary completed children, with no model calls."""

from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from benchmarks.issue_312.online_policy import (
    COMPRESSION_KNOTS,
    OnlinePolicy,
    initial_policy,
    predict_table,
)


@dataclass(frozen=True, slots=True)
class OnlineController:
    """All adaptive state that must accompany the production State on resume."""

    policy: OnlinePolicy
    anchors: np.ndarray  # [64, 2]
    learned: np.ndarray  # [64]
    counts: np.ndarray  # [64, 5]
    jump_sums: np.ndarray  # [64, 5]
    work_sums: np.ndarray  # [64, 5]
    exposure: np.ndarray  # [64], pending comparison exposure
    total_exposure: np.ndarray  # [64], cumulative completed-chain exposure
    generations: np.ndarray  # [64]
    next_sample: int
    deepest_compression: float

    @classmethod
    def initialise(cls, initial_roots: int) -> 'OnlineController':
        return cls(initial_policy(), np.zeros((64, 2)), np.zeros(64, bool),
                   np.zeros((64, 5), np.int64), np.zeros((64, 5)),
                   np.zeros((64, 5), np.int64), np.zeros(64), np.zeros(64), np.zeros(64, np.int64),
                   initial_roots, 0.)


def observe_completed(
    controller: OnlineController,
    *,
    start: int,
    seed_indices: np.ndarray,
    compression: np.ndarray,
    incoming_k: np.ndarray,
    choices: np.ndarray,
    squared_jumps: np.ndarray,
    work: np.ndarray,
    policy_version: int,
    adapt: bool = True,
) -> tuple[OnlineController, dict]:
    """Commit one batch's feedback, then construct the policy for the next batch.

    The old immutable object remains usable for exact replay. Newly allocated
    roots advance the ledger position but never contribute to objective or clock.
    A policy change invalidates pending comparisons wherever its predictor changed.
    """
    size = len(seed_indices)
    arrays = (compression, incoming_k, choices, squared_jumps, work)
    if any(array.shape != (size,) for array in arrays):
        raise ValueError('Feedback fields must share the committed batch width.')
    if start != controller.next_sample:
        raise ValueError('Feedback must commit the next contiguous sample prefix exactly once.')
    if policy_version != int(controller.policy.version):
        raise ValueError('Feedback policy version differs from its dispatch snapshot.')
    child = seed_indices >= 0
    if (np.any(~np.isfinite(compression[child])) or np.any(compression[child] < 0)
            or np.any(incoming_k[child] <= 0) or np.any(work <= 0)
            or np.any(~np.isfinite(squared_jumps)) or np.any(squared_jumps < 0)
            or np.any((choices[child] < 0) | (choices[child] > 4))):
        raise ValueError('Invalid compression, work, jump or randomized setting in feedback.')
    counts, jumps, costs = (array.copy() for array in (
        controller.counts, controller.jump_sums, controller.work_sums,
    ))
    exposure = controller.exposure.copy()
    total_exposure = controller.total_exposure.copy()
    strata = np.clip(np.floor(compression[child] / 2).astype(np.int64), 0, 63)
    np.add.at(counts, (strata, choices[child]), 1)
    np.add.at(jumps, (strata, choices[child]), squared_jumps[child])
    np.add.at(costs, (strata, choices[child]), work[child])
    increments = np.log1p(1. / incoming_k[child])
    np.add.at(exposure, strata, increments)
    np.add.at(total_exposure, strata, increments)
    anchors, learned = controller.anchors.copy(), controller.learned.copy()
    old_table = np.asarray(controller.policy.log_parameters)
    changes = []
    eligibility = []
    for stratum in np.flatnonzero(counts.sum(axis=1)):
        reasons = []
        if exposure[stratum] < 1.:
            reasons.append('exposure_below_one')
        if counts[stratum, 0] < 64:
            reasons.append('fewer_than_64_central')
        if min(counts[stratum, 1], counts[stratum, 2]) < 8:
            reasons.append('insufficient_h_pair')
        if min(counts[stratum, 3], counts[stratum, 4]) < 8:
            reasons.append('insufficient_duration_pair')
        eligibility.append(dict(stratum=int(stratum), reasons=reasons,
                                pending_exposure=float(exposure[stratum]),
                                cumulative_exposure=float(total_exposure[stratum])))
    if adapt:
        for stratum in np.flatnonzero((exposure >= 1.) & (counts[:, 0] >= 64)):
            centre = 2. * stratum + 1.
            old = np.array([np.interp(centre, COMPRESSION_KNOTS, old_table[:, d])
                            for d in range(2)])
            candidate = old.copy()
            eligible = []
            rates = np.divide(jumps[stratum], costs[stratum], out=np.zeros(5),
                              where=costs[stratum] > 0)
            for dimension, (minus, plus) in enumerate(((1, 2), (3, 4))):
                if min(counts[stratum, minus], counts[stratum, plus]) < 8:
                    continue
                denominator = rates[plus] + rates[minus]
                contrast = 0. if denominator == 0. else (rates[plus] - rates[minus]) / denominator
                candidate[dimension] += np.log(1.25) * contrast
                eligible.append(dimension)
            if not eligible:
                continue
            anchors[stratum] = candidate
            learned[stratum] = True
            changes.append(dict(stratum=int(stratum), dimensions=eligible,
                                previous=old.tolist(), proposed=candidate.tolist(),
                                counts=counts[stratum].tolist(), jumps=jumps[stratum].tolist(),
                                work=costs[stratum].tolist(), exposure=float(exposure[stratum])))
    generations = controller.generations.copy()
    policy = controller.policy
    reset = np.zeros(64, bool)
    maximum_log_change = np.zeros(2)
    if changes:
        table, support = predict_table(anchors, learned)
        maximum_log_change = np.max(np.abs(table - old_table), axis=0)
        changed_knots = np.any(table != old_table, axis=1)
        reset = changed_knots[:-1] | changed_knots[1:]
        reset[[item['stratum'] for item in changes]] = True
        counts[reset], jumps[reset], costs[reset], exposure[reset] = 0, 0., 0, 0.
        generations[reset] += 1
        policy = OnlinePolicy(jnp.asarray(table), jnp.asarray(support),
                              controller.policy.version + 1)
    deepest = max(controller.deepest_compression, float(compression[child].max(initial=0.)))
    updated = OnlineController(policy, anchors, learned, counts, jumps, costs, exposure,
                               total_exposure, generations, start + size, deepest)
    receipt = dict(start=start, stop=start + size, dispatch_version=policy_version,
                   next_version=int(policy.version), children=int(child.sum()),
                   roots=int((~child).sum()), changes=changes, eligibility=eligibility,
                   reset_strata=np.flatnonzero(reset).tolist(), deepest_compression=deepest,
                   maximum_log_parameter_change=maximum_log_change.tolist(),
                   real_parameter_change=bool(np.any(maximum_log_change > 1e-12)))
    return updated, receipt
