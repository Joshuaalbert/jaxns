"""Offline acceleration of the existing evidence law; no sampler changes.

Use the production sparse event plan and Kish gate. SciPy applies the same
cluster-weighted interval counts, while NumPy draws Gamma variables efficiently
on CPU. Only p_> enters evidence: Gamma(alpha_=) + Gamma(alpha_<) is exactly
Gamma(alpha_= + alpha_<), independently at each block. This identity avoids
sampling an unused split of the complementary mass.
"""

from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import logsumexp

from benchmarks.issue_312.allocation_phantoms import EvidenceInputs
from jaxns.shrinkage.classic import classic_dirichlet_concentrations
from jaxns.shrinkage.phantom import PhantomEvents, _prepare_phantom_events


class OfflineEvidence(NamedTuple):
    log_Z_samples: np.ndarray
    log_dZ_mean: np.ndarray
    phantom_gate_active: np.ndarray
    kish_participating_cluster_counts: np.ndarray


prepare_events = jax.jit(_prepare_phantom_events)


def count_operators(events: PhantomEvents) -> tuple[csr_matrix, csr_matrix]:
    """Encode the production A/B difference arrays as sparse linear maps."""
    clusters = len(events.cluster_presence)
    blocks = len(events.gate)
    operators = []
    for start_counts, end, active in (
        (events.count_A_start, events.event_a_hi, events.event_A_active),
        (events.count_B_start, events.event_b_hi, events.event_B_active),
    ):
        matrix = coo_matrix((
            np.concatenate([start_counts, -np.asarray(active, dtype=float)]),
            (np.concatenate([events.start_idx, end]),
             np.concatenate([np.arange(clusters), events.event_cluster_idx])),
        ), shape=(blocks + 1, clusters)).tocsr()
        matrix.eliminate_zeros()
        operators.append(matrix)
    return operators[0], operators[1]


def evidence_from_probabilities(
    p_gt: np.ndarray, log_likelihoods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the production evidence quadrature to valid blocks only."""
    log_volume = np.cumsum(np.log(np.clip(p_gt, 1e-300, 1.0)), axis=1)
    previous = np.concatenate([np.zeros((len(p_gt), 1)), log_volume[:, :-1]], axis=1)
    with np.errstate(divide="ignore"):
        log_dz = previous + np.log1p(-np.exp(log_volume - previous)) + log_likelihoods
    return logsumexp(log_dz, axis=1), log_dz


def draw_numpy_evidence(
    inputs: EvidenceInputs, seed: np.random.SeedSequence, *, phantoms: bool, draws: int = 4096,
) -> OfflineEvidence:
    """Draw the same joint evidence law with bounded 64-draw working batches."""
    block = inputs.block_state
    valid = np.asarray(block.valid)
    block_count = int(valid.sum())
    assert np.all(valid[:block_count]) and not np.any(valid[block_count:])
    log_l = np.asarray(block.log_L_blocks)[:block_count]
    concentration = classic_dirichlet_concentrations(block)
    alpha_gt = np.asarray(concentration.alpha_gt)[:block_count]
    alpha_complement = np.asarray(concentration.alpha_eq + concentration.alpha_lt)[:block_count]
    gate = np.zeros(len(valid), dtype=bool)
    kish = np.zeros(len(valid))
    if phantoms:
        events = jax.device_get(prepare_events(
            block_state=block, log_L_constraints=inputs.log_L_constraints,
            valid_phantom=inputs.valid_phantom, log_L_phantom=inputs.log_L_phantom,
            sample_mask=(jnp.arange(len(inputs.K_classic)) < inputs.num_samples)
            & (inputs.K_classic > 0), C_min=20.0,
        ))
        operator_a, operator_b = count_operators(events)
        gate = np.asarray(events.gate)
        kish = np.asarray(events.kish)
    rng = np.random.default_rng(seed)
    evidence = []
    log_sum_dz = np.full(block_count, -np.inf)
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        shape = (size, block_count)
        if phantoms:
            race_gt = rng.gamma(alpha_gt, size=shape)
            race_complement = rng.gamma(alpha_complement, size=shape)
            # One weight per chain per draw, reused at every block and for
            # every retained state in the chain, including self-transitions.
            weights = rng.exponential(size=(len(events.cluster_presence), size))
            weights *= events.cluster_presence[:, None]
            weighted_a = np.cumsum((operator_a @ weights)[:block_count], axis=0).T
            weighted_b = np.cumsum((operator_b @ weights)[:block_count], axis=0).T
            p_gt = (race_gt + weighted_b * gate[:block_count]) / (
                race_gt + race_complement + weighted_a * gate[:block_count]
            )
        else:
            p_gt = rng.beta(alpha_gt, alpha_complement, size=shape)
        log_z, log_dz = evidence_from_probabilities(p_gt, log_l)
        evidence.append(log_z)
        log_sum_dz = np.logaddexp(log_sum_dz, logsumexp(log_dz, axis=0))
    means = np.full(len(valid), -np.inf)
    means[:block_count] = log_sum_dz - np.log(draws)
    return OfflineEvidence(np.concatenate(evidence), means, gate, kish)
