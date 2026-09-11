"""Classic stopping and accuracy metrics shared by phantom-seeding benchmarks."""

from typing import NamedTuple

import jax
from jax import numpy as jnp

from jaxns.algorithm.race_tree import build_block_state
from jaxns.shrinkage.classic import (
    classic_dirichlet_concentrations,
    expected_evidence_summary,
    expected_log_posterior_weights,
)
from jaxns.state import State
from jaxns.stats_utils import effective_sample_size_kish


class ClassicMetrics(NamedTuple):
    log_Z: jax.Array
    log_Z_uncert: jax.Array
    kish_ess: jax.Array


@jax.jit
def classic_metrics(state: State) -> ClassicMetrics:
    """Match State.to_result's classic moments and ESS without X transforms."""
    blocks = build_block_state(
        state.samples, root_out_degree=state.root_out_degree,
        num_samples=state.num_samples, likelihood_order=state.likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(blocks)
    evidence = expected_evidence_summary(blocks, concentrations)
    log_weights = expected_log_posterior_weights(blocks, concentrations)
    log_weights = jnp.where(
        jnp.arange(state.samples.log_likelihoods.size) < state.num_samples,
        log_weights, -jnp.inf,
    )  # [capacity], classic slots only; padded storage has zero weight
    return ClassicMetrics(evidence.log_Z_mean, evidence.log_Z_uncert,
                          effective_sample_size_kish(log_weights))
