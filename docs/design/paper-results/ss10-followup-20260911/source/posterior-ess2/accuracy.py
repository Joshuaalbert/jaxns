"""Classic posterior accuracy for the paper's repository-defined SS10."""

import time

import jax
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from benchmarks.paper_reproduction.posterior_cases import reference_values
from benchmarks.paper_reproduction.posterior_cases import ss10_parameters
from benchmarks.paper_reproduction.run_posterior import classic_metrics
from jaxns.state import State


def ss10_accuracy(state: State) -> dict[str, float | list[float]]:
    """Compare classic evidence and component responsibilities to finite-box truth."""
    started = time.perf_counter()
    metrics = classic_metrics(state)
    results = state.to_result().trim()
    x = np.asarray(jax.tree.leaves(results.X_samples)[0])  # [N,10], classics only
    log_weights = np.asarray(results.log_dp)  # [N]
    weights = np.exp(log_weights - logsumexp(log_weights))
    means, scales = ss10_parameters("repository")
    component_log_L = np.stack([
        norm.logpdf(x, loc=mean, scale=scale).sum(axis=1)
        for mean, scale in zip(means, scales, strict=True)
    ], axis=1)  # [N,2]
    responsibility = np.exp(component_log_L[:, 0] - logsumexp(component_log_L, axis=1))
    spike_mass = float(weights @ responsibility)
    reference = reference_values("ss10", "repository")
    return dict(
        log_Z=float(metrics.log_Z), log_Z_uncert=float(metrics.log_Z_uncert),
        log_Z_error=float(metrics.log_Z) - reference["log_Z"],
        spike_mass=spike_mass, spike_mass_error=spike_mass - reference["spike_mass"],
        true_log_Z=reference["log_Z"], true_spike_mass=reference["spike_mass"],
        kish_ess=float(metrics.kish_ess), posterior_mean=(weights @ x).tolist(),
        analysis_seconds=time.perf_counter() - started,
    )
