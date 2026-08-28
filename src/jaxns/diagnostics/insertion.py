"""Statistical diagnostics for nested-sampling insertion indices."""

import jax
import numpy as np
from jax import numpy as jnp
from scipy.stats import kstwobign

from jaxns.types import IntArray


def insert_index_diagnostic(insert_indices: IntArray, num_live_points: int) -> np.ndarray:
    """
    Compute the insert index diagnostic of Fowlie et al. (2020).
    Note: JAXNS doesn't hold number of live points constant over the run, so this diagnostic is not directly applicable as implemented.

    Args:
        insert_indices: [N] array of insert indices
        num_live_points: number of live points, must be constant over run.

    Returns:
        p-value of the insert index being uniformly distributed.
    """
    # TODO(joshuaalbert): ask Andrew if he thinks this is still a useful diagnostic to implement, and if so, how to adapt it to the case of varying number of live points.

    insert_indices = jax.tree.map(np.asarray, insert_indices)
    if len(np.shape(insert_indices)) != 1:
        raise ValueError(f"Expected 1D array, got {np.shape(insert_indices)}")

    def _get_p_value(indices):
        # Get expected CDF
        expected_cdf = np.arange(num_live_points) / num_live_points
        # Get observed CDF
        observed_cdf = np.bincount(indices, minlength=num_live_points) / num_live_points
        observed_cdf = np.cumsum(observed_cdf)
        observed_cdf = observed_cdf[:num_live_points]
        observed_cdf /= observed_cdf[-1]
        # Compute KS statistic
        ks_statistic = np.max(jnp.abs(observed_cdf - expected_cdf))
        #  We convert the test-statistic into a p-value using an asymptotic approximation of the Kolmogorov distribution
        # P(KS > ks_statistic) = 1 - CDF(ks_statistic)
        p_value = kstwobign.sf(ks_statistic * np.sqrt(num_live_points))
        return p_value

    # Break into chunks
    chunk_size = num_live_points
    # For each chunk compute p-value
    p_values = []
    for i in range(0, np.size(insert_indices), chunk_size):
        # Compute p-value
        p_value = _get_p_value(indices=insert_indices[i:i + chunk_size])  # []
        p_values.append(p_value)
    # Compute minimum p-value adjusted for multiple tests
    min_p_value = np.min(np.stack(p_values))
    num_chunks = len(p_values)
    # Return adjusted p-value
    return 1. - (1. - min_p_value) ** num_chunks
