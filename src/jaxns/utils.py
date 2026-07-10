from functools import partial
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx import CtxParams
from scipy.stats import kstwobign

from jaxns.log_semiring import LogSpace, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import pytree_ravel
from jaxns.random_utils import resample_indicies
from jaxns.types import FloatArray, IntArray, XType

__all__ = [
    'bruteforce_posterior_samples',
    'bruteforce_evidence',
    'resample',
    '_bit_mask',
    'save_pytree',
    'load_pytree',
    'insert_index_diagnostic'
]


def _bit_mask(int_mask, width=8):
    return list(map(int, '{:0{size}b}'.format(int_mask, size=width)))[::-1]


def resample(key, samples, log_weights, S: int | None = None, replace: bool = True):
    if S is None:
        S = int(np.size(log_weights))
    idx = resample_indicies(key=key, log_weights=log_weights, S=S, replace=replace)
    return jax.tree.map(lambda x: x[idx, ...], samples)


def save_pytree(pytree: Any, filename: str):
    path = Path(filename)
    with path.open('wb') as f:
        pickle.dump(pytree, f)


def load_pytree(filename: str):
    path = Path(filename)
    with path.open('rb') as f:
        return pickle.load(f)


def _isinstance_namedtuple(obj) -> bool:
    """
    Check if object is a namedtuple.

    Args:
        obj: object

    Returns:
        bool
    """
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )


@partial(jax.jit, inline=True, static_argnames=['grid_res', 'batch_size'])
def bruteforce_posterior_samples(model: Model, args=(), params: CtxParams | None = None, grid_res: int = 60, batch_size: int | None = None) -> tuple[
    XType, LogSpace]:
    """
    Compute the posterior with brute-force over a regular grid.

    Args:
        model: model
        args: model args
        params: model params
        grid_res: resolution of grid

    Returns:
        samples, dp
    """
    u_vec = jnp.linspace(jnp.finfo(mp_policy.measure_dtype).eps, 1. - jnp.finfo(mp_policy.measure_dtype).eps, grid_res)
    du = u_vec[1] - u_vec[0]
    u_example = model.sample_U(jax.random.PRNGKey(0), args=args, params=params)
    u_example_flat, unravel_fn = pytree_ravel(u_example)
    U_ndims = model.U_ndims(args=args, params=params)
    u_flat = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * U_ndims, indexing='ij')], axis=-1)
    x, log_L = jax.lax.map(
        lambda u: (model.transform_to_X(unravel_fn(u), args=args, params=params),
                   model.log_likelihood(unravel_fn(u), args=args, params=params, allow_nan=False)),
        xs=u_flat, batch_size=batch_size)
    dZ = LogSpace(log_L) * LogSpace(jnp.log(du)) ** U_ndims
    dZ = normalise_log_space(dZ)
    return x, dZ


@partial(jax.jit, inline=True, static_argnames=['grid_res', 'batch_size'])
def bruteforce_evidence(model: Model, args=(), params: CtxParams | None = None, grid_res: int = 60, batch_size: int | None = None) -> FloatArray:
    """
    Compute the evidence with brute-force over a regular grid.

    Args:
        model: model
        args: model args
        params: model params
        grid_res: resolution of grid
        batch_size: optional, how many points to process in a batch when applying the model.

    Returns:
        log(Z)
    """

    u_vec = jnp.linspace(jnp.finfo(mp_policy.measure_dtype).eps, 1. - jnp.finfo(mp_policy.measure_dtype).eps, grid_res)
    du = u_vec[1] - u_vec[0]
    u_example = model.sample_U(jax.random.PRNGKey(0), args=args, params=params)
    u_example_flat, unravel_fn = pytree_ravel(u_example)
    U_ndims = model.U_ndims(args=args, params=params)
    u_flat = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * U_ndims, indexing='ij')], axis=-1)
    log_L = jax.lax.map(
        lambda u: model.log_likelihood(unravel_fn(u), args=args, params=params, allow_nan=False),
        xs=u_flat, batch_size=batch_size)
    dZ = LogSpace(log_L) * LogSpace(jnp.log(du)) ** U_ndims
    return dZ.nansum().log_abs_val


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
        N = np.size(indices)
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
