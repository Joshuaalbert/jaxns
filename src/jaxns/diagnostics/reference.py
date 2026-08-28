"""Small-grid reference calculations used to validate scientific models."""

from functools import partial

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxctx import CtxParams

from jaxns.log_semiring import LogSpace, normalise_log_space
from jaxns.model import Model
from jaxns.types import FloatArray, XType


def bruteforce_posterior_samples(
    model: Model,
    args=(),
    params: CtxParams | None = None,
    grid_res: int = 60,
    batch_size: int | None = None,
) -> tuple[XType, LogSpace]:
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
    u_example = model.sample_U(jax.random.PRNGKey(0), args=args, params=params)
    u_example_flat, unravel_fn = ravel_pytree(u_example)
    # The regular grid must follow the model's U-space dtype contract. The
    # standard unravel function also restores each leaf's original dtype when
    # a model mixes precisions, as required by jaxctx's prior realisation.
    # Midpoint quadrature avoids inverse-CDF endpoints while assigning exactly
    # unit total prior volume: ``grid_res`` cells each have width ``1/N``.
    du = jnp.asarray(1. / grid_res, dtype=u_example_flat.dtype)
    u_vec = (
        jnp.arange(grid_res, dtype=u_example_flat.dtype) + 0.5
    ) * du
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

    u_example = model.sample_U(jax.random.PRNGKey(0), args=args, params=params)
    u_example_flat, unravel_fn = ravel_pytree(u_example)
    # Keep the grid compatible with model U-space and let the standard
    # unravel function restore heterogeneous leaf dtypes before evaluation.
    # Midpoint quadrature avoids inverse-CDF endpoints while assigning exactly
    # unit total prior volume: ``grid_res`` cells each have width ``1/N``.
    du = jnp.asarray(1. / grid_res, dtype=u_example_flat.dtype)
    u_vec = (
        jnp.arange(grid_res, dtype=u_example_flat.dtype) + 0.5
    ) * du
    U_ndims = model.U_ndims(args=args, params=params)
    u_flat = jnp.stack([x.flatten() for x in jnp.meshgrid(*[u_vec] * U_ndims, indexing='ij')], axis=-1)
    log_L = jax.lax.map(
        lambda u: model.log_likelihood(unravel_fn(u), args=args, params=params, allow_nan=False),
        xs=u_flat, batch_size=batch_size)
    dZ = LogSpace(log_L) * LogSpace(jnp.log(du)) ** U_ndims
    return dZ.nansum().log_abs_val
