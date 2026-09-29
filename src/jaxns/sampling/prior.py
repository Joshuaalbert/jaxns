"""Unconditional prior draws for children of the negative sentinel."""

from jax import numpy as jnp
from jaxctx import CtxParams

from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType


def sample_prior(
        key: PRNGKey,
        model: Model,
        args: tuple,
        params: CtxParams | None,
        *,
        valid: BoolArray = True,
) -> tuple[UType, FloatArray, IntArray]:
    """Draw one sentinel child, retaining zero and NaN-mapped likelihoods.

    The model maps NaN to -inf. Redrawing either kind of zero would condition
    the prior and lose its normalization. Inactive vector lanes are only filler.
    """
    U = model.sample_U(key, args=args, params=params)
    log_L = model.log_likelihood(
        U, args=args, params=params,
    ).astype(mp_policy.measure_dtype)
    return U, log_L, jnp.asarray(valid, mp_policy.count_dtype)
