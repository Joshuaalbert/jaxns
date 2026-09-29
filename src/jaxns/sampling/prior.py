"""Unconditional prior draws for children of the negative sentinel."""

import jax
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
    """Draw one sentinel child, retaining true zero likelihoods.

    Invalid model evaluations retain the existing retry stream. Keeping NaNs
    distinct from log(0) is essential: rejecting zeros conditions the prior
    and loses its normalization. Inactive vector lanes are only filler.
    """
    def draw(draw_key):
        U = model.sample_U(draw_key, args=args, params=params)
        log_L = model.log_likelihood(
            U, args=args, params=params, allow_nan=True,
        ).astype(mp_policy.measure_dtype)
        return draw_key, U, log_L, jnp.asarray(valid, mp_policy.count_dtype)

    def invalid(carry):
        return valid & jnp.isnan(carry[2])

    def redraw(carry):
        old_key, _, _, old_evals = carry
        next_key, proposal_key = jax.random.split(old_key)
        _, U, log_L, _ = draw(proposal_key)
        return next_key, U, log_L, old_evals + 1

    _, U, log_L, num_evals = jax.lax.while_loop(
        invalid, redraw, draw(key),
    )
    return U, log_L, num_evals
