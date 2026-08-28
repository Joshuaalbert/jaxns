"""Stationary seed selection helpers for constrained chains."""

from functools import partial

import jax

from jaxns.random_utils import sample_uniformly_masked
from jaxns.samples import Samples, SeedPoint
from jaxns.types import FloatArray, PRNGKey


@partial(jax.jit, inline=True)
def get_seed_point(
        key: PRNGKey,
        samples: Samples,
        log_L_constraint: FloatArray,
) -> SeedPoint:
    """Choose uniformly among existing samples above the strict contour."""
    select_mask = samples.log_likelihoods > log_L_constraint
    return sample_uniformly_masked(
        key=key,
        v=SeedPoint(U0=samples.U_samples, log_L0=samples.log_likelihoods),
        select_mask=select_mask,
        num_samples=1,
        squeeze=True,
    )
