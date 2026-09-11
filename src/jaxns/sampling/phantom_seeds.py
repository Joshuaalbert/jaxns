"""Exact all-phantom eligibility counts and original-order rank selection."""

import jax
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.samples import Samples, SeedPoint
from jaxns.types import FloatArray, IntArray


def seed_stride(samples: Samples) -> int:
    """Use capacity-independent identities: row * (P + 1) + point slot."""
    if samples.phantom_samples.U_samples is None:
        return 1
    return samples.phantom_samples.log_L.shape[1] + 1


def phantom_seed_cumulative(
        samples: Samples,
        num_samples: IntArray,
        constraint: FloatArray,
) -> IntArray:
    """Count all eligible phantoms per classic row in O(N log P) work."""
    phantoms = samples.phantom_samples
    sorted_log_L = phantoms.seed_log_L_sorted  # [N, P]
    if sorted_log_L is None:
        raise ValueError("Phantom coordinates require an eligibility index.")
    count = sorted_log_L.shape[1] - jax.vmap(
        lambda row: jnp.searchsorted(row, constraint, side="right")
    )(sorted_log_L)  # [N]
    valid = (
        (jnp.arange(count.size) < num_samples)
        & (samples.log_L_constraints <= constraint)
    )  # [N]
    return jnp.cumsum(jnp.where(valid, count, 0), dtype=mp_policy.index_dtype)


def phantom_rank_to_identity(
        samples: Samples,
        cumulative: IntArray,
        constraint: FloatArray,
        ranks: IntArray,
) -> IntArray:
    """Map ranks to every eligible phantom, row then transition order.

    The sorted likelihood cache only counts eligibility. The selected slots
    use the original transition order, so likelihood does not favour a point.
    No coordinate population or [N, P, proposals] mask is materialised.
    """
    phantoms = samples.phantom_samples
    rows = jnp.minimum(
        jnp.searchsorted(cumulative, ranks, side="right"),
        cumulative.size - 1,
    )  # [C]
    previous = jnp.where(rows > 0, cumulative[jnp.maximum(rows - 1, 0)], 0)
    eligible = (
        phantoms.valid_mask[rows] & (phantoms.log_L[rows] > constraint)
    )  # [C, P]
    slots = jnp.argmax(
        jnp.cumsum(eligible, axis=1) > (ranks - previous)[:, None], axis=1,
    )  # [C]
    return (rows * seed_stride(samples) + slots + 1).astype(
        mp_policy.index_dtype,
    )


def phantom_seed_cumulative_batch(
        samples: Samples,
        num_samples: IntArray,
        constraints: FloatArray,
) -> IntArray:
    """Share likelihood-cache reads across all contour queries in a batch.

    This is the transpose of independently calling phantom_seed_cumulative
    for every constraint. Integer counts and cumulative ranks are identical.
    Counts are processed in small row tiles. The complete rank index is
    [S, N]; there is never a [N, P, S] eligibility array.
    """
    sorted_log_L = samples.phantom_samples.seed_log_L_sorted  # [N, P]
    if sorted_log_L is None:
        raise ValueError("Phantom coordinates require an eligibility index.")
    tile_size = 256
    capacity, num_phantom = sorted_log_L.shape
    num_tiles = (capacity + tile_size - 1) // tile_size
    offsets = jnp.arange(tile_size, dtype=mp_policy.index_dtype)
    cumulative = jnp.zeros(
        (constraints.size, num_tiles * tile_size), mp_policy.index_dtype,
    )  # [S, rounded N]

    def add_tile(tile_idx, carry):
        cdf, previous_count = carry
        rows = tile_idx * tile_size + offsets  # [tile]
        safe_rows = jnp.minimum(rows, capacity - 1)
        # Keep one small likelihood tile in cache while querying every lane.
        # Binary-search and scan temporaries are bounded by tile*S, not N*S.
        count = num_phantom - jax.vmap(
            lambda row: jnp.searchsorted(row, constraints, side="right")
        )(sorted_log_L[safe_rows])  # [tile, S]
        eligible = (
            (rows < num_samples)[:, None]
            & (samples.log_L_constraints[safe_rows, None]
               <= constraints[None, :])
        )  # [tile, S]
        tile_cdf = jnp.cumsum(
            jnp.where(eligible, count, 0).T, axis=1,
            dtype=mp_policy.index_dtype,
        ) + previous_count[:, None]  # [S, tile]
        cdf = jax.lax.dynamic_update_slice(
            cdf, tile_cdf, (0, tile_idx * tile_size),
        )
        return cdf, tile_cdf[:, -1]

    cumulative, _ = jax.lax.fori_loop(
        0, num_tiles, add_tile,
        (cumulative, jnp.zeros(constraints.size, mp_policy.index_dtype)),
    )
    return cumulative[:, :capacity]  # [S, N]


def gather_seed_points(samples: Samples, identities: IntArray) -> SeedPoint:
    """Decode the same seed identities for local and distributed requests."""
    stride = seed_stride(samples)
    rows, slots = identities // stride, identities % stride  # [S]
    classic_U = jax.tree.map(lambda u: u[rows], samples.U_samples)
    classic_log_L = samples.log_likelihoods[rows]
    if stride == 1:
        return SeedPoint(U0=classic_U, log_L0=classic_log_L)
    phantom_slot = jnp.maximum(slots - 1, 0)  # [S]
    phantom_U = jax.tree.map(
        lambda u: u[rows, phantom_slot], samples.phantom_samples.U_samples,
    )
    return SeedPoint(
        U0=jax.tree.map(
            lambda classic, phantom: jnp.where(
                (slots > 0).reshape((-1,) + (1,) * (classic.ndim - 1)),
                phantom, classic,
            ),
            classic_U, phantom_U,
        ),
        log_L0=jnp.where(
            slots > 0, samples.phantom_samples.log_L[rows, phantom_slot],
            classic_log_L,
        ),
    )
