"""Exact append-order phantom counts, cached in fixed blocks of chains.

This index belongs to sampler state, not scientific sample rows: sorting a
posterior view must not permute a cache whose entries describe whole blocks.
Likelihood endpoints count membership only. Rank lookup retains the original
row/transition order used by the all-phantom selector.
"""

import dataclasses
from functools import partial

import jax
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import Samples
from jaxns.types import FloatArray, IntArray


@dataclasses.dataclass(frozen=True, slots=True)
class PhantomSeedIndex(PureDataclassPytree):
    """Immutable counts for sealed blocks; the unfinished tail stays visible."""

    birth_sorted: FloatArray  # [blocks, B]
    birth_count: IntArray  # [blocks, B], cumulative valid interval counts
    likelihood_sorted: FloatArray  # [blocks, B*P], invalid endpoints padded +inf
    num_sealed: IntArray  # [], number of complete indexed blocks


PhantomSeedIndex.register_pytree()


@partial(jax.jit, inline=True, static_argnames=("block_size",))
def build_phantom_seed_index(
        samples: Samples,
        num_samples: IntArray,
        *,
        block_size: int = 256,
) -> PhantomSeedIndex:
    """Build a cache from the accepted prefix, without changing sample records."""
    if block_size < 1:
        raise ValueError("Phantom seed block size must be positive.")
    capacity, num_phantom = samples.phantom_samples.log_L.shape
    if capacity < 1 or num_phantom < 1:
        raise ValueError("Phantom indexing requires nonempty row and slot capacities.")
    num_blocks = (capacity + block_size - 1) // block_size
    index = PhantomSeedIndex(
        birth_sorted=jnp.full(
            (num_blocks, block_size), jnp.inf, samples.log_L_constraints.dtype,
        ),
        birth_count=jnp.zeros((num_blocks, block_size), mp_policy.index_dtype),
        likelihood_sorted=jnp.full(
            (num_blocks, block_size * num_phantom), jnp.inf,
            samples.phantom_samples.log_L.dtype,
        ),
        num_sealed=jnp.asarray(0, mp_policy.index_dtype),
    )
    return update_phantom_seed_index(index, samples, num_samples)


@partial(jax.jit, inline=True)
def update_phantom_seed_index(
        index: PhantomSeedIndex,
        samples: Samples,
        num_samples: IntArray,
) -> PhantomSeedIndex:
    """Seal newly completed blocks once; leave every partial-tail point visible."""
    block_size = index.birth_sorted.shape[1]
    offsets = jnp.arange(block_size, dtype=mp_policy.index_dtype)
    num_sealed = (num_samples // block_size).astype(mp_policy.index_dtype)

    def seal(block, current):
        rows = block * block_size + offsets  # [B]
        # Clipping permits tracing the zero-iteration loop at capacities < B.
        safe_rows = jnp.minimum(rows, samples.log_likelihoods.size - 1)
        birth = samples.log_L_constraints[safe_rows]  # [B]
        likelihood = samples.phantom_samples.log_L[safe_rows]  # [B,P]
        valid = (
            (rows < num_samples)[:, None]
            & samples.phantom_samples.valid_mask[safe_rows]
            & (birth[:, None] < likelihood)
        )  # [B,P]; zero-width intervals can never supply a seed
        count = jnp.sum(valid, axis=1, dtype=mp_policy.index_dtype)  # [B]
        birth, count = jax.lax.sort((birth, count), num_keys=1)
        count = jnp.cumsum(count, dtype=mp_policy.index_dtype)
        endpoints = jnp.sort(jnp.where(valid, likelihood, jnp.inf).reshape(-1))
        return PhantomSeedIndex(
            birth_sorted=current.birth_sorted.at[block].set(birth),
            birth_count=current.birth_count.at[block].set(count),
            likelihood_sorted=current.likelihood_sorted.at[block].set(endpoints),
            num_sealed=current.num_sealed,
        )

    updated = jax.lax.fori_loop(index.num_sealed, num_sealed, seal, index)
    return dataclasses.replace(updated, num_sealed=num_sealed)


@partial(jax.jit, inline=True, static_argnames=("capacity",))
def resize_phantom_seed_index(
        index: PhantomSeedIndex,
        capacity: int,
) -> PhantomSeedIndex:
    """Resize only block storage, preserving completed counts and identities."""
    block_size = index.birth_sorted.shape[1]
    num_blocks = (capacity + block_size - 1) // block_size
    previous_blocks = index.birth_sorted.shape[0]
    if num_blocks <= previous_blocks:
        return PhantomSeedIndex(
            birth_sorted=index.birth_sorted[:num_blocks],
            birth_count=index.birth_count[:num_blocks],
            likelihood_sorted=index.likelihood_sorted[:num_blocks],
            num_sealed=jnp.minimum(index.num_sealed, capacity // block_size),
        )
    padding = ((0, num_blocks - previous_blocks), (0, 0))
    return PhantomSeedIndex(
        birth_sorted=jnp.pad(index.birth_sorted, padding, constant_values=jnp.inf),
        birth_count=jnp.pad(index.birth_count, padding),
        likelihood_sorted=jnp.pad(
            index.likelihood_sorted, padding, constant_values=jnp.inf,
        ),
        num_sealed=index.num_sealed,
    )


@partial(jax.jit, inline=True)
def phantom_block_cumulative(
        samples: Samples,
        num_samples: IntArray,
        constraints: FloatArray,
        index: PhantomSeedIndex,
) -> IntArray:
    """Count exact eligible identities per block for a batch of contours."""
    block_size = index.birth_sorted.shape[1]

    def count_block(birth, birth_count, likelihood):
        reached = jnp.searchsorted(birth, constraints, side="right")  # [S]
        born = jnp.where(reached > 0, birth_count[jnp.maximum(reached - 1, 0)], 0)
        # At +inf searchsorted includes padding; only actual endpoints die.
        died = jnp.minimum(
            jnp.searchsorted(likelihood, constraints, side="right"), birth_count[-1],
        )
        return born - died

    counts = jax.vmap(count_block)(
        index.birth_sorted, index.birth_count, index.likelihood_sorted,
    ).T  # [S,blocks]
    counts = jnp.where(jnp.arange(counts.shape[1]) < index.num_sealed, counts, 0)

    rows = index.num_sealed * block_size + jnp.arange(block_size)  # [B]
    safe_rows = jnp.minimum(rows, samples.log_likelihoods.size - 1)
    birth = samples.log_L_constraints[safe_rows]  # [B]
    likelihood = samples.phantom_samples.log_L[safe_rows]  # [B,P]
    valid = (rows < num_samples)[:, None] & samples.phantom_samples.valid_mask[safe_rows]

    def count_tail(constraint):
        return jnp.sum(
            valid & (birth[:, None] <= constraint) & (likelihood > constraint),
            dtype=mp_policy.index_dtype,
        )

    # A scalar map bounds scratch by B*P, even for a wide shell. Tail points
    # participate immediately, including before their block can be sealed.
    tail_count = jax.lax.map(count_tail, constraints)  # [S]
    counts = counts.at[:, index.num_sealed].add(tail_count, mode="drop")
    return jnp.cumsum(counts, axis=1, dtype=mp_policy.index_dtype)  # [S,blocks]


@partial(jax.jit, inline=True)
def phantom_block_rank_to_identity(
        samples: Samples,
        num_samples: IntArray,
        cumulative: IntArray,
        constraint: FloatArray,
        ranks: IntArray,
        index: PhantomSeedIndex,
) -> IntArray:
    """Map phantom ranks in original order using bounded one-block scratch."""
    block_size = index.birth_sorted.shape[1]
    num_phantom = samples.phantom_samples.log_L.shape[1]
    offsets = jnp.arange(block_size, dtype=mp_policy.index_dtype)

    def select(rank):
        block = jnp.minimum(
            jnp.searchsorted(cumulative, rank, side="right"), cumulative.size - 1,
        )
        previous = jnp.where(block > 0, cumulative[jnp.maximum(block - 1, 0)], 0)
        rows = block * block_size + offsets  # [B]
        safe_rows = jnp.minimum(rows, samples.log_likelihoods.size - 1)
        eligible = (
            (rows < num_samples)[:, None]
            & (samples.log_L_constraints[safe_rows, None] <= constraint)
            & samples.phantom_samples.valid_mask[safe_rows]
            & (samples.phantom_samples.log_L[safe_rows] > constraint)
        )  # [B,P], original transition order
        flat_slot = jnp.argmax(
            jnp.cumsum(eligible.reshape(-1), dtype=mp_policy.index_dtype) > rank - previous,
        )
        row = block * block_size + flat_slot // num_phantom
        return (row * (num_phantom + 1) + flat_slot % num_phantom + 1).astype(
            mp_policy.index_dtype,
        )

    # Do not vmap: an eager retry batch must not materialise [proposals,B,P].
    return jax.lax.map(select, ranks)
