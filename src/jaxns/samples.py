import dataclasses
from functools import partial

import jax.lax
import jax.tree
from jax import numpy as jnp
from jax import random

from jaxns.cumulative_ops import scan_or_while_loop
from jaxns.logging import jaxns_logger
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.types import BoolArray, FloatArray, IntArray, UType


@dataclasses.dataclass(slots=True, frozen=True)
class SeedPoint(PureDataclassPytree):
    U0: UType  # [...] one point in the unit-hypercube pytree
    log_L0: FloatArray  # []


SeedPoint.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class PhantomSamples(PureDataclassPytree):
    U_samples: UType | None  # [P, ...] unit-hypercube pytree leaves
    valid_mask: BoolArray  # [P]
    log_L: FloatArray  # [P]


PhantomSamples.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class Samples(PureDataclassPytree):
    """Append-order scientific samples sufficient to reconstruct the race.

    The parent contour and per-sample out-degree are persistent. A concrete
    parent storage index is deliberately absent: it is needed only transiently
    when a child increments its parent's degree, and would become stale if
    these rows were sorted.
    """

    log_L_constraints: FloatArray  # [N]
    log_likelihoods: FloatArray  # [N]
    U_samples: UType  # [N, ...] unit-hypercube pytree leaves
    out_degree: IntArray  # [N]
    num_likelihood_evaluations: IntArray  # [N]
    phantom_samples: PhantomSamples  # [N, P, ...]

    def __len__(self):
        return self.log_likelihoods.shape[0]

    def __getitem__(self, item):
        return _get_items(self, item)

    def slice(self, start: IntArray, size: int) -> 'Samples':
        return _slice(self, start, size)

    def set_slice(self, start: IntArray, other: 'Samples') -> 'Samples':
        return _set_slice(self, start, other)

    def concat(self, other: 'Samples') -> 'Samples':
        return _concat(self, other)

    def sort(self) -> 'Samples':
        return _sort(self)

    def perm_sort(self, key) -> 'Samples':
        return _perm_sort(self, key)

    def compute_num_live_points_per_sample(self, root_out_degree: IntArray,
                                           num_samples: IntArray | None = None) -> IntArray:
        return _compute_num_live_points_per_sample(self, root_out_degree, num_samples)

    def append_samples(self, insert_idx: IntArray, parent_idxs: IntArray, samples: 'Samples',
                       delta_parent_out_degree: IntArray) -> 'Samples':
        """Append a fixed batch and apply its transient parent degree updates."""
        return _append_samples(self, insert_idx, parent_idxs, samples, delta_parent_out_degree)

    def resize(self, max_samples: int) -> 'Samples':
        return _resize(self, max_samples)


Samples.register_pytree()


@partial(jax.jit, inline=True)
def _get_items(self: Samples, item):
    return jax.tree.map(lambda x: x[item], self)


@partial(jax.jit, inline=True, static_argnames=['size'])
def _slice(self: Samples, start: IntArray, size: int) -> Samples:
    return jax.tree.map(lambda x: jax.lax.dynamic_slice(x, (start,) + (0,) * (x.ndim - 1), (size,) + x.shape[1:]), self)


@partial(jax.jit, inline=True)
def _set_slice(self: Samples, start: IntArray, other: Samples) -> Samples:
    return jax.tree.map(
        lambda x, y: jax.lax.dynamic_update_slice(x, y, (start,) + (0,) * (x.ndim - 1)),
        self, other
    )


@partial(jax.jit, inline=True)
def _concat(self: Samples, other: Samples) -> Samples:
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), self, other)


@partial(jax.jit, inline=True)
def _sort(self: Samples) -> Samples:
    iota = jnp.arange(len(self.log_likelihoods))
    (log_likelihoods, idxs) = jax.lax.sort(
        (self.log_likelihoods, iota),
        is_stable=False, num_keys=1)
    sorted_samples = jax.tree.map(lambda x: x[idxs], self)
    return dataclasses.replace(sorted_samples, log_likelihoods=log_likelihoods)


@partial(jax.jit, inline=True)
def _perm_sort(self: Samples, key) -> Samples:
    sort_keys = random.randint(
        key,
        shape=jnp.shape(self.log_likelihoods),
        minval=0,
        maxval=jnp.iinfo(jnp.uint32).max,
        dtype=jnp.uint32
    )
    iota = jnp.arange(len(self.log_likelihoods))
    (log_likelihoods, _, idxs) = jax.lax.sort(
        (self.log_likelihoods, sort_keys, iota),
        is_stable=False, num_keys=2)
    sorted_samples = jax.tree.map(lambda x: x[idxs], self)
    return dataclasses.replace(sorted_samples, log_likelihoods=log_likelihoods)


@partial(jax.jit, inline=True)
def _compute_num_live_points_per_sample(self: Samples, root_out_degree: IntArray,
                                        num_samples: IntArray | None = None) -> IntArray:
    # Cumulatively apply K[i+1] = K[i] - 1 + d(i)
    def scan_fn(carry, out_degree_i):
        K_i, = carry
        K_ip1 = K_i - jnp.ones((), K_i.dtype) + out_degree_i
        return (K_ip1,), K_i

    _, K_values = scan_or_while_loop(scan_fn, (root_out_degree,), self.out_degree.astype(root_out_degree.dtype),
                                     length=num_samples, unroll=1)
    return K_values


@partial(jax.jit, inline=True)
def _append_samples(self: Samples, insert_idx: IntArray, parent_idxs: IntArray, samples: Samples,
                    delta_parent_out_degree: IntArray) -> Samples:
    # Parent storage indices exist only for this scatter. The persistent child
    # records its parent likelihood contour, while the updated degree is all
    # that later race reconstruction and shrinkage require.
    batch_size = samples.log_likelihoods.shape[0]

    def append_full_batch(_):
        # Preserve the single dynamic-update operation on the ordinary hot
        # path; a general scatter would add indexing work to every batch.
        return self.set_slice(insert_idx, samples)

    def append_partial_tail(_):
        # A finite hard maximum need not be divisible by the static replacement
        # width. Drop out-of-range inactive lanes so the valid prefix can fill
        # the final physical slots without an S-1 allocation beyond the limit.
        sample_indices = insert_idx + jnp.arange(
            batch_size,
            dtype=mp_policy.index_dtype,
        )
        return jax.tree.map(
            lambda destination, source: destination.at[sample_indices].set(
                source,
                mode="drop",
            ),
            self,
            samples,
        )

    appended = jax.lax.cond(
        insert_idx + batch_size <= self.log_likelihoods.shape[0],
        append_full_batch,
        append_partial_tail,
        operand=None,
    )
    out_degree = appended.out_degree.at[parent_idxs].add(
        delta_parent_out_degree
    )
    return dataclasses.replace(appended, out_degree=out_degree)


@partial(jax.jit, inline=True, static_argnames=['max_samples'])
def _resize(self: Samples, max_samples: int) -> Samples:
    if len(self) >= max_samples:
        jaxns_logger.warning(
            f"Samples.resize called with max_samples={max_samples} less than current size {len(self)}. No resize performed.")
        return self

    def _concat(x, fill_value):
        return jax.tree.map(
            lambda x, y: jnp.concatenate([
                x,
                jnp.repeat(y[None, ...], repeats=(max_samples - len(self)), axis=0)
            ], axis=0),
            x,
            fill_value
        )

    sample_atom = Samples(
        log_L_constraints=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
        log_likelihoods=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
        out_degree=jnp.asarray(0, mp_policy.count_dtype),
        num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype),
        U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), self.U_samples),
        phantom_samples=PhantomSamples(
            U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), self.phantom_samples.U_samples),
            log_L=jnp.full_like(self.phantom_samples.log_L[0], -jnp.inf),
            valid_mask=jnp.zeros_like(self.phantom_samples.valid_mask[0])
        ),
    )

    return _concat(self, sample_atom)
