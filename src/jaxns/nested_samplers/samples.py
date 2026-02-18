import dataclasses
from functools import partial
from typing import Any

import jax.lax
import jax.tree
import numpy as np
from jax import numpy as jnp
from jax._src import prng

from jaxns.nested_samplers.pytree import PureDataclassPytree
from jaxns.nested_samplers.logging import jaxns_logger
from jaxns.nested_samplers.types import FloatArray, IntArray
from jaxns.nested_samplers.utils import scan_or_while_loop

UType = Any


@dataclasses.dataclass(slots=True)
class Samples(PureDataclassPytree):
    log_likelihoods: FloatArray  # [max_samples]
    U_samples: UType  # [max_samples, ...]
    out_degree: IntArray  # [max_samples]
    num_likelihood_evaluations: IntArray  # [max_samples]

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

    def append_samples(self, insert_idx: IntArray, parent_idx: IntArray, samples: 'Samples',
                       delta_parent_out_degree: IntArray) -> 'Samples':
        return _append_samples(self, insert_idx, parent_idx, samples, delta_parent_out_degree)

    def resize(self, max_samples: int) -> 'Samples':
        return _resize(self, max_samples)


Samples.register_pytree()


@partial(jax.jit, inline=True)
def _get_items(self, item):
    return jax.tree.map(lambda x: x[item], self)


@partial(jax.jit, inline=True)
def _slice(self, start: IntArray, size: int) -> 'Samples':
    return jax.tree.map(lambda x: jax.lax.dynamic_slice(x, (start,) + (0,) * (x.ndim - 1), (size,) + x.shape[1:]),
                        self)


@partial(jax.jit, inline=True)
def _set_slice(self, start: IntArray, other: 'Samples') -> 'Samples':
    return jax.tree.map(
        lambda x, y: jax.lax.dynamic_update_slice(x, y, (start,) + (0,) * (x.ndim - 1)),
        self, other
    )


@partial(jax.jit, inline=True)
def _concat(self, other: 'Samples') -> 'Samples':
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), self, other)


@partial(jax.jit, inline=True)
def _sort(self) -> 'Samples':
    iota = jnp.arange(len(self.log_likelihoods))
    (log_likeihoods, out_degree, num_likelihood_evaluations, idxs) = jax.lax.sort(
        (self.log_likelihoods, self.out_degree, self.num_likelihood_evaluations, iota),
        is_stable=False, num_keys=1)
    return Samples(
        log_likelihoods=log_likeihoods,
        U_samples=jax.tree.map(lambda x: x[idxs], self.U_samples),
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations,
    )


@partial(jax.jit, inline=True)
def _perm_sort(self, key) -> 'Samples':
    sort_keys = prng.random_bits(key, bit_width=32, shape=np.shape(self.log_likelihoods))
    iota = jnp.arange(len(self.log_likelihoods))
    (log_likeihoods, _, out_degree, num_likelihood_evaluations, idxs) = jax.lax.sort(
        (self.log_likelihoods, sort_keys, self.out_degree, self.num_likelihood_evaluations, iota),
        is_stable=False, num_keys=2)
    return Samples(
        log_likelihoods=log_likeihoods,
        U_samples=jax.tree.map(lambda x: x[idxs], self.U_samples),
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations,
    )


@partial(jax.jit, inline=True)
def _compute_num_live_points_per_sample(self, root_out_degree: IntArray,
                                        num_samples: IntArray | None = None) -> IntArray:
    # Cumulatively apply K[i] = K[i-1] - 1 + d(i)
    def scan_fn(carry, out_degree):
        K_prev, = carry
        K_new = K_prev - jnp.ones((), root_out_degree.dtype) + out_degree
        return (K_new,), K_new

    _, K_values = scan_or_while_loop(scan_fn, (root_out_degree,), self.out_degree.astype(root_out_degree.dtype),
                                     length=num_samples, unroll=1)
    return K_values


@partial(jax.jit, inline=True)
def _append_samples(self, insert_idx: IntArray, parent_idx: IntArray, samples: 'Samples',
                    delta_parent_out_degree: IntArray) -> 'Samples':
    samples = self.set_slice(insert_idx, samples)
    out_degree = samples.out_degree.at[parent_idx].add(delta_parent_out_degree)
    return Samples(
        log_likelihoods=samples.log_likelihoods,
        U_samples=samples.U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=samples.num_likelihood_evaluations,
    )


@partial(jax.jit, inline=True)
def _resize(self, max_samples: int) -> 'Samples':
    if len(self) >= max_samples:
        jaxns_logger.warning(
            f"Samples.resize called with max_samples={max_samples} less than current size {len(self)}. No resize performed.")
        return self

    def _concat(x, fill_value):
        return jax.tree.map(
            lambda x: jnp.concatenate([
                x,
                jnp.full((max_samples - len(self),) + x.shape[1:], fill_value, dtype=x.dtype)
            ], axis=0),
            x
        )

    return Samples(
        log_likelihoods=_concat(self.log_likelihoods, -jnp.inf),
        out_degree=_concat(self.out_degree, 0),
        num_likelihood_evaluations=_concat(self.num_likelihood_evaluations, 0),
        U_samples=_concat(self.U_samples, 0),
    )
