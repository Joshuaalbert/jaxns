import dataclasses
from functools import partial

import jax.lax
import jax.tree
import numpy as np
from jax import numpy as jnp
from jax._src import prng

from jaxns.logging import jaxns_logger
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.types import FloatArray, IntArray, UType, BoolArray
from jaxns.cumulative_ops import scan_or_while_loop


@dataclasses.dataclass(slots=True)
class SeedPoint(PureDataclassPytree):
    U0: UType
    log_L0: FloatArray


SeedPoint.register_pytree()


@dataclasses.dataclass(slots=True)
class PhantomSamples(PureDataclassPytree):
    U_samples: UType | None  # [num_phantom, ...]
    valid_mask: BoolArray  # [num_phantom] whether the phantom sample is valid or not, used for book-keeping when resizing phantom samples
    log_L: FloatArray  # [num_phantom] the likelihoods of the phantom samples, must be > than log_L_constraint


PhantomSamples.register_pytree()

@dataclasses.dataclass(slots=True)
class Samples(PureDataclassPytree):
    log_likelihoods: FloatArray  # [max_samples]
    U_samples: UType  # [max_samples, ...]
    out_degree: IntArray  # [max_samples]
    num_likelihood_evaluations: IntArray  # [max_samples] incorperates
    phantom_samples: PhantomSamples  # [max_samples, ...]

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
        return _append_samples(self, insert_idx, parent_idxs, samples, delta_parent_out_degree)

    def resize(self, max_samples: int) -> 'Samples':
        return _resize(self, max_samples)


Samples.register_pytree()


@partial(jax.jit, inline=True)
def _get_items(self: Samples, item):
    return jax.tree.map(lambda x: x[item], self)


@partial(jax.jit, inline=True)
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
    self = jax.tree.map(lambda x: x[idxs], self)
    self.log_likelihoods = log_likelihoods
    return self


@partial(jax.jit, inline=True)
def _perm_sort(self: Samples, key) -> Samples:
    sort_keys = prng.random_bits(key, bit_width=32, shape=np.shape(self.log_likelihoods))
    iota = jnp.arange(len(self.log_likelihoods))
    (log_likelihoods, _, idxs) = jax.lax.sort(
        (self.log_likelihoods, sort_keys, iota),
        is_stable=False, num_keys=2)
    self = jax.tree.map(lambda x: x[idxs], self)
    self.log_likelihoods = log_likelihoods
    return self


@partial(jax.jit, inline=True)
def _compute_num_live_points_per_sample(self: Samples, root_out_degree: IntArray,
                                        num_samples: IntArray | None = None) -> IntArray:
    # Cumulatively apply K[i+1] = K[i] - 1 + d(i)
    def scan_fn(carry, out_degree):
        K_i, = carry
        K_ip1 = K_i - jnp.ones((), K_i.dtype) + out_degree
        return (K_ip1,), K_i

    _, K_values = scan_or_while_loop(scan_fn, (root_out_degree,), self.out_degree.astype(root_out_degree.dtype),
                                     length=num_samples, unroll=1)
    return K_values


@partial(jax.jit, inline=True)
def _append_samples(self: Samples, insert_idx: IntArray, parent_idxs: IntArray, samples: Samples,
                    delta_parent_out_degree: IntArray) -> Samples:
    samples = self.set_slice(insert_idx, samples)
    samples.out_degree = samples.out_degree.at[parent_idxs].add(delta_parent_out_degree)
    return samples


@partial(jax.jit, inline=True)
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
        log_likelihoods=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        out_degree=jnp.asarray(0, mp_policy.count_dtype),
        num_likelihood_evaluations=jnp.asarray(0, mp_policy.count_dtype),
        U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), self.U_samples),
        phantom_samples=PhantomSamples(
            U_samples=jax.tree.map(lambda x: jnp.zeros_like(x[0]), self.phantom_samples.U_samples),
            log_L=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
            valid_mask=jnp.asarray(False, mp_policy.bool_dtype)
        )
    )

    return _concat(self, sample_atom)
