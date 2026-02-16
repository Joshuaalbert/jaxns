import dataclasses
from functools import partial
from typing import Tuple

import jax.random
import jax.tree
import numpy as np
from jax import numpy as jnp

from jaxns.nested_samplers.log_semiring import LogSpace
from jaxns.nested_samplers.mixed_precision import mp_policy
from jaxns.nested_samplers.pytree import PureDataclassPytree
from jaxns.nested_samplers.evidence_calculation import EvidenceCalculation
from jaxns.nested_samplers.samples import Samples
from jaxns.nested_samplers.types import IntArray, FloatArray
from jaxns.nested_samplers.utils import scan_or_while_loop


@dataclasses.dataclass(slots=True, frozen=True)
class State(PureDataclassPytree):
    root_out_degree: IntArray  # scalar
    samples: Samples
    num_samples: IntArray  # scalar

    def merge(self, other: 'State') -> 'State':
        return _merge(self, other)

    def determine_parent_graph(self):
        return _determine_parent_graph(self)

    def ensure_consistency(self):
        return _ensure_consistency(self)

    def evaluate_evidence(self) -> Tuple[EvidenceCalculation, EvidenceCalculation]:
        return _evaluate_evidence(self)

    def sample_logZ(self, key, num_samples: int) -> FloatArray:
        return _sample_logZ(self, key, num_samples)


State.register_pytree()


@partial(jax.jit, inline=True)
def _merge(self, other: 'State') -> 'State':
    return State(
        root_out_degree=self.root_out_degree + other.root_out_degree,
        samples=self.samples.concat(other.samples),
        num_samples=self.num_samples + other.num_samples
    )


@partial(jax.jit, inline=True)
def _determine_parent_graph(self):
    # Determine parent graph from out-degrees along
    samples = self.samples.sort()
    # Carry:
    # next_parent_idx, remaining_out_degrees

    carry_init = (jnp.asarray(-1, mp_policy.count_dtype), self.root_out_degree)

    def scan_fn(carry, x):
        next_parent_idx, remaining_out_degrees = carry
        child_node_idx, = x
        y = (next_parent_idx, child_node_idx)
        remaining_out_degrees = remaining_out_degrees - 1
        next_parent_idx = jnp.where(remaining_out_degrees <= 0, next_parent_idx + 1, next_parent_idx)
        remaining_out_degrees = jnp.where(remaining_out_degrees <= 0, samples.out_degree[next_parent_idx], remaining_out_degrees)
        return (next_parent_idx, remaining_out_degrees), y

    _, parent_edges = scan_or_while_loop(scan_fn, carry_init, (jnp.arange(self.samples.log_likelihoods.shape[0]),), length=self.num_samples, unroll=1)
    return parent_edges


@partial(jax.jit, inline=True)
def _ensure_consistency(self):
    # Every non-root has exactly one parent, so sum of out-degrees = num_samples
    assert self.root_out_degree + np.sum(self.samples.out_degree) == self.num_samples
    # You must have a replacement child in order to die
    K_samples = self.samples.sort().compute_num_live_points_per_sample(self.root_out_degree, self.num_samples)
    assert K_samples[self.num_samples - 1] == 0
    K_pre = K_samples + 1 - self.samples.out_degree
    assert np.all(K_pre[:self.num_samples] > 0)

    # determine parent graph
    samples = jax.tree.map(np.asarray, self.samples.sort())
    next_parent_idx, remaining_out_degrees = (-1, float(self.root_out_degree))
    child_node_idx = 0
    num_samples = int(self.num_samples)
    parent_edges = []
    while child_node_idx < num_samples:
        if remaining_out_degrees <= 0:
            raise ValueError(f"Invalid graph, contains a broken lineage.")
        parent_edges.append([next_parent_idx, child_node_idx])
        remaining_out_degrees -= 1
        if remaining_out_degrees == 0:
            next_parent_idx = next_parent_idx + 1
            remaining_out_degrees = samples.out_degree[next_parent_idx]
        child_node_idx += 1


@partial(jax.jit, inline=True)
def _evaluate_evidence(self) -> Tuple[EvidenceCalculation, EvidenceCalculation]:
    # Evaluate evidence calculation over all samples
    def single_register_update(carry, x):
        K_total, evidence_calculation, log_L = carry
        out_degree, = x
        evidence_calculation = evidence_calculation.update_evidence(K_total, log_L)
        K_total = K_total - 1 + out_degree
        return (K_total, evidence_calculation), evidence_calculation

    carry_init = (self.root_out_degree, EvidenceCalculation.initialise())
    xs = (self.samples.out_degree, self.samples.log_likelihoods)
    (_, final_evidence_calculation), evidence_calculations = scan_or_while_loop(
        single_register_update,
        carry_init,
        xs,
        length=self.num_samples,
        unroll=1
    )
    return (final_evidence_calculation, evidence_calculations)


@partial(jax.jit, inline=True)
def _sample_logZ(self, key, num_samples: int) -> FloatArray:
    def single_sample_logZ(key) -> LogSpace:
        samples = self.samples.perm_sort(key)
        s0 = jnp.zeros((), dtype=samples.log_likelihoods.dtype)
        K_samples = samples.compute_num_live_points_per_sample(self.root_out_degree)
        Z_acc = LogSpace(jnp.array(-jnp.inf, dtype=samples.log_likelihoods.dtype))
        L_samples = LogSpace(samples.log_likelihoods)
        carry_init = (s0, Z_acc)
        xs = (L_samples, K_samples)

        def scan_fn(carry, x):
            s_prev, Z_acc = carry
            L, K = x
            ds = jax.random.exponential(key, ()) / K
            s_new = s_prev + ds
            # dX = jnp.exp(-s_new) - jnp.exp(-s_prev) =  jnp.exp(-(s_prev + ds)) - jnp.exp(-s_prev)
            #      = jnp.exp(-s_prev) * (jnp.exp(-ds) - 1)
            dX = LogSpace(jnp.log(jnp.expm1(-ds)) - s_prev)
            dZ = dX * L
            Z_acc = Z_acc + dZ
            return (s_new, Z_acc), None

        (_, Z_sample), _ = scan_or_while_loop(scan_fn, carry_init, xs, length=num_samples, unroll=1)
        return Z_sample

    keys = jax.random.split(key, num_samples)
    Z_samples = jax.vmap(single_sample_logZ)(keys)
    logZ = Z_samples.log_abs_val
    return logZ
