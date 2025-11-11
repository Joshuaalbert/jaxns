import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Tuple

import jax.numpy as jnp
import jax.random
import numpy as np
from jax._src import prng

from jaxns.framework.new_model import Model
from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.pytree import PureDataclassPytree
from jaxns.internals.stats import linear_to_log_stats, effective_sample_size_kish
from jaxns.internals.types import FloatArray, IntArray, BoolArray
from jaxns.logging import jaxns_logger

"""
We formulate nested sampling as an exponential race of lineages.

All node indices are in sorted order, so i < j implies L(i) >= L(j) (where = only for plateaus).
We handle pleateaus by marginalising the race over permutations within plateaus.

A parent graph is defined,

    p(i) -> i means that i is a child of p(i), which mans i was sampled from {L > L(p(i))}.
    
A lineage is maximal chain of nodes where p(i_{k}}) = i_{k-1}.

A dummy root node, 0, is defined with L(0) = 0.

Suppose we have N samples, indexed 1,...,N, via argsort(L) + 1 (stable sort so within plateau ordering preserved).

define s_i = -log(X_i) where X_i is the prior volume associated with sample i. 
Set s_0 = 0.

Define the number of active children of node i at index j>i, as C(i,j) = |{ k : p(k) = i, k >= j}|.

Define the active parent set A(i-1) = {j : j=1..i-1, C(j,i) > 0} which is the set of nodes before i which have active children at or after i.

Each parent in A(i-1) defines a lineage, which at least one active child at or after i.

For node i, sample ds_i ~ Exponential(K(i)) where K(i)=sum_{j in A(i-1)} C(j,i) is the total number of active children over all active lineages A(i-1).

The principle property of exponential races is that the minimum of independent exponentials is itself an exponential with rate equal to the sum of the rates.

So we set s_i = s_{i-1} + ds_i, and assign the race winner to the observed winning lineage (unconditionally we would need to sample which lineage won, but nested sampling conditions on this).
In the case of plateaus we marginalise over the permutations within the plateau, which is equivalent to assigning winners in every possible order and averaging the results.
We rely on permutations and stable sorting to ensure that the order of nodes within plateaus is consistent across all computations.

Now, we don't need to actually compute C(j,i), nor maintain A(i-1) explicitly.
We can maintain a running count of the number of active children for each node, and update this as we move through the samples.
Define out-degree d(i) = |{ j : p(j) = i }|, and set d(0) = K, which K is the number of live points.

When we process node i, we have K(i) = sum_{j in A(i-1)} C(j,i) as the total number of active children over all active lineages.
We then decrement C(p(i),i+1) = C(p(i),i) - 1, removing one active child from the parent lineage of i, and then add d(i) active children to C(i+1) = C(i) + d(i).

So in total after consuming node i, perform the update:

    K(i) = K(i-1) - 1 + d(i)
    
Thus, we only need to maintain d(i) for each node, and not actually the graph {p(i)->i}, since the only thing we do with that graph is compute the out-degrees d(i).

We are thus free to arbitrarily add samples to the graph from any node in any lineage, including the dummy node. 
This implies creating new lineages is possibly by creating d(i) > 1 for some i.

We have two modalities of evidence estimation: expectation-based and sampling-based.

In the sampling based approach, we compute the trajectory of s_i many times, and then x_i = exp(-s_i) for each sample, and use these to sample evidence.
When doing this for each sample we permutate the samples first, after which stable sorting preserves the that permuatation within plateaus.
Thus each evidence sample corresponds to a different trajectory of s_i handling plateaus correctly.

For the expectation-based approach, we track several sufficient statistics to compute E[logZ], Var[logZ], E[logH] etc, marginalising over permutations within plateaus.
These are ...

Now the above tells us how to compute the prior volumes X_i associated with each sample i, but we now turn to how to sample.

We begin with a dummy node 0, and K children sampled iid from the prior (constrained to L>L(0)), forming the initial live point set.
We then apply the above tracking of K(i) in an online manner starting from node 0 with K(0) = K.
We introduce parallelism by discarding m in (1, K/2) points at a time, and outsourcing the sampling of their replacements to different devices.
That is we apply the above logic, to sequentially select nodes 1..m, each time updating K(i) = K(i-1) - 1 + d(i) as above, and updating the sufficient statistics for evidence estimation after each discard.
Once we have selected m nodes to discard, we send their replacement sampling to different devices, sampling from within the last discarded point {L>L(m)}.
"""

UType = Any


def scan_or_while_loop(scan_fn, carry_init, xs, length: IntArray | None = None, unroll: int = 1) -> tuple:
    # if length is None or static use scan, other wise use while_loop
    if length is None or isinstance(length, int):
        return jax.lax.scan(scan_fn, carry_init, xs, length=length, unroll=unroll)

    def cond_fn(carry):
        i, _, _ = carry
        return i < length

    def body_fn(carry):
        i, carry_inner, ys = carry
        x = jax.tree.map(lambda x: x[i], xs)
        carry_inner, y = scan_fn(carry_inner, x)
        ys = jax.tree.map(lambda y, y_new: y.at[i, ...].set(y_new), ys, y)
        return (i + 1, carry_inner, ys)

    # aeval to build ys structure
    max_length = jax.tree.leaves(xs)[0].shape[0]
    carry_struct, ys_struct = jax.eval_shape(scan_fn, carry_init, jax.tree.map(lambda x: x[0], xs))
    ys_init = jax.tree.map(lambda y: jnp.zeros((max_length,) + y.shape, dtype=y.dtype), ys_struct)
    carry = (0, carry_init, ys_init)
    _, carry_inner, ys = jax.lax.while_loop(cond_fn, body_fn, carry)
    return carry_inner, ys


def test_scan_or_while_loop():
    def scan_fn(carry, x):
        return carry + x, carry + x

    carry_init = 0
    xs = jnp.arange(10)

    # Test with static length
    _, ys_scan = scan_or_while_loop(scan_fn, carry_init, xs, length=10)
    assert jnp.all(ys_scan == jnp.cumsum(xs))

    # Test with dynamic length
    length = jnp.array(10)
    _, ys_while = scan_or_while_loop(scan_fn, carry_init, xs, length=length)
    assert jnp.all(ys_while == jnp.cumsum(xs))

    # Test with dynamic length
    length = jnp.array(7)
    _, ys_while = scan_or_while_loop(scan_fn, carry_init, xs, length=length)
    assert jnp.all(ys_while == jnp.cumsum(xs).at[7:].set(0))


@dataclasses.dataclass(slots=True)
class EvidenceCalculation(PureDataclassPytree):
    """
    Contains a running estimate of evidence and related quantities.
    """
    L: LogSpace
    X_mean: LogSpace
    X2_mean: LogSpace
    Z_mean: LogSpace
    ZX_mean: LogSpace
    Z2_mean: LogSpace
    dZ_mean: LogSpace
    dZ2_mean: LogSpace

    @staticmethod
    def initialise() -> 'EvidenceCalculation':
        return EvidenceCalculation(
            L=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            X_mean=LogSpace(jnp.array(0., dtype=mp_policy.measure_dtype)),
            X2_mean=LogSpace(jnp.array(0., dtype=mp_policy.measure_dtype)),
            Z_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            ZX_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            Z2_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            dZ_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype)),
            dZ2_mean=LogSpace(jnp.array(-jnp.inf, dtype=mp_policy.measure_dtype))
        )

    def update_evidence(self, K_total: IntArray, log_L_next: FloatArray) -> 'EvidenceCalculation':
        next_L = LogSpace(log_L_next)

        # num_live_points = jnp.maximum(y.num_live_points, jnp.zeros_like(y.num_live_points))
        log_num_live_points = jnp.log(K_total)
        log_num_live_points_p1 = jnp.log(K_total + np.asarray(1., K_total.dtype))
        log_num_live_points_p2 = jnp.log(K_total + np.asarray(2., K_total.dtype))

        # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
        # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
        T_mean = LogSpace(- jnp.logaddexp(0., -log_num_live_points))
        # T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
        t_mean = LogSpace(- log_num_live_points_p1)
        # T2_mean = LogSpace(jnp.log(num_live_points) - jnp.log( num_live_points + 2.))
        # T2_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 2./num_live_points))
        T2_mean = LogSpace(- jnp.logaddexp((0.), np.log(2.) - log_num_live_points))
        # T2_mean = LogSpace(- jnp.logaddexp(jnp.log(2.), -jnp.log(num_live_points)))
        t2_mean = LogSpace(jnp.log(2.) - log_num_live_points_p1 - log_num_live_points_p2)
        # tT_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
        # tT_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points) - jnp.log(num_live_points + 2.))
        tT_mean = LogSpace(- jnp.logaddexp(0., -log_num_live_points) - log_num_live_points_p2)
        # tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))

        midL = LogSpace(np.log(0.5)) * (next_L + self.L)
        dZ_mean = self.X_mean * t_mean * midL
        next_X_mean = self.X_mean * T_mean
        next_X2_mean = self.X2_mean * T2_mean
        next_Z_mean = self.Z_mean + dZ_mean
        next_ZX_mean = self.ZX_mean * T_mean + self.X2_mean * tT_mean * midL
        next_Z2_mean = self.Z2_mean + LogSpace(np.log(2.)) * self.ZX_mean * t_mean * midL + (
                self.X2_mean * t2_mean * midL ** 2)
        next_dZ2_mean = self.dZ2_mean + (self.X2_mean * t2_mean * midL ** 2)

        return EvidenceCalculation(
            L=next_L,
            X_mean=next_X_mean,
            X2_mean=next_X2_mean,
            Z_mean=next_Z_mean,
            ZX_mean=next_ZX_mean,
            Z2_mean=next_Z2_mean,
            dZ_mean=dZ_mean,
            dZ2_mean=next_dZ2_mean
        )


EvidenceCalculation.register_pytree()


@dataclasses.dataclass(slots=True)
class TerminationCondition(PureDataclassPytree):
    """
    Contains the termination conditions for the nested sampling run.

    Args:
        ess: The effective sample size, if the ESS (Kish's estimate) is greater than this the run will terminate.
        evidence_uncert: The uncertainty in the evidence, if the uncertainty is less than this the run will terminate.
        dlogZ: Terminate if log(Z_current + Z_remaining) - log(Z_current) < dlogZ. Default log(1 + 1e-2)
        max_samples: Terminate if the number of samples exceeds this.
        max_num_likelihood_evaluations: Terminate if the number of likelihood evaluations exceeds this.
        log_L_contour: Terminate if this log(L) contour is reached. A contour is reached if any dead point
            has log(L) > log_L_contour. Uncollected live points are not considered.
        efficiency_threshold: Terminate if the efficiency (num_samples / num_likelihood_evaluations) is less than this,
            for the last shrinkage iteration.
        rtol: finish when the relative value 2*|log_L_max - log_L_min|/|log_L_max + log_L_min| < rol
        atol: finish when the absolute |log_L_max - log_L_min| < atol
        cummax_XL_frac: Terminate when XL < cummax(XL) * cummax_XL_frac
    """
    ess: FloatArray | None = None
    evidence_uncert: FloatArray | None = None
    dlogZ: FloatArray | None = None  #
    max_samples: IntArray | None = None
    max_num_likelihood_evaluations: IntArray | None = None
    log_L_contour: FloatArray | None = None
    efficiency_threshold: FloatArray | None = None
    rtol: FloatArray | None = None
    atol: FloatArray | None = None
    cummax_XL_frac: FloatArray | None = None


TerminationCondition.register_pytree()


@dataclasses.dataclass(slots=True)
class TerminationRegister(PureDataclassPytree):
    num_samples_used: IntArray
    evidence_calc: EvidenceCalculation
    evidence_calc_with_remaining: EvidenceCalculation
    num_likelihood_evaluations: IntArray
    log_L_contour: FloatArray
    efficiency: FloatArray
    plateau: BoolArray
    no_seed_points: BoolArray
    relative_spread: FloatArray
    absolute_spread: FloatArray
    cummax_log_XL: FloatArray

    @staticmethod
    def initialise():
        zero_count = jnp.asarray(0, mp_policy.count_dtype)
        init_evidence_calc = EvidenceCalculation.initialise()
        return TerminationRegister(
            num_samples_used=zero_count,
            evidence_calc=init_evidence_calc,
            evidence_calc_with_remaining=init_evidence_calc,
            num_likelihood_evaluations=zero_count,
            log_L_contour=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
            efficiency=jnp.asarray(0., mp_policy.measure_dtype),
            plateau=jnp.asarray(False, mp_policy.bool_dtype),
            no_seed_points=jnp.asarray(False, mp_policy.bool_dtype),
            relative_spread=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
            absolute_spread=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
            cummax_log_XL=jnp.asarray(-jnp.inf, mp_policy.measure_dtype)
        )

    def is_done(self, term_cond: TerminationCondition) -> Tuple[BoolArray, IntArray]:
        """
        Determine if termination should happen. Termination Flags are bits:
            0-bit -> 1: used maximum allowed number of samples
            1-bit -> 2: evidence uncert below threshold
            2-bit -> 4: live points evidence below threshold
            3-bit -> 8: effective sample size big enough
            4-bit -> 16: used maxmimum allowed number of likelihood evaluations
            5-bit -> 32: maximum log-likelihood contour reached
            6-bit -> 64: sampler efficiency too low
            7-bit -> 128: all lineages on a single plateau
            8-bit -> 256: relative spread of live points < rtol
            9-bit -> 512: absolute spread of live points < atol
            10-bit -> 1024: no seed points left
            11-bit -> 2048: XL < cummax(XL) * peak_XL_frac

        Multiple flags are summed together

        Args:
            term_cond: termination condition

        Returns:
            boolean done signal, and termination reason
        """

        termination_reason = jnp.asarray(0, mp_policy.count_dtype)
        done = jnp.asarray(False, jnp.bool_)

        def _set_done_bit(bit_done, bit_reason, done, termination_reason):
            if bit_done.size > 1:
                raise RuntimeError("bit_done must be a scalar.")
            done = jnp.bitwise_or(bit_done, done)
            termination_reason += jnp.where(bit_done,
                                            jnp.asarray(2 ** bit_reason, mp_policy.count_dtype),
                                            jnp.asarray(0, mp_policy.count_dtype))
            return done, termination_reason

        if term_cond.max_samples is not None:
            # used all points
            reached_max_samples = self.num_samples_used >= term_cond.max_samples
            done, termination_reason = _set_done_bit(reached_max_samples, 0,
                                                     done=done, termination_reason=termination_reason)

        if term_cond.evidence_uncert is not None:
            _, log_Z_var = linear_to_log_stats(
                log_f_mean=self.evidence_calc_with_remaining.Z_mean.log_abs_val,
                log_f2_mean=self.evidence_calc_with_remaining.Z2_mean.log_abs_val)
            evidence_uncert_low_enough = log_Z_var <= jnp.square(term_cond.evidence_uncert)
            done, termination_reason = _set_done_bit(evidence_uncert_low_enough, 1,
                                                     done=done, termination_reason=termination_reason)

        if term_cond.dlogZ is not None:
            # (Z_remaining + Z_current) / Z_remaining < exp(dlogZ)
            log_Z_mean_1, log_Z_var_1 = linear_to_log_stats(
                log_f_mean=self.evidence_calc_with_remaining.Z_mean.log_abs_val,
                log_f2_mean=self.evidence_calc_with_remaining.Z2_mean.log_abs_val)

            log_Z_mean_0, log_Z_var_0 = linear_to_log_stats(
                log_f_mean=self.evidence_calc.Z_mean.log_abs_val,
                log_f2_mean=self.evidence_calc.Z2_mean.log_abs_val)

            small_remaining_evidence = jnp.less(
                log_Z_mean_1 - log_Z_mean_0, term_cond.dlogZ
            )
            done, termination_reason = _set_done_bit(small_remaining_evidence, 2,
                                                     done=done, termination_reason=termination_reason)

        if term_cond.ess is not None:
            # Kish's ESS = [sum weights]^2 / [sum weights^2]
            ess = effective_sample_size_kish(self.evidence_calc_with_remaining.Z_mean.log_abs_val,
                                             self.evidence_calc_with_remaining.dZ2_mean.log_abs_val)
            ess_reached = ess >= term_cond.ess
            done, termination_reason = _set_done_bit(ess_reached, 3,
                                                     done=done, termination_reason=termination_reason)

        if term_cond.max_num_likelihood_evaluations is not None:
            num_likelihood_evaluations = jnp.sum(self.num_likelihood_evaluations)
            too_max_likelihood_evaluations = num_likelihood_evaluations >= term_cond.max_num_likelihood_evaluations
            done, termination_reason = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                     done=done, termination_reason=termination_reason)

        if term_cond.log_L_contour is not None:
            likelihood_contour_reached = self.log_L_contour >= term_cond.log_L_contour
            done, termination_reason = _set_done_bit(likelihood_contour_reached, 5,
                                                     done=done, termination_reason=termination_reason)

        if term_cond.efficiency_threshold is not None:
            efficiency_too_low = self.efficiency < term_cond.efficiency_threshold
            done, termination_reason = _set_done_bit(efficiency_too_low, 6,
                                                     done=done, termination_reason=termination_reason)

        done, termination_reason = _set_done_bit(self.plateau, 7,
                                                 done=done, termination_reason=termination_reason)

        if term_cond.rtol is not None:
            relative_spread_low = self.relative_spread < term_cond.rtol
            done, termination_reason = _set_done_bit(relative_spread_low, 8,
                                                     done=done, termination_reason=termination_reason)

        if term_cond.atol is not None:
            absolute_spread_low = self.absolute_spread < term_cond.atol
            done, termination_reason = _set_done_bit(absolute_spread_low, 9,
                                                     done=done, termination_reason=termination_reason)

        # Sampler relies upon this termination condition to stop when no seed points are left.
        done, termination_reason = _set_done_bit(self.no_seed_points, 10,
                                                 done=done, termination_reason=termination_reason)

        if term_cond.cummax_XL_frac is not None:
            log_XL = self.evidence_calc.X_mean.log_abs_val + self.evidence_calc.L.log_abs_val
            peak_log_XL = self.cummax_log_XL
            XL_reduction_reached = log_XL < peak_log_XL + jnp.log(term_cond.cummax_XL_frac)
            done, termination_reason = _set_done_bit(XL_reduction_reached, 11,
                                                     done=done, termination_reason=termination_reason)

        return done, termination_reason


TerminationRegister.register_pytree()


@dataclasses.dataclass(slots=True)
class Samples(PureDataclassPytree):
    log_likelihoods: FloatArray  # [max_samples]
    U_samples: UType  # [max_samples, ...]
    out_degree: IntArray  # [max_samples]
    num_likelihood_evaluations: IntArray  # [max_samples]

    def __len__(self):
        return self.log_likelihoods.shape[0]

    def __getitem__(self, item):
        return jax.tree.map(lambda x: x[item], self)

    def slice(self, start: IntArray, size: int) -> 'Samples':
        return jax.tree.map(lambda x: jax.lax.dynamic_slice(x, (start,) + (0,) * (x.ndim - 1), (size,) + x.shape[1:]),
                            self)

    def set_slice(self, start: IntArray, other: 'Samples') -> 'Samples':
        return jax.tree.map(
            lambda x, y: jax.lax.dynamic_update_slice(x, y, (start,) + (0,) * (x.ndim - 1)),
            self, other
        )

    def concat(self, other: 'Samples') -> 'Samples':
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), self, other)

    def sort(self) -> 'Samples':
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

    def perm_sort(self, key) -> 'Samples':
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

    def compute_num_live_points_per_sample(self, root_out_degree: IntArray,
                                           num_samples: IntArray | None = None) -> IntArray:
        # Cumulatively apply K[i] = K[i-1] - 1 + d(i)
        def scan_fn(carry, out_degree):
            K_prev, = carry
            K_new = K_prev - jnp.ones((), root_out_degree.dtype) + out_degree
            return (K_new,), K_new

        _, K_values = scan_or_while_loop(scan_fn, (root_out_degree,), self.out_degree.astype(root_out_degree.dtype),
                                         length=num_samples, unroll=1)
        return K_values

    def append_samples(self, insert_idx: IntArray, parent_idx: IntArray, samples: 'Samples',
                       delta_parent_out_degree: IntArray) -> 'Samples':
        samples = self.set_slice(insert_idx, samples)
        samples.out_degree = samples.out_degree.at[parent_idx].add(delta_parent_out_degree)
        return samples

    def resize(self, max_samples: int) -> 'Samples':
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


Samples.register_pytree()


@dataclasses.dataclass(slots=True)
class State(PureDataclassPytree):
    root_out_degree: IntArray
    samples: Samples
    num_samples: IntArray

    def ensure_consistency(self):
        # Every non-root has exactly one parent, so sum of out-degrees = num_samples
        assert self.root_out_degree + np.sum(self.samples.out_degree) == self.num_samples
        # You must have a replacement child in order to die
        K_samples = self.samples.sort().compute_num_live_points_per_sample(self.root_out_degree, self.num_samples)
        assert K_samples[self.num_samples - 1] == 0
        K_pre = K_samples + 1 - self.samples.out_degree
        assert np.all(K_pre[:self.num_samples] > 0)

    def evaluate_evidence(self) -> Tuple[EvidenceCalculation, EvidenceCalculation]:
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

    def sample_logZ(self, key, num_samples: int) -> FloatArray:
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


State.register_pytree()


class AbstractSampler(ABC):

    @abstractmethod
    def get_sample(self, key, log_L_constraint: FloatArray, seed_point: UType) -> Tuple[UType, FloatArray, IntArray]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            seed_point: a seed point to begin sampling from

        Returns:
            U_sample: an i.i.d. sample within the constraint
            log_L: the log-likelihood of the sample
            num_likelihood_evaluations: number of likelihood evaluations used to produce the sample
        """
        ...


def sample_init_state(key, num_live_points: int, max_samples: int, model: Model, args=(),
                      params=None,
                      batch_size: int | None = None) -> State:
    def single_sample(key):
        key, subkey = jax.random.split(key)
        U_sample = model.sample_U(subkey)
        log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False)
        num_likelihood_evaluations = jnp.array(1, dtype=mp_policy.count_dtype)
        carry = (key, U_sample, log_L, num_likelihood_evaluations)

        def cond_fn(carry):
            _, _, log_L, _ = carry
            return log_L <= -jnp.inf

        def body_fn(carry):
            key, _, _, num_likelihood_evaluations = carry
            key, subkey = jax.random.split(key)
            U_sample = model.sample_U(subkey)
            log_L = model.log_likelihood(U_sample, args=args, params=params, allow_nan=False)
            num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
            return (key, U_sample, log_L, num_likelihood_evaluations)

        key, U_sample, log_L, num_likelihood_evaluations = jax.lax.while_loop(cond_fn, body_fn, carry)
        return U_sample, log_L, num_likelihood_evaluations

    U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
        single_sample,
        jax.random.split(key, num_live_points),
        batch_size=batch_size
    )

    # extend each to max_samples
    def _concat(x, fill_value):
        return jax.tree.map(
            lambda x: jnp.concatenate([
                x,
                jnp.full((max_samples - num_live_points,) + x.shape[1:], fill_value, dtype=x.dtype)
            ], axis=0),
            x
        )

    U_samples = _concat(U_samples, 0.0)
    log_likelihoods = _concat(log_likelihoods, jnp.inf)
    num_likelihood_evaluations = _concat(num_likelihood_evaluations, 0)
    out_degree = jnp.zeros((max_samples,), dtype=mp_policy.count_dtype)
    samples = Samples(
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=out_degree,
        num_likelihood_evaluations=num_likelihood_evaluations
    )
    state = State(
        root_out_degree=jnp.array(num_live_points, dtype=mp_policy.count_dtype),
        samples=samples,
        num_samples=jnp.array(num_live_points, dtype=mp_policy.count_dtype)
    )
    # Sort samples into increasing log-likelihood order
    state.samples = state.samples.sort()
    return state


def single_ns_run(key, root_out_degree: int, max_samples: int, shell_size: int, model: Model, args=(),
                  sampler: AbstractSampler | None = None,
                  params=None,
                  termination_condition: TerminationCondition | None = None,
                  batch_size: int | None = None) -> State:
    """
    Perform a single nested sampling run, using a shell-based parallel nested sampling algorithm.

    The branching strategy is as follows:
    0 -> {r(1),...,r(K)}
    S -> {r(1) ,..., r(S)}
    2S -> {r(S), ..., r(2S)}
    ...
    nS -> {r((n-1)S), ..., r(nS)}

    where r(i) is the likelihood rank of the i-th sample.


    Args:
        key: PRNG key
        root_out_degree: the number of live points to use off root
        max_samples: the maximum number of samples to store
        shell_size: the number of samples to discard and replenish per iteration
        model: the model to sample from
        args: arguments to pass to the model
        sampler: the sampler to use to produce i.i.d. samples within likelihood constraints
        params: parameters to pass to the model
        termination_condition: the termination condition to use
        batch_size: how many likelihood evaluations to batch together

    Returns:
        A final state object containing all samples and relevant information for resuming, or evidence calculation.
    """
    # Initialise the state
    key, init_key = jax.random.split(key)
    state = sample_init_state(
        key=init_key,
        num_live_points=root_out_degree,
        max_samples=max_samples,
        model=model,
        args=args,
        params=params,
        batch_size=batch_size
    )

    # Initialize register
    register = TerminationRegister.initialise()
    register.num_samples_used = root_out_degree
    register.num_likelihood_evaluations = jnp.sum(state.samples.num_likelihood_evaluations)
    # register.log_L_contour = register.evidence_calc.L.log_abs_val
    register.efficiency = register.num_samples_used / register.num_likelihood_evaluations
    register.plateau = (log_L0 := state.samples.log_likelihoods[0]) == (
        log_L1 := state.samples.log_likelihoods[root_out_degree - 1])
    register.no_seed_points = register.plateau
    register.absolute_spread = jnp.abs(log_L1 - log_L0)
    register.relative_spread = 2. * register.absolute_spread / jnp.abs(log_L0 + log_L1)
    register.cummax_log_XL = jnp.maximum(register.cummax_log_XL,
                                         (register.evidence_calc.X_mean * register.evidence_calc.L).log_abs_val)
    register.evidence_calc_with_remaining = register.evidence_calc
    K_total_tmp = state.root_out_degree
    for idx in range(root_out_degree):
        register.evidence_calc_with_remaining = register.evidence_calc_with_remaining.update_evidence(
            K_total_tmp,
            state.samples.log_likelihoods[idx]
        )
        K_total_tmp = K_total_tmp - 1 + state.samples.out_degree[idx]

    # Sequentially discard shell, and replenish until termination condition
    discard_idx = 0
    K_total = state.root_out_degree
    done = register.is_done(termination_condition)
    while not done:
        for _ in range(shell_size):
            # Partial register update per discard
            register.evidence_calc = register.evidence_calc.update_evidence(
                K_total, state.samples.log_likelihoods[discard_idx])
            K_total = K_total - 1 + state.samples.out_degree[discard_idx]
            discard_idx += 1

        # Last discarded sample sets the likelihood constraint
        parent_idx = discard_idx - 1
        insert_idx = discard_idx

        # Replenish discarded samples, by merging and sorting with active samples
        kept_size = root_out_degree - shell_size
        active_samples = state.samples.slice(insert_idx, kept_size)

        log_L_constraint = state.samples.log_likelihoods[parent_idx]
        # When there are no seeds, we reparent off the root.
        # However, since this retroactively changes the out-degree of the root,
        # the evidence calculation needs to be recalculated.
        # It also becomes unclear where to continue sampling from, since there were no seeds in the active set.
        # We therefore stop the run if there are no seeds.
        # The user can then use adaptive refinement to continue improving if desired.
        no_seeds = jnp.all(active_samples.log_likelihoods <= log_L_constraint)
        log_L_constraint = jnp.where(no_seeds, jnp.asarray(-jnp.inf, dtype=mp_policy.measure_dtype), log_L_constraint)
        delta_root_out_degree = jnp.where(no_seeds, shell_size, 0).astype(mp_policy.count_dtype)
        delta_parent_out_degree = jnp.where(no_seeds, 0, shell_size).astype(mp_policy.count_dtype)

        # TODO: modularise out the sampling distribution strategy.

        def get_sample(key, log_L_constraint, active_samples: Samples):
            seed_key, sample_key = jax.random.split(key)
            seed_select_idx = jax.random.randint(seed_key, (), 0, len(active_samples))
            seed_point = active_samples.U_samples[seed_select_idx]
            return sampler.get_sample(
                sample_key, log_L_constraint, seed_point
            )

        key, subkey = jax.random.split(key)
        U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
            lambda key: get_sample(key, log_L_constraint, active_samples),
            jax.random.split(subkey, shell_size),
            batch_size=batch_size
        )

        new_samples = Samples(
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations
        )
        # If no reparenting off root, then this is sufficient to maintain ordering.
        joint_samples = active_samples.concat(new_samples).sort()
        assert len(joint_samples) == root_out_degree
        state.samples = state.samples.append_samples(
            insert_idx=insert_idx,
            parent_idx=parent_idx,
            samples=new_samples,
            delta_parent_out_degree=delta_parent_out_degree
        )
        state.num_samples += shell_size
        state.root_out_degree += delta_root_out_degree
        K_total += delta_root_out_degree

        # Update register now rest of the way (post sampling)
        register.num_samples_used = state.num_samples
        register.num_likelihood_evaluations += delta_num_like_evals := jnp.sum(new_samples.num_likelihood_evaluations)
        register.log_L_contour = register.evidence_calc.L.log_abs_val
        register.efficiency = 0.1 * register.efficiency + 0.9 * (shell_size / delta_num_like_evals)
        register.plateau = (log_L0 := joint_samples.log_likelihoods[0]) == (log_L1 := joint_samples.log_likelihoods[-1])
        register.no_seed_points = no_seeds
        register.absolute_spread = jnp.abs(log_L1 - log_L0)
        register.relative_spread = 2. * register.absolute_spread / jnp.abs(log_L0 + log_L1)
        register.cummax_log_XL = jnp.maximum(register.cummax_log_XL,
                                             (register.evidence_calc.X_mean * register.evidence_calc.L).log_abs_val)

        K_total_tmp = K_total
        register.evidence_calc_with_remaining = register.evidence_calc
        for idx in range(len(joint_samples)):
            register.evidence_calc_with_remaining = register.evidence_calc_with_remaining.update_evidence(
                K_total_tmp,
                joint_samples.log_likelihoods[idx]
            )
            K_total_tmp = K_total_tmp - 1 + joint_samples.out_degree[idx]

        # Check termination condition
        done = register.is_done(termination_condition)

    state.samples = state.samples.sort()
    return state


def resume_ns_run(key, state: State, log_L_start: FloatArray, log_L_end: FloatArray, model: Model, args=(),
                  sampler: AbstractSampler | None = None,
                  params=None,
                  max_samples: int | None = None) -> State:
    ...


def refine_ns_run(
        key, state: State,
        log_L_start: FloatArray, log_L_end: FloatArray,
        target_live_points: IntArray,
        model: Model, args=(), sampler: AbstractSampler | None = None,
        params=None,
        max_samples: int | None = None,
        batch_size: int | None = None
) -> State:
    """
    Adds children off parents in the specified log-likelihood range, until each contour within that range has at least
    the target number of live points, or until there are no seed points.
    Greedily adds children off the lowest live point parents first.

    Args:
        key: PRNG key
        state: the current state of the nested sampling run
        log_L_start: the starting log-likelihood contour to refine from
        log_L_end: the ending log-likelihood contour to refine to
        target_live_points: the target number of live points per contour
        model: the model to sample from
        args: arguments to pass to the model
        sampler: the sampler to use to produce i.i.d. samples within likelihood constraints
        params: parameters to pass to the model
        max_samples: the maximum number of samples to store
        batch_size: how many likelihood evaluations to batch together

    Returns:
        A final state object containing all samples and relevant information for resuming, or evidence calculation.
    """
    if max_samples is not None:
        state.samples = state.samples.resize(max_samples)
    done = ...
    while not done:
        state.samples = state.samples.sort()
        K_total_per_sample = state.samples.compute_num_live_points_per_sample(
            root_out_degree=state.root_out_degree,
            num_samples=state.num_samples
        )  # [max_samples]
        sample_mask = jnp.logical_and(
            K_total_per_sample < target_live_points,
            (state.samples.log_likelihoods >= log_L_start) &
            (state.samples.log_likelihoods <= log_L_end)
        )
        key, select_key = jax.random.split(key)
        parent_idxs = jax.random.choice(
            select_key, len(state.samples), replace=True,
            shape=(batch_size,),
            p=jnp.where(
                sample_mask.astype(jnp.float32),
                target_live_points - K_total_per_sample,
                0
            )
        )
        empty_sample_mask = ~jnp.any(sample_mask)
        log_L_constraints = jnp.where(
            empty_sample_mask,
            state.samples.log_likelihoods[parent_idxs],
            -jnp.inf
        )

        def get_sample(key, log_L_constraint, active_samples: Samples):
            seed_key, sample_key = jax.random.split(key)
            seed_select_idx = jax.random.randint(seed_key, (), 0, len(active_samples))
            seed_point = active_samples.U_samples[seed_select_idx]
            return sampler.get_sample(
                sample_key, log_L_constraint, seed_point
            )

        key, subkey = jax.random.split(key)
        U_samples, log_likelihoods, num_likelihood_evaluations = jax.lax.map(
            lambda key, log_L_constraint: get_sample(key, log_L_constraint, active_samples),
            jax.random.split(subkey, shell_size),
            batch_size=batch_size
        )

        new_samples = Samples(
            log_likelihoods=log_likelihoods,
            U_samples=U_samples,
            out_degree=jnp.zeros((shell_size,), dtype=mp_policy.count_dtype),
            num_likelihood_evaluations=num_likelihood_evaluations
        )
