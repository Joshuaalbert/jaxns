import dataclasses
from functools import partial

import jax.random
import jax.tree
import numpy as np
from jax import numpy as jnp
from jaxctx import CtxParams

from jaxns.cumulative_ops import scan_or_while_loop
from jaxns.evidence_calculation import EvidenceCalculation
from jaxns.log_semiring import LogSpace, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import Samples, UType
from jaxns.stats_utils import linear_to_log_stats, effective_sample_size_kish
from jaxns.termination_condition import TerminationRegister
from jaxns.types import IntArray, FloatArray
from jaxns.results import NestedSamplerResults


@dataclasses.dataclass(slots=True)
class State(PureDataclassPytree):
    root_out_degree: IntArray  # scalar
    samples: Samples
    num_samples: IntArray  # scalar

    log_L_supremum: FloatArray  # scalar, the maximum likelihood value seen so far, used for termination conditions and evidence calculation
    U_supremum: UType

    termination_reason: IntArray

    model: Model
    args: tuple = ()
    params: CtxParams | None = None

    def merge(self, other: 'State') -> 'State':
        """
        Merge samples from another state into this state. This is used for merging results from parallel nested sampling runs.

        Args:
            other: another state to merge with this state. Must have the same model, args, and params.

        Returns:
            a new state with the samples from both states merged together.
        """
        return _merge(self, other)

    def determine_parent_graph(self):
        """
        Compute the parent graph from the out-degrees along the lineage. This is needed for book-keeping and for computing the evidence.

        Returns:
            parent_edges: [num_samples, 2] array of (parent_idx, child_idx) edges, where parent_idx is the index of the parent sample, and child_idx is the
            index of the child sample. The root sample has parent_idx = -1.
        """
        return _determine_parent_graph(self)

    def ensure_consistency(self):
        """
        Ensures the parent graph is consistent with the out-degrees, and that every non-root has a replacement child.

        Raises:
            AssertionError: if the parent graph is not consistent with the out-degrees, or if any non-root sample does not have a replacement child.
        """
        return _ensure_consistency(self)

    def evaluate_evidence(self) -> tuple[EvidenceCalculation, EvidenceCalculation]:
        """
        Evaluate evidence over the current state.

        Returns:
            an evidence calculation, cumulative evdience calculation
        """
        return _evaluate_evidence(self)

    def sample_logZ(self, key, num_samples: int) -> FloatArray:
        """
        Samples log-evidence from the current state.

        Args:
            key: PRNGKey
            num_samples: how many sampels to produce.

        Returns:
            samples of log-evidence.
        """
        return _sample_logZ(self, key, num_samples)

    def compute_termination_register(self, target_num_live_points: int) -> TerminationRegister:
        """
        Compute the termination register, which contains all the information needed to evaluate the termination condition, and to compute the evidence if the run is terminated.

        Args:
            target_num_live_points: the number of live points to use off root, which is needed to compute the evidence shrinkage and efficiency shrinkage.

        Returns:
            a TerminationRegister containing all the information needed to evaluate the termination condition, and to compute the evidence if the run is terminated.
        """
        return _compute_termination_register(self, target_num_live_points)

    def to_result(self: NestedSamplerResults) -> NestedSamplerResults:
        """
            Convert the current state to a NestedSamplerResults object, which contains all the information about the samples, evidence, and diagnostics.

        Returns:
            result object
        """
        return _to_result(self)




State.register_pytree()

@partial(jax.jit, inline=True)
def _to_result(self: State) -> NestedSamplerResults:
    max_samples = self.samples.log_likelihoods.shape[0]
    total_num_samples = self.num_samples.astype(mp_policy.count_dtype)
    sample_mask = jnp.arange(max_samples) < total_num_samples
    evidence_calc, cum_evidence_calc = self.evaluate_evidence()
    log_Z_mean, log_Z_var = linear_to_log_stats(evidence_calc.Z_mean.log_abs_val, log_f2_mean=evidence_calc.Z2_mean.log_abs_val)
    log_Z_uncert = jnp.sqrt(jnp.maximum(0, log_Z_var))
    ess = effective_sample_size_kish(
        evidence_calc.Z_mean.log_abs_val,
        evidence_calc.dZ2_mean.log_abs_val
    )
    log_dZ_mean = jnp.where(sample_mask, cum_evidence_calc.dZ_mean.log_abs_val, jnp.asarray(-jnp.inf, mp_policy.measure_dtype))
    dp_mean = normalise_log_space(LogSpace(log_dZ_mean))
    # E[log(L) - log(Z)]
    H_mean_instable = (
            (
                    dp_mean * LogSpace.from_signed_value(
                jnp.where(jnp.isneginf(dp_mean.log_abs_val), 0., self.samples.log_likelihoods)
            )
            ).sum().value - log_Z_mean
    )
    # H \approx E[-log(compression)] = E[-log(X)] (More stable than E[log(L) - log(Z)] but biased)
    H_mean_stable = (dp_mean * LogSpace(jnp.log(-cum_evidence_calc.X_mean.log_abs_val))).sum().value
    H_mean = jnp.where(jnp.isfinite(H_mean_instable), H_mean_instable, H_mean_stable)
    U_samples = self.samples.U_samples
    X_samples = jax.lax.map(lambda u: self.model.transform_to_X(u, args=self.args, params=self.params), U_samples)
    X_samples = jax.tree.map(lambda x: jnp.where(sample_mask.reshape([-1] + [1] * (len(x.shape) - 1)), x, 0.), X_samples)
    log_L = self.samples.log_likelihoods
    log_dp = dp_mean.log_abs_val
    log_X_mean = cum_evidence_calc.X_mean.log_abs_val
    log_posterior_density = log_L + jax.lax.map(
        lambda u: self.model.log_prior(u, args=self.args, params=self.params), U_samples
    ) - log_Z_mean
    log_posterior_density = jnp.where(sample_mask, log_posterior_density, -jnp.inf)
    num_live_points_per_sample = self.samples.compute_num_live_points_per_sample(root_out_degree=self.root_out_degree, num_samples=self.num_samples)
    num_likelihood_evaluations_per_sample = self.samples.num_likelihood_evaluations.astype(mp_policy.count_dtype)
    total_phantom_samples = jnp.sum(self.samples.phantom_samples.valid_mask).astype(mp_policy.count_dtype)
    total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
    log_efficiency = jnp.log(total_num_samples) - jnp.log(total_num_likelihood_evaluations)
    log_L_constraints = self.samples.log_L_constraints
    log_L_phantom = self.samples.phantom_samples.log_L
    # TODO: Currently only accept clusters which are completely valid.
    phantom_valid_mask = self.samples.phantom_samples.valid_mask
    if phantom_valid_mask.shape[-1] == 0:
        valid_phantom = jnp.zeros(phantom_valid_mask.shape[:-1], dtype=mp_policy.bool_dtype)
    else:
        valid_phantom = jnp.all(phantom_valid_mask, axis=-1)

    X_supremum = self.model.transform_to_X(self.U_supremum, args=self.args, params=self.params)
    map_idx = jnp.argmax(log_posterior_density)
    log_L_map = log_L[map_idx]
    U_map = jax.tree.map(lambda u: u[map_idx], U_samples)
    X_map = jax.tree.map(lambda x: x[map_idx], X_samples)
    return NestedSamplerResults(
        U_samples=U_samples,
        X_samples=X_samples,
        log_L=log_L,
        log_dp=log_dp,
        log_X_mean=log_X_mean,
        log_posterior_density=log_posterior_density,
        num_live_points_per_sample=num_live_points_per_sample,
        num_likelihood_evaluations_per_sample=num_likelihood_evaluations_per_sample,
        log_Z_mean=log_Z_mean,
        log_Z_uncert=log_Z_uncert,
        ess=ess,
        H_mean=H_mean,
        total_num_samples=total_num_samples,
        total_phantom_samples=total_phantom_samples,
        total_num_likelihood_evaluations=total_num_likelihood_evaluations,
        log_efficiency=log_efficiency,
        termination_reason=self.termination_reason,
        log_L_supremum=self.log_L_supremum,
        U_supremum=self.U_supremum,
        X_supremum=X_supremum,
        log_L_map=log_L_map,
        U_map=U_map,
        X_map=X_map,
        log_L_constraints=log_L_constraints,
        log_L_phantom=log_L_phantom,
        valid_phantom=valid_phantom
    )

@partial(jax.jit, inline=True)
def _merge(self: State, other: State) -> 'State':
    assert jax.tree.structure(self.model) == jax.tree.structure(other.model), "Cannot merge states with different models"
    assert jax.tree.structure(self.args) == jax.tree.structure(other.args), "Cannot merge states with different args"
    assert jax.tree.structure(self.params) == jax.tree.structure(other.params), "Cannot merge states with different params"
    return State(
        root_out_degree=self.root_out_degree + other.root_out_degree,
        samples=self.samples.concat(other.samples),
        num_samples=self.num_samples + other.num_samples,
        log_L_supremum=jnp.maximum(self.log_L_supremum, other.log_L_supremum),
        U_supremum=jax.tree.map(lambda x, y: jnp.where(self.log_L_supremum >= other.log_L_supremum, x, y), self.U_supremum, other.U_supremum),
        termination_reason=jnp.where(self.termination_reason != 0, self.termination_reason, other.termination_reason),
        model=self.model,
        args=self.args,
        params=self.params
    )


@partial(jax.jit, inline=True)
def _determine_parent_graph(self: State):
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


def _ensure_consistency(self: State):
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
def _evaluate_evidence(self: State) -> tuple[EvidenceCalculation, EvidenceCalculation]:
    evidence_calc, cum_evidence = EvidenceCalculation.initialise().update_from_samples(
        samples=self.samples, root_out_degree=self.root_out_degree, num_samples=self.num_samples)
    return evidence_calc, cum_evidence


@partial(jax.jit, inline=True, static_argnames=['num_samples'])
def _sample_logZ(self: State, key, num_samples: int) -> FloatArray:
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

        (_, Z_sample), _ = scan_or_while_loop(scan_fn, carry_init, xs, length=self.num_samples, unroll=1)
        return Z_sample

    keys = jax.random.split(key, num_samples)
    Z_samples = jax.vmap(single_sample_logZ)(keys)
    logZ = Z_samples.log_abs_val
    return logZ


@partial(jax.jit, inline=True, static_argnames=['target_num_live_points'])
def _compute_termination_register(state: State, target_num_live_points: int) -> TerminationRegister:
    evidence_calc, cum_evidence_calc = state.evaluate_evidence()
    Z_0 = cum_evidence_calc.Z_mean[state.num_samples - target_num_live_points]
    Z2_0 = cum_evidence_calc.Z2_mean[state.num_samples - target_num_live_points]
    Z_1 = cum_evidence_calc.Z_mean[state.num_samples - 1]
    Z2_1 = cum_evidence_calc.Z2_mean[state.num_samples - 1]
    log_Z_mean_0, log_Z_var_0 = linear_to_log_stats(Z_0.log_abs_val, log_f2_mean=Z2_0.log_abs_val)
    log_Z_mean_1, log_Z_var_1 = linear_to_log_stats(Z_1.log_abs_val, log_f2_mean=Z2_1.log_abs_val)
    dZ_shrinkage = LogSpace(log_Z_mean_1) - LogSpace(log_Z_mean_0)
    active_live_points = state.samples.slice(state.num_samples - target_num_live_points, target_num_live_points)
    num_likelihood_evaluations = jnp.sum(state.samples.num_likelihood_evaluations)
    efficiency_shrinkage = target_num_live_points / jnp.sum(active_live_points.num_likelihood_evaluations)
    cummax_XL = (cum_evidence_calc.X_mean * evidence_calc.L).max()
    log_L_max_active = active_live_points.log_likelihoods[-1]
    log_L_min_active = active_live_points.log_likelihoods[0]
    absolute_spread = jnp.abs(log_L_max_active - log_L_min_active)
    relative_spread = 2. * absolute_spread / jnp.abs(log_L_max_active + log_L_min_active)
    plateau = log_L_max_active == log_L_min_active
    no_seed_points = plateau
    register = TerminationRegister(
        num_samples_used=state.num_samples,
        evidence_calc=evidence_calc,
        dZ_shrinkage=dZ_shrinkage,
        num_likelihood_evaluations=num_likelihood_evaluations,
        log_L_max=state.log_L_supremum,
        log_L_contour_max=evidence_calc.L.log_abs_val,
        efficiency_shrinkage=efficiency_shrinkage,
        plateau=plateau,
        no_seed_points=no_seed_points,
        relative_spread=relative_spread,
        absolute_spread=absolute_spread,
        cummax_XL=cummax_XL
    )

    return register
