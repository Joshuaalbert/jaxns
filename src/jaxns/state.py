import dataclasses
from functools import partial
from typing import Literal

import jax.random
import jax.tree
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jaxctx import CtxParams

from jaxns.cumulative_ops import scan_or_while_loop
from jaxns.evidence_calculation import EvidenceCalculation
from jaxns.log_semiring import LogSpace, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.phantom_eval import EvidenceSamples
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import BlockState, LikelihoodOrder, build_block_state
from jaxns.results import NestedSamplerResults
from jaxns.samples import Samples, UType
from jaxns.stats_utils import effective_sample_size_kish
from jaxns.termination_condition import TerminationRegister
from jaxns.types import FloatArray, IntArray
from jaxns.v3_shrinkage import (
    classic_dirichlet_concentrations,
    dirichlet_probability_means,
    expected_v3_evidence_summary,
    expected_v3_log_posterior_weights,
    sample_v3_evidence,
    validate_lineage_capacity,
)


@dataclasses.dataclass(slots=True, frozen=True)
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
    goal_loop_iter: IntArray = dataclasses.field(
        default_factory=lambda: jnp.asarray(0, mp_policy.count_dtype)
    )
    depth_loop_iter: IntArray = dataclasses.field(
        default_factory=lambda: jnp.asarray(0, mp_policy.count_dtype)
    )
    likelihood_order: LikelihoodOrder | None = None
    random_key: jax.Array | None = None
    num_reparented: IntArray = dataclasses.field(
        default_factory=lambda: jnp.asarray(0, mp_policy.count_dtype)
    )
    num_reused_seeds: IntArray = dataclasses.field(
        default_factory=lambda: jnp.asarray(0, mp_policy.count_dtype)
    )

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
        _validate_evidence_block_capacity(self)
        return _sample_logZ(self, key, num_samples)

    def sample_evidence_mc(
            self,
            num_samples: int,
            *,
            conditioning: Literal["classic", "phantom"],
            key: jax.Array,
            batch_size: int | None = None,
            C_min: float = 20,
    ) -> EvidenceSamples:
        """Draw final evidence samples from this immutable state.

        This is a thin state-level forwarder to the same result implementation
        used after a run. ``conditioning`` must explicitly be ``"classic"``
        or ``"phantom"``; the depth-loop expectation register is not used.
        """
        return self.to_result().trim().sample_evidence_mc(
            num_samples=num_samples,
            conditioning=conditioning,
            key=key,
            batch_size=batch_size,
            C_min=C_min,
        )

    def compute_termination_register(
            self,
            target_num_live_points: int | None = None,
    ) -> TerminationRegister:
        """
        Compute the termination register, which contains all the information needed to evaluate the termination condition, and to compute the evidence if the run is terminated.

        Args:
            target_num_live_points: Deprecated compatibility argument. The v3
                register is derived from the current race blocks.

        Returns:
            a TerminationRegister containing all the information needed to evaluate the termination condition, and to compute the evidence if the run is terminated.
        """
        del target_num_live_points
        return _compute_termination_register(self)

    def to_result(self: "State") -> NestedSamplerResults:
        """Convert this state to user-facing samples and diagnostics.

        Returns:
            Immutable nested-sampling results.
        """
        _validate_evidence_block_capacity(self)
        return _to_result(self)

    def trim(self) -> "State":
        """Return a state whose sample buffers contain only valid samples."""
        return _trim(self)

    def resize(self, max_samples: int) -> "State":
        """Return a state with larger preallocated sample buffers."""
        return dataclasses.replace(
            self,
            samples=self.samples.resize(max_samples),
            likelihood_order=(
                None
                if self.likelihood_order is None
                else self.likelihood_order.resize(max_samples)
            ),
        )


State.register_pytree()


def _trim(self: State) -> State:
    num_samples = int(np.asarray(self.num_samples))
    return dataclasses.replace(
        self,
        samples=self.samples.slice(
            jnp.asarray(0, mp_policy.index_dtype),
            num_samples,
        ),
        num_samples=jnp.asarray(num_samples, mp_policy.count_dtype),
        likelihood_order=(
            None
            if self.likelihood_order is None
            else LikelihoodOrder(
                sample_indices=self.likelihood_order.sample_indices[
                    :num_samples
                ]
            )
        ),
    )


def _validate_evidence_block_capacity(self: State) -> None:
    block_state = build_block_state(
        self.samples,
        root_out_degree=self.root_out_degree,
        num_samples=self.num_samples,
        likelihood_order=self.likelihood_order,
        validate=True,
    )
    validate_lineage_capacity(block_state)


@partial(jax.jit, inline=True)
def _to_result(self: State) -> NestedSamplerResults:
    max_samples = self.samples.log_likelihoods.shape[0]
    total_num_samples = self.num_samples.astype(mp_policy.count_dtype)
    sample_mask = jnp.arange(max_samples) < total_num_samples
    log_L = self.samples.log_likelihoods
    block_state = build_block_state(
        self.samples,
        root_out_degree=self.root_out_degree,
        num_samples=self.num_samples,
        likelihood_order=self.likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    evidence_summary = expected_v3_evidence_summary(block_state, concentrations)
    log_Z_mean = evidence_summary.log_Z_mean
    log_Z_uncert = evidence_summary.log_Z_uncert
    ess = effective_sample_size_kish(
        evidence_summary.log_Z_linear_mean,
        evidence_summary.log_dZ2_sum,
    )
    log_dp = expected_v3_log_posterior_weights(block_state, concentrations)
    log_dp = jnp.where(sample_mask, log_dp, jnp.asarray(-jnp.inf, mp_policy.measure_dtype))
    dp_mean = normalise_log_space(LogSpace(log_dp))
    # E[log(L) - log(Z)]
    log_l_for_h = jnp.where(
        jnp.isneginf(dp_mean.log_abs_val),
        0.0,
        self.samples.log_likelihoods,
    )
    H_mean = (
        (dp_mean * LogSpace.from_signed_value(log_l_for_h)).sum().value
        - log_Z_mean
    )
    U_samples = self.samples.U_samples
    X_samples = jax.vmap(
        lambda u: self.model.transform_to_X(
            u,
            args=self.args,
            params=self.params,
        )
    )(U_samples)
    X_samples = jax.tree.map(
        lambda x: jnp.where(
            sample_mask.reshape([-1] + [1] * (len(x.shape) - 1)),
            x,
            0.0,
        ),
        X_samples,
    )
    log_X_mean = jnp.where(sample_mask, evidence_summary.log_X_mean, -jnp.inf)
    log_posterior_density = log_L + jax.vmap(
        lambda u: self.model.log_prior(
            u,
            args=self.args,
            params=self.params,
        )
    )(U_samples) - log_Z_mean
    log_posterior_density = jnp.where(sample_mask, log_posterior_density, -jnp.inf)
    # Storage order is stable append order, not likelihood order. Map each
    # sample to its strict likelihood block instead of sorting the coordinate
    # pytree merely to recover K_g.
    sample_block_idx = jnp.searchsorted(
        block_state.log_L_blocks,
        self.samples.log_likelihoods,
        side="left",
    )
    sample_block_idx = jnp.clip(
        sample_block_idx,
        0,
        block_state.log_L_blocks.shape[0] - 1,
    )
    num_live_points_per_sample = jnp.where(
        sample_mask,
        block_state.incoming_K[sample_block_idx],
        jnp.asarray(0, mp_policy.count_dtype),
    )
    num_likelihood_evaluations_per_sample = (
        self.samples.num_likelihood_evaluations.astype(mp_policy.count_dtype)
    )
    total_phantom_samples = jnp.sum(self.samples.phantom_samples.valid_mask).astype(mp_policy.count_dtype)
    total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
    log_efficiency = jnp.log(total_num_samples) - jnp.log(total_num_likelihood_evaluations)
    log_L_constraints = self.samples.log_L_constraints
    log_L_phantom = self.samples.phantom_samples.log_L
    # A cluster is one statistical unit with a shared gamma weight. Partially
    # populated rows are excluded rather than silently changing cluster size.
    phantom_valid_mask = self.samples.phantom_samples.valid_mask
    if phantom_valid_mask.shape[-1] == 0:
        valid_phantom = jnp.zeros(phantom_valid_mask.shape[:-1], dtype=mp_policy.bool_dtype)
    else:
        valid_phantom = jnp.all(phantom_valid_mask, axis=-1)
    block_p_gt_mean, block_p_eq_mean, _ = dirichlet_probability_means(concentrations)

    X_supremum = self.model.transform_to_X(self.U_supremum, args=self.args, params=self.params)
    map_idx = jnp.argmax(log_posterior_density)
    log_L_map = log_L[map_idx]
    U_map = jax.tree.map(lambda u: u[map_idx], U_samples)
    X_map = jax.tree.map(lambda x: x[map_idx], X_samples)
    return NestedSamplerResults(
        U_samples=U_samples,
        X_samples=X_samples,
        log_L=log_L,
        log_L_blocks=block_state.log_L_blocks,
        log_dp=log_dp,
        v3_log_posterior_weights=log_dp,
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
        valid_phantom=valid_phantom,
        block_first_idx=block_state.block_first_idx,
        block_size=block_state.block_size,
        block_incoming_K=block_state.incoming_K,
        block_out_degree=block_state.block_out_degree,
        block_start=block_state.block_start,
        block_stop=block_state.block_stop,
        block_sample_indices=block_state.block_sample_indices,
        block_classic_alpha_gt=concentrations.alpha_gt,
        block_classic_alpha_eq=concentrations.alpha_eq,
        block_classic_alpha_lt=concentrations.alpha_lt,
        block_epsilon=concentrations.epsilon,
        block_classic_p_gt_mean=jnp.where(
            block_state.valid,
            block_p_gt_mean,
            jnp.nan,
        ),
        block_classic_p_eq_mean=jnp.where(
            block_state.valid,
            block_p_eq_mean,
            jnp.nan,
        ),
        # Phantom count matrices are potentially large and are not required
        # for classic result construction. They are evaluated lazily by
        # ``phantom_conditioning_diagnostics`` or MC shrinkage after trimming.
        block_phantom_A=None,
        block_phantom_B=None,
        block_phantom_E=None,
        block_phantom_R=None,
        block_kish_participating_cluster_counts=None,
        block_phantom_gate_active=None,
        effective_parent_idx=self.samples.parent_idx,
        requested_parent_idx=self.samples.requested_parent_idx,
        requested_log_L_constraint=(
            self.samples.requested_log_L_constraint
        ),
        phantom_seed_idx=self.samples.seed_idx,
        num_reparented=self.num_reparented,
        num_reused_seeds=self.num_reused_seeds,
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
        params=self.params,
        goal_loop_iter=self.goal_loop_iter + other.goal_loop_iter,
        depth_loop_iter=self.depth_loop_iter + other.depth_loop_iter,
        num_reparented=self.num_reparented + other.num_reparented,
        num_reused_seeds=self.num_reused_seeds + other.num_reused_seeds,
        # Merging changes append identities, so rebuild this optional cache on
        # first use instead of pretending either input ordering is still valid.
        likelihood_order=None,
        random_key=self.random_key,
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
    num_samples = int(np.asarray(self.num_samples))
    max_samples = self.samples.log_likelihoods.shape[0]
    if num_samples < 0 or num_samples > max_samples:
        raise ValueError(
            f"num_samples={num_samples} is outside the available sample "
            f"range [0, {max_samples}]."
        )
    if num_samples == 0:
        return

    root_out_degree = int(np.asarray(self.root_out_degree))
    if root_out_degree <= 0:
        raise ValueError("root_out_degree must be positive for a non-empty race tree.")

    out_degree = np.asarray(self.samples.out_degree[:num_samples])
    if np.any(out_degree < 0):
        bad = np.where(out_degree < 0)[0][0]
        raise ValueError(f"Invalid race tree: out_degree[{bad}] is negative.")

    total_children = root_out_degree + int(np.sum(out_degree))
    if total_children != num_samples:
        raise ValueError(
            "Invalid race tree out-degree total: "
            f"root_out_degree + sum(out_degree) = {total_children}, "
            f"expected num_samples={num_samples}."
        )

    log_likelihoods = np.asarray(self.samples.log_likelihoods[:num_samples])
    constraints = np.asarray(self.samples.log_L_constraints[:num_samples])
    if np.any(log_likelihoods <= constraints):
        bad = np.where(log_likelihoods <= constraints)[0][0]
        raise ValueError(
            "Strict contour violation for sample "
            f"{bad}: log_likelihood={log_likelihoods[bad]} must be greater "
            f"than log_L_constraint={constraints[bad]}."
        )

    build_block_state(
        samples=self.samples,
        root_out_degree=self.root_out_degree,
        num_samples=self.num_samples,
        validate=True,
    )
    return


@partial(jax.jit, inline=True)
def _evaluate_evidence(self: State) -> tuple[EvidenceCalculation, EvidenceCalculation]:
    evidence_calc, cum_evidence = EvidenceCalculation.initialise().update_from_samples(
        samples=self.samples, root_out_degree=self.root_out_degree, num_samples=self.num_samples)
    return evidence_calc, cum_evidence


@partial(jax.jit, inline=True, static_argnames=['num_samples'])
def _sample_logZ(self: State, key, num_samples: int) -> FloatArray:
    block_state = build_block_state(
        self.samples,
        root_out_degree=self.root_out_degree,
        num_samples=self.num_samples,
        likelihood_order=self.likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    return sample_v3_evidence(
        key=key,
        block_state=block_state,
        concentrations=concentrations,
        num_samples=num_samples,
    ).log_Z_samples


@partial(jax.jit, inline=True)
def _compute_termination_register(state: State) -> TerminationRegister:
    """Build a linear-memory expectation register for a depth-loop check."""
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        likelihood_order=state.likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    alpha0 = (
        concentrations.alpha_gt
        + concentrations.alpha_eq
        + concentrations.alpha_lt
    )
    valid = block_state.valid & (alpha0 > 0.0)
    p_gt = jnp.where(valid, concentrations.alpha_gt / alpha0, 1.0)
    p_gt = jnp.clip(p_gt, 1e-300, 1.0)
    X = jnp.cumprod(jnp.where(valid, p_gt, 1.0))
    X_prev = jnp.concatenate(
        [jnp.ones((1,), X.dtype), X[:-1]],
        axis=0,
    )
    shell_mass = jnp.maximum(X_prev - X, 0.0)
    return termination_register_from_volume_path(
        state,
        block_state,
        X,
        shell_mass,
    )


def termination_register_from_volume_path(
        state: State,
        block_state: BlockState,
        X: FloatArray,
        shell_mass: FloatArray,
) -> TerminationRegister:
    """Summarise a shared deterministic volume path for loop conditions."""
    valid = block_state.valid
    log_X = jnp.where(
        valid & (X > 0.0),
        jnp.log(X),
        -jnp.inf,
    )
    log_dX = jnp.where(
        valid & (shell_mass > 0.0),
        jnp.log(shell_mass),
        -jnp.inf,
    )
    log_dZ = jnp.where(
        valid,
        block_state.log_L_blocks + log_dX,
        -jnp.inf,
    )
    log_Z = logsumexp(log_dZ)
    posterior_weights = jnp.exp(log_dZ - log_Z)
    ess = 1.0 / jnp.sum(jnp.square(posterior_weights))
    information = jnp.sum(
        jnp.where(
            valid,
            posterior_weights * (block_state.log_L_blocks - log_Z),
            0.0,
        )
    )
    mean_K = jnp.sum(
        posterior_weights
        * jnp.maximum(block_state.incoming_K, 1).astype(log_X.dtype)
    )
    log_Z_uncert = jnp.sqrt(jnp.maximum(information, 0.0) / mean_K)

    num_blocks = block_state.num_blocks
    last_idx = jnp.maximum(num_blocks - 1, 0)
    log_remaining = block_state.log_L_blocks[last_idx] + log_X[last_idx]
    remaining_fraction = jnp.exp(
        log_remaining - jnp.logaddexp(log_Z, log_remaining)
    )
    log_XL = jnp.where(
        valid,
        block_state.log_L_blocks + log_X,
        -jnp.inf,
    )
    posterior_tail_fraction = jnp.exp(
        log_XL[last_idx] - jnp.max(log_XL)
    )

    num_likelihood_evaluations = jnp.sum(
        jnp.where(
            jnp.arange(state.samples.log_likelihoods.shape[0])
            < state.num_samples,
            state.samples.num_likelihood_evaluations,
            0,
        )
    )
    efficiency = state.num_samples.astype(log_X.dtype) / jnp.maximum(
        num_likelihood_evaluations.astype(log_X.dtype),
        1.0,
    )
    first_log_L = block_state.log_L_blocks[0]
    last_log_L = block_state.log_L_blocks[last_idx]
    absolute_spread = jnp.abs(last_log_L - first_log_L)
    relative_spread = 2.0 * absolute_spread / jnp.maximum(
        jnp.abs(last_log_L + first_log_L),
        jnp.finfo(log_X.dtype).eps,
    )
    plateau = num_blocks == 1
    return TerminationRegister(
        num_samples_used=state.num_samples,
        num_likelihood_evaluations=num_likelihood_evaluations,
        log_Z_mean=log_Z,
        log_Z_uncert=log_Z_uncert,
        remaining_evidence_fraction=remaining_fraction,
        posterior_tail_fraction=posterior_tail_fraction,
        ess=ess,
        log_L_max=state.log_L_supremum,
        log_L_contour_max=last_log_L,
        efficiency_shrinkage=efficiency,
        plateau=plateau,
        no_seed_points=jnp.asarray(False, mp_policy.bool_dtype),
        relative_spread=relative_spread,
        absolute_spread=absolute_spread,
    )
