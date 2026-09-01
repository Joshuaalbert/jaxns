import dataclasses
import operator
from functools import partial
from typing import Literal

import jax.random
import jax.tree
import numpy as np
from jax import numpy as jnp
from jaxctx import CtxParams

from jaxns.algorithm.race_tree import LikelihoodOrder, build_block_state
from jaxns.algorithm.scheduler import ThreadSchedule
from jaxns.cumulative_ops import scan_or_while_loop
from jaxns.log_semiring import LogSpace, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, pytree_ravel
from jaxns.results import BlockData, NestedSamplerResults
from jaxns.samples import Samples, UType
from jaxns.sampling.ellipsoid import (
    SamplerData,
    empty_sampler_data,
    update_sampler_data,
)
from jaxns.shrinkage.classic import (
    classic_dirichlet_concentrations,
    dirichlet_probability_means,
    expected_evidence_summary,
    expected_log_posterior_weights,
    sample_evidence,
    validate_lineage_capacity,
)
from jaxns.shrinkage.online import EvidenceCalculation
from jaxns.shrinkage.phantom import EvidenceSamples
from jaxns.stats_utils import effective_sample_size_kish
from jaxns.types import BoolArray, FloatArray, IntArray


@dataclasses.dataclass(slots=True, frozen=True)
class State(PureDataclassPytree):
    root_out_degree: IntArray  # []
    samples: Samples
    num_samples: IntArray  # []

    log_L_supremum: FloatArray  # [] maximum likelihood seen so far
    U_supremum: UType  # [...] one point in the unit-hypercube pytree

    termination_reason: IntArray  # [] zero or a hard-stop reason code

    model: Model
    args: tuple = ()
    params: CtxParams | None = None
    goal_loop_iter: IntArray = dataclasses.field(  # []
        default_factory=lambda: jnp.asarray(0, mp_policy.count_dtype)
    )
    # Allocation targets may advance several times inside one expected-depth
    # traversal (for example, to resolve a terminal plateau) without exposing
    # an intermediate state to the user goal.
    allocation_loop_iter: IntArray = dataclasses.field(  # []
        default_factory=lambda: jnp.asarray(0, mp_policy.count_dtype)
    )
    depth_loop_iter: IntArray = dataclasses.field(  # []
        default_factory=lambda: jnp.asarray(0, mp_policy.count_dtype)
    )
    likelihood_order: LikelihoodOrder | None = None
    random_key: jax.Array | None = None  # [2]
    # While growth interrupts a depth epoch, this preserves the already-split
    # key for the next Python goal boundary.
    goal_key: jax.Array | None = None  # [2]
    needs_growth: BoolArray = dataclasses.field(  # []
        default_factory=lambda: jnp.asarray(False, mp_policy.bool_dtype)
    )
    # True only at a Python goal boundary after the expected-depth cut is met.
    # Filling one allocation target without reaching that cut advances
    # allocation_loop_iter internally and remains invisible to the user goal.
    depth_reached: BoolArray = dataclasses.field(  # []
        default_factory=lambda: jnp.asarray(False, mp_policy.bool_dtype)
    )
    # Direction geometry is continuation data, not a user-facing scientific
    # result. Keeping it on State makes capacity growth and checkpoint/resume
    # reproduce the same future transition kernels without hidden mutation.
    sampler_data: SamplerData | None = None
    # A frozen schedule is present only while one planning round is incomplete.
    # It makes storage growth and checkpoint resume transparent without adding
    # parent identities to scientific samples or results.
    scheduler_data: ThreadSchedule | None = None

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
        Reconstruct one compatible parent graph from likelihoods and out-degrees.

        The graph is an optional derived view for inspection. Parent indices
        are not persistent state and are not needed for evidence calculation.

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

    @property
    def expected_log_Z_mean(self) -> FloatArray:
        """Return the deterministic block-moment estimate of ``E[log Z]``.

        This goal-loop view uses only the immutable race-tree fields already
        maintained by the core. It intentionally avoids constructing the
        transformed samples and posterior arrays owned by :meth:`to_result`.
        """
        mean, _ = _expected_evidence_scalars(self)
        return mean

    @property
    def expected_log_Z_uncert(self) -> FloatArray:
        """Return the deterministic block-moment uncertainty approximation.

        The final user-facing uncertainty may instead be sampled with
        :meth:`sample_evidence_mc`; this scalar exists for inexpensive Python
        goal conditions between compiled depth epochs.
        """
        _, uncertainty = _expected_evidence_scalars(self)
        return uncertainty

    @property
    def total_num_likelihood_evaluations(self) -> IntArray:
        """Return the logical likelihood work from valid classic rows only."""
        return _total_likelihood_evaluations(self)

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

    def fit_gmm_directions(
            self,
            *,
            num_components: int | None = None,
            num_iterations: int = 10,
            iso_prob: float = 1e-2,
            regularisation: float = 1e-6,
    ) -> "State":
        """Fit and enable a GMM direction law from every stored classic.

        Posterior-weighted classics fit the component locations and covariance
        geometry in homogeneous U-space. Their already stored likelihoods fit
        each component's value at its mean, without calling the user's
        likelihood. Fitting is explicit and never occurs inside a
        nested-sampling depth loop.

        Args:
            num_components: Number of Gaussian components. ``None`` preserves
                an existing fit width, or uses one for the first fit. One is
                the conservative default: extra components should represent
                modes the user has scientific reason to resolve.
            num_iterations: Number of warm-started EM iterations.
            iso_prob: Probability of an exactly isotropic direction while the
                fitted GMM direction law is enabled.
            regularisation: Covariance ridge relative to component variance.

        Returns:
            A new frozen state with the fitted direction law enabled.

        Raises:
            ValueError: If configuration is invalid, the U-space is periodic,
                or the fit cannot produce full-dimensional geometry.
        """
        return _fit_gmm_directions_public(
            self,
            num_components=num_components,
            num_iterations=num_iterations,
            iso_prob=iso_prob,
            regularisation=regularisation,
        )

    def iso_directions(self) -> "State":
        """Return a state using exactly isotropic directions.

        Any fitted GMM is retained so :meth:`gmm_directions` can restore it
        without refitting.
        """
        if self.sampler_data is None:
            return self
        return dataclasses.replace(
            self,
            sampler_data=dataclasses.replace(
                self.sampler_data,
                enabled=jnp.asarray(False, mp_policy.bool_dtype),
            ),
        )

    def gmm_directions(self) -> "State":
        """Return a state using its previously fitted GMM direction law.

        Raises:
            ValueError: If no successful GMM fit is available.
        """
        if self.sampler_data is None or not bool(np.any(
                np.asarray(self.sampler_data.valid)
        )):
            raise ValueError(
                "No fitted GMM directions are available; call "
                "state.fit_gmm_directions() first."
            )
        return dataclasses.replace(
            self,
            sampler_data=dataclasses.replace(
                self.sampler_data,
                enabled=jnp.asarray(True, mp_policy.bool_dtype),
            ),
        )

    def sample_evidence_mc(
            self,
            num_samples: int,
            *,
            conditioning: Literal["classic", "phantom"],
            key: jax.Array,
            num_phantoms: int | None = None,
            batch_size: int | None = None,
            C_min: float = 20,
            diagnostics: bool = False,
    ) -> EvidenceSamples:
        """Draw final evidence samples from this immutable state.

        This is a thin state-level forwarder to the same result implementation
        used after a run. ``conditioning`` must explicitly be ``"classic"``
        or ``"phantom"``; the depth-loop expectation register is not used.

        Args:
            num_samples: Number of evidence draws.
            conditioning: Whether to use only the classic race or condition
                on retained phantom clusters.
            key: Explicit JAX random key.
            num_phantoms: Number of retained states to use from the start of
                each phantom cluster. ``None`` uses every saved state.
            batch_size: Maximum simultaneous evidence draws. ``None`` uses
                the bounded result-level default.
            C_min: Minimum participating-cluster Kish count.
            diagnostics: Whether to retain full per-draw, per-block arrays.

        Returns:
            The evidence ensemble and its block-aligned summaries.
        """
        return self.to_result().trim().sample_evidence_mc(
            num_samples=num_samples,
            conditioning=conditioning,
            key=key,
            num_phantoms=num_phantoms,
            batch_size=batch_size,
            C_min=C_min,
            diagnostics=diagnostics,
        )

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
        """Return a state with larger sample-indexed buffers.

        Control status and the continuation key are preserved. The Python
        runner, which owns the resize boundary, clears ``needs_growth`` only
        immediately before it resumes the interrupted depth epoch.

        Args:
            max_samples: New physical leading dimension.

        Returns:
            A state with every sample-indexed buffer grown consistently.

        Raises:
            ValueError: If ``max_samples`` would shrink existing storage.
        """
        current_size = self.samples.log_likelihoods.shape[0]
        if max_samples < current_size:
            raise ValueError(
                "State.resize only supports growth: "
                f"current size is {current_size}, requested {max_samples}."
            )
        if max_samples == current_size:
            return self
        return dataclasses.replace(
            self,
            samples=self.samples.resize(max_samples),
            likelihood_order=(
                None
                if self.likelihood_order is None
                else self.likelihood_order.resize(max_samples)
            ),
            scheduler_data=(
                None
                if self.scheduler_data is None
                else self.scheduler_data.resize(max_samples)
            ),
        )


State.register_pytree()


def _fit_gmm_directions_public(
        state: State,
        *,
        num_components: int | None,
        num_iterations: int,
        iso_prob: float,
        regularisation: float,
) -> State:
    """Validate one explicit fit and publish only successful geometry."""
    if num_components is None:
        num_components = (
            1
            if state.sampler_data is None
            else state.sampler_data.centres.shape[0]
        )
    try:
        num_components = operator.index(num_components)
        num_iterations = operator.index(num_iterations)
    except TypeError as error:
        raise TypeError(
            "num_components and num_iterations must be Python integers."
        ) from error
    if num_components < 1:
        raise ValueError("num_components must be positive.")
    if num_iterations < 1:
        raise ValueError("num_iterations must be positive.")
    if not 0.0 <= iso_prob <= 1.0:
        raise ValueError("iso_prob must be between zero and one.")
    if regularisation <= 0.0:
        raise ValueError("regularisation must be positive.")
    periodic = state.model._periodic_coordinates(state.args, state.params)
    if any(periodic):
        raise ValueError(
            "GMM directions do not yet support periodic U-space; keep exact "
            "isotropic directions for this state."
        )

    # Fit against a compact view so explicit fitting pays only for collected
    # scientific rows, not unused geometric-growth capacity. The returned
    # state retains its original buffers and continuation capacity.
    fit_state = state.trim()
    flat_supremum, _ = pytree_ravel(fit_state.U_supremum)
    dimension = flat_supremum.shape[0]
    if int(np.asarray(fit_state.num_samples)) < num_components * (dimension + 1):
        raise ValueError(
            "A full-covariance GMM needs at least D + 1 classics per "
            f"component; got {int(np.asarray(fit_state.num_samples))} "
            f"classics for D={dimension} and K={num_components}."
        )

    data = state.sampler_data
    if data is None or data.centres.shape[0] != num_components:
        data = empty_sampler_data(num_components, dimension)
    previous_updates = int(np.asarray(data.num_updates))
    fitted = _fit_gmm_directions(
        fit_state,
        data,
        num_iterations=num_iterations,
        iso_prob=iso_prob,
        regularisation=regularisation,
    )
    if int(np.asarray(fitted.num_updates)) == previous_updates:
        raise ValueError(
            "The GMM fit did not produce full-dimensional direction geometry."
        )
    return dataclasses.replace(state, sampler_data=fitted)


@partial(
    jax.jit,
    inline=True,
    static_argnames=(
        "num_iterations",
        "iso_prob",
        "regularisation",
    ),
)
def _fit_gmm_directions(
        state: State,
        data: SamplerData,
        *,
        num_iterations: int,
        iso_prob: float,
        regularisation: float,
) -> SamplerData:
    """Fit all compact classic rows to one evidence-scaled GMM surrogate."""
    likelihood_order = (
        state.likelihood_order
        if state.scheduler_data is None
        else None
    )
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
        likelihood_order=likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    log_weights = expected_log_posterior_weights(
        block_state,
        concentrations,
    )  # [N]
    points = jax.vmap(lambda sample: pytree_ravel(sample)[0])(
        state.samples.U_samples
    )  # [N, D]
    mask = jnp.ones((points.shape[0],), mp_policy.bool_dtype)  # [N]
    return update_sampler_data(
        jax.random.PRNGKey(0),
        data,
        points,
        state.samples.log_likelihoods,
        log_weights,
        mask,
        state.num_samples,
        n_iters=num_iterations,
        iso_prob=iso_prob,
        regularisation=regularisation,
    )


def _trim(self: State) -> State:
    num_samples = int(np.asarray(self.num_samples))
    scheduler_data = self.scheduler_data
    if scheduler_data is not None:
        if bool(np.asarray(scheduler_data.active)):
            scheduler_data = scheduler_data.resize(num_samples)
        else:
            scheduler_data = None
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
        scheduler_data=scheduler_data,
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
def _expected_evidence_scalars(
        self: State,
) -> tuple[FloatArray, FloatArray]:
    """Reduce a trusted immutable core state to its two goal-loop scalars."""
    # A completed Python goal boundary has published every accepted row into
    # likelihood_order. A single-iteration capacity return can expose an
    # active frozen schedule before that merge, so its diagnostic path must
    # reconstruct an exact order rather than silently omit committed rows.
    likelihood_order = (
        self.likelihood_order
        if self.scheduler_data is None
        else None
    )
    block_state = build_block_state(
        self.samples,
        root_out_degree=self.root_out_degree,
        num_samples=self.num_samples,
        likelihood_order=likelihood_order,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    summary = expected_evidence_summary(block_state, concentrations)
    return summary.log_Z_mean, summary.log_Z_uncert


@partial(jax.jit, inline=True)
def _total_likelihood_evaluations(self: State) -> IntArray:
    """Sum logical work without materialising a complete Results object."""
    sample_slots = jnp.arange(
        self.samples.num_likelihood_evaluations.shape[0],
        dtype=mp_policy.index_dtype,
    )
    valid = sample_slots < self.num_samples
    return jnp.sum(
        jnp.where(
            valid,
            self.samples.num_likelihood_evaluations,
            jnp.asarray(0, mp_policy.count_dtype),
        ),
        dtype=mp_policy.count_dtype,
    )


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
    evidence_summary = expected_evidence_summary(block_state, concentrations)
    log_Z_mean = evidence_summary.log_Z_mean
    log_Z_uncert = evidence_summary.log_Z_uncert
    log_dp = expected_log_posterior_weights(block_state, concentrations)
    log_dp = jnp.where(
        sample_mask,
        log_dp,
        jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
    )
    dp_mean = normalise_log_space(LogSpace(log_dp))
    # ESS describes the posterior representation, so derive it only from the
    # classic posterior weights. Retained phantom states are not posterior
    # samples and therefore cannot increase this quantity.
    ess = effective_sample_size_kish(log_dp)
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
    # Physical storage tail rows may contain ignored scheduler work or stale
    # overwritten batches. They are not classic samples and must contribute
    # neither a per-sample logical count nor the user-facing total.
    num_likelihood_evaluations_per_sample = (
        jnp.where(
            sample_mask,
            self.samples.num_likelihood_evaluations,
            jnp.asarray(0, mp_policy.count_dtype),
        ).astype(mp_policy.count_dtype)
    )
    # Phantom validity follows the same logical sample prefix; ignored static
    # lanes cannot create user-visible clusters merely because their storage
    # rows still contain a shaped phantom buffer.
    phantom_valid_mask = (
        self.samples.phantom_samples.valid_mask
        & sample_mask[:, None]
    )
    total_phantom_samples = jnp.sum(phantom_valid_mask).astype(
        mp_policy.count_dtype
    )
    total_num_likelihood_evaluations = jnp.sum(num_likelihood_evaluations_per_sample)
    log_efficiency = jnp.log(total_num_samples) - jnp.log(total_num_likelihood_evaluations)
    log_L_constraints = self.samples.log_L_constraints
    log_L_phantom = self.samples.phantom_samples.log_L
    # A cluster is one statistical unit with a shared gamma weight. Partially
    # populated rows are excluded rather than silently changing cluster size.
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
        valid_phantom=valid_phantom,
        block_data=BlockData(
            log_L=block_state.log_L_blocks,
            first_idx=block_state.block_first_idx,
            size=block_state.block_size,
            incoming_K=block_state.incoming_K,
            out_degree=block_state.block_out_degree,
            valid=block_state.valid,
            start=block_state.block_start,
            stop=block_state.block_stop,
            sample_indices=block_state.block_sample_indices,
            alpha_gt=concentrations.alpha_gt,
            alpha_eq=concentrations.alpha_eq,
            alpha_lt=concentrations.alpha_lt,
            epsilon=concentrations.epsilon,
            p_gt_mean=jnp.where(
                block_state.valid,
                block_p_gt_mean,
                jnp.nan,
            ),
            p_eq_mean=jnp.where(
                block_state.valid,
                block_p_eq_mean,
                jnp.nan,
            ),
            # Phantom count matrices are potentially large and are evaluated
            # lazily by result-level diagnostics or MC shrinkage.
        ),
    )


@partial(jax.jit, inline=True)
def _merge(self: State, other: State) -> 'State':
    assert jax.tree.structure(self.model) == jax.tree.structure(other.model), "Cannot merge states with different models"
    assert jax.tree.structure(self.args) == jax.tree.structure(other.args), "Cannot merge states with different args"
    assert jax.tree.structure(self.params) == jax.tree.structure(other.params), "Cannot merge states with different params"
    termination_reason = jnp.bitwise_or(
        self.termination_reason,
        other.termination_reason,
    )
    terminal = termination_reason != 0
    needs_growth = jnp.logical_and(
        jnp.logical_not(terminal),
        jnp.logical_or(self.needs_growth, other.needs_growth),
    )
    # A merge is a completed Python-level operation. If neither input is
    # terminal nor requesting growth, the merged continuation is at a normal
    # depth boundary rather than an ambiguous all-false control state.
    depth_reached = jnp.logical_not(terminal | needs_growth)
    if self.sampler_data is None:
        sampler_data = other.sampler_data
    elif other.sampler_data is None:
        sampler_data = self.sampler_data
    else:
        # A merged race remains valid under either symmetric direction law.
        # Continue from the more recently informed geometry, while preserving
        # cumulative work counters from both independent runs.
        use_self = (
            self.sampler_data.num_samples
            >= other.sampler_data.num_samples
        )
        sampler_data = jax.tree.map(
            lambda left, right: jnp.where(use_self, left, right),
            self.sampler_data,
            other.sampler_data,
        )
        sampler_data = dataclasses.replace(
            sampler_data,
            num_updates=(
                self.sampler_data.num_updates
                + other.sampler_data.num_updates
            ),
            num_directions=(
                self.sampler_data.num_directions
                + other.sampler_data.num_directions
            ),
            num_isotropic=(
                self.sampler_data.num_isotropic
                + other.sampler_data.num_isotropic
            ),
        )

    return State(
        root_out_degree=self.root_out_degree + other.root_out_degree,
        samples=self.samples.concat(other.samples),
        num_samples=self.num_samples + other.num_samples,
        log_L_supremum=jnp.maximum(self.log_L_supremum, other.log_L_supremum),
        U_supremum=jax.tree.map(lambda x, y: jnp.where(self.log_L_supremum >= other.log_L_supremum, x, y), self.U_supremum, other.U_supremum),
        termination_reason=termination_reason,
        model=self.model,
        args=self.args,
        params=self.params,
        goal_loop_iter=self.goal_loop_iter + other.goal_loop_iter,
        allocation_loop_iter=(
            self.allocation_loop_iter + other.allocation_loop_iter
        ),
        depth_loop_iter=self.depth_loop_iter + other.depth_loop_iter,
        # Merging changes append identities, so rebuild this optional cache on
        # first use instead of pretending either input ordering is still valid.
        likelihood_order=None,
        random_key=self.random_key,
        goal_key=self.goal_key,
        needs_growth=needs_growth,
        depth_reached=depth_reached,
        sampler_data=sampler_data,
        scheduler_data=None,
    )


@partial(jax.jit, inline=True)
def _determine_parent_graph(self: State):
    # This compatibility view chooses the likelihood-sorted nuisance ordering
    # within plateaus. Scientific state deliberately stores only contours and
    # out-degrees because concrete row identities would become stale on sort.
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
    return sample_evidence(
        key=key,
        block_state=block_state,
        concentrations=concentrations,
        num_samples=num_samples,
    ).log_Z_samples
