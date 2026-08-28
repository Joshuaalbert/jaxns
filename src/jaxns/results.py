import dataclasses
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, TextIO, TypeVar

import jax
from jax import numpy as jnp

from jaxns.cumulative_ops import batch_reduce
from jaxns.diagnostics.plotting import (
    plot_cornerplot,
    plot_diagnostics,
)
from jaxns.diagnostics.summary import _summary
from jaxns.log_semiring import LogSpace
from jaxns.mixed_precision import mp_policy
from jaxns.phantom_eval import (
    EvidenceSamples,
    compute_phantom_count_matrices,
    sample_mc_shrinkage,
    validate_sample_mc_shrinkage_inputs,
)
from jaxns.pytree import PureDataclassPytree
from jaxns.algorithm.race_tree import BlockState
from jaxns.random_utils import resample_indicies
from jaxns.shrinkage import DirichletConcentrations, PhantomCountMatrices
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType, XType

MF = TypeVar('MF')
EvidenceConditioning = Literal["classic", "phantom"]
DEFAULT_MC_BATCH_SIZE = 64


@dataclasses.dataclass(slots=True, frozen=True)
class BlockData(PureDataclassPytree):
    """Block-aligned race-tree data and shrinkage diagnostics.

    Keeping this secondary schema nested prevents implementation-facing
    block buffers from obscuring the sample, evidence, and posterior fields
    that scientific users access most often.
    """

    log_L: FloatArray  # [G]
    first_idx: IntArray  # [G]
    size: IntArray  # [G]
    incoming_K: IntArray  # [G]
    out_degree: IntArray  # [G]
    valid: BoolArray  # [G]
    start: IntArray | None = None  # [G]
    stop: IntArray | None = None  # [G]
    sample_indices: IntArray | None = None  # [N] or [G, M_g^max]
    alpha_gt: FloatArray | None = None  # [G]
    alpha_eq: FloatArray | None = None  # [G]
    alpha_lt: FloatArray | None = None  # [G]
    epsilon: FloatArray | None = None  # [G]
    p_gt_mean: FloatArray | None = None  # [G]
    p_eq_mean: FloatArray | None = None  # [G]
    phantom_A: FloatArray | None = None  # [G]
    phantom_B: FloatArray | None = None  # [G]
    phantom_E: FloatArray | None = None  # [G]
    phantom_R: FloatArray | None = None  # [G]
    kish_participating_cluster_counts: FloatArray | None = None  # [G]
    phantom_gate_active: BoolArray | None = None  # [G]

    @property
    def m_g(self) -> IntArray:
        """Block multiplicities."""
        return self.size

    @property
    def K_g(self) -> IntArray:
        """Incoming active lineage counts."""
        return self.incoming_K

    @property
    def L(self) -> FloatArray:
        """Likelihood-scale block levels."""
        return jnp.exp(self.log_L)

    @property
    def A_g(self) -> FloatArray | None:
        return self.phantom_A

    @property
    def B_g(self) -> FloatArray | None:
        return self.phantom_B

    @property
    def E_g(self) -> FloatArray | None:
        return self.phantom_E

    @property
    def R_g(self) -> FloatArray | None:
        return self.phantom_R

    @property
    def concentrations(self) -> DirichletConcentrations | None:
        """Classic block Dirichlet concentrations."""
        if self.alpha_gt is None:
            return None
        return DirichletConcentrations(
            alpha_gt=self.alpha_gt,
            alpha_eq=self.alpha_eq,
            alpha_lt=self.alpha_lt,
            epsilon=self.epsilon,
        )

    def to_block_state(self) -> BlockState:
        """Return the scheduler/shrinkage view of these blocks."""
        return BlockState(
            log_L_blocks=self.log_L,
            block_first_idx=self.first_idx,
            block_size=self.size,
            incoming_K=self.incoming_K,
            block_out_degree=self.out_degree,
            valid=self.valid,
            block_start=self.start,
            block_stop=self.stop,
            block_sample_indices=self.sample_indices,
        )

    def trim(self, size: int) -> "BlockData":
        """Trim padded block buffers along their leading axis."""
        return jax.tree.map(lambda value: value[:size, ...], self)


BlockData.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class NestedSamplerResults(PureDataclassPytree):
    """
    Results of the nested sampling run.
    """
    log_Z_mean: FloatArray  # [] estimate of E[log(Z)]
    log_Z_uncert: FloatArray  # [] estimate of StdDev[log(Z)]
    ess: FloatArray  # [] estimate of Kish's effective sample size
    H_mean: FloatArray  # [] estimate of E[int log(L) L dp/Z]
    total_num_samples: IntArray  # [] number of classic samples collected
    total_phantom_samples: IntArray  # [] number of phantom samples collected
    total_num_likelihood_evaluations: IntArray  # []
    log_efficiency: FloatArray  # [] log(N / likelihood evaluations)
    termination_reason: IntArray  # [] bit mask

    U_samples: UType  # [N, ...] unit-hypercube pytree leaves
    X_samples: XType  # [N, ...] transformed parameter pytree leaves
    log_L_constraints: FloatArray  # [N]
    log_L_phantom: FloatArray  # [N, P]
    valid_phantom: BoolArray  # [N]
    log_L: FloatArray  # [N]
    log_dp: FloatArray  # [N] plateau-aware log posterior weights
    log_X_mean: FloatArray  # [N]
    log_posterior_density: FloatArray  # [N]
    num_live_points_per_sample: IntArray  # [N]
    num_likelihood_evaluations_per_sample: IntArray  # [N]

    # Pointwise estimates.
    # max(L)
    log_L_supremum: FloatArray  # [] max(log L)
    U_supremum: UType  # [...] unit-hypercube pytree point at max(log L)
    X_supremum: XType  # [...] parameter pytree point at max(log L)
    # max(L p)
    log_L_map: FloatArray  # [] log likelihood at the sampled MAP point
    U_map: UType  # [...] unit-hypercube pytree MAP point
    X_map: XType  # [...] parameter pytree MAP point
    block_data: BlockData | None = None

    @property
    def expected_log_Z_mean(self) -> FloatArray:
        """Deterministic block-moment estimate used for lightweight output."""
        return self.log_Z_mean

    @property
    def expected_log_Z_uncert(self) -> FloatArray:
        """Deterministic block-moment uncertainty approximation."""
        return self.log_Z_uncert

    def trim(self) -> 'NestedSamplerResults':
        num_samples = int(self.total_num_samples)
        initial_size = self.log_L.shape[0]
        if num_samples > initial_size:
            raise ValueError(
                f"num_samples ({num_samples}) is greater than the number of samples collected ({initial_size}). You probably set max_samples too low.")
        sample_data = {
            "U_samples": self.U_samples,
            "X_samples": self.X_samples,
            "log_L": self.log_L,
            "log_dp": self.log_dp,
            "log_X_mean": self.log_X_mean,
            "log_posterior_density": self.log_posterior_density,
            "num_live_points_per_sample": self.num_live_points_per_sample,
            "num_likelihood_evaluations_per_sample": self.num_likelihood_evaluations_per_sample,
            "log_L_constraints": self.log_L_constraints,
            "log_L_phantom": self.log_L_phantom,
            "valid_phantom": self.valid_phantom,
        }
        sample_data = jax.tree.map(lambda s: s[:num_samples, ...], sample_data)
        return dataclasses.replace(
            self,
            **sample_data,
            block_data=(
                None
                if self.block_data is None
                else self.block_data.trim(num_samples)
            ),
        )

    def summary(self, f_obj: str | TextIO | Path | None = None):
        """
        Gives a summary of the results of a nested sampling run.

        Args:
            f_obj: file-like object to write summary to. If None, prints to stdout.
        """
        return _summary(self, f_obj=f_obj)

    def plot_diagnostics(self, save_file: str | Path | None = None):
        """
        Plot diagnostics of the nested sampling run.

        Args:
            save_file: file to save figure to. If None, shows the figure.
        """
        plot_diagnostics(self, save_file=save_file)

    def plot_cornerplot(self, variables: list[str] | None = None, save_name: str | Path | None = None, kde_overlay: bool = False):
        """
        Plots a cornerplot of the posterior samples.
        """
        plot_cornerplot(self, variables=variables, save_name=save_name, kde_overlay=kde_overlay)

    def resample(self, num_samples: int, replace: bool = True, key: PRNGKey | None = None) -> 'NestedSamplerResults':
        """
        Resamples the nested sampling results according to the posterior weights.

        Args:
            key: PRNGKey for resampling
            num_samples: number of samples to resample
            replace: whether to sample with replacement or not

        Returns:
            a new NestedSamplerResults object with resampled samples.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        return _resample(self, key, num_samples, replace)

    def integrate_fn_over_posterior(self, fn: Callable[[XType], MF], *, semi_positive: bool = False, batch_size: int | None = None) -> MF:
        """
        Computes the marginalised value of a function over the samples in X space, using the posterior weights.
        This can be used to compute posterior expectations of functions of the parameters, e.g. posterior mean, variance, or more complicated functions.

        Args:
            fn: function to integrate, should take in an XType and return a value that can be averaged over samples.
            semi_positive: set True iff the function is known to be semi-positive, i.e. fn(X) >= 0 for all X.
            batch_size: optional, how many samples to process in a batch when applying the function.

        Returns:
            pytree output of the function, averaged over the posterior distribution represented by the samples.
        """

        return _integrate_fn_over_posterior(self, fn, semi_positive=semi_positive, batch_size=batch_size)

    def sample_evidence(self, num_samples: int, batch_size: int | None = None, key: PRNGKey | None = None) -> FloatArray:
        """
        Sample the evidence using the shrinkage method.

        Args:
            num_samples: number of evidence samples to draw
            batch_size: Maximum simultaneous evidence draws. ``None`` uses
                the same bounded 64-draw default as ``sample_evidence_mc``.
            key: optional, PRNGKey for resampling

        Returns:
            array of shape [num_samples] containing samples of the evidence.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        return _sample_evidence(self, num_samples=num_samples, batch_size=batch_size, key=key)

    def sample_mc_shrinkage(
            self,
            num_samples: int,
            batch_size: int | None = None,
            key: PRNGKey | None = None,
            C_min: float = 20,
            conditioning: Literal["auto", "classic", "phantom"] = "auto",
            diagnostics: bool = True,
    ) -> EvidenceSamples:
        """
        Sample the evidence using the MC shrinkage method.

        Args:
            num_samples: number of evidence samples to draw
            batch_size: optional, how many samples to process in a batch when applying the function.
            key: optional, PRNGKey for resampling
            C_min: Minimum participating-cluster Kish count.
            conditioning: Whether to use retained phantom clusters.
            diagnostics: Whether to retain full per-draw, per-block arrays.
                These arrays have shape ``[num_samples, num_blocks]``.

        Returns:
            EvidenceSamples object containing samples of the evidence and related statistics.
        """
        if conditioning not in ("auto", "classic", "phantom"):
            raise ValueError(
                "conditioning must be 'auto', 'classic', or 'phantom'."
            )
        if key is None:
            key = jax.random.PRNGKey(42)
        if conditioning == "classic":
            results = dataclasses.replace(
                self,
                valid_phantom=jnp.zeros_like(self.valid_phantom),
                log_L_phantom=self.log_L_phantom[:, :0],
            )
        else:
            results = self
        block_state = _block_state_from_results(self)
        validate_sample_mc_shrinkage_inputs(
            log_L_constraints=results.log_L_constraints,
            log_L_classic=results.log_L,
            K_classic=results.num_live_points_per_sample,
            valid_phantom=results.valid_phantom,
            log_L_phantom=results.log_L_phantom,
            num_samples=results.total_num_samples,
            block_state=block_state,
        )
        if block_state is not None:
            return _sample_mc_shrinkage_with_block_state(
                results=results,
                block_state=block_state,
                num_samples=num_samples,
                batch_size=batch_size,
                key=key,
                C_min=C_min,
                diagnostics=diagnostics,
            )
        return _sample_mc_shrinkage(
            results,
            num_samples=num_samples,
            batch_size=batch_size,
            key=key,
            C_min=C_min,
            diagnostics=diagnostics,
        )

    def sample_evidence_mc(
            self,
            num_samples: int,
            *,
            conditioning: EvidenceConditioning,
            key: PRNGKey,
            batch_size: int | None = None,
            C_min: float = 20,
            diagnostics: bool = False,
    ) -> EvidenceSamples:
        """Draw the authoritative final evidence ensemble.

        Args:
            num_samples: Number of shrinkage/evidence draws.
            conditioning: ``"classic"`` ignores retained phantoms;
                ``"phantom"`` uses retained clusters subject to the Kish gate.
            key: Explicit JAX PRNG key.
            batch_size: Maximum number of draws evaluated at once. ``None``
                uses an automatically bounded batch of at most 64 draws.
            C_min: Minimum participating-cluster Kish count for conditioning.
            diagnostics: Whether to retain full ``[num_samples, num_blocks]``
                probability and phantom-addition arrays. Defaults to the
                economical evidence-summary path.

        Returns:
            Evidence draws whose ``log_Z_mean`` and ``log_Z_uncert``
            properties are the final Monte Carlo evidence summary.
        """
        if conditioning not in ("classic", "phantom"):
            raise ValueError(
                "conditioning must be explicitly 'classic' or 'phantom'."
            )
        if conditioning == "phantom" and self.log_L_phantom.shape[1] == 0:
            raise ValueError(
                "Phantom conditioning was requested, but no phantom slots "
                "were collected."
            )
        if batch_size is None:
            batch_size = min(num_samples, DEFAULT_MC_BATCH_SIZE)
        return self.sample_mc_shrinkage(
            num_samples=num_samples,
            batch_size=batch_size,
            key=key,
            C_min=C_min,
            conditioning=conditioning,
            diagnostics=diagnostics,
        )

    def phantom_conditioning_diagnostics(
            self,
            C_min: float = 20,
    ) -> PhantomCountMatrices:
        """Return block-aligned gamma phantom-conditioning diagnostics."""
        num_samples = self.total_num_samples.astype(mp_policy.count_dtype)
        sample_mask = (
            jnp.arange(self.log_L.shape[0], dtype=mp_policy.count_dtype)
            < num_samples
        )
        block_state = _block_state_from_results(self)
        if block_state is None:
            log_L_blocks = jnp.unique(
                self.log_L,
                size=self.log_L.shape[0],
                fill_value=jnp.inf,
            )
            block_valid_mask = jnp.isfinite(log_L_blocks)
        else:
            log_L_blocks = block_state.log_L_blocks
            block_valid_mask = block_state.valid
        return compute_phantom_count_matrices(
            log_L_blocks=log_L_blocks,
            block_valid_mask=block_valid_mask,
            log_L_constraints=self.log_L_constraints,
            valid_phantom=self.valid_phantom,
            log_L_phantom=self.log_L_phantom,
            sample_mask=sample_mask,
            C_min=C_min,
        )

    def ess_with_phantom(self, num_samples: int = 512, batch_size: int | None = None, key: PRNGKey | None = None) -> FloatArray:
        """
        Compute the ESS including phantoms.

        Args:
            num_samples: number of samples to draw
            batch_size: optional, how many samples to process in a batch when applying the function.
            key: optional, PRNGKey for resampling

        Returns:
            scalar, the ESS including phantoms.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        return _ess_with_phantom(self, num_samples=num_samples, batch_size=batch_size, key=key)


NestedSamplerResults.register_pytree()


def _block_state_from_results(self: NestedSamplerResults) -> BlockState | None:
    if self.block_data is None:
        return None
    return self.block_data.to_block_state()


@partial(jax.jit, inline=True, static_argnames=['num_samples', 'batch_size'])
def _ess_with_phantom(self: NestedSamplerResults, num_samples: int, batch_size: int | None = None, key: PRNGKey | None = None) -> FloatArray:
    evidence_samples = self.sample_mc_shrinkage(num_samples=num_samples, batch_size=batch_size, key=key)
    # make sure finite mask is same for both numerator and denominator
    finite_mask = jnp.isfinite(evidence_samples.H_samples) & jnp.isfinite(evidence_samples.log_Z_samples)
    H_mean = jnp.nanmean(jnp.where(finite_mask, evidence_samples.H_samples, jnp.nan))
    log_Z_var = jnp.nanvar(jnp.where(finite_mask, evidence_samples.log_Z_samples, jnp.nan))
    return H_mean / log_Z_var


def _posterior_log_weights(self: NestedSamplerResults) -> FloatArray:
    return self.log_dp


@partial(jax.jit, inline=True, static_argnames=['num_samples', 'replace'])
def _resample(self: NestedSamplerResults, key: PRNGKey, num_samples: int, replace: bool = True) -> NestedSamplerResults:
    log_weights = _posterior_log_weights(self)
    idx = resample_indicies(key, log_weights, S=num_samples, replace=replace)
    # after resampling, the weights are uniform, so log_dp = log(1/N) = -log(N)
    sample_data = {
        "U_samples": self.U_samples,
        "X_samples": self.X_samples,
        "log_L": self.log_L,
        "log_dp": jnp.full(
            self.log_dp.shape,
            -jnp.log(num_samples),
            mp_policy.measure_dtype,
        ),
        "log_X_mean": self.log_X_mean,
        "log_posterior_density": self.log_posterior_density,
        "num_live_points_per_sample": self.num_live_points_per_sample,
        "num_likelihood_evaluations_per_sample": self.num_likelihood_evaluations_per_sample,
        "log_L_constraints": self.log_L_constraints,
        "log_L_phantom": self.log_L_phantom,
        "valid_phantom": self.valid_phantom,
    }
    sample_data = jax.tree.map(lambda s: s[idx, ...], sample_data)
    sort_idxs = jnp.argsort(sample_data['log_L'])
    sample_data = jax.tree.map(lambda s: s[sort_idxs, ...], sample_data)
    return dataclasses.replace(
        self,
        **sample_data,
        total_num_samples=jnp.asarray(num_samples, dtype=mp_policy.count_dtype),
        block_data=None,
    )

@partial(jax.jit, inline=True, static_argnames=['fn', 'semi_positive', 'batch_size'])
def _integrate_fn_over_posterior(self: NestedSamplerResults, fn: Callable[[XType], MF], *, semi_positive: bool = False, batch_size: int | None = None) -> MF:
    def kernel(x):
        weight, X = x
        Y = fn(X)

        def _increment(y):
            if semi_positive:
                f = LogSpace(y)
            else:
                f = LogSpace.from_signed_value(y)
            return (weight * f).value

        return jax.tree.map(_increment, Y)

    weights = LogSpace(_posterior_log_weights(self))
    return batch_reduce(kernel, xs=(weights, self.X_samples), reduce_fn=jnp.sum, batch_size=batch_size, vectorised_kernel=False)


def _sample_evidence(self: NestedSamplerResults,
                     num_samples: int = 100, batch_size: int | None = None, key: PRNGKey | None = None) -> FloatArray:
    if batch_size is None:
        batch_size = min(num_samples, DEFAULT_MC_BATCH_SIZE)
    evidence_samples = sample_mc_shrinkage(
        key=key,
        log_L_constraints=self.log_L_constraints,
        log_L_classic=self.log_L,
        K_classic=self.num_live_points_per_sample,
        valid_phantom=self.valid_phantom,
        log_L_phantom=self.log_L_phantom,
        num_samples=self.total_num_samples,
        num_Z_samples=num_samples,
        block_state=_block_state_from_results(self),
        batch_size=batch_size,
        diagnostics=False,
    )
    return evidence_samples.log_Z_samples


def _sample_mc_shrinkage(self: NestedSamplerResults,
                         num_samples: int = 100, batch_size: int | None = None,
                         key: PRNGKey | None = None,
                         C_min: float = 20,
                         diagnostics: bool = True) -> EvidenceSamples:
    evidence_samples = sample_mc_shrinkage(
        key=key,
        log_L_constraints=self.log_L_constraints,
        log_L_classic=self.log_L,
        K_classic=self.num_live_points_per_sample,
        valid_phantom=self.valid_phantom,
        log_L_phantom=self.log_L_phantom,
        num_samples=self.total_num_samples,
        num_Z_samples=num_samples,
        block_state=_block_state_from_results(self),
        batch_size=batch_size,
        C_min=C_min,
        diagnostics=diagnostics,
    )
    return evidence_samples


def _sample_mc_shrinkage_with_block_state(
        *,
        results: NestedSamplerResults,
        block_state: BlockState,
        num_samples: int = 100,
        batch_size: int | None = None,
        key: PRNGKey,
        C_min: float = 20,
        diagnostics: bool = True,
) -> EvidenceSamples:
    return sample_mc_shrinkage(
        key=key,
        log_L_constraints=results.log_L_constraints,
        log_L_classic=results.log_L,
        K_classic=results.num_live_points_per_sample,
        valid_phantom=results.valid_phantom,
        log_L_phantom=results.log_L_phantom,
        num_samples=results.total_num_samples,
        num_Z_samples=num_samples,
        block_state=block_state,
        batch_size=batch_size,
        C_min=C_min,
        diagnostics=diagnostics,
    )
