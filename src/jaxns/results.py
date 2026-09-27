import dataclasses
import operator
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, TextIO, TypeVar

import jax
from jax import numpy as jnp

from jaxns.algorithm.race_tree import BlockState
from jaxns.diagnostics.plotting import (
    plot_cornerplot,
    plot_diagnostics,
    plot_evidence,
)
from jaxns.diagnostics.summary import _summary
from jaxns.mixed_precision import mp_policy
from jaxns.posterior import PosteriorSamples, _integrate_posterior
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample_indicies
from jaxns.shrinkage.classic import DirichletConcentrations, PhantomCountMatrices
from jaxns.shrinkage.phantom import (
    EvidenceSamples,
    compute_phantom_count_matrices,
    sample_mc_shrinkage,
)
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType, XType

MF = TypeVar('MF')
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

    ``log_Z_mean`` and ``log_Z_uncert`` are classic expectation-based
    estimates. ``sample_evidence`` supplies Monte Carlo summaries without
    changing these fields or the represented posterior measure.
    """
    log_Z_mean: FloatArray  # [] estimate of E[log(Z)]
    log_Z_uncert: FloatArray  # [] estimate of StdDev[log(Z)]
    ess: FloatArray  # [] estimate of Kish's effective sample size
    H_mean: FloatArray  # [] estimate of E[int log(L) L dp/Z]
    total_num_samples: IntArray  # [] number of classic samples collected
    total_phantom_samples: IntArray  # [] number of phantom samples collected
    total_num_likelihood_evaluations: IntArray  # []
    log_efficiency: FloatArray  # [] log(N / likelihood evaluations)
    termination_reason: IntArray  # [] zero or a hard-stop reason code

    U_samples: UType  # [N, ...] unit-hypercube pytree leaves
    X_samples: XType  # [N, ...] transformed parameter pytree leaves
    log_L_constraints: FloatArray  # [N]
    log_L_phantom: FloatArray  # [N, P]
    valid_phantom: BoolArray  # [N]
    log_L: FloatArray  # [N]
    log_dp: FloatArray  # [N] plateau-aware log posterior weights
    log_X_mean: FloatArray  # [N]
    log_posterior_density: FloatArray  # [N]
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
    block_data: BlockData

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
            "num_likelihood_evaluations_per_sample": self.num_likelihood_evaluations_per_sample,
            "log_L_constraints": self.log_L_constraints,
            "log_L_phantom": self.log_L_phantom,
            "valid_phantom": self.valid_phantom,
        }
        sample_data = jax.tree.map(lambda s: s[:num_samples, ...], sample_data)
        return dataclasses.replace(
            self,
            **sample_data,
            block_data=self.block_data.trim(num_samples),
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

    def plot_evidence(
            self,
            *,
            num_samples: int = 512,
            conditionings: tuple[Literal["classic", "phantom"], ...] = ("classic",),
            key: PRNGKey | None = None,
            exact_log_Z: float | None = None,
            save_name: str | Path | None = None,
    ) -> None:
        """Plot Monte Carlo log-evidence ensembles.

        Args:
            num_samples: Number of shrinkage draws per conditioning mode.
            conditionings: Explicit evidence conditioning modes to compare.
            key: Optional base random key. Defaults to a fixed plotting key.
            exact_log_Z: Optional known log-evidence to mark for calibration.
            save_name: File to save the figure to. If None, shows the figure.
        """
        plot_evidence(
            self,
            num_samples=num_samples,
            conditionings=conditionings,
            key=key,
            exact_log_Z=exact_log_Z,
            save_name=save_name,
        )

    def plot_cornerplot(
            self,
            variables: list[str] | None = None,
            save_name: str | Path | None = None,
            kde_overlay: bool = False,
    ) -> None:
        """Plot posterior samples using classic expected shrinkage weights."""
        plot_cornerplot(
            self,
            variables=variables,
            save_name=save_name,
            kde_overlay=kde_overlay,
        )

    def resample(
            self,
            num_samples: int,
            *,
            key: PRNGKey,
            replace: bool = True,
    ) -> PosteriorSamples:
        """Draw equally weighted posterior samples using an explicit key.

        The returned empirical measure supports posterior integration and
        carries no evidence estimates or race-tree diagnostics. With
        ``replace=False``, samples are selected without replacement and are
        dependent rather than independent draws from the posterior measure.
        """
        num_samples = operator.index(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if not replace and num_samples > int(self.total_num_samples):
            raise ValueError("Cannot draw more samples than exist without replacement.")
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

        return _integrate_posterior(
            self.X_samples, self.log_dp, fn,
            semi_positive=semi_positive, batch_size=batch_size,
        )

    def sample_evidence(
            self,
            num_samples: int,
            *,
            phantom_conditioning: bool = False,
            key: PRNGKey,
            num_phantoms: int | None = None,
            batch_size: int | None = None,
            C_min: float = 20,
            diagnostics: bool = False,
    ) -> EvidenceSamples:
        """Draw a Monte Carlo evidence ensemble from the classic race tree.

        Args:
            num_samples: Number of shrinkage/evidence draws.
            phantom_conditioning: Opt in to conditioning on retained phantom
                clusters, subject to the Kish gate. False uses only classics.
            key: Explicit JAX PRNG key.
            num_phantoms: Number of retained states to use from the start of
                each phantom cluster. ``None`` uses every saved state. This is
                only valid with ``phantom_conditioning=True``.
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
        if type(phantom_conditioning) is not bool:
            raise TypeError("phantom_conditioning must be a bool.")
        results = self
        if not phantom_conditioning:
            # A zero-width axis keeps classic inference independent of the
            # retained phantom capacity, including its compiled workspace.
            results = dataclasses.replace(
                self,
                valid_phantom=jnp.zeros_like(self.valid_phantom),
                log_L_phantom=self.log_L_phantom[:, :0],
            )
            if num_phantoms is not None:
                raise ValueError(
                    "num_phantoms is only valid with phantom conditioning."
                )
        else:
            retained_phantoms = self.log_L_phantom.shape[1]
            if retained_phantoms == 0:
                raise ValueError(
                    "Phantom conditioning was requested, but no phantom "
                    "slots were collected."
                )
            if num_phantoms is None:
                num_phantoms = retained_phantoms
            else:
                try:
                    num_phantoms = operator.index(num_phantoms)
                except TypeError as error:
                    raise TypeError(
                        "num_phantoms must be an integer or None."
                    ) from error
                if num_phantoms <= 0:
                    raise ValueError(
                        "num_phantoms must be positive for phantom "
                        "conditioning."
                    )
                if num_phantoms > retained_phantoms:
                    raise ValueError(
                        "num_phantoms cannot exceed the retained phantom "
                        f"capacity of {retained_phantoms}."
                    )
            # This physical prefix slice changes the static P axis seen by
            # the jitted MC kernel. An unused suffix therefore contributes no
            # device work or compiler memory, rather than merely being masked.
            results = dataclasses.replace(
                self,
                log_L_phantom=self.log_L_phantom[:, :num_phantoms],
            )
        if batch_size is None:
            batch_size = min(num_samples, DEFAULT_MC_BATCH_SIZE)
        block_state = results.block_data.to_block_state()
        incoming_lineages = _incoming_lineages_per_sample(results)
        # The shared shrinkage entry point owns metadata validation.
        return sample_mc_shrinkage(
            key=key,
            log_L_constraints=results.log_L_constraints,
            log_L_classic=results.log_L,
            K_classic=incoming_lineages,
            valid_phantom=results.valid_phantom,
            log_L_phantom=results.log_L_phantom,
            num_samples=results.total_num_samples,
            num_Z_samples=num_samples,
            block_state=block_state,
            batch_size=batch_size,
            C_min=C_min,
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
        log_L_blocks = self.block_data.log_L
        block_valid_mask = self.block_data.valid
        return compute_phantom_count_matrices(
            log_L_blocks=log_L_blocks,
            block_valid_mask=block_valid_mask,
            log_L_constraints=self.log_L_constraints,
            valid_phantom=self.valid_phantom,
            log_L_phantom=self.log_L_phantom,
            sample_mask=sample_mask,
            C_min=C_min,
        )


NestedSamplerResults.register_pytree()


def _incoming_lineages_per_sample(results: NestedSamplerResults) -> IntArray:
    """Derive the shrinkage kernel's row view from the owning block counts."""
    return _expand_block_lineages(
        results.log_L, results.total_num_samples,
        results.block_data.log_L, results.block_data.incoming_K,
    )


@partial(jax.jit, inline=True)
def _expand_block_lineages(
        log_L: FloatArray,
        num_samples: IntArray,
        block_log_L: FloatArray,
        incoming_K: IntArray,
) -> IntArray:
    # Only lineage inputs enter this compilation boundary. A changed phantom
    # prefix or parameter tree must not retrace this unrelated derived view.
    # Samples stay append ordered, so match likelihoods without sorting payloads.
    block_idx = jnp.searchsorted(block_log_L, log_L, side="left")
    block_idx = jnp.clip(block_idx, 0, block_log_L.shape[0] - 1)
    valid = jnp.arange(log_L.shape[0]) < num_samples
    return jnp.where(valid, incoming_K[block_idx], 0)


@partial(jax.jit, inline=True, static_argnames=['num_samples', 'replace'])
def _resample(
        results: NestedSamplerResults,
        key: PRNGKey,
        num_samples: int,
        replace: bool,
) -> PosteriorSamples:
    indices = resample_indicies(key, results.log_dp, S=num_samples, replace=replace)
    # Retain draw order and only data belonging to the empirical posterior.
    # Run-level uncertainty and lineage metadata cannot be resampled this way.
    return PosteriorSamples(
        U_samples=jax.tree.map(lambda x: x[indices], results.U_samples),
        X_samples=jax.tree.map(lambda x: x[indices], results.X_samples),
        log_L=results.log_L[indices],
    )
