import dataclasses
import io
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, TextIO, TypeVar

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx.context import CtxParams
from scipy.stats import gaussian_kde

from jaxns.cumulative_ops import batch_reduce
from jaxns.log_semiring import LogSpace, cumulative_logsumexp, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.optional import import_matplotlib
from jaxns.phantom_eval import (
    EvidenceSamples,
    compute_phantom_count_matrices,
    sample_mc_shrinkage,
    validate_sample_mc_shrinkage_inputs,
)
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import BlockState
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


def _bit_mask(int_mask, width=8):
    """
    Convert an integer mask into a bit-mask. I.e. convert an integer into list of left-starting bits.

    Examples:

    1 -> [1,0,0,0,0,0,0,0]
    2 -> [0,1,0,0,0,0,0,0]
    3 -> [1,1,0,0,0,0,0,0]

    Args:
        int_mask: int
        width: number of output bits

    Returns:
        List of bits from left
    """
    return list(map(int, '{:0{size}b}'.format(int_mask, size=width)))[::-1]


def _summary(results: NestedSamplerResults, f_obj: str | TextIO | None = None):
    """
    Gives a summary of the results of a nested sampling run.

    Args:
        results (NestedSamplerResults): Nested sampler result
        f_obj: file-like object to write summary to. If None, prints to stdout.
    """

    main_s = []

    def _print(s):
        if f_obj is None:
            # It goes to file instead
            print(s)
        main_s.append(s)

    def _round(v, uncert_v):
        v = float(v)
        uncert_v = float(uncert_v)
        try:
            sig_figs = -int(f"{uncert_v:e}".split('e')[1]) + 1
            return round(float(v), sig_figs)
        except (OverflowError, ValueError):
            return float(v)

    def _print_termination_reason(_termination_reason: int):
        termination_bit_mask = _bit_mask(int(_termination_reason), width=13)

        for bit, condition in zip(termination_bit_mask, [
            'Reached max samples',
            'Evidence uncertainty low enough',
            'Small remaining evidence',
            'Reached ESS',
            "Used max num likelihood evaluations",
            'Likelihood supremum reached',
            'Likelihood contour reached',
            'Sampler efficiency too low',
            'All live-points are on a single plateau (sign of possible precision error)',
            'relative spread of live points < rtol',
            'absolute spread of live points < atol',
            'no seed points left (consider decreasing shell_fraction)',
            'XL < max(XL) * peak_XL_frac'
        ]):
            if bit == 1:
                _print(condition)

    _print("--------")
    _print("Termination Conditions:")
    if np.size(results.termination_reason) > 1:  # Reasons for each parallel sampler
        print(results.termination_reason)
        for sampler_idx in range(np.size(results.termination_reason)):
            _print(f"Sampler {sampler_idx}:")
            _print_termination_reason(int(results.termination_reason[sampler_idx]))
    else:
        _print_termination_reason(int(results.termination_reason))
    _print("--------")
    _print(f"likelihood evals: {int(results.total_num_likelihood_evaluations):d}")
    _print(f"classic samples: {int(results.total_num_samples):d}")
    _print(f"phantom samples: {int(results.total_phantom_samples):d}")
    _print(
        f"likelihood evals / sample: {float(results.total_num_likelihood_evaluations / results.total_num_samples):.1f}"
    )
    _print("--------")
    _print(
        f"logZ (classic)={_round(results.log_Z_mean, results.log_Z_uncert)} +- {_round(results.log_Z_uncert, results.log_Z_uncert)}"
    )
    if results.total_phantom_samples > 0:
        log_Z_samples = results.sample_evidence(num_samples=512)
        _log_Z_samples_mean = jnp.nanmean(log_Z_samples)
        _log_Z_samples_uncert = jnp.nanstd(log_Z_samples)
        _print(
            f"logZ (with phantom)={_round(_log_Z_samples_mean, _log_Z_samples_uncert)} +- {_round(_log_Z_samples_uncert, _log_Z_samples_uncert)}"
        )
    _print(
        f"max(logL)={_round(results.log_L_supremum, results.log_Z_uncert)}"
    )
    # _print("H={} +- {}".format(
    #     _round(results.H_mean, results.H_uncert), _round(results.H_uncert, results.H_uncert)))
    _print(
        f"H={_round(results.H_mean, 0.1)}"
    )
    _print(
        f"effective sample size (classic)={results.ess:.1f}"
    )
    if results.total_phantom_samples > 0:
        ess_with_phantom = results.ess_with_phantom()
        _print(
            f"effective sample size (with phantom)={ess_with_phantom:.1f}"
        )
    _print(
        f"likelihood evals / ess(classic): {float(results.total_num_likelihood_evaluations / results.ess):.1f}"
    )
    if results.total_phantom_samples > 0:
        _print(
            f"likelihood evals / ess(with phantom): {float(results.total_num_likelihood_evaluations / ess_with_phantom):.1f}"
        )

    def moments(x):
        x2 = jax.tree.map(jnp.square, x)
        return x, x2

    x_mean, x2_mean = jax.tree.map(np.asarray, results.integrate_fn_over_posterior(moments))
    x_std: CtxParams = jax.tree.map(lambda m, m2: np.sqrt(np.maximum(0., m2 - np.square(m))), x_mean, x2_mean)

    for name, _ in x_std.iter_items():
        _x_mean = x_mean.get_dotted(name).reshape((-1,))
        _x_std = x_std.get_dotted(name).reshape((-1,))
        _x_map = results.X_map.get_dotted(name).reshape((-1,))
        _x_ml = results.X_supremum.get_dotted(name).reshape((-1,))
        ndims = _x_mean.shape[0]
        _print("--------")
        var_name = name if ndims == 1 else f"{name}[#]"
        _print(
            f"{var_name}: mean +- std.dev. | MAP est. | max(L) est."
        )
        for dim in range(ndims):
            _uncert = _x_std[dim]
            # two sig-figs based on uncert
            sig_figs = 1 - int(f"{_uncert:e}".split('e')[1])

            def _round(ar, digits=sig_figs):
                return round(float(ar), digits)

            _print("{}: {} +- {} | {} | {}".format(
                name if ndims == 1 else f"{name}[{dim}]",
                _round(_x_mean[dim]), _round(_uncert),
                _round(_x_map[dim]),
                _round(_x_ml[dim])
            ))
    _print("--------")
    if f_obj is not None:
        out = "\n".join(main_s)
        if isinstance(f_obj, (str, Path)):
            with open(f_obj, 'w') as f:
                f.write(out)
        elif isinstance(f_obj, io.TextIOBase):
            f_obj.write(out)
        else:
            raise TypeError(f"Invalid f_obj: {type(f_obj)}")


def plot_diagnostics(results: NestedSamplerResults, save_file=None):
    """
    Plot diagnostics of the nested sampling run.

    Args:
        results: NestedSamplerResult
        save_file: file to save figure to.
    """

    # Plotting is optional so importing the scientific result type does not
    # force non-plotting users to install a GUI-oriented dependency stack.
    plt = import_matplotlib()
    num_samples = int(results.total_num_samples)
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(8, 15))
    log_X = np.asarray(results.log_X_mean[:num_samples])
    num_live_points_per_sample = np.asarray(results.num_live_points_per_sample[:num_samples])
    log_L = np.asarray(results.log_L[:num_samples])
    max_log_likelihood = np.max(log_L)
    log_dp = np.asarray(_posterior_log_weights(results)[:num_samples])
    log_cum_evidence = cumulative_logsumexp(log_dp)
    cum_evidence = np.exp(log_cum_evidence)
    log_Z_mean = np.asarray(results.log_Z_mean)
    num_likelihood_evaluations_per_sample = np.asarray(results.num_likelihood_evaluations_per_sample[:num_samples])
    mean_efficiency = np.exp(results.log_efficiency)
    if np.any(num_likelihood_evaluations_per_sample == 0):
        warnings.warn("Found samples with zero likelihood evaluations.")
        efficiency = np.reciprocal(
            np.where(
                num_likelihood_evaluations_per_sample == 0,
                np.nan,
                num_likelihood_evaluations_per_sample
            )
        )
    else:
        efficiency = 1. / num_likelihood_evaluations_per_sample

    # Plot the number of live points
    axs[0].plot(-log_X, num_live_points_per_sample, c='black')
    axs[0].set_ylabel(r'$n_{\rm live}$')
    # detect if too small log likelihood
    rel_log_L = log_L - max_log_likelihood
    axs[1].plot(-log_X, np.exp(rel_log_L), c='black')
    axs[1].axhline(1., color='black', ls='dashed',
                   label=rf"$\log L_{{\rm max}}={max_log_likelihood:.1f}$")
    axs[1].set_ylabel(r'$L/L_{\rm max}$')
    axs[1].legend()
    axs[2].plot(-log_X, np.exp(log_dp), c='black')
    axs[2].axvline(results.H_mean, color='black', ls='dashed',
                   label=rf'$H={results.H_mean:.1f}$')
    axs[2].set_ylabel(r'$Z^{-1}L dX$')
    axs[2].legend()
    axs[3].plot(-log_X, cum_evidence, c='black')
    axs[3].axhline(1., color='black', ls='dashed',
                   label=rf"$\log Z={log_Z_mean:.1f}$")
    axs[3].set_ylabel(r'$Z(\lambda > L)/Z$')
    axs[3].legend()
    axs[4].scatter(-log_X, efficiency, s=2, c='black')

    axs[4].axhline(mean_efficiency, color='black', ls='dashed',
                   label=f'avg. eff.={mean_efficiency:.3f}')
    axs[4].set_ylabel("sampler efficiency")
    axs[4].set_ylim(0., 1.05)
    axs[4].legend()

    # Plot X*L vs -log(X)
    XL = (LogSpace(log_X) * LogSpace(log_L)).value
    axs[5].plot(-log_X, XL, c='black')
    axs[5].set_ylabel(r'$X L$')

    axs[5].set_xlabel(r'$- \log X$')
    if save_file is not None:
        fig.savefig(save_file, bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.close(fig)
    else:
        plt.show()


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


def _tuple_prod(t):
    """
    Product of shape tuple

    Args:
        t: tuple

    Returns:
        int
    """
    if len(t) == 0:
        return 1
    res = t[0]
    for a in t[1:]:
        res *= a
    return res


def plot_cornerplot(results: NestedSamplerResults, variables: list[str] | None = None, save_name: str | None = None, kde_overlay: bool = False):
    """
    Plots a cornerplot of the posterior samples.

    Args:
        results: NestedSamplerResult
        variables: list of variable names to plot. Plots all collected samples by default.
        save_name: file to save result to.
        kde_overlay: whether to overlay a KDE on the histograms.
    """
    # Keep the optional dependency behind the operation that needs it and
    # provide one consistent installation remedy when it is unavailable.
    plt = import_matplotlib()
    num_samples = int(results.total_num_samples)
    sample_items = list(results.X_samples.iter_items())
    sample_map = dict(sample_items)
    for name, sample in sample_items:
        sample_map[name] = np.asarray(sample[:num_samples])

    if variables is None:
        variables = list(sample_map.keys())
    unknown_variables = [name for name in variables if name not in sample_map]
    if len(unknown_variables) > 0:
        raise ValueError(f"Unknown variables requested: {unknown_variables}")

    ndims = sum([_tuple_prod(sample_map[key].shape[1:]) for key in variables], 0)

    for key in variables:
        if sample_map[key].shape[0] != num_samples:
            raise ValueError(f"Expected all samples to have the same number of samples, "
                             f"got {key} with {sample_map[key].shape[0]} samples, "
                             f"expected {num_samples} samples.")

    # Get the leaves of the tree, and concatenate into [num_samples, ndims] shape
    leaves = np.concatenate(
        [sample_map[key].reshape((num_samples, -1)) for key in variables],
        axis=-1
    )

    # Create a parameter for each column. For scalar parameters, we just use the name of the parameter.
    # For vector we use name[i,j,...] etc.
    parameters = []
    for key in variables:
        shape = sample_map[key].shape[1:]
        if _tuple_prod(shape) == 1:
            parameters.append(key)
        else:
            # Loop over each dimension of the parameter, and create a parameter for each index
            for i in range(_tuple_prod(shape)):
                indices = np.unravel_index(i, shape)
                parameters.append(f"{key}[{','.join([str(j) for j in indices])}]")

    # Get the maximum likelihood and MAP samples
    log_L_samples = np.asarray(results.log_L[:num_samples])
    log_posterior_density = np.asarray(results.log_posterior_density[:num_samples])
    max_like_idx = np.argmax(log_L_samples)
    map_idx = np.argmax(log_posterior_density)
    max_like_sample = leaves[max_like_idx]
    map_sample = leaves[map_idx]

    # Get the weight of each sample
    log_weights = np.asarray(
        normalise_log_space(
            LogSpace(_posterior_log_weights(results)[:num_samples]),
            norm_type='max',
        ).log_abs_val)

    figsize = min(20, max(4, int(2 * ndims)))
    fig, axs = plt.subplots(ndims, ndims, figsize=(figsize, figsize), squeeze=False)

    # Get the number of bins for the histograms based on the effective sample size
    if np.isnan(np.asarray(results.ess)):
        nbins = 10
    else:
        nbins = max(10, int(jnp.sqrt(results.ess)))

    # Loop over the variables, and plot the marginal distributions on the diagonal setting a title above
    # each plot with the mean+-stddev, 5%/50%/95%, and MAP
    param_limits = {}  # Store the 1_per and 99_per for each parameter
    for row in range(ndims):
        for col in range(ndims):
            ax = axs[row][col]
            if row != col:  # i == j ==> plot the marginal distribution
                continue
            # Plot the marginal distribution
            _samples = leaves[:, row]  # [num_samples]
            _parameter = parameters[row]
            _log_weights = log_weights
            is_finite = np.isfinite(_samples)
            if np.bitwise_not(np.all(is_finite)):
                warnings.warn(f"Found {np.sum(np.bitwise_not(is_finite))} non-finite samples for {_parameter}")
                _samples = _samples[is_finite]
                _log_weights = _log_weights[is_finite]
            _weights = np.exp(_log_weights)
            # Percentiles
            per_1, per_5, per_50, per_95, per_99 = _weighted_percentile(_samples, _log_weights,
                                                                        [1, 5, 50, 95, 99])
            # Plot the histogram, from 1_per to 99_per
            ax.hist(_samples, bins=nbins, fc='None', edgecolor='black', density=True, weights=_weights,
                    range=(per_1, per_99))
            # Plot the maximum likelihood and MAP samples
            ax.axvline(max_like_sample[row], color='green')
            ax.axvline(map_sample[row], color='red')
            # Plot the mean and standard deviation
            sample_mean = np.average(_samples, weights=_weights)
            sample_std = np.sqrt(np.average((_samples - sample_mean) ** 2, weights=_weights))
            ax.axvline(sample_mean, linestyle='dashed', color='red')
            ax.axvline(sample_mean + sample_std, linestyle='dotted', color='red')
            ax.axvline(sample_mean - sample_std, linestyle='dotted', color='red')

            # Set the title
            title = [
                rf"${per_50:.2f}_{{{per_5:.2f}}}^{{{per_95:.2f}}}$",
                rf"${sample_mean:.2f}\pm{sample_std:.2f}$",
                rf"MAP ${map_sample[row]:.2f}$ | ML ${max_like_sample[row]:.2f}$"
            ]
            ax.set_title("\n".join(title))
            # Set the limits to 1 to 99 percentiles
            ax.set_xlim(per_1, per_99)
            param_limits[_parameter] = (per_1, per_99)

    # Plot the 2D histograms on lower-diagonal.
    for row in range(ndims):
        for col in range(ndims):
            ax = axs[row][col]
            if col >= row:
                continue

            # Get the samples for the 2D histogram
            _samples = leaves[:, [row, col]]  # [num_samples, 2]
            _log_weights = log_weights
            is_finite = np.all(np.isfinite(_samples), axis=-1)  # [num_samples]
            if np.bitwise_not(np.all(is_finite)):
                warnings.warn(
                    f"Found {np.sum(np.bitwise_not(is_finite))} non-finite samples for {parameters[row]} and {parameters[col]}")
                _samples = _samples[is_finite]
                _log_weights = _log_weights[is_finite]
            _weights = np.exp(_log_weights)

            # Plot the 2D histogram, over ranges set by the 1_per and 99_per of each parameter
            ranges = [param_limits[parameters[col]], param_limits[parameters[row]]]
            ax.hist2d(_samples[:, 1], _samples[:, 0], bins=(nbins, nbins), density=True,
                      cmap="bone_r",
                      weights=_weights, range=ranges)

            if kde_overlay:  # Put KDE contour on the 2D histograms

                # Calculate the point density
                x = _samples[:, 1]
                y = _samples[:, 0]
                xy = np.vstack([x, y])

                x_array = np.linspace(*param_limits[parameters[col]], 128)
                y_array = np.linspace(*param_limits[parameters[row]], 128)
                X, Y = np.meshgrid(x_array, y_array)
                xy_eval = np.vstack([X.ravel(), Y.ravel()])

                z = gaussian_kde(xy, weights=_weights)(xy_eval)
                z = z.reshape(X.shape)
                ax.contour(X, Y, z, levels=6, alpha=0.5)

            # Plot the maximum likelihood and MAP samples
            ax.scatter(max_like_sample[col], max_like_sample[row], color='green', marker='x')
            ax.scatter(map_sample[col], map_sample[row], color='red', marker='x')

            # Set the limits to 1 to 99 percentiles
            ax.set_xlim(param_limits[parameters[col]])
            ax.set_ylim(param_limits[parameters[row]])
    # Remove spacing
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    # Remove x ticks for all but bottom row
    for row in range(ndims - 1):
        for col in range(ndims):
            axs[row][col].set_xticks([])
            axs[row][col].set_xticklabels([])
    # Remove y ticks for all but left column
    for row in range(ndims):
        for col in range(1, ndims):
            axs[row][col].set_yticks([])
            axs[row][col].set_yticklabels([])
    # Set the labels on the bottom row and left column
    for i in range(ndims):
        axs[-1][i].set_xlabel(parameters[i], rotation=30, ha='right')
        axs[i][0].set_ylabel(parameters[i], rotation=30, ha='right')
    # Remove upper diagonal
    for row in range(ndims):
        for col in range(ndims):
            if col <= row:
                continue
            axs[row][col].remove()
    # Save the figure
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.close(fig)
    else:
        plt.show()


def _weighted_percentile(samples: np.ndarray, log_weights: np.ndarray,
                         percentiles: list[float | int]) -> np.ndarray:
    """
    Compute weighted percentiles of a set of samples.

    Args:
        samples: weighted samples
        log_weights: log weights of samples
        percentiles: list of percentiles to compute

    Returns:
        weighted percentiles
    """
    if len(percentiles) == 0:
        raise ValueError("percentiles must be a non-empty list")
    # Convert log weights to actual weights
    weights = LogSpace(log_weights - np.max(log_weights))  # Subtract max to avoid overflow
    weights = normalise_log_space(weights, norm_type='sum')  # Normalize weights

    # Sort samples and weights
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = sorted_weights.cumsum()
    cumulative_weights = cumulative_weights - cumulative_weights[0]
    cumulative_weights = cumulative_weights / cumulative_weights[-1]
    # Add zero to start of cumulative weights

    # Compute weighted percentiles
    percentile_values = np.interp(np.asarray(percentiles) / 100.0, cumulative_weights.value, sorted_samples)
    return percentile_values
