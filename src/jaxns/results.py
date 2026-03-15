import dataclasses
import io
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar, TextIO, Optional, Union

import jax
import numpy as np
import pylab as plt
from jax import numpy as jnp
from jaxctx.context import CtxParams
from scipy.stats import gaussian_kde

from jaxns.cumulative_ops import batch_reduce
from jaxns.log_semiring import LogSpace, cumulative_logsumexp, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.phantom_eval import EvidenceSamples, sample_mc_shrinkage
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import resample_indicies
from jaxns.types import FloatArray, IntArray, UType, XType, PRNGKey, BoolArray

MF = TypeVar('MF')


@dataclasses.dataclass(slots=True)
class NestedSamplerResults(PureDataclassPytree):
    """
    Results of the nested sampling run.
    """
    log_Z_mean: FloatArray  # estimate of E[log(Z)]
    log_Z_uncert: FloatArray  # estimate of StdDev[log(Z)]
    ess: FloatArray  # estimate of Kish's effective sample size
    H_mean: FloatArray  # estimate of E[int log(L) L dp/Z]
    total_num_samples: IntArray  # int, the total number of samples collected.
    total_phantom_samples: IntArray  # int, the total number of phantom samples collected.
    total_num_likelihood_evaluations: IntArray  # how many likelihood evaluations were made in total
    log_efficiency: FloatArray  # log(total_num_samples / total_num_likelihood_evaluations)
    termination_reason: IntArray  # this will be an int reflecting the reason for termination

    U_samples: UType  # [num_samples] samples in homogeneous unit hypercube.
    X_samples: XType  # [num_samples] samples in the constrained space.
    log_L_constraints: FloatArray  # [num_samples] the likelihood constraint for each sample, i.e. the likelihood contour that the sample was drawn from.
    log_L_phantom: FloatArray  # [num_samples, num_phantom] the likelihood of the phantom point used for each sample, or -inf if no phantom point was used.
    valid_phantom: BoolArray  # [num_samples] whether the phantoms are valid for each sample, i.e. whether the phantom points were used for the sample or not.
    log_L: FloatArray  # log(L) of each sample
    log_dp: FloatArray  # log(weight) of each sample, i.e. log(dp) = log(L) + log(dX) - log(Z)
    log_X_mean: FloatArray  # log(E[U]) of each sample
    log_posterior_density: FloatArray  # log(P( theta | D )) log posteriori density
    num_live_points_per_sample: IntArray  # how many live points were taken for the samples.
    num_likelihood_evaluations_per_sample: IntArray  # how many likelihood evaluations were made per sample

    ## pointwise estimates
    # max(L)
    log_L_supremum: FloatArray  # max(L) of the samples, used for diagnostics and resampling
    U_supremum: UType  # the U point with the largest likelihood, used for diagnostics and resampling
    X_supremum: XType  # the X point with the largest likelihood, used for diagnostics and resampling
    # max(L p)
    log_L_map: FloatArray  # max(L p) of the samples, used for diagnostics and resampling
    U_map: UType  # the U point with the largest posterior density, used for diagnostics and resampling
    X_map: XType  # the X point with the largest posterior density, used for diagnostics and resampling

    def trim(self) -> 'NestedSamplerResults':
        num_samples = int(self.total_num_samples)
        sample_data = dict(
            U_samples=self.U_samples,
            X_samples=self.X_samples,
            log_L=self.log_L,
            log_dp=self.log_dp,
            log_X_mean=self.log_X_mean,
            log_posterior_density=self.log_posterior_density,
            num_live_points_per_sample=self.num_live_points_per_sample,
            num_likelihood_evaluations_per_sample=self.num_likelihood_evaluations_per_sample,
            log_L_constraints=self.log_L_constraints,
            log_L_phantom=self.log_L_phantom,
            valid_phantom=self.valid_phantom,
        )
        sample_data = jax.tree.map(lambda s: s[:num_samples, ...], sample_data)
        return dataclasses.replace(self, **sample_data)

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
        Sample the evidence using the MC shrinkage method.

        Args:
            num_samples: number of evidence samples to draw
            batch_size: optional, how many samples to process in a batch when applying the function.
            key: optional, PRNGKey for resampling

        Returns:
            array of shape [num_samples] containing samples of the evidence.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        return _sample_evidence(self, num_samples=num_samples, batch_size=batch_size, key=key)

    def sample_mc_shrinkage(self, num_samples: int, batch_size: int | None = None, key: PRNGKey | None = None) -> EvidenceSamples:
        """
        Sample the evidence using the MC shrinkage method.

        Args:
            num_samples: number of evidence samples to draw
            batch_size: optional, how many samples to process in a batch when applying the function.
            key: optional, PRNGKey for resampling

        Returns:
            EvidenceSamples object containing samples of the evidence and related statistics.
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        return _sample_mc_shrinkage(self, num_samples=num_samples, batch_size=batch_size, key=key)


NestedSamplerResults.register_pytree()


@partial(jax.jit, inline=True, static_argnames=['num_samples', 'replace'])
def _resample(self: NestedSamplerResults, key: PRNGKey, num_samples: int, replace: bool = True) -> NestedSamplerResults:
    idx = resample_indicies(key, self.log_dp, S=num_samples, replace=replace)
    # after resampling, the weights are uniform, so log_dp = log(1/N) = -log(N)
    sample_data = dict(
        U_samples=self.U_samples,
        X_samples=self.X_samples,
        log_L=self.log_L,
        log_dp=jnp.full(self.log_dp.shape, -jnp.log(num_samples), mp_policy.measure_dtype),
        log_X_mean=self.log_X_mean,
        log_posterior_density=self.log_posterior_density,
        num_live_points_per_sample=self.num_live_points_per_sample,
        num_likelihood_evaluations_per_sample=self.num_likelihood_evaluations_per_sample,
    )
    sample_data = jax.tree.map(lambda s: s[idx, ...], sample_data)
    sort_idxs = jnp.argsort(sample_data['log_L'])
    sample_data = jax.tree.map(lambda s: s[sort_idxs, ...], sample_data)
    return dataclasses.replace(self, **sample_data)


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

    weights = LogSpace(self.log_dp)
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
            sig_figs = -int("{:e}".format(uncert_v).split('e')[1]) + 1
            return round(float(v), sig_figs)
        except:
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
    _print(f"samples: {int(results.total_num_samples):d}")
    _print(f"phantom samples: {int(results.total_phantom_samples):d}")
    _print(
        f"likelihood evals / sample: {float(results.total_num_likelihood_evaluations / results.total_num_samples):.1f}"
    )
    _print(
        f"phantom fraction (%): {100 * float(results.total_phantom_samples / results.total_num_samples):.1f}%"
    )
    _print("--------")
    _print(
        f"logZ={_round(results.log_Z_mean, results.log_Z_uncert)} +- {_round(results.log_Z_uncert, results.log_Z_uncert)}"
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
        f"effective sample size={results.ess:.1f}"
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
        var_name = name if ndims == 1 else "{}[#]".format(name)
        _print(
            f"{var_name}: mean +- std.dev. | MAP est. | max(L) est."
        )
        for dim in range(ndims):
            _uncert = _x_std[dim]
            # two sig-figs based on uncert
            sig_figs = 1 - int(f"{_uncert:e}".split('e')[1])

            def _round(ar):
                return round(float(ar), sig_figs)

            _print("{}: {} +- {} | {} | {}".format(
                name if ndims == 1 else "{}[{}]".format(name, dim),
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

    num_samples = int(results.total_num_samples)
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(8, 15))
    log_X = np.asarray(results.log_X_mean[:num_samples])
    num_live_points_per_sample = np.asarray(results.num_live_points_per_sample[:num_samples])
    log_L = np.asarray(results.log_L[:num_samples])
    max_log_likelihood = np.max(log_L)
    log_dp = np.asarray(results.log_dp[:num_samples])
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
    axs[2].axvline(-results.H_mean, color='black', ls='dashed',
                   label=rf'$-H={-results.H_mean:.1f}$')
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


@partial(jax.jit, inline=True, static_argnames=['num_samples', 'batch_size'])
def _sample_evidence(self: NestedSamplerResults,
                     num_samples: int = 100, batch_size: int | None = None, key: PRNGKey | None = None) -> FloatArray:
    evidence_samples = sample_mc_shrinkage(
        key=key,
        log_L_constraints=self.log_L_constraints,
        log_L_classic=self.log_L,
        K_classic=self.num_live_points_per_sample,
        valid_phantom=self.valid_phantom,
        log_L_phantom=self.log_L_phantom,
        num_samples=self.total_num_samples,
        num_Z_samples=num_samples,
        batch_size=batch_size
    )
    return evidence_samples.log_Z_samples


@partial(jax.jit, inline=True, static_argnames=['num_samples', 'batch_size'])
def _sample_mc_shrinkage(self: NestedSamplerResults,
                         num_samples: int = 100, batch_size: int | None = None, key: PRNGKey | None = None) -> EvidenceSamples:
    evidence_samples = sample_mc_shrinkage(
        key=key,
        log_L_constraints=self.log_L_constraints,
        log_L_classic=self.log_L,
        K_classic=self.num_live_points_per_sample,
        valid_phantom=self.valid_phantom,
        log_L_phantom=self.log_L_phantom,
        num_samples=self.total_num_samples,
        num_Z_samples=num_samples,
        batch_size=batch_size
    )
    return evidence_samples


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


def plot_cornerplot(results: NestedSamplerResults, variables: Optional[list[str]] = None, save_name: Optional[str] = None, kde_overlay: bool = False):
    """
    Plots a cornerplot of the posterior samples.

    Args:
        results: NestedSamplerResult
        variables: list of variable names to plot. Plots all collected samples by default.
        save_name: file to save result to.
        kde_overlay: whether to overlay a KDE on the histograms.
    """
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
        normalise_log_space(LogSpace(results.log_dp[:num_samples]), norm_type='max').log_abs_val)

    figsize = min(20, max(4, int(2 * ndims)))
    fig, axs = plt.subplots(ndims, ndims, figsize=(figsize, figsize), squeeze=False)

    # Get the number of bins for the histograms based on the effective sample size
    if np.isnan(np.asarray(results.ess)):
        nbins = 10
    else:
        nbins = max(10, int(jnp.sqrt(results.ess)))

    # Loop over the variables, and plot the marginal distributions on the diagonal setting a title above
    # each plot with the mean+-stddev, 5%/50%/95%, and MAP
    param_limits = dict()  # Store the 1_per and 99_per for each parameter
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
                         percentiles: list[Union[float, int]]) -> np.ndarray:
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
