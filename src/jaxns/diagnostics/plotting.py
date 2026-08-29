"""Optional plotting diagnostics for nested-sampling results."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
from scipy.stats import gaussian_kde

from jaxns.log_semiring import (
    LogSpace,
    cumulative_logsumexp,
    normalise_log_space,
)
from jaxns.optional import import_matplotlib

if TYPE_CHECKING:
    from jaxns.results import NestedSamplerResults


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
    negative_log_X = -log_X
    # V3 retains samples in append order. Every panel below is a function of
    # prior volume, so align all quantities under one stable volume ordering.
    sample_order = np.argsort(negative_log_X, kind="stable")
    log_X = log_X[sample_order]
    negative_log_X = negative_log_X[sample_order]
    num_live_points_per_sample = np.asarray(
        results.num_live_points_per_sample[:num_samples]
    )[sample_order]
    log_L = np.asarray(results.log_L[:num_samples])[sample_order]
    max_log_likelihood = np.max(log_L)
    log_dp = np.asarray(results.log_dp[:num_samples])[sample_order]
    # Accumulating in append order would mix unrelated volume regions.
    log_cum_evidence = cumulative_logsumexp(log_dp)
    cum_evidence = np.exp(np.asarray(log_cum_evidence))
    log_Z_mean = np.asarray(results.log_Z_mean)
    num_likelihood_evaluations_per_sample = np.asarray(
        results.num_likelihood_evaluations_per_sample[:num_samples]
    )[sample_order]
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
    axs[0].plot(negative_log_X, num_live_points_per_sample, c='black')
    axs[0].set_ylabel(r'$n_{\rm live}$')
    # detect if too small log likelihood
    rel_log_L = log_L - max_log_likelihood
    axs[1].plot(negative_log_X, np.exp(rel_log_L), c='black')
    axs[1].axhline(1., color='black', ls='dashed',
                   label=rf"$\log L_{{\rm max}}={max_log_likelihood:.1f}$")
    axs[1].set_ylabel(r'$L/L_{\rm max}$')
    axs[1].legend()
    axs[2].plot(negative_log_X, np.exp(log_dp), c='black')
    axs[2].axvline(results.H_mean, color='black', ls='dashed',
                   label=rf'$H={results.H_mean:.1f}$')
    axs[2].set_ylabel(r'$Z^{-1}L dX$')
    axs[2].legend()
    axs[3].plot(negative_log_X, cum_evidence, c='black')
    axs[3].axhline(1., color='black', ls='dashed',
                   label=rf"$\log Z={log_Z_mean:.1f}$")
    axs[3].set_ylabel('cumulative evidence / Z')
    axs[3].legend()
    axs[4].scatter(negative_log_X, efficiency, s=2, c='black')

    axs[4].axhline(mean_efficiency, color='black', ls='dashed',
                   label=f'avg. eff.={mean_efficiency:.3f}')
    axs[4].set_ylabel("sampler efficiency")
    axs[4].set_ylim(0., 1.05)
    axs[4].legend()

    # Plot X*L vs -log(X)
    XL = (LogSpace(log_X) * LogSpace(log_L)).value
    axs[5].plot(negative_log_X, XL, c='black')
    axs[5].set_ylabel(r'$X L$')

    axs[5].set_xlabel(r'$- \log X$')
    if save_file is not None:
        fig.savefig(save_file, bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.close(fig)
    else:
        plt.show()


def plot_evidence(
        results: NestedSamplerResults,
        *,
        num_samples: int = 512,
        conditionings: tuple[str, ...] = ("classic",),
        key=None,
        exact_log_Z: float | None = None,
        save_name: str | Path | None = None,
) -> None:
    """Plot explicitly conditioned Monte Carlo log-evidence ensembles."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if len(conditionings) == 0:
        raise ValueError("conditionings must be non-empty.")
    if len(set(conditionings)) != len(conditionings):
        raise ValueError("conditionings must not contain duplicates.")
    invalid_conditionings = [
        conditioning
        for conditioning in conditionings
        if conditioning not in ("classic", "phantom")
    ]
    if len(invalid_conditionings) > 0:
        raise ValueError(
            f"Unsupported evidence conditionings: {invalid_conditionings}"
        )
    if key is None:
        key = jax.random.PRNGKey(42)

    # Give each displayed ensemble an independent, reproducible stream.
    keys = jax.random.split(key, len(conditionings))
    log_Z_ensembles = []
    labels = []
    for conditioning, conditioning_key in zip(
            conditionings,
            keys,
            strict=True,
    ):
        evidence_samples = results.sample_evidence_mc(
            num_samples=num_samples,
            conditioning=conditioning,
            key=conditioning_key,
            diagnostics=False,
        )
        log_Z_samples = np.asarray(evidence_samples.log_Z_samples)
        finite = np.isfinite(log_Z_samples)
        if not np.all(finite):
            warnings.warn(
                f"Ignoring {np.sum(~finite)} non-finite {conditioning} "
                "log-evidence draws."
            )
            log_Z_samples = log_Z_samples[finite]
        if log_Z_samples.size == 0:
            raise ValueError(
                f"No finite {conditioning} log-evidence draws to plot."
            )
        label = (
            "phantom-conditioned"
            if conditioning == "phantom"
            else "classic"
        )
        labels.append(
            f"{label}: {np.mean(log_Z_samples):.3f} "
            f"± {np.std(log_Z_samples):.3f}"
        )
        log_Z_ensembles.append(log_Z_samples)

    plt = import_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    num_bins = max(10, min(80, int(np.sqrt(num_samples))))
    ax.hist(
        log_Z_ensembles,
        bins=num_bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=labels,
    )
    if exact_log_Z is not None:
        ax.axvline(
            exact_log_Z,
            color="black",
            linestyle="dashed",
            label=rf"exact $\log Z={exact_log_Z:.3f}$",
        )
    ax.set_xlabel(r"sampled $\log Z$")
    ax.set_ylabel("density")
    ax.legend()
    if save_name is not None:
        fig.savefig(save_name, bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close(fig)
    else:
        plt.show()


def _tuple_prod(shape: tuple[int, ...]) -> int:
    """Return the number of scalar parameters represented by one leaf."""
    product = 1
    for size in shape:
        product *= size
    return product


def plot_cornerplot(
        results: NestedSamplerResults,
        variables: list[str] | None = None,
        save_name: str | Path | None = None,
        kde_overlay: bool = False,
) -> None:
    """Plot posterior samples using classic expected shrinkage weights.

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

    # Use the authoritative point estimates retained by result construction.
    # Recomputing sample argmax indices can disagree for resampled results.
    max_like_point = dict(results.X_supremum.iter_items())
    map_point = dict(results.X_map.iter_items())
    max_like_sample = np.concatenate(
        [np.asarray(max_like_point[key]).reshape((-1,)) for key in variables]
    )
    map_sample = np.concatenate(
        [np.asarray(map_point[key]).reshape((-1,)) for key in variables]
    )

    # Get the weight of each sample
    log_weights = np.asarray(
        normalise_log_space(
            LogSpace(results.log_dp[:num_samples]),
            norm_type='max',
        ).log_abs_val)

    figsize = min(20, max(5, 2.5 * ndims))
    fig, axs = plt.subplots(ndims, ndims, figsize=(figsize, figsize), squeeze=False)
    title_fontsize = max(7, min(10, 11 - 0.25 * ndims))

    # Get the number of bins for the histograms based on the effective sample size
    posterior_ess = float(results.ess)
    if not np.isfinite(posterior_ess) or posterior_ess <= 0.0:
        nbins = 10
    else:
        # Bound visual noise and the square 2D histogram allocation.
        nbins = max(10, min(100, int(np.sqrt(posterior_ess))))

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
            ax.axvline(sample_mean, linestyle='dashed', color='blue')
            ax.axvline(sample_mean + sample_std, linestyle='dotted', color='blue')
            ax.axvline(sample_mean - sample_std, linestyle='dotted', color='blue')

            # Set the title
            lower_interval = per_50 - per_5
            upper_interval = per_95 - per_50
            title = [
                (
                    rf"${per_50:.2f}_{{-{lower_interval:.2f}}}^"
                    rf"{{+{upper_interval:.2f}}}$"
                ),
                rf"${sample_mean:.2f}\pm{sample_std:.2f}$",
                rf"MAP ${map_sample[row]:.2f}$ | ML ${max_like_sample[row]:.2f}$"
            ]
            ax.set_title("\n".join(title), fontsize=title_fontsize, pad=8)
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
        axs[-1][i].set_xlabel(parameters[i])
        if i == 0:
            axs[i][0].set_ylabel("density")
        else:
            axs[i][0].set_ylabel(parameters[i])
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
    samples = np.asarray(samples)
    log_weights = np.asarray(log_weights)
    percentiles_array = np.asarray(percentiles, dtype=float)
    if samples.ndim != 1 or log_weights.ndim != 1:
        raise ValueError("samples and log_weights must be one-dimensional")
    if samples.size == 0 or samples.size != log_weights.size:
        raise ValueError(
            "samples and log_weights must be non-empty and have equal size"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must be finite")
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError("log_weights must be finite or negative infinity")
    if not np.any(np.isfinite(log_weights)):
        raise ValueError("at least one log weight must be finite")
    if np.any((percentiles_array < 0.0) | (percentiles_array > 100.0)):
        raise ValueError("percentiles must lie between zero and 100")

    # Work relative to the largest log weight, then discard exact numerical
    # zeros so the interpolation coordinates remain strictly increasing.
    weights = np.exp(log_weights - np.max(log_weights))

    # Sort samples and weights
    sorted_indices = np.argsort(samples, kind="stable")
    sorted_samples = samples[sorted_indices]
    sorted_weights = weights[sorted_indices]
    positive_weight = sorted_weights > 0.0
    sorted_samples = sorted_samples[positive_weight]
    sorted_weights = sorted_weights[positive_weight]

    # Center each sample within its probability mass. Subtracting the first
    # weight would silently discard the lowest sample's mass.
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_midpoints = cumulative_weights - 0.5 * sorted_weights
    cumulative_midpoints /= cumulative_weights[-1]
    percentile_values = np.interp(
        percentiles_array / 100.0,
        cumulative_midpoints,
        sorted_samples,
    )
    return percentile_values
