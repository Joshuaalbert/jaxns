import warnings
from typing import Optional, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import pylab as plt
from jax import random
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

from jaxns.internals.log_semiring import cumulative_logsumexp, LogSpace, normalise_log_space
from jaxns.internals.shapes import tuple_prod
from jaxns.nested_samplers.common.types import NestedSamplerResults
from jaxns.internals.mixed_precision import int_type
from jaxns.utils import resample

__all__ = ['plot_diagnostics',
           'plot_cornerplot']


def plot_diagnostics(results: NestedSamplerResults, save_name=None):
    """
    Plot diagnostics of the nested sampling run.

    Args:
        results: NestedSamplerResult
        save_name: file to save figure to.
    """

    num_samples = int(results.total_num_samples)
    if results.log_L_samples.shape[0] != num_samples:
        raise ValueError(f"Expected all samples to have the same number of samples, "
                         f"got log_L_samples with {results.log_L_samples.shape[0]} samples, "
                         f"expected {num_samples} samples.")
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
    log_X = np.asarray(results.log_X_mean)
    num_live_points_per_sample = np.asarray(results.num_live_points_per_sample)
    log_L = np.asarray(results.log_L_samples)
    max_log_likelihood = np.max(log_L)
    log_dp_mean = np.asarray(results.log_dp_mean)
    log_cum_evidence = cumulative_logsumexp(log_dp_mean)
    cum_evidence = np.exp(log_cum_evidence)
    log_Z_mean = np.asarray(results.log_Z_mean)
    num_likelihood_evaluations_per_sample = np.asarray(results.num_likelihood_evaluations_per_sample)
    if np.any(num_likelihood_evaluations_per_sample == 0):
        warnings.warn("Found samples with zero likelihood evaluations.")
        efficiency = np.where(
            num_likelihood_evaluations_per_sample == 0,
            np.nan,
            1. / num_likelihood_evaluations_per_sample
        )
    else:
        efficiency = 1. / num_likelihood_evaluations_per_sample
    mean_efficiency = np.exp(results.log_efficiency)
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
    axs[2].plot(-log_X, np.exp(log_dp_mean), c='black')
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
    axs[4].set_xlabel(r'$- \log X$')
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.show()


def plot_cornerplot(results: NestedSamplerResults, variables: Optional[List[str]] = None,
                    with_parametrised: bool = False,
                    save_name: Optional[str] = None, kde_overlay: bool = False):
    """
    Plots a cornerplot of the posterior samples.

    Args:
        results: NestedSamplerResult
        variables: list of variable names to plot. Plots all collected samples by default.
        save_name: file to save result to.
        kde_overlay: whether to overlay a KDE on the histograms.
    """
    samples = results.samples
    if with_parametrised:
        samples.update(results.parametrised_samples)
    # Plot all variables by default
    if variables is None:
        variables = list(samples.keys())
    variables = sorted(filter(lambda v: v in samples, variables))
    ndims = sum([tuple_prod(samples[key].shape[1:]) for key in variables], 0)

    num_samples = int(results.total_num_samples)
    for key in variables:
        if samples[key].shape[0] != num_samples:
            raise ValueError(f"Expected all samples to have the same number of samples, "
                             f"got {key} with {samples[key].shape[0]} samples, "
                             f"expected {num_samples} samples.")

    # Get the leaves of the tree, and concatenate into [num_samples, ndims] shape
    leaves = np.concatenate(
        [np.asarray(samples[key]).reshape((num_samples, -1)) for key in variables],
        axis=-1
    )

    # Create a parameter for each column. For scalar parameters, we just use the name of the parameter.
    # For vector we use name[i,j,...] etc.
    parameters = []
    for key in variables:
        shape = samples[key].shape[1:]
        if tuple_prod(shape) == 1:
            parameters.append(key)
        else:
            # Loop over each dimension of the parameter, and create a parameter for each index
            for i in range(tuple_prod(shape)):
                indices = np.unravel_index(i, shape)
                parameters.append(f"{key}[{','.join([str(j) for j in indices])}]")

    # Get the maximum likelihood and MAP samples
    log_L_samples = np.asarray(results.log_L_samples)
    log_posterior_density = np.asarray(results.log_posterior_density)
    max_like_idx = np.argmax(log_L_samples)
    map_idx = np.argmax(log_posterior_density)
    max_like_sample = leaves[max_like_idx]
    map_sample = leaves[map_idx]

    # Get the weight of each sample
    log_weights = np.asarray(normalise_log_space(LogSpace(results.log_dp_mean), norm_type='max').log_abs_val)

    figsize = min(20, max(4, int(2 * ndims)))
    fig, axs = plt.subplots(ndims, ndims, figsize=(figsize, figsize), squeeze=False)

    # Get the number of bins for the histograms based on the effective sample size
    nbins = max(10, int(jnp.sqrt(results.ESS)))

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
            per_1, per_5, per_50, per_95, per_99 = weighted_percentile(_samples, _log_weights,
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
        axs[-1][i].set_xlabel(parameters[i])
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
    plt.show()


def weighted_percentile(samples: np.ndarray, log_weights: np.ndarray,
                        percentiles: List[Union[float, int]]) -> np.ndarray:
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


def plot_samples_development(results, variables=None, save_name=None):
    """
    Animate the live points in a corner plot, visualising how the algorithm proceeds.
    Caution, this can be very slow as it plots a frame per sample.

    Args:
        results: NestedSamplingResult
        vars: list of variable names to plot, or None
        save_name: '.mp4' file to save animation to.
    """
    if save_name is None:
        raise ValueError("In order to plot the animation we must save it.")
    # Plot all variables by default
    if variables is None:
        variables = list(results.samples.keys())
    variables = sorted(filter(lambda v: v in results.samples, variables))
    ndims = sum([tuple_prod(results.samples[key].shape[1:]) for key in variables], 0)
    figsize = min(20, max(4, int(2 * ndims)))
    fig, axs = plt.subplots(ndims, ndims, figsize=(figsize, figsize))
    if ndims == 1:
        axs = [[axs]]
    weights = jnp.exp(results.log_p_mean)
    max_samples = weights.size
    norm = plt.Normalize(weights.min(), weights.max())
    to_colour = lambda w: plt.cm.jet(norm(w))

    def _get_artists(artists, start, stop):
        lims = {}
        dim = 0
        for key in variables:  # sorted(results.samples.keys()):
            n1 = tuple_prod(results.samples[key].shape[1:])
            for i in range(n1):
                samples1 = results.samples[key].reshape((max_samples, -1))[:, i]
                samples1 = samples1[start:stop]
                dim2 = 0
                for key2 in variables:  # sorted(results.samples.keys()):
                    n2 = tuple_prod(results.samples[key2].shape[1:])
                    for i2 in range(n2):
                        ax = axs[dim][dim2]
                        if dim2 > dim:
                            dim2 += 1
                            ax.set_xticks([])
                            ax.set_xticklabels([])
                            ax.set_yticks([])
                            ax.set_yticklabels([])
                            continue
                        if n2 > 1:
                            title2 = "{}[{}]".format(key2, i2)
                        else:
                            title2 = "{}".format(key2)
                        if n1 > 1:
                            title1 = "{}[{}]".format(key, i)
                        else:
                            title1 = "{}".format(key)
                        # ax.set_title('{} {}'.format(title1, title2))
                        if dim == dim2:
                            _, _, new_patches = ax.hist(samples1)
                            artists = artists + list(new_patches)
                            lims[dim] = ax.get_xlim()
                        else:
                            samples2 = results.samples[key2].reshape((max_samples, -1))[:, i2]
                            samples2 = samples2[start:stop]

                            sc = ax.scatter(samples2, samples1, marker='+', c=to_colour(weights[start:stop]), alpha=0.3)
                            artists.append(sc)
                        if dim == ndims - 1:
                            ax.set_xlabel("{}".format(title2))
                        if dim2 == 0:
                            ax.set_ylabel("{}".format(title1))

                        dim2 += 1
                dim += 1
        for dim in range(ndims):
            for dim2 in range(ndims):
                if dim == dim2:
                    continue
                ax = axs[dim][dim2] if ndims > 1 else axs[0]
                if dim in lims.keys():
                    ax.set_ylim(lims[dim])
                if dim2 in lims.keys():
                    ax.set_xlim(lims[dim2])
        return artists

    def init():
        start = 0
        stop = start + results.n_per_sample[start].astype(int_type)
        for i in range(ndims):
            for j in range(ndims):
                axs[i][j].clear()

        artists = []

        artists = _get_artists(artists, start, stop)
        return artists

    def update(start):
        stop = start + results.n_per_sample[start].astype(int_type)
        for i in range(ndims):
            for j in range(ndims):
                axs[i][j].clear()

        artists = []

        artists = _get_artists(artists, start, stop)
        return artists

    ani = FuncAnimation(fig, update, frames=jnp.arange(1, results.num_samples),
                        init_func=init, blit=True)

    ani.save(save_name, fps=results.n_per_sample[0] / 2.)


def add_colorbar_to_axes(ax, cmap, norm=None, vmin=None, vmax=None, label=None):
    """
    Add colorbar to axes easily.

    Args:
        ax: Axes
        cmap: str or cmap
        norm: Normalize or None
        vmin: lower limit of color if norm is None
        vmax: upper limit of color if norm is None
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if norm is None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm, cmap=plt.colormaps.get_cmap(cmap))
    if label is None:
        ax.figure.colorbar(sm, cax=cax, orientation='vertical')
    else:
        ax.figure.colorbar(sm, cax=cax, orientation='vertical', label=label)


def corner_cornerplot(results: NestedSamplerResults):
    try:
        import corner
    except ImportError:
        warnings.warn("You must run `pip install corner`")
        exit(0)
    try:
        import arviz as az
    except ImportError:
        warnings.warn("You must run `pip install arviz`")
        exit(0)
    samples = resample(random.PRNGKey(42), results.samples, results.log_dp_mean, S=int(results.ESS))
    corner.corner(az.from_dict(posterior=jax.tree.map(lambda x: x[None], samples)), )
    plt.show()
