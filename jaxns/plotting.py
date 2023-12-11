import logging

import numpy as np
import pylab as plt
import jax.numpy as jnp
from jax import random
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jaxns.utils import resample
from jaxns.internals.log_semiring import cumulative_logsumexp
from jaxns.internals.shapes import tuple_prod
from jaxns.internals.types import NestedSamplerResults, int_type

logger = logging.getLogger('jaxns')

__all__ = ['plot_diagnostics',
           'plot_cornerplot']


def plot_diagnostics(results: NestedSamplerResults, save_name=None):
    """
    Plot diagnostics of the nested sampling run.

    Args:
        results: NestedSamplerResult
        save_name: file to save figure to.
    """
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
    log_X = results.log_X_mean
    axs[0].plot(-log_X, results.num_live_points_per_sample, c='black')
    axs[0].set_ylabel(r'$n_{\rm live}(U)$')
    # detect if too small log likelihood
    log_likelihood = results.log_L_samples
    max_log_likelihood = jnp.max(log_likelihood)
    rel_log_likelihood = log_likelihood - max_log_likelihood
    axs[1].plot(-log_X, rel_log_likelihood, c='black')
    axs[1].hlines(0., jnp.min(-log_X),
                  jnp.max(-log_X), colors='black', ls='dashed',
                  label=r"$\log L_{{\rm max}}={:.1f}$".format(max_log_likelihood))
    axs[1].set_ylabel(r'$\log \left(L(X)/L_{\rm max}\right)$')
    axs[1].legend()
    axs[2].plot(-log_X, jnp.exp(results.log_dp_mean), c='black')
    # axs[2].vlines(-results.H_mean, 0., jnp.exp(jnp.max(results.log_dp_mean)), colors='black', ls='dashed',
    #               label='-logX=-H={:.1f}'.format(-results.H_mean))
    axs[2].set_ylabel(r'$Z^{-1}L(U) dX$')
    axs[2].legend()
    log_cum_evidence = cumulative_logsumexp(results.log_dp_mean)
    cum_evidence = jnp.exp(log_cum_evidence)
    axs[3].plot(-log_X, cum_evidence, c='black')
    axs[3].hlines(1., jnp.min(-log_X),
                  jnp.max(-log_X), colors='black', ls='dashed',
                  label=r"$\log Z={:.1f}$".format(results.log_Z_mean))
    axs[3].set_ylabel(r'$Z(x > U)/Z$')
    axs[3].legend()
    axs[4].scatter(-log_X, 1. / results.num_likelihood_evaluations_per_sample, s=2, c='black')
    axs[4].hlines(jnp.exp(results.log_efficiency), jnp.min(-log_X),
                  jnp.max(-log_X), colors='black', ls='dashed',
                  label='avg. eff.={:.3f}'.format(jnp.exp(results.log_efficiency)))
    axs[4].set_ylabel("sampler efficiency")
    axs[4].set_ylim(0., 1.05)
    axs[4].legend()
    axs[4].set_xlabel(r'$- \log X$')
    if save_name is not None:
        fig.savefig(save_name)
    plt.show()


def plot_cornerplot(results: NestedSamplerResults, vars=None, save_name=None):
    """
    Plots a cornerplot of the posterior samples.

    Args:
        results: NestedSamplerResult
        vars: list of variable names to plot, or None.
        save_name: file to save result to.
    """
    rkey0 = random.PRNGKey(123496)
    vars = _get_vars(results, vars)
    ndims = _get_ndims(results, vars)
    figsize = min(20, max(4, int(2 * ndims)))
    fig, axs = plt.subplots(ndims, ndims, figsize=(figsize, figsize))
    if ndims == 1:
        axs = [[axs]]
    nsamples = results.total_num_samples
    max_like_idx = jnp.argmax(results.log_L_samples)
    map_idx = jnp.argmax(results.log_posterior_density)
    log_p = results.log_dp_mean
    nbins = max(10, int(jnp.sqrt(results.ESS)) + 1)
    lims = {}
    dim = 0
    for key in vars:  # sorted(results.samples.keys()):
        n1 = tuple_prod(results.samples[key].shape[1:])
        for i in range(n1):
            samples1 = results.samples[key].reshape((nsamples, -1))[:, i]
            if jnp.std(samples1) == 0.:
                dim += 1
                continue
            weights = jnp.where(jnp.isfinite(samples1), jnp.exp(log_p), 0.)
            log_weights = jnp.where(jnp.isfinite(samples1), log_p, -jnp.inf)
            samples1 = jnp.where(jnp.isfinite(samples1), samples1, 0.)
            # kde1 = gaussian_kde(samples1, weights=weights, bw_method='silverman')
            # samples1_resampled = kde1.resample(size=int(results.ESS))
            rkey0, rkey = random.split(rkey0, 2)
            samples1_resampled = resample(rkey, samples1, log_weights, S=max(10, int(results.ESS)), replace=True)
            samples1_max_like = samples1[max_like_idx]
            samples1_map_point = samples1[map_idx]
            binsx = jnp.linspace(*jnp.percentile(samples1_resampled, jnp.asarray([0, 100])), 2 * nbins)
            dim2 = 0
            for key2 in vars:  # sorted(results.samples.keys()):
                n2 = tuple_prod(results.samples[key2].shape[1:])
                for i2 in range(n2):
                    ax = axs[dim][dim2]
                    if dim2 > dim:
                        dim2 += 1
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                        ax.remove()
                        continue
                    if dim < ndims - 1:
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                    if dim2 > 0:
                        ax.set_yticks([])
                        ax.set_yticklabels([])
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
                        # ax.plot(binsx, kde1(binsx))
                        ax.hist(np.asarray(samples1_resampled), bins='auto', fc='None', edgecolor='black', density=True)
                        ax.axvline(samples1_max_like, color='green')
                        sample_mean = jnp.average(samples1, weights=weights)
                        sample_std = jnp.sqrt(jnp.average((samples1 - sample_mean) ** 2, weights=weights))
                        ax.set_title(
                            r"${:.2f}_{{{:.2f}}}^{{{:.2f}}}$".format(
                                *jnp.percentile(samples1_resampled, jnp.asarray([50, 5, 95]))) + \
                            "\n" + r"${:.2f}\pm{:.2f}$".format(sample_mean, sample_std) + \
                            "\n" + r"MAP ${:.2f}$ | ML ${:.2f}$".format(samples1_map_point, samples1_max_like))
                        # ax.set_title(r"{}: ${:.2f}\pm{:.2f}$".format(title1, sample_mean, sample_std))
                        # ax.text(0., 1., r"${:.2f}_{{{:.2f}}}^{{{:.2f}}}$".format(*jnp.percentile(samples1_resampled, [50, 5, 95])),
                        #      verticalalignment = 'top', horizontalalignment='left', transform = ax.transAxes,
                        #         bbox=dict(facecolor='grey', alpha=0.5))

                        ax.axvline(sample_mean, linestyle='dashed', color='red')
                        ax.axvline(sample_mean + sample_std,
                                   linestyle='dotted', color='red')
                        ax.axvline(sample_mean - sample_std,
                                   linestyle='dotted', color='red')
                        ax.set_xlim(binsx.min(), binsx.max())
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                        lims[dim] = ax.get_xlim()
                    else:
                        samples2 = results.samples[key2].reshape((nsamples, -1))[:, i2]
                        if jnp.std(samples2) == 0.:
                            dim2 += 1
                            continue
                        weights = jnp.where(jnp.isfinite(samples2), jnp.exp(log_p), 0.)
                        log_weights = jnp.where(jnp.isfinite(samples2), log_p, -jnp.inf)
                        samples2 = jnp.where(jnp.isfinite(samples2), samples2, 0.)
                        # kde2 = gaussian_kde(jnp.stack([samples1, samples2], axis=0),
                        #                     weights=weights,
                        #                     bw_method='silverman')
                        # samples2_resampled = kde2.resample(size=int(results.ESS))
                        rkey0, rkey = random.split(rkey0, 2)
                        samples2_resampled = resample(rkey, jnp.stack([samples1, samples2], axis=-1), log_weights,
                                                      S=max(10, int(results.ESS)), replace=True)
                        # norm = plt.Normalize(log_weights.min(), log_weights.max())
                        # color = jnp.atleast_2d(plt.cm.jet(norm(log_weights)))
                        ax.hist2d(samples2_resampled[:, 1], samples2_resampled[:, 0], bins=(nbins, nbins), density=True,
                                  cmap=plt.cm.bone_r)
                        # ax.scatter(samples2_resampled[:, 1], samples2_resampled[:, 0], marker='+', c='black', alpha=0.5)
                        # binsy = jnp.linspace(*jnp.percentile(samples2_resampled[:, 1], [0, 100]), 2 * nbins)
                        # U, Y = jnp.meshgrid(binsx, binsy, indexing='ij')
                        # ax.contour(kde2(jnp.stack([U.flatten(), Y.flatten()], axis=0)).reshape((2 * nbins, 2 * nbins)),
                        #            extent=(binsy.min(), binsy.max(),
                        #                    binsx.min(), binsx.max()),
                        #            origin='lower')
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
    if save_name is not None:
        fig.savefig(save_name)
    plt.show()


def _get_ndims(results, vars):
    ndims = int(sum([tuple_prod(v.shape[1:]) for k, v in results.samples.items() if (k in vars)]))
    return ndims


def _get_vars(results, vars):
    if vars is None:
        vars = [k for k, v in results.samples.items()]
    vars = [v for v in vars if v in results.samples.keys()]
    vars = sorted(vars)
    return vars


def plot_samples_development(results, vars=None, save_name=None):
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
    vars = _get_vars(results, vars)
    ndims = _get_ndims(results, vars)
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
        for key in vars:  # sorted(results.samples.keys()):
            n1 = tuple_prod(results.samples[key].shape[1:])
            for i in range(n1):
                samples1 = results.samples[key].reshape((max_samples, -1))[:, i]
                samples1 = samples1[start:stop]
                dim2 = 0
                for key2 in vars:  # sorted(results.samples.keys()):
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
    sm = plt.cm.ScalarMappable(norm, cmap=plt.cm.get_cmap(cmap))
    if label is None:
        ax.figure.colorbar(sm, cax=cax, orientation='vertical')
    else:
        ax.figure.colorbar(sm, cax=cax, orientation='vertical', label=label)


def corner_cornerplot(results: NestedSamplerResults):
    try:
        import corner
    except ImportError:
        logger.warning("You must run `pip install corner`")
        exit(0)
    try:
        import arviz as az
    except ImportError:
        logger.warning("You must run `pip install arviz`")
        exit(0)
    from jax import tree_map
    samples = resample(random.PRNGKey(42), results.samples, results.log_dp_mean, S=int(results.ESS))
    corner.corner(az.from_dict(posterior=tree_map(lambda x: x[None], samples)), )
    plt.show()
