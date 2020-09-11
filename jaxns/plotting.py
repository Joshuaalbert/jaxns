import pylab as plt
import jax.numpy as jnp
from scipy.stats.kde import gaussian_kde
from jaxns.utils import safe_gaussian_kde


def plot_diagnostics(results, save_name=None):
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
    axs[0].plot(-results.log_X[:results.num_samples], results.n_per_sample[:results.num_samples])
    axs[0].set_ylabel(r'$n(X)$')
    axs[1].plot(-results.log_X[:results.num_samples], jnp.exp(results.log_L_samples[:results.num_samples]))
    axs[1].set_ylabel(r'$L(X)$')
    axs[2].plot(-results.log_X[:results.num_samples], jnp.exp(results.log_p[:results.num_samples]))
    axs[2].set_ylabel(r'$Z^{-1}L(X) dX$')
    axs[3].plot(-results.log_X[:results.num_samples],
                jnp.exp(results.logZ) * jnp.cumsum(jnp.exp(results.log_p[:results.num_samples])))
    axs[3].set_ylabel(r'$Z(x > X)$')
    axs[4].plot(-results.log_X[:results.num_samples], results.sampler_efficiency[:results.num_samples])
    axs[4].hlines(results.efficiency, jnp.min(-results.log_X[:results.num_samples]),
                  jnp.max(-results.log_X[:results.num_samples]), colors='black', ls='dashed',
                  label='avg. eff.={:.3f}'.format(results.efficiency))
    axs[4].set_ylabel("sampler efficiency")
    axs[4].set_xlabel(r'$-\log X$')
    axs[4].legend()
    if save_name is not None:
        fig.savefig(save_name)
    plt.show()


def plot_cornerplot(results, vars=None, save_name=None):
    if vars is None:
        vars = [k for k, v in results.samples.items()]
    vars = [v for v in vars if v in results.samples.keys()]
    vars = sorted(vars)
    ndims = int(sum([jnp.prod(v.shape[1:]) for k, v in results.samples.items() if (k in vars)]))
    figsize = min(20, max(4, int(2 * ndims)))
    fig, axs = plt.subplots(ndims, ndims, figsize=(figsize, figsize))
    # if not isinstance(axs, list):
    #     axs = [axs]
    weights = jnp.exp(results.log_p)
    nsamples = weights.size
    nbins = int(jnp.sqrt(results.ESS)) + 1
    lims = {}
    dim = 0
    for key in vars:  # sorted(results.samples.keys()):
        n1 = jnp.prod(results.samples[key].shape[1:])
        for i in range(n1):
            samples1 = results.samples[key].reshape((nsamples, -1))[:, i]
            if jnp.std(samples1) == 0.:
                dim += 1
                continue
            weights = jnp.where(jnp.isfinite(samples1), jnp.exp(results.log_p), 0.)
            samples1 = jnp.where(jnp.isfinite(samples1), samples1, 0.)
            kde1 = gaussian_kde(samples1, weights=weights, bw_method='silverman')
            # kde1 = safe_gaussian_kde(samples1, weights=weights)
            samples1_resampled = kde1.resample(size=int(results.ESS))
            binsx = jnp.linspace(*jnp.percentile(samples1_resampled, [0, 100]), 2 * nbins)
            dim2 = 0
            for key2 in vars:  # sorted(results.samples.keys()):
                n2 = jnp.prod(results.samples[key2].shape[1:])
                for i2 in range(n2):
                    ax = axs[dim][dim2] if ndims > 1 else axs[0]
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
                    ax.set_title('{} {}'.format(title1, title2))
                    if dim == dim2:
                        ax.plot(binsx, kde1(binsx))
                        sample_mean = jnp.average(samples1, weights=weights)
                        sample_std = jnp.sqrt(jnp.average((samples1 - sample_mean) ** 2, weights=weights))
                        ax.set_title("{:.2f}:{:.2f}:{:.2f}\n{:.2f}+-{:.2f}".format(
                            *jnp.percentile(samples1_resampled, [5, 50, 95]), sample_mean, sample_std))
                        ax.vlines(sample_mean, *ax.get_ylim(), linestyles='solid', colors='red')
                        ax.vlines([sample_mean - sample_std, sample_mean + sample_std],
                                  *ax.get_ylim(), linestyles='dotted', colors='red')
                        ax.set_xlim(binsx.min(), binsx.max())
                        lims[dim] = ax.get_xlim()
                    else:
                        samples2 = results.samples[key2].reshape((nsamples, -1))[:, i2]
                        if jnp.std(samples2) == 0.:
                            dim2 += 1
                            continue
                        weights = jnp.where(jnp.isfinite(samples2), weights, 0.)
                        samples2 = jnp.where(jnp.isfinite(samples2), samples2, 0.)
                        kde2 = gaussian_kde(jnp.stack([samples1, samples2], axis=0),
                                            weights=weights,
                                            bw_method='silverman')
                        samples2_resampled = kde2.resample(size=int(results.ESS))
                        ax.scatter(samples2_resampled[1, :], samples2_resampled[0, :], marker='+', c='black', alpha=0.5)
                        binsy = jnp.linspace(*jnp.percentile(samples2_resampled[1, :], [0, 100]), 2 * nbins)
                        X, Y = jnp.meshgrid(binsx, binsy, indexing='ij')
                        ax.contour(kde2(jnp.stack([X.flatten(), Y.flatten()], axis=0)).reshape((2 * nbins, 2 * nbins)),
                                   extent=(binsy.min(), binsy.max(),
                                           binsx.min(), binsx.max()),
                                   origin='lower')
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
