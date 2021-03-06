from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior
import pylab as plt
from jax import jit
from jax import numpy as jnp, random
import numpy as np
from timeit import default_timer


def wrap(phi):
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


def generate_data(key, tec, const, clock, uncert, num_outliers=3):
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec / freqs * TEC_CONV + const + (1e-9 * freqs * jnp.pi) * clock  # added a constant term
    phase = phase.at[:num_outliers * 2:2].set(1. * random.normal(key, shape=(num_outliers,)))
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + uncert * random.normal(key, shape=Y.shape)
    amp = jnp.ones_like(phase)
    return Y_obs, amp, tec, const, clock, freqs


def log_normal(x, mean, scale):
    dx = (x - mean) / scale
    return jnp.sum(-0.5 * jnp.log(2. * jnp.pi) - jnp.log(scale)) \
           - 0.5 * dx @ dx


def log_laplace(x, mean, scale):
    dx = jnp.abs(x - mean) / scale
    return jnp.sum(- jnp.log(2. * scale) - dx)


@jit
def run(key, tec, const, clock, uncert):
    Y_obs, amp, tec, const, clock, freqs = generate_data(key, tec, const, clock, uncert, num_outliers=0)
    phase_obs = jnp.arctan2(Y_obs[freqs.size:], Y_obs[:freqs.size])

    def log_likelihood(tec, const, clock, uncert, smooth, **kwargs):
        TEC_CONV = -8.4479745e6  # mTECU/Hz
        # uncert = jnp.repeat(uncert,8)
        phase = tec * (TEC_CONV / freqs) + const + clock * (2e-9 * freqs) * jnp.pi
        dphase = wrap(wrap(phase) - phase_obs)
        return log_laplace(dphase, 0., uncert) + log_laplace(jnp.diff(dphase), 0., smooth)

    def Y(tec, const, clock, **kwargs):
        TEC_CONV = -8.4479745e6  # mTECU/Hz
        phase = tec * (TEC_CONV / freqs) + const + clock * (2e-9 * freqs) * jnp.pi
        Y = jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=-1)
        return Y

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -300., 300.)) \
        .push(UniformPrior('clock', -3., 3.)) \
        .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
        .push(HalfLaplacePrior('uncert', uncert)) \
        .push(HalfLaplacePrior('smooth', uncert))

    print("Probabilistic model:\n{}".format(prior_chain))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice',
                       num_live_points=2000,
                       max_samples=1e6,
                       collect_samples=True,
                       num_parallel_samplers=1,
                       sampler_kwargs=dict(depth=5, num_slices=prior_chain.U_ndims * 5),
                       marginalised=dict(tec_mean=lambda tec, **kw: tec,
                                         const_mean=lambda const, **kw: const,
                                         Y_mean=Y,
                                         Y2_mean=lambda *args, **kwargs: Y(*args, **kwargs) ** 2)
                       )

    results = ns(key=key, termination_frac=0.001)

    return results, Y_obs, phase_obs, freqs


def main():
    results, Y_obs, phase_obs, freqs = run(random.PRNGKey(3245), 100., -1., 0., 0.5)
    x_uncert = jnp.sqrt(results.marginalised['Y2_mean'][:24] - results.marginalised['Y_mean'][:24] ** 2)
    y_uncert = jnp.sqrt(results.marginalised['Y2_mean'][24:] - results.marginalised['Y_mean'][24:] ** 2)
    plt.plot(results.marginalised['Y_mean'][:24], results.marginalised['Y_mean'][24:])
    plt.errorbar(results.marginalised['Y_mean'][:24], results.marginalised['Y_mean'][24:], xerr=x_uncert, yerr=y_uncert)
    plt.plot(Y_obs[:24], Y_obs[24:])
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
