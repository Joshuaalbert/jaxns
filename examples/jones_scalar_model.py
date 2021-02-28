from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior
import pylab as plt
from jax import jit
from jax import numpy as jnp, random
import numpy as np
from timeit import default_timer


def generate_data(key, tec, const, uncert, num_outliers=3):
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec / freqs * TEC_CONV + const  # added a constant term
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + uncert * random.normal(key, shape=Y.shape)

    Y_obs = Y_obs.at[:num_outliers * 2:2].set(1. * random.normal(key, shape=(num_outliers,)))
    amp = jnp.ones_like(phase)
    return Y_obs, amp, tec, const, freqs


def log_normal(x, mean, scale):
    dx = (x - mean) / scale
    return -0.5 * x.size * jnp.log(2. * jnp.pi) - x.size * jnp.log(scale) \
           - 0.5 * dx @ dx


def log_laplace(x, mean, scale):
    dx = jnp.abs(x - mean) / scale
    return - x.size * jnp.log(2. * scale) - jnp.sum(dx)


@jit
def run(key, tec, const, uncert):
    Y_obs, amp, tec, const, freqs = generate_data(key, tec, const, uncert, num_outliers=10)

    def log_likelihood(tec, const, uncert, **kwargs):
        TEC_CONV = -8.4479745e6  # mTECU/Hz
        phase = tec * (TEC_CONV / freqs) + const
        Y = jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=-1)
        log_prob = log_normal(Y, Y_obs, uncert)
        return log_prob

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -300., 300.)) \
        .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
        .push(HalfLaplacePrior('uncert', uncert))

    print("Probabilistic model:\n{}".format(prior_chain))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice',
                       num_live_points=1000,
                       max_samples=1e5,
                       collect_samples=True,
                       sampler_kwargs=dict(depth=2, num_slices=3),
                       tec_mean=lambda tec, **kw: tec,
                       const_mean=lambda const, **kw: const
                       )

    results = ns(key=key, termination_frac=0.01)

    return results


def main():

    results = run(random.PRNGKey(3245), 100., -1., 0.2)

    plot_diagnostics(results)
    plot_cornerplot(results)



if __name__ == '__main__':
    main()
