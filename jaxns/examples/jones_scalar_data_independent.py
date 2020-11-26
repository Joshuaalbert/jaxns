from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, NormalPrior, SymmetricUniformWalkPrior, DiagGaussianWalkPrior

from jax.scipy.linalg import solve_triangular
from jax import jit, vmap, soft_pmap
from jax import numpy as jnp, random
import pylab as plt
from timeit import default_timer

from jax._src.scipy.signal import _convolve_nd

# import os
# os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=10"

def generate_data():
    T = 100
    tec = 30.+jnp.cumsum(10. * random.normal(random.PRNGKey(87658), shape=(T,)))
    print(tec)
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV  + jnp.linspace(-jnp.pi, jnp.pi, T)[:, None]
    phase = phase.at[:,:24:8].set(0.)
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * random.normal(random.PRNGKey(75467), shape=Y.shape)
    # Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape_dict)
    Sigma = 0.25 ** 2 * jnp.eye(48)
    amp = jnp.ones_like(phase)
    return Sigma, T, Y_obs, amp, tec, freqs

def windowed_mean(a, w, mode='reflect'):
    if w is None:
        return jnp.broadcast_to(jnp.mean(a, axis=0, keepdims=True), a.shape)
    dims = len(a.shape)
    a = a
    kernel = jnp.reshape(jnp.ones(w)/w, [w]+[1]*(dims-1))
    _w1 = (w-1)//2
    _w2 = _w1 if (w%2 == 1) else _w1 + 1
    pad_width = [(_w1, _w2)] + [(0,0)]*(dims-1)
    a = jnp.pad(a, pad_width=pad_width, mode=mode)
    return _convolve_nd(a,kernel, mode='valid', precision=None)

def log_laplace(x, mean, uncert):
    dx = (x - mean) / uncert
    return - x.size * jnp.log(2.*uncert) \
           - jnp.sum(jnp.abs(dx))

def main():
    Sigma, T, Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def single_solve1(key, Y_obs, amp):
        def log_likelihood(tec, const, uncert, **kwargs):
            phase = tec * (TEC_CONV / freqs)  + const #+ clock
            Y = jnp.concatenate([amp*jnp.cos(phase), amp*jnp.sin(phase)], axis=-1)
            return log_laplace(Y, Y_obs, uncert[0])

        prior_chain = PriorChain() \
            .push(UniformPrior('tec', -300., 300.)) \
            .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
            .push(HalfLaplacePrior('uncert', 0.2))

        ns = NestedSampler(log_likelihood, prior_chain,
                           sampler_name='slice',
                           tec_mean=lambda tec,**kwargs: tec,
                           tec2_mean=lambda tec, **kwargs: tec**2,
                           const_mean=lambda const, **kwargs: const,
                           const2_mean=lambda const, **kwargs: const ** 2
                           )

        results = ns(key=key,
                      num_live_points=500,
                      max_samples=1e5,
                      collect_samples=True,
                      only_marginalise=False,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                      sampler_kwargs=dict(depth=2, num_slices=3))
        return results

    def single_solve2(key, Y_obs, amp, const_mu, const_std):
        def log_likelihood(tec, const, uncert, **kwargs):
            phase = tec * (TEC_CONV / freqs)  + const #+ clock
            Y = jnp.concatenate([amp*jnp.cos(phase), amp*jnp.sin(phase)], axis=-1)
            return log_laplace(Y, Y_obs, uncert[0])

        prior_chain = PriorChain() \
            .push(UniformPrior('tec', -300., 300.)) \
            .push(NormalPrior('const', const_mu, const_std)) \
            .push(HalfLaplacePrior('uncert', 0.2))

        ns = NestedSampler(log_likelihood, prior_chain,
                           sampler_name='slice',
                           tec_mean=lambda tec,**kwargs: tec,
                           tec2_mean=lambda tec, **kwargs: tec**2,
                           const_mean=lambda const, **kwargs: const,
                           const2_mean=lambda const, **kwargs: const ** 2
                           )

        results = ns(key=key,
                      num_live_points=500,
                      max_samples=1e5,
                      collect_samples=True,
                      only_marginalise=False,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                      sampler_kwargs=dict(depth=2, num_slices=3))
        return results

    f1 = jit(single_solve1)
    f2 = jit(single_solve2)

    t0 = default_timer()
    results = list(map(f1,random.split(random.PRNGKey(235235), T), Y_obs, amp))
    const_mean = jnp.concatenate([result.marginalised['const_mean'] for result in results])
    # const_mean = windowed_mean(const_mean, 50, 'reflect')

    const2_mean = jnp.concatenate([result.marginalised['const2_mean'] for result in results])
    const_var = const2_mean - const_mean**2
    # const_var = windowed_mean(const_var, 50, 'reflect')/50.
    const_std = jnp.sqrt(const_var)

    results = list(map(f2,random.split(random.PRNGKey(9876), T), Y_obs, amp, const_mean, const_std))
    print("Time with no compile", default_timer() - t0)

    tec_mean = jnp.concatenate([result.marginalised['tec_mean'] for result in results])
    tec_std = jnp.sqrt(jnp.concatenate([result.marginalised['tec2_mean'] for result in results]) - tec_mean**2)
    plt.plot(tec)
    plt.errorbar(jnp.arange(T), tec_mean, yerr=tec_std,label='RBE-exact')
    plt.xlabel('timestep [30sec]')
    plt.ylabel('TEC [mTECU')
    plt.legend()
    plt.show()
    plt.plot(tec_mean-tec)
    plt.show()

    const_mean = jnp.concatenate([result.marginalised['const_mean'] for result in results])
    const_std = jnp.sqrt(jnp.concatenate([result.marginalised['const2_mean'] for result in results]) - const_mean ** 2)
    plt.errorbar(jnp.arange(T), const_mean, yerr=const_std, label='RBE-exact')
    plt.xlabel('timestep [30sec]')
    plt.ylabel('const [rad]')
    plt.legend()
    plt.show()
    ###

    # plot_diagnostics(results)
    # plot_cornerplot(results, vars=['tec', 'uncert'])


if __name__ == '__main__':
    main()
