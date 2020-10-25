from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, LaplacePrior, MVNDiagPrior

from jax.scipy.linalg import solve_triangular
import pylab as plt
from jax import numpy as jnp, random, vmap, disable_jit


def generate_data():
    T=5
    tec = 10. * jnp.cumsum(random.normal(random.PRNGKey(0), shape=(T,)))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec[:,None] / freqs * TEC_CONV #- 0.2  # + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + 0.25 * random.normal(random.PRNGKey(1), shape=Y.shape)
    amp = jnp.ones_like(phase)
    return Y_obs, amp, tec, freqs


def main():
    Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    T = tec.size

    def log_mvnormal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    def log_normal(x, mean, uncert):
        dx = (x - mean)/uncert
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(uncert)) \
               - 0.5 * dx @ dx

    def log_likelihood(tec, const, uncert, **kwargs):
        phase = tec[:, None] * (TEC_CONV / freqs)#  + const# + clock *(jnp.pi*2)*freqs#+ clock
        Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        return jnp.sum(vmap(lambda Y, Y_obs: log_normal(Y, Y_obs, uncert * jnp.ones(2 * freqs.size)))(Y,Y_obs))

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -200.*jnp.ones(T), 200.*jnp.ones(T))) \
        .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
        .push(UniformPrior('uncert', 0, 1))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice',
                       tec_mean=lambda tec,**kwargs: tec,
                       const_mean=lambda const,**kwargs: const)

    results = ns(key=random.PRNGKey(0),
                      num_live_points=300,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                 sampler_kwargs=dict(depth=3, num_slices=1))

    ###
    plt.plot(results.marginalised['tec_mean'])
    plt.plot(tec)
    plt.show()
    print(tec, results.marginalised['tec_mean'])
    print(results.marginalised['const_mean'])

    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
