from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, LaplacePrior, MVNDiagPrior, SymmetricUniformWalkPrior

from jax.scipy.linalg import solve_triangular
from jax import jit, vmap
from jax import numpy as jnp, random
import pylab as plt
from timeit import default_timer


def generate_data():
    T = 10
    # var = T
    dw = jnp.sqrt(T)
    phase = dw * jnp.cumsum(random.normal(random.PRNGKey(87658), shape=(T,)))
    phase = tec[:, None] / freqs * TEC_CONV  # + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * random.normal(random.PRNGKey(75467), shape=Y.shape)
    # Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape_dict)
    Sigma = 0.25 ** 2 * jnp.eye(48)
    amp = jnp.ones_like(phase)
    return Sigma, T, Y_obs, amp, tec, freqs


def main():
    Sigma, T, Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz


    def log_normal(x, mean, uncert):
        dx = (x - mean)/uncert
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - x.size * jnp.log(uncert) \
               - 0.5 * dx @ dx

    def log_likelihood(tec, uncert, **kwargs):
        # tec = x[0]  # [:, 0]
        # uncert = x[1]  # [:, 1]
        # clock = x[2] * 1e-9
        # uncert = 0.25#x[2]
        phase = tec * (TEC_CONV / freqs)  # + clock *(jnp.pi*2)*freqs#+ clock
        Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        return jnp.sum(vmap(lambda Y, Y_obs: log_normal(Y, Y_obs, uncert))(Y, Y_obs))

    # prior_transform = MVNDiagPrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    # prior_transform = LaplacePrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    prior_chain = PriorChain() \
        .push(SymmetricUniformWalkPrior('tec', T, UniformPrior('tec0', -100., 100.),
                                        UniformPrior('half_width', 0., 20.))) \
        .push(UniformPrior('uncert', 0.01, 1.))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice',
                       tec_mean=lambda tec,**kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec**2)

    @jit
    def run(key):
        return ns(key=key,
                  num_live_points=1000,
                  max_samples=1e5,
                  collect_samples=True,
                  termination_frac=0.01,
                  stoachastic_uncertainty=False,
                  sampler_kwargs=dict(depth=5, num_slices=1))

    # with disable_jit():
    t0 = default_timer()
    results = run(random.PRNGKey(0))
    print("Time with compile efficiency normalised", results.efficiency * (default_timer() - t0))
    print("Time with compile", default_timer() - t0)
    t0 = default_timer()
    results = run(random.PRNGKey(1))
    print("Time no compile efficiency normalised", results.efficiency * (default_timer() - t0))
    print("Time no compile", default_timer() - t0)

    tec_mean = results.marginalised['tec_mean'][:, 0]
    tec_std = jnp.sqrt(results.marginalised['tec2_mean'][:,0] - tec_mean**2)
    plt.plot(tec)
    plt.errorbar(jnp.arange(T), tec_mean, yerr=tec_std)
    plt.show()
    plt.plot(tec_mean-tec)
    plt.show()
    ###

    plot_diagnostics(results)
    plot_cornerplot(results, vars=['tec0', 'half_width', 'uncert'])


if __name__ == '__main__':
    main()
