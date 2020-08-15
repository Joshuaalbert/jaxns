from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, LaplacePrior, MVNPrior

from jax.scipy.linalg import solve_triangular
from jax import numpy as jnp, random


def generate_data():
    tec = 10. * random.normal(random.PRNGKey(0))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec / freqs * TEC_CONV - 0.2  # + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + 0.25 * random.normal(random.PRNGKey(1), shape=Y.shape)
    amp = jnp.ones_like(phase)
    return Y_obs, amp, tec, freqs


def main():
    Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    def log_likelihood(param, uncert, **kwargs):
        # tec = x[0]  # [:, 0]
        # uncert = x[1]  # [:, 1]
        # clock = x[2] * 1e-9
        # uncert = 0.25#x[2]
        tec = param[0]
        const = param[1]
        clock = param[2]*1e-9
        phase = tec * (TEC_CONV / freqs)  + const + clock *(jnp.pi*2)*freqs#+ clock
        Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        return log_normal(Y, Y_obs, uncert ** 2 * jnp.eye(2 * freqs.size))

    prior_chain = PriorChain() \
        .push(MVNPrior('param', jnp.zeros(3), jnp.diag(jnp.array([100., jnp.pi, 2.])**2))) \
        .push(UniformPrior('uncert', 0, 1))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='whitened_ellipsoid')

    results = ns(key=random.PRNGKey(0),
                      num_live_points=300,
                      max_samples=1e6,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=True)

    ###

    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
