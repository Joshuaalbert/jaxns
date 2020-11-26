from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, LaplacePrior, HalfLaplacePrior

from jax.scipy.linalg import solve_triangular
from jax import jit, vmap, disable_jit
from jax import numpy as jnp, random
from timeit import default_timer


def generate_data():
    tec = 50. * random.normal(random.PRNGKey(0))
    print("ground truth tec", tec)
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec / freqs * TEC_CONV
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + 0.5 * random.normal(random.PRNGKey(1452), shape=Y.shape)
    amp = jnp.ones_like(phase)
    return Y_obs, amp, tec, freqs


def main():
    Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_normal(x, mean, scale):
        dx = (x - mean)/scale
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - x.size*jnp.log(scale) \
               - 0.5 * dx @ dx

    def log_likelihood(tec, uncert, **kwargs):
        phase = tec * (TEC_CONV / freqs)
        Y = jnp.concatenate([amp*jnp.cos(phase), amp*jnp.sin(phase)], axis=-1)
        log_prob = log_normal(Y, Y_obs, uncert[0])
        return log_prob

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -100., 100.)) \
        .push(HalfLaplacePrior('uncert', 0.5))

    print("Probabilistic model:\n{}".format(prior_chain))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice',
                       tec_mean=lambda tec, **kw: tec#I would like to this function over the posterior
                       )

    run = jit(lambda key: ns(key=key,
                      num_live_points=1000,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                 sampler_kwargs=dict(depth=4, num_slices=1)))

    t0 = default_timer()
    results = run(random.PRNGKey(2364))
    print(results.efficiency)
    print("Time compile", default_timer() - t0)

    t0 = default_timer()
    results = run(random.PRNGKey(1324))
    print(results.efficiency)
    print("Time no compile",default_timer() - t0)


    ###
    print(results.marginalised['tec_mean'])
    plot_diagnostics(results)
    plot_cornerplot(results)


if __name__ == '__main__':
    main()
