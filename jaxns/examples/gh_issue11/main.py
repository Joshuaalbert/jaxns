from jax import numpy as jnp, random
from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import NormalPrior, HalfLaplacePrior, DeterministicTransformPrior, PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics


def log_laplace(x, mean, uncert):
    dx = (x - mean) / uncert
    return - jnp.log(2. * uncert) - jnp.abs(dx)


def constrained_solve(freqs, key, Y_obs, amp, tec_mean, tec_std, const_mu, clock_mu):
    """
    Perform constrained solve with better priors, including outlier detection, for tec, const, and clock.

    Args:
        freqs: [Nf] frequencies
        key: PRNG key
        Y_obs: [2Nf] obs in real, imag order
        amp: [Nf] smoothed amplitudes
        tec_mean: prior tec mean
        tec_std: prior tec uncert
        const_mu: delta const prior
        clock_mu: delta clock prior

    Returns:
        tec_mean post. tec
        tec_std post. uncert
        phase_mean post. phase
    """
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_likelihood(Y, uncert, **kwargs):
        return jnp.sum(log_laplace(Y, Y_obs, uncert))

    def Y_transform(tec):
        phase = tec * (TEC_CONV / freqs) + const_mu + clock_mu * 1e-9 * (2. * jnp.pi * freqs)
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)])

    tec = NormalPrior('tec', tec_mean, tec_std)
    # increasing the scale slightly or using Uniform solves the problem immediately.
    # tec = UniformPrior('tec', -300., 300.)
    prior_chain = PriorChain() \
        .push(HalfLaplacePrior('uncert', 0.2)) \
        .push(DeterministicTransformPrior('Y', Y_transform, (freqs.size * 2,), tec, tracked=False))

    # prior_chain.test_prior(key, 500, log_likelihood=log_likelihood)
    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       tec_mean=lambda tec, **kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec ** 2
                       )

    results = ns(key=key,
                 num_live_points=100,
                 max_samples=1e5,
                 collect_samples=True,
                 termination_frac=0.01,
                 sampler_kwargs=dict(depth=2, num_slices=3))

    plot_diagnostics(results)
    plot_cornerplot(results)


def main():
    Y_obs = jnp.array([-0.00976067, 0.83416892, 0.21166622, 0.21733296, 0.49917201, 0.73596325,
                       0.8118896, 0.87541932, 0.71782782, 0.91435638, 0.89906082, 0.85542426,
                       0.7099701, 0.84393383, 0.46801934, 0.38645841, 0.87008515, - 0.74630165,
                       -0.91772599, 0.14500923, 0.06121048, -0.37406732, - 0.37266085, - 0.84109397,
                       -0.81871003, -0.02246329, -0.82197054, -0.83395193, -0.7169895, -0.49045282,
                       -0.37488623, - 0.2228393, -0.56222012, 0.09993957, 0.22805479, 0.37788126,
                       0.62048593, 0.43819284, 0.83751466, 0.88816856, 0.44812023, 0.65039152,
                       0.40357663, 1.00638594, 1.03110006, 0.98244398, 1.005249, 0.70243257])
    amp = jnp.array([0.81876821, 0.83447132, 0.84878629, 0.86180592, 0.87363988, 0.88441272,
                     0.89426194, 0.90333623, 0.91179397, 0.91980188, 0.92753402, 0.93517106,
                     0.94289996, 0.95091392, 0.95941279, 0.96860389, 0.97870318, 0.98993701,
                     1.00254431, 1.01677939, 1.03291532, 1.05124808, 1.07210152, 1.09583328])
    tec_mean = 42.74039154004245
    # I randomly had chosen tec_std=10.0 in bayes_gain_screens, and this resulting in this error which seems to be an
    # information gain problem. Satirically, When I looked at when value greater than 10 it started to work again I
    # I found somewhere in the interval [10.065, 10.125], making it quite coincidental to have chosen 10.0.
    tec_std = 10.0
    const_mean = 0.1959228515625
    clock_mean = -0.022598743438720703
    key = random.PRNGKey(2345236)
    freqs = jnp.linspace(121e6, 166e6, 24)
    constrained_solve(freqs, key, Y_obs, amp, tec_mean, tec_std, const_mean, clock_mean)


if __name__ == '__main__':
    main()
