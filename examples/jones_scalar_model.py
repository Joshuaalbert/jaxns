from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary
from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, CauchyPrior
from jax import jit
from jax import numpy as jnp, random

TEC_CONV = -8.4479745#rad*MHz/mTECU
CLOCK_CONV = (2e-3*jnp.pi)#rad/MHz/ns
CUBIC_TERM = 0.1*140.**3#rad*MHz^3/arb.

def wrap(phi):
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi

def generate_data(key, uncert):
    """
    Generate gain data where the phase have a clock const and tec component, where the clock are shared for all data,
    and the tec are different:

    for i=1..n
        phase[i] = tec[i] * tec_conv + clock * clock_conv + const

    then the gains are:

        gains[i] = {cos(phase[i]), sin(phase[i])}

    Args:
        key:
        n: number of consequtive gains, i.e. tec values with shared const and clock terms.

    Returns:
        Y_obs, phase_obs, freqs
    """
    freqs = jnp.linspace(121, 166, 24)#MHz
    tec = 90.#mTECU
    const = 2.#rad
    clock = 0.5#ns
    phase = tec * (TEC_CONV / freqs) + clock * (CLOCK_CONV * freqs) + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + uncert * random.normal(key, shape=Y.shape)
    phase_obs = jnp.arctan2(Y_obs[..., freqs.size:], Y_obs[..., :freqs.size])
    return Y_obs, phase_obs, freqs


def log_normal(x, mean, scale):
    dx = (x - mean) / scale
    return -0.5 * jnp.log(2. * jnp.pi) - jnp.log(scale)  - 0.5 * dx * dx

@jit
def solve_with_clock(key, freqs, Y_obs):
    def log_likelihood(tec, const, clock, uncert, **kwargs):
        phase = tec * (TEC_CONV / freqs) + const + clock * (CLOCK_CONV * freqs)# + cubic * (CUBIC_TERM / freqs**3)
        Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        logL = log_normal(Y, Y_obs, uncert)
        return jnp.sum(logL)

    prior_chain = PriorChain(CauchyPrior('tec', 0., 100.),
                             UniformPrior('const', -jnp.pi, jnp.pi),
                             UniformPrior('clock', -2., 2.),
                             # CauchyPrior('cubic', 0., 0.5),
                             HalfLaplacePrior('uncert', 0.5))

    #logZ=78.16 +- 0.075 with cubic
    #logZ=78.253 +- 0.084 without cubic
    ns = NestedSampler(log_likelihood, prior_chain,
                       num_live_points=prior_chain.U_ndims*500)

    results = ns(key)

    return results


def run(key):
    key,data_key = random.split(key)
    Y_obs, phase_obs, freqs = generate_data(data_key, 0.05)

    results = solve_with_clock(key, freqs, Y_obs)
    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)


def main():
    run(random.PRNGKey(3245))


if __name__ == '__main__':
    main()
