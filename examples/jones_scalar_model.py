from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, marginalise_static
from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, DeltaPrior
from jax import jit
from jax import numpy as jnp, random, flatten_util, jacfwd
import pylab as plt

TEC_CONV = -8.4479745e6


def wrap(phi):
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


def generate_data(key, n, uncert):
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
    freqs = jnp.linspace(121e6, 166e6, 24)
    tec = jnp.arange(n)*10+100.
    const = 2.#rad
    clock = 0.5#ns
    phase = tec[:, None] * (TEC_CONV / freqs) + const + (1e-9 * freqs) * jnp.pi * clock
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + uncert * random.normal(key, shape=Y.shape)
    phase_obs = jnp.arctan2(Y_obs[..., freqs.size:], Y_obs[..., :freqs.size])
    return Y_obs, phase_obs, freqs, tec, const, clock, uncert


def log_normal(x, mean, scale):
    dx = (x - mean) / scale
    return -0.5 * jnp.log(2. * jnp.pi) - jnp.log(scale)  - 0.5 * dx * dx


def log_laplace(x, mean, scale):
    dx = jnp.abs(x - mean) / scale
    return - jnp.log(2. * scale) - dx


def log_cauchy(x, mean, scale):
    dx = (x - mean) / scale
    return -jnp.log(jnp.pi) - jnp.log(scale) - jnp.log1p(dx ** 2)

@jit
def solve_with_clock(key, freqs, Y_obs, true_clock, true_tec, true_const, true_uncert):
    def log_likelihood(tec, const, clock, uncert, **kwargs):
        """
        Attentional mechanism for outliers.
        weight(y) = L(y)/sum_y L(y)
        P(Y) = prod_y weight(y) L(y)
        log P(Y) = sum_y log weight(y) + log L(y) = sum_y 2. * log L(y) - log sum_y L(y)
        Args:
            tec:
            const:
            uncert:
            **kwargs:

        Returns:

        """
        phase = tec[:, None] * (TEC_CONV / freqs) + const + clock * (2e-9 * freqs) * jnp.pi
        Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        logL = log_normal(Y, Y_obs, uncert)
        return jnp.sum(logL)

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -300. * jnp.ones(Y_obs.shape[0]), 300. * jnp.ones(Y_obs.shape[0]))) \
        .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
        .push(UniformPrior('clock', -1., 1.)) \
        .push(HalfLaplacePrior('uncert', 0.5))

    # print("Probabilistic model:\n{}".format(prior_chain))

    def var_tec(tec, const, clock, uncert, **kwargs):
        return (tec - true_tec) ** 2

    def var_const(tec, const, clock, uncert, **kwargs):
        return (const - true_const) ** 2

    def var_clock(tec, const, clock, uncert, **kwargs):
        return (clock - true_clock) ** 2

    def var_uncert(tec, const, clock, uncert, **kwargs):
        return (uncert - true_uncert) ** 2

    ns = NestedSampler(log_likelihood, prior_chain,
                       num_live_points=100*prior_chain.U_ndims,
                       marginalised=dict(var_tec=var_tec, var_const=var_const, var_clock=var_clock, var_uncert=var_uncert)
                       )

    results = ns(key, 0.001)

    return results, (results.marginalised['var_tec'].mean(), results.marginalised['var_const'],
                     results.marginalised['var_clock'], results.marginalised['var_uncert'])


def run(key):
    n_array = [1,2,3,4]
    for n in n_array:
        for uncert in [0.4]:
            print(f"Solving with clock, uncert={uncert}, n={n}")
            key,data_key = random.split(key)
            Y_obs, phase_obs, freqs, true_tec, true_const, true_clock, true_uncert = generate_data(data_key, n, uncert)

            results, variance = solve_with_clock(key, freqs, Y_obs, true_clock, true_tec, true_const,
                                                 true_uncert)
            summary(results)
            plot_diagnostics(results)
            plot_cornerplot(results)


def main():
    run(random.PRNGKey(3245))


if __name__ == '__main__':
    main()
