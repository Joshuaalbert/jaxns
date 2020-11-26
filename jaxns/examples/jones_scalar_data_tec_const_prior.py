from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.prior_transforms import UniformPrior, PriorChain, DeltaPrior, LaplacePrior, HalfLaplacePrior
import pylab as plt
from jax.scipy.linalg import solve_triangular
from jax import jit, vmap, disable_jit
from jax import numpy as jnp, random
import numpy as np
from timeit import default_timer


def generate_data(key, tec, const, uncert, num_outliers=3):
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec / freqs * TEC_CONV + const# added a constant term
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + uncert * random.normal(key, shape=Y.shape)

    Y_obs = Y_obs.at[:num_outliers*2:2].set(1.*random.normal(key, shape=(num_outliers,)))
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
        Y = jnp.concatenate([amp*jnp.cos(phase), amp*jnp.sin(phase)], axis=-1)
        # log_prob = log_laplace(Y, Y_obs, uncert[0])
        log_prob = log_normal(Y, Y_obs, uncert[0])
        return log_prob

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -300., 300.)) \
        .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
        .push(HalfLaplacePrior('uncert', uncert))

    print("Probabilistic model:\n{}".format(prior_chain))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice',
                       tec_mean=lambda tec, **kw: tec,
                       const_mean=lambda const, **kw: const
                       )

    results = ns(key=key,
                      num_live_points=100,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                 sampler_kwargs=dict(depth=2, num_slices=3))

    dtec = tec - jnp.median(results.samples['tec'][:,0], axis=0)
    dconst = const - jnp.median(results.samples['const'][:,0], axis=0)
    duncert = uncert - jnp.median(results.samples['uncert'])
    return results, dtec, dconst, duncert

def main():
    # run_on_grid()
    t0 = default_timer()
    results,_,_,_ = run(random.PRNGKey(3245), 100.,-1.,0.2)
    print(results.efficiency)
    print(default_timer() - t0)

    t0 = default_timer()
    results, _, _, _ = run(random.PRNGKey(3245), 100., -1., 0.2)
    print(results.efficiency)
    print(default_timer() - t0)

    ###
    # print(results.marginalised['tec_mean'])
    # print(results.marginalised['const_mean'])
    plot_diagnostics(results)
    plot_cornerplot(results)


def run_on_grid():
    tec_array = np.linspace(-90., 90., 2)
    const_array = np.linspace(-jnp.pi, jnp.pi, 2)
    uncert_array = np.linspace(0.05, 1., 2)
    params = np.stack([v.flatten() for v in np.meshgrid(
        tec_array, const_array, uncert_array, indexing='ij')], axis=-1)
    dtec, dconst, duncert = [], [], []
    t0 = default_timer()
    for i in range(params.shape[0]):
        _, _dtec, _dconst, _duncert = run(random.PRNGKey(3245 + i), params[i, 0], params[i, 1], params[i, 2])
        dtec.append(_dtec)
        dconst.append(_dconst)
        duncert.append(_duncert)
    print("Time", default_timer() - t0)
    print("Time per run", (default_timer() - t0) / params.shape[0])
    dtec = np.array(dtec)
    dconst = np.array(dconst)
    duncert = np.array(duncert)
    plt.hist(dtec)
    plt.show()
    plt.hist(dconst)
    plt.show()
    plt.hist(duncert)
    plt.show()


if __name__ == '__main__':
    main()
