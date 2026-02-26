import jax
import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.results import _sample_evidence


def test_nested_sampling_run_results(all_run_results):
    for name, (log_Z_true, results) in all_run_results:
        print(f"Checking {name}")
        # Use numpy testing

        assert not np.isnan(results.log_Z_mean)
        assert not np.isnan(results.log_Z_uncert)

        log_Z_samples = _sample_evidence(alpha=results.num_live_points_per_sample,
                                         log_L=results.log_L_samples,
                                         num_samples=1000)

        # Filter outliers
        select_mask = jnp.bitwise_and(
            log_Z_samples > jnp.percentile(log_Z_samples, 5),
            log_Z_samples < jnp.percentile(log_Z_samples, 95)
        )
        log_Z_samples = log_Z_samples[select_mask]

        import pylab as plt
        plt.hist(log_Z_samples, bins='auto')
        plt.show()
        log_Z_ensemble_mean = jnp.mean(log_Z_samples)
        log_Z_ensemble_std = jnp.std(log_Z_samples)
        np.testing.assert_allclose(log_Z_ensemble_mean, log_Z_true, atol=3.0 * results.log_Z_uncert)
        np.testing.assert_allclose(results.log_Z_mean, log_Z_ensemble_mean, atol=3. * results.log_Z_uncert)
        np.testing.assert_allclose(results.log_Z_uncert, log_Z_ensemble_std,
                                   atol=np.sqrt(results.log_Z_uncert ** 2 + log_Z_ensemble_std ** 2))


def test_gh108():
    import tensorflow_probability.substrates.jax as tfp
    from jax import random

    import psutil
    import os
    tfpd = tfp.distributions

    def nested_sampling(key):

        def prior_model():
            x = Prior(tfpd.Uniform(0., 1.)).realise()
            return x

        model = Model(prior_model=prior_model)

        ns = NestedSampler(model=model)

        state = ns.run()
        results = state.to_result()

    pid = os.getpid()
    python_process = psutil.Process(pid)

    ram_py = []
    jax.clear_caches()
    for i in range(3):
        nested_sampling(random.PRNGKey(i))
        jax.clear_caches()
        ram_py.append(python_process.memory_info()[0] / 2 ** 30)
        # print(ram_py[-1])

    # plt.plot(ram_py, 'k.-')
    # plt.xlabel('runs', fontsize=12)
    # plt.ylabel('python RAM usage(GB)', fontsize=12)
    # plt.show()

    np.testing.assert_allclose(ram_py, ram_py[0], atol=2e-3)

    ns_compile = jax.jit(nested_sampling).lower(random.PRNGKey(0)).compile()

    ram_py = []
    ram_py.append(python_process.memory_info()[0] / 2 ** 30)
    for i in range(3):
        ns_compile(random.PRNGKey(i))
        ram_py.append(python_process.memory_info()[0] / 2 ** 30)
        # print(ram_py[-1])

    np.testing.assert_allclose(ram_py, ram_py[0], atol=1e-6)
