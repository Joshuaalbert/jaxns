import jax
import numpy as np
from jax import random, numpy as jnp

from jaxns.nested_samplers.common.types import TerminationCondition
from jaxns.utils import sample_evidence


def test_nested_sampling_run_results(all_run_results):
    for name, (log_Z_true, results) in all_run_results:
        print(f"Checking {name}")
        # Use numpy testing

        assert not np.isnan(results.log_Z_mean)
        assert not np.isnan(results.log_Z_uncert)

        log_Z_samples = sample_evidence(random.PRNGKey(42),
                                        results.num_live_points_per_sample,
                                        results.log_L_samples,
                                        S=1000)

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

    from jaxns import NestedSampler
    from jaxns import Model
    from jaxns import Prior

    import psutil
    import os
    tfpd = tfp.distributions

    def nested_sampling(key):

        def log_likelihood(theta):
            return 0.

        def prior_model():
            x = yield Prior(tfpd.Uniform(0., 1.))
            return x

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

        ns = NestedSampler(model=model, max_samples=1e4)

        termination_reason, state = ns(key, term_cond=TerminationCondition())
        results = ns.to_results(termination_reason=termination_reason, state=state, trim=False)

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
