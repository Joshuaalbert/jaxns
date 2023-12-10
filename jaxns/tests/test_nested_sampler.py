import os

from jax import random, numpy as jnp

# Force 2 jax  hosts
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from jaxns.utils import sample_evidence
import numpy as np


def test_nested_sampling_run_results(all_run_results):
    for name, (log_Z_true, results) in all_run_results:
        print(f"Checking {name}")
        # Use numpy testing

        assert not np.isnan(results.log_Z_mean)
        np.testing.assert_allclose(results.log_Z_mean, log_Z_true, atol=2.0 * results.log_Z_uncert)

        log_Z_samples = sample_evidence(random.PRNGKey(42),
                                        results.num_live_points_per_sample,
                                        results.log_L_samples,
                                        S=1000)
        import pylab as plt
        plt.hist(log_Z_samples, bins='auto')
        plt.show()
        log_Z_ensemble_mean = jnp.mean(log_Z_samples)
        log_Z_ensemble_std = jnp.std(log_Z_samples)
        np.testing.assert_allclose(log_Z_ensemble_mean, log_Z_true, atol=2 * results.log_Z_uncert)
        np.testing.assert_allclose(results.log_Z_mean, log_Z_ensemble_mean, atol=1.75 * results.log_Z_uncert)
        np.testing.assert_allclose(results.log_Z_uncert, log_Z_ensemble_std, atol=results.log_Z_uncert)
