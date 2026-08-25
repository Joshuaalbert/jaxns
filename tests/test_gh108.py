import gc
import os

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jaxctx.priors.prior import Prior

from jaxns.core import NestedSampler
from jaxns.model import Model


def _process_ram_gb():
    return psutil.Process(os.getpid()).memory_info()[0] / 2 ** 30


@jax.jit
def _baseline_compile_pressure(x):
    return jnp.sin(x) + 1.


def _measure_ram_growth(run_once, num_runs: int, *, clear_caches_between: bool) -> np.ndarray:
    ram_py = []
    if clear_caches_between:
        jax.clear_caches()
    for _ in range(num_runs):
        run_once()
        gc.collect()
        if clear_caches_between:
            jax.clear_caches()
        ram_py.append(_process_ram_gb())
    return np.asarray(ram_py)


def test_gh108():
    import tensorflow_probability.substrates.jax as tfp

    tfpd = tfp.distributions

    def nested_sampling():
        def prior_model():
            x = Prior(tfpd.Uniform(0., 1.), name='x').realise()
            return x

        model = Model(prior_model=prior_model)

        ns = NestedSampler(model=model)

        state = ns.run()
        results = state.to_result()
        return results

    def baseline_compile():
        _baseline_compile_pressure(jnp.arange(1024., dtype=jnp.float32)).block_until_ready()

    baseline_ram = _measure_ram_growth(baseline_compile, num_runs=3, clear_caches_between=True)
    nested_sampling_ram = _measure_ram_growth(nested_sampling, num_runs=3, clear_caches_between=True)
    baseline_drift = np.max(baseline_ram) - baseline_ram[0]
    nested_sampling_drift = np.max(nested_sampling_ram) - nested_sampling_ram[0]

    # RSS includes allocator and compiler bookkeeping that Python garbage
    # collection cannot release deterministically. A 20 MiB envelope still
    # catches the unbounded growth from GH108 while tolerating runner drift.
    ram_tolerance_gb = 2e-2
    assert nested_sampling_drift <= baseline_drift + ram_tolerance_gb

    ram_py = np.asarray([_process_ram_gb()])
    for _ in range(3):
        nested_sampling()
        gc.collect()
        ram_py = np.append(ram_py, _process_ram_gb())
    np.testing.assert_allclose(ram_py, ram_py[0], atol=ram_tolerance_gb)
