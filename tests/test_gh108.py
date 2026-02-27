import jax
import numpy as np
from jaxctx.priors.prior import Prior

from jaxns.core import NestedSampler
from jaxns.model import Model


def test_gh108():
    import tensorflow_probability.substrates.jax as tfp

    import psutil
    import os
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

    pid = os.getpid()
    python_process = psutil.Process(pid)

    ram_py = []
    jax.clear_caches()
    for i in range(3):
        nested_sampling()
        jax.clear_caches()
        ram_py.append(python_process.memory_info()[0] / 2 ** 30)
        # print(ram_py[-1])

    # plt.plot(ram_py, 'k.-')
    # plt.xlabel('runs', fontsize=12)
    # plt.ylabel('python RAM usage(GB)', fontsize=12)
    # plt.show()

    np.testing.assert_allclose(ram_py, ram_py[0], atol=2e-3)

    ram_py = [python_process.memory_info()[0] / 2 ** 30]
    for i in range(3):
        nested_sampling()
        ram_py.append(python_process.memory_info()[0] / 2 ** 30)

    np.testing.assert_allclose(ram_py, ram_py[0], atol=2e-3)
