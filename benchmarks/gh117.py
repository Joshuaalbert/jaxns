import time

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import random

from jaxns import Model, Prior

tfpd = tfp.distributions


def run_model(max_samples: int):
    def log_likelihood(x):
        return 0.

    def prior_model():
        x = yield Prior(tfpd.Uniform(0., 1.))  # , name='x')
        return x

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    from jaxns import DefaultNestedSampler

    # Create the nested sampler class. In this case without any tuning.
    exact_ns = DefaultNestedSampler(model=model, max_samples=max_samples)

    termination_reason, state = exact_ns(random.PRNGKey(42))
    return termination_reason


def performance_benchmark():
    max_samples = int(1e7)
    m = 3
    run_model_aot = jax.jit(lambda: run_model(max_samples=max_samples)).lower().compile()
    t0 = time.time()
    for _ in range(m):
        termination_reason = run_model_aot()
        termination_reason.block_until_ready()
    print(f"Time taken: {(time.time() - t0) / m:.5f} seconds.")


if __name__ == '__main__':
    performance_benchmark()
