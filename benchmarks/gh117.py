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
    m = 10
    run_model_aot = jax.jit(lambda: run_model(max_samples=max_samples)).lower().compile()
    dt = []

    for _ in range(m):
        t0 = time.time()
        termination_reason = run_model_aot()
        termination_reason.block_until_ready()
        t1 = time.time()
        dt.append(t1 - t0)
    total_time = sum(dt)
    print(f"Avg. time taken: {total_time / m:.5f} seconds.")
    best_3 = sum(sorted(dt)[:3]) / 3.
    print(f"The best 3 of {m} runs took {best_3:.5f} seconds.")


# _inter_sync_shrinkage_process unroll=1
# get_sample_from_seed unroll=1
# Avg. time taken: 4.79353 seconds.
# The best 3 of 10 runs took 4.63075 seconds.

# _inter_sync_shrinkage_process unroll=2
# get_sample_from_seed unroll=1
# Avg. time taken: 5.04382 seconds.
# The best 3 of 10 runs took 4.74833 seconds.

# _inter_sync_shrinkage_process unroll=1
# get_sample_from_seed unroll=2
# Avg. time taken: 4.40303 seconds.
# The best 3 of 10 runs took 4.37935 seconds.

# With fix
# Avg. time taken: 0.00562 seconds.
# The best 3 of 10 runs took 0.00478 seconds.


if __name__ == '__main__':
    performance_benchmark()
