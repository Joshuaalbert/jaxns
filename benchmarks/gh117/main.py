import time

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import random
from jaxctx.priors.prior import Prior

from jaxns.core import NestedSampler
from jaxns.model import Model


tfpd = tfp.distributions


def run_model(max_samples: int):
    def prior_model():
        Prior(tfpd.Uniform(0., 1.)).realise()  # , name='x')
        return 0.

    model = Model(prior_model=prior_model)


    # Create the nested sampler class. In this case without any tuning.
    exact_ns = NestedSampler(model=model, max_samples=max_samples)

    state = exact_ns.run(random.PRNGKey(42))
    return state.termination_reason


def main():
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


# Before fix
# Avg. time taken: 4.40303 seconds.
# The best 3 of 10 runs took 4.37935 seconds.

# After fix
# Avg. time taken: 0.00562 seconds.
# The best 3 of 10 runs took 0.00478 seconds.

# 2.6.2
# Avg. time taken: 0.00012 seconds.
# The best 3 of 10 runs took 0.00007 seconds.


if __name__ == '__main__':
    main()
