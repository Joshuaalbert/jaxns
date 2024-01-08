import jax
import tensorflow_probability.substrates.jax as tfp
from jax import random

from jaxns import Model, Prior

tfpd = tfp.distributions


def run_model(max_samples: int):
    def log_likelihood(x):
        return 0.

    def prior_model():
        x = yield Prior(tfpd.Uniform(0., 1.), name='x')
        return x

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    from jaxns import DefaultNestedSampler

    # Create the nested sampler class. In this case without any tuning.
    exact_ns = DefaultNestedSampler(model=model, max_samples=max_samples)

    termination_reason, state = exact_ns(random.PRNGKey(42))
    return termination_reason


def performance_benchmark():
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        termination_reason = run_model(max_samples=100000000)
        termination_reason.block_until_ready()


if __name__ == '__main__':
    performance_benchmark()