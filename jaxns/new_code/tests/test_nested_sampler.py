import tensorflow_probability.substrates.jax as tfp

from jaxns.new_code.model import Model
from jaxns.new_code.nested_sampler import ApproximateNestedSampler, ExactNestedSampler
from jaxns.new_code.prior import PriorModelGen, Prior
from jax import random, numpy as jnp

from jaxns.new_code.types import TerminationCondition

tfpd = tfp.distributions


def test_approximate_nested_sampler():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -z ** 2

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    approx_ns = ApproximateNestedSampler(model=model, num_live_points=50, num_parallel_samplers=1,
                                                         max_samples=1000)
    termination_reason, state = approx_ns(random.PRNGKey(42),
                                          term_cond=TerminationCondition(live_evidence_frac=1e-4))
    # print(termination_reason)
    # print(state)
    results = approx_ns.to_results(state, termination_reason)
    print(results)


def test_exact_nested_sampler():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -z ** 2

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    exact_ns = ExactNestedSampler(model=model, num_live_points=50, num_parallel_samplers=1,
                                                         max_samples=1000)
    termination_reason, state = exact_ns(random.PRNGKey(42),
                                          term_cond=TerminationCondition(live_evidence_frac=1e-4))
    # print(termination_reason)
    # print(state)
    results = exact_ns.to_results(state, termination_reason)
    print(results)
