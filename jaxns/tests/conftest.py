import os
from time import monotonic_ns

import jax

# Force 2 jax  hosts
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import pytest
from jax import numpy as jnp, random
from jax._src.scipy.linalg import solve_triangular
from tensorflow_probability.substrates import jax as tfp

from jaxns.framework.bases import PriorModelGen
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns.nested_sampler.standard_static import StandardStaticNestedSampler
from jaxns.public import DefaultNestedSampler
from jaxns.samplers.multi_ellipsoidal_samplers import MultiEllipsoidalSampler
from jaxns.internals.types import TerminationCondition
from jaxns.utils import bruteforce_evidence, summary

# from jaxns.nested_sampler import ApproximateNestedSampler, ExactNestedSampler

tfpd = tfp.distributions


class Timer:
    def __enter__(self):
        self.t0 = monotonic_ns()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time to execute: {(monotonic_ns() - self.t0) / 1e9} seconds.")


@pytest.fixture(scope='package')
def basic_model():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1), name='x')
        return x

    def log_likelihood(x):
        return - jnp.sum(x ** 2)

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    log_Z_true = bruteforce_evidence(model=model, S=200)
    return model, log_Z_true


@pytest.fixture(scope='package')
def basic_run_results(basic_model):
    model, log_Z_true = basic_model

    ns = DefaultNestedSampler(model=model, max_samples=1000)
    ns_jit = jax.jit(lambda key: ns(key))
    ns_compiled = ns_jit.lower(random.PRNGKey(42)).compile()
    with Timer():
        termination_reason, state = ns_compiled(random.PRNGKey(42))
        termination_reason.block_until_ready()
    results = ns.to_results(termination_reason=termination_reason, state=state)
    # exact_ns.plot_diagnostics(results)
    ns.summary(results)
    # exact_ns.plot_cornerplot(results)
    return log_Z_true, results


@pytest.fixture(scope='package')
def basic2_model():
    n = 2

    # Prior is uniform in U[0,1]
    # Likelihood is 1 - x**n
    # Z = 1 - 1/n+1

    log_Z_true = jnp.log(1. - 1. / (n + 1))

    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1))
        return x

    def log_likelihood(x):
        return jnp.log(1. - x ** n)

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    return model, log_Z_true


@pytest.fixture(scope='package')
def basic2_run_results(basic2_model):
    model, log_Z_true = basic2_model

    ns = DefaultNestedSampler(model=model, max_samples=1000)

    ns_jit = jax.jit(lambda key: ns(key))
    ns_compiled = ns_jit.lower(random.PRNGKey(42)).compile()
    with Timer():
        termination_reason, state = ns_compiled(random.PRNGKey(42))
        termination_reason.block_until_ready()
    results = ns.to_results(termination_reason=termination_reason, state=state)
    # exact_ns.plot_diagnostics(results)
    ns.summary(results)
    # exact_ns.plot_cornerplot(results)
    return log_Z_true, results


@pytest.fixture(scope='package')
def basic3_model():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -z ** 2

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    log_Z_true = bruteforce_evidence(model=model, S=200)
    return model, log_Z_true


@pytest.fixture(scope='package')
def basic3_run_results(basic3_model):
    model, log_Z_true = basic3_model

    ns = DefaultNestedSampler(model=model, max_samples=1000)

    ns_jit = jax.jit(lambda key: ns(key))
    ns_compiled = ns_jit.lower(random.PRNGKey(42)).compile()
    with Timer():
        termination_reason, state = ns_compiled(random.PRNGKey(42))
        termination_reason.block_until_ready()
    results = ns.to_results(termination_reason=termination_reason, state=state)
    # exact_ns.plot_diagnostics(results)
    ns.summary(results)
    # exact_ns.plot_cornerplot(results)
    return log_Z_true, results


@pytest.fixture(scope='package')
def plateau_model():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1))
        return x

    def log_likelihood(x):
        return 0.

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    log_Z_true = jnp.asarray(0.)
    return model, log_Z_true


@pytest.fixture(scope='package')
def plateau_run_results(plateau_model):
    model, log_Z_true = plateau_model
    ns = DefaultNestedSampler(
        model=model,
        max_samples=1000
    )

    ns_jit = jax.jit(lambda key: ns(key))
    ns_compiled = ns_jit.lower(random.PRNGKey(42)).compile()
    with Timer():
        termination_reason, state = ns_compiled(random.PRNGKey(42))
        termination_reason.block_until_ready()
    results = ns.to_results(termination_reason=termination_reason, state=state)
    # exact_ns.plot_diagnostics(results)
    ns.summary(results)
    # exact_ns.plot_cornerplot(results)

    return log_Z_true, results


@pytest.fixture(scope='package')
def basic_mvn_model():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
            - 0.5 * dx @ dx

    ndims = 8
    prior_mu = 15 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.99, data_cov)

    log_Z_true = log_normal(data_mu, prior_mu, prior_cov + data_cov)
    # not super happy with this being 1.58 and being off by like 0.1. Probably related to the ESS.
    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    print(f"True post mu:{post_mu}")
    print(f"True log Z: {log_Z_true}")

    def prior_model() -> PriorModelGen:
        x = yield Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x')
        return x

    def log_likelihood(x):
        return tfpd.MultivariateNormalTriL(loc=data_mu, scale_tril=jnp.linalg.cholesky(data_cov)).log_prob(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    return log_Z_true, model


@pytest.fixture(scope='package')
def basic_mvn_run_results(basic_mvn_model):
    log_Z_true, model = basic_mvn_model

    ns = DefaultNestedSampler(model=model, max_samples=50000)

    ns_jit = jax.jit(lambda key: ns(key))
    ns_compiled = ns_jit.lower(random.PRNGKey(42)).compile()
    with Timer():
        termination_reason, state = ns_compiled(random.PRNGKey(42))
        termination_reason.block_until_ready()
    results = ns.to_results(termination_reason=termination_reason, state=state)
    # exact_ns.plot_diagnostics(results)
    ns.summary(results)
    # exact_ns.plot_cornerplot(results)

    return log_Z_true, results


@pytest.fixture(scope='package')
def basic_mvn_run_results_parallel(basic_mvn_model):
    log_Z_true, model = basic_mvn_model

    ns = DefaultNestedSampler(model=model, max_samples=50000, num_parallel_workers=2)

    ns_jit = jax.jit(lambda key: ns(key))
    ns_compiled = ns_jit.lower(random.PRNGKey(42)).compile()
    with Timer():
        termination_reason, state = ns_compiled(random.PRNGKey(42))
        termination_reason.block_until_ready()
    results = ns.to_results(termination_reason=termination_reason, state=state)
    ns.plot_diagnostics(results)
    ns.summary(results)
    # exact_ns.plot_cornerplot(results)

    return log_Z_true, results


@pytest.fixture(scope='package')
def multiellipsoidal_mvn_run_results(basic_mvn_model):
    log_Z_true, model = basic_mvn_model

    # model.sanity_check(random.PRNGKey(42), S=100)
    ns = StandardStaticNestedSampler(
        init_efficiency_threshold=0.1,
        model=model,
        num_live_points=model.U_ndims * 20,
        num_parallel_workers=1,
        max_samples=50000,
        sampler=MultiEllipsoidalSampler(model=model, depth=0)
    )
    ns_jit = jax.jit(lambda key: ns._run(key, term_cond=TerminationCondition()))
    ns_compiled = ns_jit.lower(random.PRNGKey(42)).compile()
    with Timer():
        termination_reason, state = ns_compiled(random.PRNGKey(42))
        termination_reason.block_until_ready()
    results = ns._to_results(termination_reason=termination_reason, state=state, trim=True)
    # plot_diagnostics(results)
    summary(results)
    # plot_cornerplot(results)

    return log_Z_true, results


@pytest.fixture(scope='package')
def all_run_results(
        basic_run_results,
        basic2_run_results,
        basic3_run_results,
        plateau_run_results,
        basic_mvn_run_results,
        # basic_mvn_run_results_parallel,
        multiellipsoidal_mvn_run_results
):
    # Return tuples with names
    return [
        ('basic', basic_run_results),
        ('basic2', basic2_run_results),
        ('basic3', basic3_run_results),
        ('plateau', plateau_run_results),
        ('basic_mvn', basic_mvn_run_results),
        # ('basic_mvn_parallel', basic_mvn_run_results_parallel),
        ('multiellipsoidal_mvn', multiellipsoidal_mvn_run_results)
    ]
