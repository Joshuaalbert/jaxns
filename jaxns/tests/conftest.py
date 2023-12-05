import os
from time import monotonic_ns

# Force 2 jax  hosts
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import pylab as plt
import pytest
from jax import numpy as jnp, random
from jax._src.scipy.linalg import solve_triangular
from tensorflow_probability.substrates import jax as tfp

from jaxns.model.bases import PriorModelGen
from jaxns.model.model import Model
from jaxns.model.prior import Prior
from jaxns.nested_sampler.standard_static import StandardStaticNestedSampler
from jaxns.plotting import plot_diagnostics
from jaxns.public import BaseNestedSampler
from jaxns.samplers.multi_ellipsoidal_samplers import MultiEllipsoidalSampler
from jaxns.types import TerminationCondition
from jaxns.utils import bruteforce_evidence, sample_evidence, summary

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
    return model


@pytest.fixture(scope='package')
def basic_run_results(basic_model):
    model = basic_model

    exact_ns = BaseNestedSampler(
        model=model,
        num_live_points=50,
        num_parallel_workers=1,
        max_samples=1000
    )
    with Timer():
        termination_reason, state = exact_ns(random.PRNGKey(42),
                                             term_cond=TerminationCondition(live_evidence_frac=1e-4))
        results = exact_ns.to_results(termination_reason=termination_reason, state=state)

        # exact_ns.plot_diagnostics(results)
        exact_ns.summary(results)
        # exact_ns.plot_cornerplot(results)

    log_Z_true = bruteforce_evidence(model=model, S=200)
    return log_Z_true, state, results


@pytest.fixture(scope='package')
def basic2_results():
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

    def exact_X(L):
        return (1. - L) ** (1. / n)

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    ns = BaseNestedSampler(
        model=model,
        num_live_points=50,
        num_parallel_workers=1,
        max_samples=1000
    )

    termination_reason, state = ns(random.PRNGKey(42),
                                   term_cond=TerminationCondition(live_evidence_frac=1e-4))
    results = ns.to_results(termination_reason=termination_reason, state=state)
    X_exact = exact_X(jnp.exp(results.log_L_samples))
    return log_Z_true, state, results, X_exact


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
    return model


@pytest.fixture(scope='package')
def plateau_run_results():
    def log_likelihood(x):
        return 0.

    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1))
        return x

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    exact_ns = BaseNestedSampler(
        model=model,
        num_live_points=50,
        max_samples=1000
    )

    termination_reason, state = exact_ns(
        random.PRNGKey(42),
        term_cond=TerminationCondition(live_evidence_frac=1e-4)
    )

    results = exact_ns.to_results(termination_reason=termination_reason, state=state)
    exact_ns.summary(results)
    exact_ns.plot_diagnostics(results)

    log_Z_samples = sample_evidence(random.PRNGKey(42),
                                    results.num_live_points_per_sample,
                                    results.log_L_samples, S=1000)
    plt.hist(log_Z_samples, bins='auto')
    plt.show()

    true_log_Z_uncert = jnp.std(log_Z_samples)
    print(f"true logZ uncert: {true_log_Z_uncert}")
    log_Z_true = 0.
    return log_Z_true, true_log_Z_uncert, state, results


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

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)
    # not super happy with this being 1.58 and being off by like 0.1. Probably related to the ESS.
    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    print(f"True post mu:{post_mu}")
    print(f"True log Z: {true_logZ}")

    def prior_model() -> PriorModelGen:
        x = yield Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x')
        return x

    def log_likelihood(x):
        return log_normal(x, data_mu, data_cov)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    ns = BaseNestedSampler(
        model=model,
        num_live_points=200,
        num_parallel_workers=1,
        max_samples=40000
    )

    return true_logZ, model, ns


@pytest.fixture(scope='package')
def basic_mvn_run_results(basic_mvn_model):
    true_logZ, model, ns = basic_mvn_model

    with Timer():
        termination_reason, state = ns(random.PRNGKey(42),
                                       term_cond=TerminationCondition(live_evidence_frac=1e-5))
        results = ns.to_results(termination_reason=termination_reason, state=state)
        ns.summary(results)
    ns.plot_diagnostics(results)
    return true_logZ, state, results


@pytest.fixture(scope='package')
def basic_mvn_run_results_parallel(basic_mvn_model):
    true_logZ, model, _ = basic_mvn_model

    ns = BaseNestedSampler(
        model=model,
        num_live_points=100,
        num_parallel_workers=2,
        max_samples=30000
    )

    with Timer():
        termination_reason, state = ns(key=random.PRNGKey(42), term_cond=TerminationCondition())
        results = ns.to_results(termination_reason=termination_reason, state=state)
        ns.summary(results)
    ns.plot_diagnostics(results)
    return true_logZ, state, results


@pytest.fixture(scope='package')
def multiellipsoidal_mvn_run_results(basic_mvn_model):
    true_logZ, model, ns = basic_mvn_model

    # model.sanity_check(random.PRNGKey(42), S=100)
    ns = StandardStaticNestedSampler(
        init_efficiency_threshold=0.1,
        model=model,
        num_live_points=100,
        num_parallel_workers=1,
        max_samples=40000,
        sampler=MultiEllipsoidalSampler(model=model, depth=0)
    )
    with Timer():
        termination_reason, state = ns._run(random.PRNGKey(41),
                                            term_cond=TerminationCondition(live_evidence_frac=1e-5))
        results = ns._to_results(termination_reason=termination_reason, state=state, trim=True)
        summary(results)
    plot_diagnostics(results)
    return true_logZ, state, results
