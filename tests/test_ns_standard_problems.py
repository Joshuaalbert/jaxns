import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.results import _sample_evidence
from jaxns.utils import bruteforce_evidence


pytestmark = pytest.mark.skip(reason="Legacy nested-sampling standard-problem tests are incompatible with the in-progress v3 rewrite.")

tfpd = tfp.distributions


def _basic_model_case():
    def prior_model():
        x = Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
        return -jnp.sum(x ** 2)

    model = Model(prior_model=prior_model)
    log_Z_true = bruteforce_evidence(model=model, grid_res=200)
    return model, log_Z_true, {}



def _basic2_model_case():
    n = 2
    log_Z_true = jnp.log(1.0 - 1.0 / (n + 1))

    def prior_model():
        x = Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
        return jnp.log(1.0 - x ** n)

    model = Model(prior_model=prior_model)
    return model, log_Z_true, {'max_samples': 1000}


def _basic3_model_case():
    def prior_model():
        x = Prior(tfpd.Uniform(low=0, high=2), name='x').realise()
        y = Prior(tfpd.Normal(loc=2, scale=x), name='y').realise()
        z = x + y
        return -z ** 2

    model = Model(prior_model=prior_model)
    log_Z_true = bruteforce_evidence(model=model, grid_res=500)
    return model, log_Z_true, {'max_samples': 2000}


def _plateau_model_case():
    def prior_model():
        Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
        return 0.0

    model = Model(prior_model=prior_model)
    log_Z_true = jnp.asarray(0.0)
    return model, log_Z_true, {'max_samples': 1000}


def _basic_mvn_model_case():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * dx @ dx

    ndims = 8
    prior_mu = 15 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0.0, 0.99, data_cov)

    log_Z_true = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    def prior_model():
        x = Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x'
        ).realise()
        return tfpd.MultivariateNormalTriL(loc=data_mu, scale_tril=jnp.linalg.cholesky(data_cov)).log_prob(x)

    model = Model(prior_model=prior_model)
    return model, log_Z_true, {'max_samples': 100000}


STANDARD_PROBLEM_CASES = [
    ('basic', _basic_model_case),
    # ('basic2', _basic2_model_case),
    # ('basic3', _basic3_model_case),
    # ('plateau', _plateau_model_case),
    # ('basic_mvn', _basic_mvn_model_case),
]



def test_nested_sampling_run_results():
    for name, build_case in STANDARD_PROBLEM_CASES:
        model, log_Z_true, ns_kwargs = build_case()
        ns = NestedSampler(model=model, **ns_kwargs)
        state = ns.run()
        results = state.to_result()

        print(f"Checking {name}")
        assert not np.isnan(results.log_Z_mean)
        assert not np.isnan(results.log_Z_uncert)

        log_Z_samples = _sample_evidence(results,
                                         alpha=results.num_live_points_per_sample,
                                         log_L=results.log_L,
                                         num_samples=1000)

        # Filter outliers
        select_mask = jnp.bitwise_and(
            log_Z_samples > jnp.percentile(log_Z_samples, 5),
            log_Z_samples < jnp.percentile(log_Z_samples, 95)
        )
        log_Z_samples = log_Z_samples[select_mask]

        log_Z_ensemble_mean = jnp.mean(log_Z_samples)
        log_Z_ensemble_std = jnp.std(log_Z_samples)
        np.testing.assert_allclose(log_Z_ensemble_mean, log_Z_true, atol=3.0 * results.log_Z_uncert)
        np.testing.assert_allclose(results.log_Z_mean, log_Z_ensemble_mean, atol=3. * results.log_Z_uncert)
        np.testing.assert_allclose(results.log_Z_uncert, log_Z_ensemble_std,
                                   atol=np.sqrt(results.log_Z_uncert ** 2 + log_Z_ensemble_std ** 2))


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
