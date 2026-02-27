from pathlib import Path

import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.utils import bruteforce_evidence

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
    return model, log_Z_true, {'max_samples': 100000, 'sampler': UniDimSliceSampler(model=model, num_slices=max(1, 10 * ndims),
                                                                                    no_step_out=True,
                                                                                    collect_phantom_samples=False)}


STANDARD_PROBLEM_CASES = [
    # ('basic', _basic_model_case),
    # ('basic2', _basic2_model_case),
    # ('basic3', _basic3_model_case),
    # ('plateau', _plateau_model_case),
    ('basic_mvn', _basic_mvn_model_case),
]


def test_nested_sampling_run_results(tmp_path):
    tmp_path = Path('.')
    for name, build_case in STANDARD_PROBLEM_CASES:
        model, log_Z_true, ns_kwargs = build_case()
        ns = NestedSampler(model=model, **ns_kwargs)
        state = ns.run()
        results = state.to_result()
        results.summary(tmp_path / f"{name}_summary.txt")

        print(f"Checking {name}")
        assert not np.isnan(results.log_Z_mean)
        assert not np.isnan(results.log_Z_uncert)
        log_Z_samples = results.sample_evidence(num_samples=1000)
        import pylab as plt
        plt.hist(log_Z_samples, bins='auto')
        plt.axvline(log_Z_true, color='k', linestyle='--', label='true log Z')
        plt.axvline(results.log_Z_mean, color='r', linestyle='--', label='estimated log Z')
        plt.legend()
        plt.title(f"{name} log Z samples")
        plt.savefig(tmp_path / f"{name}_logZ_samples.png")
        plt.close()

        results.plot_diagnostics(save_file=tmp_path / f"{name}_diagnostics.png")
        results.plot_cornerplot(save_name=tmp_path / f"{name}_cornplot.png")

        log_Z_ensemble_mean = jnp.mean(log_Z_samples)
        log_Z_ensemble_std = jnp.std(log_Z_samples)
        np.testing.assert_allclose(results.log_Z_mean, log_Z_ensemble_mean, atol=3. * results.log_Z_uncert)
        np.testing.assert_allclose(log_Z_ensemble_mean, log_Z_true, atol=3.0 * results.log_Z_uncert)
        np.testing.assert_allclose(
            results.log_Z_uncert, log_Z_ensemble_std,
            atol=np.sqrt(results.log_Z_uncert ** 2 + log_Z_ensemble_std ** 2)
        )
