from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.utils import bruteforce_evidence

# matplotlib.use("Agg")

tfpd = tfp.distributions
tfb = tfp.bijectors


@pytest.fixture(autouse=True)
def _clear_case_specific_jax_executables():
    """Bound memory while this module compiles many unrelated model shapes."""
    yield
    jax.clear_caches()


@dataclass(frozen=True, slots=True)
class StandardProblemCase:
    name: str
    build_case: Callable


def _log_normal(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    dx = x - mean
    dx = solve_triangular(L, dx, lower=True)
    return -0.5 * x.size * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * dx @ dx


def _gaussian_mixture_evidence(prior_mean, prior_cov, component_means, component_covs, weights):
    component_log_z = jnp.asarray([
        _log_normal(mean, prior_mean, prior_cov + cov)
        for mean, cov in zip(component_means, component_covs)
    ])
    return logsumexp(jnp.log(weights) + component_log_z)


def _weak_curving_bijector(ndims: int, beta: float, sigma0: float):
    zero = jnp.asarray(0.0)

    def _forward(z):
        shift = beta * (z[..., 0] ** 2 - sigma0 ** 2)
        first = z[..., :1]
        second = (z[..., 1] + shift)[..., None]
        tail = z[..., 2:]
        return jnp.concatenate([first, second, tail], axis=-1)

    def _inverse(x):
        shift = beta * (x[..., 0] ** 2 - sigma0 ** 2)
        first = x[..., :1]
        second = (x[..., 1] - shift)[..., None]
        tail = x[..., 2:]
        return jnp.concatenate([first, second, tail], axis=-1)

    def _zero_log_det(_):
        return zero

    return tfb.Inline(
        forward_fn=_forward,
        inverse_fn=_inverse,
        inverse_log_det_jacobian_fn=_zero_log_det,
        forward_log_det_jacobian_fn=_zero_log_det,
        forward_min_event_ndims=1,
        is_constant_jacobian=True,
        name=f"weak_curve_{ndims}d"
    )


def _basic_model_case():
    model = _build_basic_model()
    log_Z_true = bruteforce_evidence(model=model, grid_res=200)
    return model, log_Z_true


def _basic_prior_model():
    x = Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
    return -jnp.sum(x ** 2)


def _build_basic_model():
    return Model(prior_model=_basic_prior_model)


def _basic2_model_case():
    n = 2
    log_Z_true = jnp.log(1.0 - 1.0 / (n + 1))

    def prior_model():
        x = Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
        return jnp.log(1.0 - x ** n)

    model = Model(prior_model=prior_model)
    return model, log_Z_true


def _basic3_model_case():
    def prior_model():
        x = Prior(tfpd.Uniform(low=0, high=2), name='x').realise()
        y = Prior(tfpd.Normal(loc=2, scale=x), name='y').realise()
        z = x + y
        return -z ** 2

    model = Model(prior_model=prior_model)
    log_Z_true = bruteforce_evidence(model=model, grid_res=500)
    return model, log_Z_true


def _plateau_model_case():
    def prior_model():
        Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
        return 0.0

    model = Model(prior_model=prior_model)
    log_Z_true = jnp.asarray(0.0)
    return model, log_Z_true


def _basic_mvn_model_case():
    ndims = 8
    prior_mu = 6 * jnp.ones(ndims)
    prior_cov = np.eye(ndims)
    prior_cov[prior_cov == 0] = 0.99

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.eye(ndims)
    log_Z_true = _log_normal(data_mu, prior_mu, prior_cov + data_cov)

    def prior_model():
        x = Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mu, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x'
        ).realise()
        return tfpd.MultivariateNormalTriL(loc=data_mu, scale_tril=jnp.linalg.cholesky(data_cov)).log_prob(x)

    model = Model(prior_model=prior_model)
    return model, log_Z_true


def _spike_slab_model_case():
    model = _build_spike_slab_model()
    ndims = 8
    prior_mean = jnp.zeros(ndims)
    prior_cov = jnp.diag(4.0 * jnp.ones(ndims))

    component_means = jnp.stack([
        jnp.concatenate([3.5 * jnp.ones(4), jnp.zeros(4)]),
        jnp.concatenate([-3.0 * jnp.ones(4), 1.5 * jnp.ones(4)])
    ], axis=0)
    component_covs = jnp.stack([
        jnp.diag(jnp.concatenate([0.05 * jnp.ones(4), 0.4 * jnp.ones(4)])),
        jnp.diag(jnp.concatenate([0.6 * jnp.ones(4), 0.08 * jnp.ones(4)]))
    ], axis=0)
    weights = jnp.asarray([0.25, 0.75])

    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )
    return model, log_Z_true


def _build_spike_slab_model():
    ndims = 8
    prior_mean = jnp.zeros(ndims)
    prior_cov = jnp.diag(4.0 * jnp.ones(ndims))

    component_means = jnp.stack([
        jnp.concatenate([3.5 * jnp.ones(4), jnp.zeros(4)]),
        jnp.concatenate([-3.0 * jnp.ones(4), 1.5 * jnp.ones(4)])
    ], axis=0)
    component_covs = jnp.stack([
        jnp.diag(jnp.concatenate([0.05 * jnp.ones(4), 0.4 * jnp.ones(4)])),
        jnp.diag(jnp.concatenate([0.6 * jnp.ones(4), 0.08 * jnp.ones(4)]))
    ], axis=0)
    weights = jnp.asarray([0.25, 0.75])

    def prior_model():
        x = Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mean, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x'
        ).realise()
        mixture = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(probs=weights),
            components_distribution=tfpd.MultivariateNormalTriL(
                loc=component_means,
                scale_tril=jnp.linalg.cholesky(component_covs)
            )
        )
        return mixture.log_prob(x)

    return Model(prior_model=prior_model)


def _spike_slab10_model_case():
    ndims = 10
    prior_mean = jnp.zeros(ndims)
    prior_cov = jnp.diag(9.0 * jnp.ones(ndims))

    component_means = jnp.stack([
        jnp.concatenate([4.0 * jnp.ones(3), jnp.zeros(7)]),
        jnp.concatenate([-3.5 * jnp.ones(3), 2.0 * jnp.ones(3), jnp.zeros(4)])
    ], axis=0)
    component_covs = jnp.stack([
        jnp.diag(jnp.concatenate([0.03 * jnp.ones(3), 0.7 * jnp.ones(7)])),
        jnp.diag(jnp.concatenate([0.5 * jnp.ones(3), 0.06 * jnp.ones(3), 0.9 * jnp.ones(4)]))
    ], axis=0)
    weights = jnp.asarray([0.4, 0.6])

    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )

    def prior_model():
        x = Prior(
            tfpd.MultivariateNormalTriL(loc=prior_mean, scale_tril=jnp.linalg.cholesky(prior_cov)),
            name='x'
        ).realise()
        mixture = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(probs=weights),
            components_distribution=tfpd.MultivariateNormalTriL(
                loc=component_means,
                scale_tril=jnp.linalg.cholesky(component_covs)
            )
        )
        return mixture.log_prob(x)

    model = Model(prior_model=prior_model)
    return model, log_Z_true


def _weak_curved_mvn8_model_case():
    ndims = 8
    beta = 0.18
    prior_mean = jnp.zeros(ndims)
    prior_cov_diag = jnp.asarray([2.5, 1.7, 2.0, 1.5, 1.8, 1.6, 1.4, 1.9])
    prior_cov = jnp.diag(prior_cov_diag)
    likelihood_mean = jnp.asarray([2.0, -1.0, 0.8, -0.4, 0.2, -0.3, 1.2, -0.7])
    likelihood_cov = jnp.diag(jnp.asarray([0.25, 0.45, 0.3, 0.35, 0.5, 0.4, 0.28, 0.32]))

    log_Z_true = _log_normal(likelihood_mean, prior_mean, prior_cov + likelihood_cov)
    curve = _weak_curving_bijector(ndims=ndims, beta=beta, sigma0=jnp.sqrt(prior_cov_diag[0]))

    def prior_model():
        x = Prior(
            tfpd.TransformedDistribution(
                distribution=tfpd.MultivariateNormalTriL(
                    loc=prior_mean,
                    scale_tril=jnp.linalg.cholesky(prior_cov)
                ),
                bijector=curve
            ),
            name='x'
        ).realise()
        z = curve.inverse(x)
        return tfpd.MultivariateNormalTriL(
            loc=likelihood_mean,
            scale_tril=jnp.linalg.cholesky(likelihood_cov)
        ).log_prob(z)

    model = Model(prior_model=prior_model)
    return model, log_Z_true


def _weak_curved_spike_slab8_model_case():
    model = _build_weak_curved_spike_slab8_model()
    ndims = 8
    prior_mean = jnp.zeros(ndims)
    prior_cov_diag = jnp.asarray([4.5, 3.2, 2.8, 2.6, 2.5, 2.3, 2.1, 2.0])
    prior_cov = jnp.diag(prior_cov_diag)
    component_means = jnp.stack([
        jnp.asarray([2.6, -0.8, 1.2, 0.0, 0.4, -0.3, 0.9, -0.6]),
        jnp.asarray([-2.1, 1.0, -0.9, 0.6, -0.5, 0.7, -1.1, 0.4])
    ], axis=0)
    component_covs = jnp.stack([
        jnp.diag(jnp.asarray([0.18, 0.35, 0.26, 0.42, 0.38, 0.34, 0.29, 0.31])),
        jnp.diag(jnp.asarray([0.45, 0.16, 0.37, 0.3, 0.28, 0.24, 0.2, 0.33]))
    ], axis=0)
    weights = jnp.asarray([0.55, 0.45])

    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )
    return model, log_Z_true


def _build_weak_curved_spike_slab8_model():
    ndims = 8
    beta = 0.14
    prior_mean = jnp.zeros(ndims)
    prior_cov_diag = jnp.asarray([4.5, 3.2, 2.8, 2.6, 2.5, 2.3, 2.1, 2.0])
    prior_cov = jnp.diag(prior_cov_diag)
    component_means = jnp.stack([
        jnp.asarray([2.6, -0.8, 1.2, 0.0, 0.4, -0.3, 0.9, -0.6]),
        jnp.asarray([-2.1, 1.0, -0.9, 0.6, -0.5, 0.7, -1.1, 0.4])
    ], axis=0)
    component_covs = jnp.stack([
        jnp.diag(jnp.asarray([0.18, 0.35, 0.26, 0.42, 0.38, 0.34, 0.29, 0.31])),
        jnp.diag(jnp.asarray([0.45, 0.16, 0.37, 0.3, 0.28, 0.24, 0.2, 0.33]))
    ], axis=0)
    weights = jnp.asarray([0.55, 0.45])
    curve = _weak_curving_bijector(ndims=ndims, beta=beta, sigma0=jnp.sqrt(prior_cov_diag[0]))

    def prior_model():
        x = Prior(
            tfpd.TransformedDistribution(
                distribution=tfpd.MultivariateNormalTriL(
                    loc=prior_mean,
                    scale_tril=jnp.linalg.cholesky(prior_cov)
                ),
                bijector=curve
            ),
            name='x'
        ).realise()
        z = curve.inverse(x)
        mixture = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(probs=weights),
            components_distribution=tfpd.MultivariateNormalTriL(
                loc=component_means,
                scale_tril=jnp.linalg.cholesky(component_covs)
            )
        )
        return mixture.log_prob(z)

    return Model(prior_model=prior_model)


def _weak_curved_spike_slab10_model_case():
    ndims = 10
    beta = 0.12
    prior_mean = jnp.zeros(ndims)
    prior_cov_diag = jnp.asarray([5.0, 3.8, 3.2, 2.9, 2.7, 2.6, 2.4, 2.2, 2.0, 1.8])
    prior_cov = jnp.diag(prior_cov_diag)
    component_means = jnp.stack([
        jnp.asarray([2.8, -0.9, 1.0, 0.5, -0.2, 0.7, 0.4, -0.5, 0.8, -0.3]),
        jnp.asarray([-2.5, 1.2, -1.1, -0.4, 0.6, -0.8, -0.3, 0.9, -0.7, 0.5])
    ], axis=0)
    component_covs = jnp.stack([
        jnp.diag(jnp.asarray([0.14, 0.33, 0.22, 0.36, 0.31, 0.29, 0.35, 0.27, 0.25, 0.3])),
        jnp.diag(jnp.asarray([0.41, 0.12, 0.3, 0.28, 0.26, 0.2, 0.24, 0.18, 0.23, 0.34]))
    ], axis=0)
    weights = jnp.asarray([0.48, 0.52])

    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )
    curve = _weak_curving_bijector(ndims=ndims, beta=beta, sigma0=jnp.sqrt(prior_cov_diag[0]))

    def prior_model():
        x = Prior(
            tfpd.TransformedDistribution(
                distribution=tfpd.MultivariateNormalTriL(
                    loc=prior_mean,
                    scale_tril=jnp.linalg.cholesky(prior_cov)
                ),
                bijector=curve
            ),
            name='x'
        ).realise()
        z = curve.inverse(x)
        mixture = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(probs=weights),
            components_distribution=tfpd.MultivariateNormalTriL(
                loc=component_means,
                scale_tril=jnp.linalg.cholesky(component_covs)
            )
        )
        return mixture.log_prob(z)

    model = Model(prior_model=prior_model)
    return model, log_Z_true


STANDARD_PROBLEM_CASES = [
    StandardProblemCase('basic', _basic_model_case),
    StandardProblemCase('basic2', _basic2_model_case),
    StandardProblemCase('basic3', _basic3_model_case),
    StandardProblemCase('plateau', _plateau_model_case),
    StandardProblemCase('basic_mvn', _basic_mvn_model_case),
    StandardProblemCase('spike_slab', _spike_slab_model_case),
    StandardProblemCase('spike_slab10', _spike_slab10_model_case),
    StandardProblemCase('weak_curved_mvn8', _weak_curved_mvn8_model_case),
    StandardProblemCase('weak_curved_spike_slab8', _weak_curved_spike_slab8_model_case),
    StandardProblemCase('weak_curved_spike_slab10', _weak_curved_spike_slab10_model_case),
]

STANDARD_PROBLEM_CASES_BY_NAME = {
    case.name: case for case in STANDARD_PROBLEM_CASES
}

STANDARD_PROBLEM_MODEL_BUILDERS_BY_NAME = {
    'basic': _build_basic_model,
    'spike_slab': _build_spike_slab_model,
    'weak_curved_spike_slab8': _build_weak_curved_spike_slab8_model,
}


class StandardProblemNode:
    def __init__(self, case_name: str):
        self.case_name = case_name
        try:
            self.model = STANDARD_PROBLEM_MODEL_BUILDERS_BY_NAME[case_name]()
        except KeyError as e:
            raise ValueError(f"No model-only builder configured for distributed case '{case_name}'.") from e

    def evaluate(self, u):
        return self.model.log_likelihood(u, allow_nan=False)


def make_standard_problem_node(case_name: str) -> StandardProblemNode:
    return StandardProblemNode(case_name=case_name)


@pytest.mark.parametrize("collect_phantom_samples", [False, True])
@pytest.mark.parametrize(
    "case",
    STANDARD_PROBLEM_CASES,
    ids=lambda case: case.name,
)
def test_nested_sampling_run_results(case, collect_phantom_samples):
    """Release gate for v2 tolerances and expectation/MC agreement."""
    model, log_Z_true = case.build_case()
    ns = NestedSampler(
        model=model,
        collect_phantom_samples=collect_phantom_samples,
    )
    state = ns.run()
    results = state.to_result().trim()

    assert not np.isnan(results.log_Z_mean)
    assert not np.isnan(results.log_Z_uncert)
    if collect_phantom_samples:
        assert results.log_L_phantom.shape[1] > 0
        if int(state.depth_loop_iter) > 0:
            assert int(results.total_phantom_samples) > 0
    else:
        assert results.log_L_phantom.shape[1] == 0
        assert int(results.total_phantom_samples) == 0

    # The deterministic result is the expectation of the classic race model,
    # so validate it against a classic MC ensemble even when this run also
    # retained phantoms. Phantom conditioning is a distinct posterior update
    # and is checked separately against the known evidence below.
    classic_shrinkage = results.sample_evidence_mc(
        num_samples=1000,
        conditioning="classic",
        key=jax.random.PRNGKey(20260823),
    )
    classic_log_Z_mean = np.mean(
        np.asarray(classic_shrinkage.log_Z_samples)
    )
    np.testing.assert_allclose(
        classic_log_Z_mean,
        results.log_Z_mean,
        atol=results.log_Z_uncert,
        rtol=0,
    )

    if collect_phantom_samples:
        mc_shrinkage = results.sample_evidence_mc(
            num_samples=1000,
            conditioning="phantom",
            key=jax.random.PRNGKey(20260823),
        )
    else:
        mc_shrinkage = classic_shrinkage
    log_Z_samples = np.asarray(mc_shrinkage.log_Z_samples)
    log_Z_ensemble_mean = np.mean(log_Z_samples)
    log_Z_ensemble_std = np.std(log_Z_samples)

    # These are the unchanged v2 release tolerances. The new core must not
    # weaken them.
    np.testing.assert_allclose(
        results.log_Z_mean,
        log_Z_true,
        atol=3.0 * results.log_Z_uncert,
        rtol=0,
    )
    np.testing.assert_allclose(
        log_Z_ensemble_mean,
        log_Z_true,
        atol=2.0 * log_Z_ensemble_std,
        rtol=0,
    )
