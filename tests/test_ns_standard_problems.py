import os
from pathlib import Path
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import sitecustomize  # noqa: F401
import jax
import matplotlib
import numpy as np
import pytest
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.fabric.node import local_node_evaluator, make_model_service_factory
from jaxns.model import Model
from jaxns.runtime import LoadBalancerClient
from jaxns.termination_condition import TerminationCondition
from jaxns.utils import bruteforce_evidence

matplotlib.use("Agg", force=True)

tfpd = tfp.distributions
tfb = tfp.bijectors


@dataclass(frozen=True)
class StandardProblemCase:
    name: str
    build_case: Callable


ALLOCATION_TARGETS = [
    "uniform",
    "evidence_improving",
    "posterior_improving",
]

STANDARD_NUM_SLICES = 24
STANDARD_PHANTOM_BURN_IN = 4
STANDARD_TARGET_NUM_LIVE_POINTS = 30
STANDARD_MAX_SAMPLES = 1200
STANDARD_SHELL_SIZE = 15


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
    return tfb.Inline(
        forward_fn=partial(_weak_curve_forward, beta=beta, sigma0=sigma0),
        inverse_fn=partial(_weak_curve_inverse, beta=beta, sigma0=sigma0),
        inverse_log_det_jacobian_fn=partial(_weak_curve_zero_log_det),
        forward_log_det_jacobian_fn=partial(_weak_curve_zero_log_det),
        forward_min_event_ndims=1,
        is_constant_jacobian=True,
        name=f"weak_curve_{ndims}d"
    )


def _weak_curve_forward(z, *, beta: float, sigma0: float):
    shift = beta * (z[..., 0] ** 2 - sigma0 ** 2)
    first = z[..., :1]
    second = (z[..., 1] + shift)[..., None]
    tail = z[..., 2:]
    return jnp.concatenate([first, second, tail], axis=-1)


def _weak_curve_inverse(x, *, beta: float, sigma0: float):
    shift = beta * (x[..., 0] ** 2 - sigma0 ** 2)
    first = x[..., :1]
    second = (x[..., 1] - shift)[..., None]
    tail = x[..., 2:]
    return jnp.concatenate([first, second, tail], axis=-1)


def _weak_curve_zero_log_det(_):
    return jnp.asarray(0.0)


def _basic_model_case():
    model = _build_basic_model()
    log_Z_true = bruteforce_evidence(model=model, grid_res=200)
    return model, log_Z_true


def _basic_prior_model():
    x = Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
    return -jnp.sum(x ** 2)


def _build_basic_model():
    return Model(prior_model=_basic_prior_model)


def _basic_power_prior_model(*, n: int):
    x = Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
    return jnp.log(1.0 - x ** n)


def _basic2_prior_model():
    return _basic_power_prior_model(n=2)


def _basic2_model_case():
    n = 2
    log_Z_true = jnp.log(1.0 - 1.0 / (n + 1))
    model = Model(prior_model=_basic2_prior_model)
    return model, log_Z_true


def _basic3_prior_model():
    x = Prior(tfpd.Uniform(low=0, high=2), name='x').realise()
    y = Prior(tfpd.Normal(loc=2, scale=x), name='y').realise()
    z = x + y
    return -z ** 2


def _basic3_model_case():
    model = Model(prior_model=_basic3_prior_model)
    log_Z_true = bruteforce_evidence(model=model, grid_res=500)
    return model, log_Z_true


def _plateau_prior_model():
    Prior(tfpd.Uniform(low=0, high=1), name='x').realise()
    return 0.0


def _plateau_model_case():
    model = Model(prior_model=_plateau_prior_model)
    log_Z_true = jnp.asarray(0.0)
    return model, log_Z_true


def _mvn_prior_model(*, prior_mu, prior_cov, data_mu, data_cov):
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=prior_mu,
            scale_tril=jnp.linalg.cholesky(prior_cov),
        ),
        name='x'
    ).realise()
    return tfpd.MultivariateNormalTriL(
        loc=data_mu,
        scale_tril=jnp.linalg.cholesky(data_cov),
    ).log_prob(x)


def _basic_mvn_parameters():
    ndims = 8
    prior_mu = 6 * jnp.ones(ndims)
    prior_cov = np.eye(ndims)
    prior_cov[prior_cov == 0] = 0.99

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.eye(ndims)
    return prior_mu, prior_cov, data_mu, data_cov


def _basic_mvn_prior_model():
    prior_mu, prior_cov, data_mu, data_cov = _basic_mvn_parameters()
    return _mvn_prior_model(
        prior_mu=prior_mu,
        prior_cov=prior_cov,
        data_mu=data_mu,
        data_cov=data_cov,
    )


def _basic_mvn_model_case():
    prior_mu, prior_cov, data_mu, data_cov = _basic_mvn_parameters()
    log_Z_true = _log_normal(data_mu, prior_mu, prior_cov + data_cov)

    model = Model(prior_model=_basic_mvn_prior_model)
    return model, log_Z_true


def _spike_slab8_parameters():
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
    return prior_mean, prior_cov, component_means, component_covs, weights


def _spike_slab_model_case():
    model = _build_spike_slab_model()
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
    ) = _spike_slab8_parameters()
    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )
    return model, log_Z_true


def _spike_slab_prior_model(
        *,
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
):
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=jnp.linalg.cholesky(prior_cov),
        ),
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


def _spike_slab8_prior_model():
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
    ) = _spike_slab8_parameters()
    return _spike_slab_prior_model(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights,
    )


def _build_spike_slab_model():
    return Model(prior_model=_spike_slab8_prior_model)


def _spike_slab10_parameters():
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
    return prior_mean, prior_cov, component_means, component_covs, weights


def _spike_slab10_prior_model():
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
    ) = _spike_slab10_parameters()
    return _spike_slab_prior_model(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights,
    )


def _spike_slab10_model_case():
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
    ) = _spike_slab10_parameters()
    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )

    model = Model(prior_model=_spike_slab10_prior_model)
    return model, log_Z_true


def _weak_curved_mvn_prior_model(
        *,
        prior_mean,
        prior_cov,
        likelihood_mean,
        likelihood_cov,
        beta: float,
        sigma0: float,
):
    curve = _weak_curving_bijector(
        ndims=int(prior_mean.shape[0]),
        beta=beta,
        sigma0=sigma0,
    )
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


def _weak_curved_mvn8_parameters():
    ndims = 8
    beta = 0.18
    prior_mean = jnp.zeros(ndims)
    prior_cov_diag = jnp.asarray([2.5, 1.7, 2.0, 1.5, 1.8, 1.6, 1.4, 1.9])
    prior_cov = jnp.diag(prior_cov_diag)
    likelihood_mean = jnp.asarray([2.0, -1.0, 0.8, -0.4, 0.2, -0.3, 1.2, -0.7])
    likelihood_cov = jnp.diag(jnp.asarray([0.25, 0.45, 0.3, 0.35, 0.5, 0.4, 0.28, 0.32]))
    sigma0 = float(np.sqrt(2.5))
    return prior_mean, prior_cov, likelihood_mean, likelihood_cov, beta, sigma0


def _weak_curved_mvn8_prior_model():
    (
        prior_mean,
        prior_cov,
        likelihood_mean,
        likelihood_cov,
        beta,
        sigma0,
    ) = _weak_curved_mvn8_parameters()
    return _weak_curved_mvn_prior_model(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        likelihood_mean=likelihood_mean,
        likelihood_cov=likelihood_cov,
        beta=beta,
        sigma0=sigma0,
    )


def _weak_curved_mvn8_model_case():
    (
        prior_mean,
        prior_cov,
        likelihood_mean,
        likelihood_cov,
        _,
        _,
    ) = _weak_curved_mvn8_parameters()
    log_Z_true = _log_normal(likelihood_mean, prior_mean, prior_cov + likelihood_cov)
    model = Model(prior_model=_weak_curved_mvn8_prior_model)
    return model, log_Z_true


def _weak_curved_spike_slab8_parameters():
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
    sigma0 = float(np.sqrt(4.5))
    return (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
        beta,
        sigma0,
    )


def _weak_curved_spike_slab8_prior_model():
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
        beta,
        sigma0,
    ) = _weak_curved_spike_slab8_parameters()
    return _weak_curved_spike_slab_prior_model(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights,
        beta=beta,
        sigma0=sigma0,
    )


def _weak_curved_spike_slab8_model_case():
    model = _build_weak_curved_spike_slab8_model()
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
        _,
        _,
    ) = _weak_curved_spike_slab8_parameters()
    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )
    return model, log_Z_true


def _weak_curved_spike_slab_prior_model(
        *,
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
        beta: float,
        sigma0: float,
):
    curve = _weak_curving_bijector(
        ndims=int(prior_mean.shape[0]),
        beta=beta,
        sigma0=sigma0,
    )
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


def _build_weak_curved_spike_slab8_model():
    return Model(prior_model=_weak_curved_spike_slab8_prior_model)


def _weak_curved_spike_slab10_parameters():
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
    sigma0 = float(np.sqrt(5.0))
    return (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
        beta,
        sigma0,
    )


def _weak_curved_spike_slab10_prior_model():
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
        beta,
        sigma0,
    ) = _weak_curved_spike_slab10_parameters()
    return _weak_curved_spike_slab_prior_model(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights,
        beta=beta,
        sigma0=sigma0,
    )


def _weak_curved_spike_slab10_model_case():
    (
        prior_mean,
        prior_cov,
        component_means,
        component_covs,
        weights,
        _,
        _,
    ) = _weak_curved_spike_slab10_parameters()
    log_Z_true = _gaussian_mixture_evidence(
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        component_means=component_means,
        component_covs=component_covs,
        weights=weights
    )
    model = Model(prior_model=_weak_curved_spike_slab10_prior_model)
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


def _run_standard_problem_case(
        case: StandardProblemCase,
        allocation_target: str,
        tmp_path: Path,
):
    import pylab as plt

    assert matplotlib.get_backend().lower() == "agg"
    tmp_path = tmp_path / allocation_target / case.name
    tmp_path.mkdir(parents=True, exist_ok=True)

    print(f"Checking {case.name} with allocation_target={allocation_target}")
    model, log_Z_true = case.build_case()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=STANDARD_NUM_SLICES,
        phantom_burn_in=STANDARD_PHANTOM_BURN_IN,
        collect_phantom_samples=True,
        direction_kernel="ellipsoidal",
    )
    t0 = time.time()
    with LoadBalancerClient(address="local") as lb:
        lb.add_workers(["cpu:*:2"])
        ns = lb.get_nested_sampler(
            model=model,
            collect_phantoms=True,
            sampler=sampler,
            target_num_live_points=STANDARD_TARGET_NUM_LIVE_POINTS,
            max_samples=STANDARD_MAX_SAMPLES,
            shell_size=STANDARD_SHELL_SIZE,
        )
        assert sum(sector.num_workers for sector in lb.compute_sectors) >= 2
        state = ns.run_until_goal(
            goal_cond=lambda state: False,
            depth_cond=TerminationCondition(
                max_samples=ns.max_samples,
            ),
            allocation_target=allocation_target,
        )
    results = state.to_result().trim()
    print(f"Runtime: {time.time() - t0} seconds")
    results.summary(tmp_path / f"{case.name}_summary.txt")

    assert not np.isnan(results.log_Z_mean)
    assert not np.isnan(results.log_Z_uncert)
    assert results.log_L_phantom.shape[1] > 0, (
        "phantom conditioning diagnostics require collected phantom samples."
    )
    diagnostics = getattr(results, "execution_diagnostics", None)
    assert diagnostics is not None
    assert diagnostics.sampler.direction_kernel_mode == "ellipsoidal"
    direction_adaptation_diagnostics = (
        diagnostics.sampler.direction_adaptation_diagnostics
    )
    assert direction_adaptation_diagnostics
    assert any(
        (
            getattr(item, "active_kernel_mode", None) == "gmm"
            and not bool(getattr(item, "fallback_active", True))
            and int(getattr(item, "active_kernel_version", 0)) >= 1
        )
        for item in direction_adaptation_diagnostics
    )
    dispatch_records = tuple(diagnostics.worker_runtime.dispatch_records)
    assert dispatch_records
    mc_shrinkage_samples = results.sample_mc_shrinkage(num_samples=1000)

    log_Z_samples = mc_shrinkage_samples.log_Z_samples
    log_L_blocks = np.asarray(mc_shrinkage_samples.log_L_blocks)
    np.testing.assert_allclose(np.asarray(results.log_L_blocks), log_L_blocks)
    valid_blocks = np.isfinite(log_L_blocks)
    phantom_diag = results.phantom_conditioning_diagnostics()
    kish_counts = np.asarray(phantom_diag.kish_participating_cluster_counts)
    gate_active = np.asarray(phantom_diag.phantom_gate_active, dtype=bool)

    plt.hist(log_Z_samples, bins='auto')
    plt.axvline(log_Z_true, color='k', linestyle='--', label='true log Z')
    plt.axvline(results.log_Z_mean, color='r', linestyle='--', label='estimated log Z')
    plt.axvline(results.log_Z_mean - results.log_Z_uncert, color='r', linestyle='dotted', label='+1sigma')
    plt.axvline(results.log_Z_mean + results.log_Z_uncert, color='r', linestyle='dotted', label='-1sigma')

    plt.legend()
    plt.title(f"{case.name} log Z samples")
    plt.savefig(tmp_path / f"{case.name}_logZ_samples.png")
    plt.close()

    plt.plot(
        log_L_blocks[valid_blocks],
        kish_counts[valid_blocks],
        label='Kish participating clusters',
    )
    plt.step(
        log_L_blocks[valid_blocks],
        gate_active[valid_blocks].astype(float),
        where='post',
        label='phantom gate active',
    )
    plt.legend()
    plt.title(f"{case.name} phantom conditioning diagnostics")
    plt.savefig(tmp_path / f"{case.name}_phantom_conditioning_diagnostics.png")
    plt.close()

    results.plot_diagnostics(save_file=tmp_path / f"{case.name}_diagnostics.png")
    results.plot_cornerplot(save_name=tmp_path / f"{case.name}_cornerplot.png")

    log_Z_ensemble_mean = np.mean(log_Z_samples)
    log_Z_ensemble_std = np.std(log_Z_samples)
    assert kish_counts.shape == log_L_blocks.shape
    assert gate_active.shape == log_L_blocks.shape
    assert np.all(kish_counts[valid_blocks] >= 0.0)
    assert np.all(gate_active[valid_blocks] == (kish_counts[valid_blocks] >= 20.0))
    np.testing.assert_allclose(
        log_Z_ensemble_mean,
        log_Z_true,
        atol=3.0 * log_Z_ensemble_std,
        rtol=0,
    )


@pytest.mark.parametrize(
    "case",
    STANDARD_PROBLEM_CASES,
    ids=[case.name for case in STANDARD_PROBLEM_CASES],
)
@pytest.mark.parametrize("allocation_target", ALLOCATION_TARGETS)
def test_nested_sampling_run_results(
        tmp_path,
        case,
        allocation_target,
):
    _run_standard_problem_case(case, allocation_target, tmp_path)


def test_nested_sampling_run_results_distributed(tmp_path):
    tmp_path = tmp_path / "distributed"
    tmp_path.mkdir(parents=True, exist_ok=True)

    case_name = 'basic'
    model = _build_basic_model()
    keys = jax.random.split(jax.random.PRNGKey(42), 4)
    u_samples = [model.sample_U(key) for key in keys]
    local_log_likelihoods = [
        model.log_likelihood(u_sample, allow_nan=False)
        for u_sample in u_samples
    ]

    with local_node_evaluator(
        service_factory=make_model_service_factory(model),
        num_workers=2,
        ident_prefix=f"std-{case_name}",
        start_method="forkserver",
    ) as evaluator:
        t0 = time.time()
        distributed_log_likelihoods = [
            evaluator.evaluate(u_sample)
            for u_sample in u_samples
        ]
        print(f"Distributed runtime: {time.time() - t0} seconds")

    np.testing.assert_allclose(
        np.asarray(distributed_log_likelihoods),
        np.asarray(local_log_likelihoods),
        atol=1e-8,
        rtol=1e-8,
    )
