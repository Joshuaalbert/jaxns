"""Current evidence-paper cases: G10, CG10, and repository SS10."""

import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from scipy.special import logsumexp
from scipy.special import roots_hermitenorm
from scipy.stats import norm
from tensorflow_probability.substrates import jax as tfp

from benchmarks.paper_reproduction.posterior_cases import build_model
from benchmarks.paper_reproduction.posterior_cases import reference_values
from benchmarks.paper_reproduction.references import component_log_evidence
from jaxns.model import Model

DIMENSION = 10
CASE_REVISION = "evidence10-AC300-20260909"
CASES = ("g10", "cg10", "ss10")
GAUSSIAN_MEAN = np.array([3.] + [0.] * 9)  # [D]
GAUSSIAN_COVARIANCE = .01 * np.eye(DIMENSION) + .99  # [D,D]
CG_BETA = .4


def g10_prior_model():
    """Shifted correlated normal likelihood under N(0,I)."""
    x = Prior(tfp.distributions.MultivariateNormalDiag(
        loc=jnp.zeros(DIMENSION, jnp.float64),
        scale_diag=jnp.ones(DIMENSION, jnp.float64),
    ), name="x").realise()
    return tfp.distributions.MultivariateNormalTriL(
        loc=jnp.asarray(GAUSSIAN_MEAN),
        scale_tril=jnp.linalg.cholesky(jnp.asarray(GAUSSIAN_COVARIANCE)),
    ).log_prob(x)


def cg10_prior_model():
    """G10 with a unit-Jacobian shear centered on its component mean."""
    x = Prior(tfp.distributions.MultivariateNormalDiag(
        loc=jnp.zeros(DIMENSION, jnp.float64),
        scale_diag=jnp.ones(DIMENSION, jnp.float64),
    ), name="x").realise()
    shift = CG_BETA * ((x[0] - GAUSSIAN_MEAN[0]) ** 2
                       - GAUSSIAN_COVARIANCE[0, 0])
    z = x.at[1].add(-shift)  # [D], inverse shear in physical coordinates
    return tfp.distributions.MultivariateNormalTriL(
        loc=jnp.asarray(GAUSSIAN_MEAN),
        scale_tril=jnp.linalg.cholesky(jnp.asarray(GAUSSIAN_COVARIANCE)),
    ).log_prob(z)


def build_case(case: str) -> tuple[Model, dict]:
    """Build a durable model and its deterministic reference evidence."""
    if case == "ss10":
        return build_model("ss10", "repository"), reference_values("ss10", "repository")
    if case not in ("g10", "cg10"):
        raise ValueError(f"Unknown evidence case: {case}")
    beta = CG_BETA if case == "cg10" else 0.
    return (
        Model(prior_model=cg10_prior_model if case == "cg10" else g10_prior_model),
        {"log_Z": component_log_evidence(GAUSSIAN_MEAN, GAUSSIAN_COVARIANCE, beta),
         "method": "conditional 1D quadrature" if beta else "Gaussian conjugacy",
         "mean": GAUSSIAN_MEAN.tolist(),
         "covariance": GAUSSIAN_COVARIANCE.tolist(), "shear_beta": beta},
    )


def cg10_hermite_reference(order: int) -> float:
    """Check adaptive quadrature independently by Gaussian-weighted quadrature."""
    covariance = GAUSSIAN_COVARIANCE
    variance = covariance[0, 0]
    slope = covariance[1:, 0] / variance  # [D-1]
    conditional_covariance = covariance[1:, 1:] - np.outer(
        covariance[1:, 0], covariance[0, 1:],
    ) / variance  # [D-1,D-1]
    combined = np.eye(DIMENSION - 1) + conditional_covariance
    nodes, weights = roots_hermitenorm(order)  # [Q] each
    first = (GAUSSIAN_MEAN[0] / (1. + variance)
             + np.sqrt(variance / (1. + variance)) * nodes)  # [Q]
    means = GAUSSIAN_MEAN[1:] + (first[:, None] - GAUSSIAN_MEAN[0]) * slope
    means[:, 0] += CG_BETA * ((first-GAUSSIAN_MEAN[0])**2 - variance)
    exponent = -.5 * np.einsum("qi,ij,qj->q", means, np.linalg.inv(combined), means)
    log_integral = logsumexp(np.log(weights) + exponent) - .5 * np.log(2. * np.pi)
    return float(norm.logpdf(GAUSSIAN_MEAN[0], scale=np.sqrt(1. + variance))
                 - .5 * ((DIMENSION-1) * np.log(2. * np.pi)
                         + np.linalg.slogdet(combined)[1]) + log_integral)
