"""Importable definitions of the paper's known-evidence problems.

A complete :class:`jaxns.state.State` carries its model, so these long-running
experiments use module-level prior functions that survive durable pickle round
trips.  The paper cases are deliberately independent from the fixed standard
regression suite: changing an experimental stress test must not erase the
backward-comparison baseline.
"""

import dataclasses
from collections.abc import Callable

import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.mixed_precision import mp_policy
from jaxns.model import Model

tfb = tfp.bijectors
tfpd = tfp.distributions

DIMENSION = 8
PAPER_DTYPE = mp_policy.measure_dtype

G8_PRIOR_MEAN = jnp.zeros(DIMENSION, dtype=PAPER_DTYPE)
G8_PRIOR_COVARIANCE = jnp.eye(DIMENSION, dtype=PAPER_DTYPE)
G8_LIKELIHOOD_MEAN = jnp.asarray(
    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=PAPER_DTYPE,
)
G8_LIKELIHOOD_COVARIANCE = np.eye(DIMENSION, dtype=np.float64)
G8_LIKELIHOOD_COVARIANCE[G8_LIKELIHOOD_COVARIANCE == 0.0] = 0.99

CG8_BETA = 0.4
CG8_PRIOR_MEAN = G8_PRIOR_MEAN
CG8_PRIOR_COVARIANCE = G8_PRIOR_COVARIANCE
CG8_LIKELIHOOD_MEAN = G8_LIKELIHOOD_MEAN
CG8_LIKELIHOOD_COVARIANCE = G8_LIKELIHOOD_COVARIANCE
# Conditioning the seven remaining latent coordinates on the first reduces
# this reference evidence to deterministic one-dimensional quadrature.
CG8_LOG_EVIDENCE = jnp.asarray(-12.5580306213, dtype=PAPER_DTYPE)

SS8_PRIOR_MEAN = jnp.zeros(DIMENSION, dtype=PAPER_DTYPE)
SS8_PRIOR_COVARIANCE = jnp.eye(DIMENSION, dtype=PAPER_DTYPE)
SS8_COMPONENT_MEANS = jnp.stack(
    [
        G8_LIKELIHOOD_MEAN,
        -G8_LIKELIHOOD_MEAN,
    ]
)
SS8_COMPONENT_COVARIANCES = jnp.stack(
    [
        0.5 * jnp.eye(DIMENSION, dtype=PAPER_DTYPE),
        jnp.eye(DIMENSION, dtype=PAPER_DTYPE),
    ]
)
CSS8_PRIOR_MEAN = SS8_PRIOR_MEAN
CSS8_PRIOR_COVARIANCE = SS8_PRIOR_COVARIANCE
CSS8_COMPONENT_MEANS = SS8_COMPONENT_MEANS
CSS8_COMPONENT_COVARIANCES = jnp.stack(
    [
        0.5 * jnp.asarray(G8_LIKELIHOOD_COVARIANCE),
        jnp.asarray(G8_LIKELIHOOD_COVARIANCE),
    ]
)
@dataclasses.dataclass(frozen=True, slots=True)
class PaperCase:
    """One importable model factory and its reference log evidence."""

    name: str
    build: Callable[[], tuple[Model, jnp.ndarray]]


def _log_normal(x, mean, covariance):
    """Evaluate a multivariate normal log density."""
    lower = jnp.linalg.cholesky(covariance)
    displacement = solve_triangular(lower, x - mean, lower=True)
    return (
        -0.5 * x.size * jnp.log(2.0 * jnp.pi)
        - jnp.sum(jnp.log(jnp.diag(lower)))
        - 0.5 * displacement @ displacement
    )


def _mixture_log_evidence(
        prior_mean,
        prior_covariance,
        component_means,
        component_covariances,
):
    """Integrate a Gaussian mixture likelihood under a Gaussian prior."""
    component_log_evidence = jnp.asarray([
        _log_normal(
            mean,
            prior_mean,
            prior_covariance + covariance,
        )
        for mean, covariance in zip(
            component_means,
            component_covariances,
        )
    ])
    return logsumexp(component_log_evidence)


def _curve(beta: float, center: float, variance: float):
    """Create the unit-Jacobian shear used by the curved case."""
    zero = jnp.asarray(0.0, dtype=PAPER_DTYPE)

    def forward(z):
        shift = beta * ((z[..., 0] - center) ** 2 - variance)
        return jnp.concatenate([
            z[..., :1],
            (z[..., 1] + shift)[..., None],
            z[..., 2:],
        ], axis=-1)

    def inverse(x):
        shift = beta * ((x[..., 0] - center) ** 2 - variance)
        return jnp.concatenate([
            x[..., :1],
            (x[..., 1] - shift)[..., None],
            x[..., 2:],
        ], axis=-1)

    def zero_log_det(unused):
        del unused
        return zero

    return tfb.Inline(
        forward_fn=forward,
        inverse_fn=inverse,
        inverse_log_det_jacobian_fn=zero_log_det,
        forward_log_det_jacobian_fn=zero_log_det,
        forward_min_event_ndims=1,
        is_constant_jacobian=True,
        name="paper_weak_curve_8d",
    )


def g8_prior_model():
    """Isotropic Gaussian prior and shifted correlated likelihood."""
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=G8_PRIOR_MEAN,
            scale_tril=jnp.linalg.cholesky(G8_PRIOR_COVARIANCE),
        ),
        name="x",
    ).realise()
    return tfpd.MultivariateNormalTriL(
        loc=G8_LIKELIHOOD_MEAN,
        scale_tril=jnp.linalg.cholesky(G8_LIKELIHOOD_COVARIANCE),
    ).log_prob(x)


def cg8_prior_model():
    """Isotropic Gaussian prior and shifted, curved Gaussian likelihood."""
    curve = _curve(
        CG8_BETA,
        CG8_LIKELIHOOD_MEAN[0],
        CG8_LIKELIHOOD_COVARIANCE[0, 0],
    )
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=CG8_PRIOR_MEAN,
            scale_tril=jnp.linalg.cholesky(CG8_PRIOR_COVARIANCE),
        ),
        name="x",
    ).realise()
    # The unit-Jacobian inverse maps physical coordinates onto the same latent
    # correlated Gaussian used by G8, bending its contours without changing
    # the likelihood's density normalisation.
    z = curve.inverse(x)
    return tfpd.MultivariateNormalTriL(
        loc=CG8_LIKELIHOOD_MEAN,
        scale_tril=jnp.linalg.cholesky(CG8_LIKELIHOOD_COVARIANCE),
    ).log_prob(z)


def ss8_prior_model():
    """Separated, unequal-scale Gaussian-mixture likelihood."""
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=SS8_PRIOR_MEAN,
            scale_tril=jnp.linalg.cholesky(SS8_PRIOR_COVARIANCE),
        ),
        name="x",
    ).realise()
    components = tfpd.MultivariateNormalTriL(
        loc=SS8_COMPONENT_MEANS,
        scale_tril=jnp.linalg.cholesky(SS8_COMPONENT_COVARIANCES),
    )
    # The likelihood is the sum of the two normal densities. Treating them as
    # a normalised categorical mixture would shift the evidence by -log(2).
    return logsumexp(components.log_prob(x))


def css8_prior_model():
    """Separated, correlated spike--slab likelihood."""
    x = Prior(
        tfpd.MultivariateNormalTriL(
            loc=CSS8_PRIOR_MEAN,
            scale_tril=jnp.linalg.cholesky(CSS8_PRIOR_COVARIANCE),
        ),
        name="x",
    ).realise()
    components = tfpd.MultivariateNormalTriL(
        loc=CSS8_COMPONENT_MEANS,
        scale_tril=jnp.linalg.cholesky(CSS8_COMPONENT_COVARIANCES),
    )
    return logsumexp(components.log_prob(x))


def build_g8() -> tuple[Model, jnp.ndarray]:
    """Build G8 and its analytic log evidence."""
    truth = _log_normal(
        G8_LIKELIHOOD_MEAN,
        G8_PRIOR_MEAN,
        G8_PRIOR_COVARIANCE + G8_LIKELIHOOD_COVARIANCE,
    )
    return Model(prior_model=g8_prior_model), truth


def build_cg8() -> tuple[Model, jnp.ndarray]:
    """Build CG8 and its high-accuracy reference log evidence."""
    return Model(prior_model=cg8_prior_model), CG8_LOG_EVIDENCE


def build_ss8() -> tuple[Model, jnp.ndarray]:
    """Build SS8 and its analytic log evidence."""
    truth = _mixture_log_evidence(
        SS8_PRIOR_MEAN,
        SS8_PRIOR_COVARIANCE,
        SS8_COMPONENT_MEANS,
        SS8_COMPONENT_COVARIANCES,
    )
    return Model(prior_model=ss8_prior_model), truth


def build_css8() -> tuple[Model, jnp.ndarray]:
    """Build CSS8 and its analytic log evidence."""
    truth = _mixture_log_evidence(
        CSS8_PRIOR_MEAN,
        CSS8_PRIOR_COVARIANCE,
        CSS8_COMPONENT_MEANS,
        CSS8_COMPONENT_COVARIANCES,
    )
    return Model(prior_model=css8_prior_model), truth


PAPER_CASES = {
    "basic_mvn": PaperCase("basic_mvn", build_g8),
    "weak_curved_mvn8": PaperCase("weak_curved_mvn8", build_cg8),
    "spike_slab": PaperCase("spike_slab", build_ss8),
    "correlated_spike_slab8": PaperCase(
        "correlated_spike_slab8",
        build_css8,
    ),
}
