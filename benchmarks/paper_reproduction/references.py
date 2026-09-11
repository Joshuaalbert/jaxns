"""Deterministic Gaussian and curved-Gaussian evidence references."""

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm


def component_log_evidence(
        mean: np.ndarray, covariance: np.ndarray, beta: float = 0.,
) -> float:
    """Integrate a component under N(0,I), with its own centered shear.

    The shear adds beta * ((z[0]-mean[0])**2-covariance[0,0]) to z[1].
    Conditioning on z[0] leaves a Gaussian integral in all other coordinates.
    """
    dimension = mean.size
    if beta == 0:
        return float(multivariate_normal.logpdf(
            mean, cov=np.eye(dimension) + covariance,
        ))
    variance = covariance[0, 0]
    slope = covariance[1:, 0] / variance  # [D-1]
    conditional_covariance = (
        covariance[1:, 1:]
        - np.outer(covariance[1:, 0], covariance[0, 1:]) / variance
    )  # [D-1, D-1]
    combined = np.eye(dimension - 1) + conditional_covariance
    precision = np.linalg.inv(combined)
    log_normalizer = -.5 * (
        (dimension - 1) * np.log(2 * np.pi) + np.linalg.slogdet(combined)[1]
    )
    # Factor the product of the prior and component first-coordinate normals.
    posterior_mean = mean[0] / (1 + variance)
    posterior_std = np.sqrt(variance / (1 + variance))

    def integrand(first: float) -> float:
        conditional_mean = mean[1:] + slope * (first - mean[0])
        conditional_mean[0] += beta * ((first - mean[0])**2 - variance)
        return float(
            norm.pdf(first, loc=posterior_mean, scale=posterior_std)
            * np.exp(-.5 * conditional_mean @ precision @ conditional_mean)
        )

    value, error = quad(
        integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12, limit=200,
    )
    if value <= 0 or error / value > 1e-9:
        raise ArithmeticError('Curved-component evidence did not converge.')
    return float(
        norm.logpdf(mean[0], scale=np.sqrt(1 + variance))
        + log_normalizer + np.log(value)
    )


def component_log_likelihoods(
        samples: np.ndarray, means: np.ndarray,
        covariances: np.ndarray, beta: float = 0.,
) -> np.ndarray:
    """Evaluate components in their own latent coordinates for mode labels."""
    values = []
    for mean, covariance in zip(means, covariances, strict=True):
        coordinates = samples.copy()  # [N, D]
        coordinates[:, 1] -= beta * (
            (coordinates[:, 0] - mean[0])**2 - covariance[0, 0]
        )
        values.append(np.atleast_1d(multivariate_normal.logpdf(
            coordinates, mean=mean, cov=covariance,
        )))
    return np.stack(values, axis=1)  # [N, 2]
