"""Analytic posterior diagnostics shared by the v2 and v3 producers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GaussianReference:
    """Gaussian prior and Gaussian-mixture likelihood in latent coordinates."""

    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    component_means: np.ndarray
    component_covariances: np.ndarray
    component_weights: np.ndarray
    curve_beta: float | None = None


def _diagonal(values: list[float]) -> np.ndarray:
    return np.diag(np.asarray(values, dtype=float))


REFERENCES = {
    "basic_mvn": GaussianReference(
        prior_mean=6.0 * np.ones(8),
        prior_covariance=np.eye(8) + 0.99 * (1.0 - np.eye(8)),
        component_means=np.zeros((1, 8)),
        component_covariances=np.eye(8)[None, ...],
        component_weights=np.ones(1),
    ),
    "spike_slab": GaussianReference(
        prior_mean=np.zeros(8),
        prior_covariance=4.0 * np.eye(8),
        component_means=np.stack([
            np.concatenate([3.5 * np.ones(4), np.zeros(4)]),
            np.concatenate([-3.0 * np.ones(4), 1.5 * np.ones(4)]),
        ]),
        component_covariances=np.stack([
            _diagonal([0.05] * 4 + [0.4] * 4),
            _diagonal([0.6] * 4 + [0.08] * 4),
        ]),
        component_weights=np.asarray([0.25, 0.75]),
    ),
    "spike_slab10": GaussianReference(
        prior_mean=np.zeros(10),
        prior_covariance=9.0 * np.eye(10),
        component_means=np.stack([
            np.concatenate([4.0 * np.ones(3), np.zeros(7)]),
            np.concatenate([
                -3.5 * np.ones(3),
                2.0 * np.ones(3),
                np.zeros(4),
            ]),
        ]),
        component_covariances=np.stack([
            _diagonal([0.03] * 3 + [0.7] * 7),
            _diagonal([0.5] * 3 + [0.06] * 3 + [0.9] * 4),
        ]),
        component_weights=np.asarray([0.4, 0.6]),
    ),
    "weak_curved_mvn8": GaussianReference(
        prior_mean=np.zeros(8),
        prior_covariance=_diagonal(
            [2.5, 1.7, 2.0, 1.5, 1.8, 1.6, 1.4, 1.9]
        ),
        component_means=np.asarray([
            [2.0, -1.0, 0.8, -0.4, 0.2, -0.3, 1.2, -0.7],
        ]),
        component_covariances=np.asarray([
            _diagonal([0.25, 0.45, 0.3, 0.35, 0.5, 0.4, 0.28, 0.32]),
        ]),
        component_weights=np.ones(1),
        curve_beta=0.18,
    ),
    "weak_curved_spike_slab8": GaussianReference(
        prior_mean=np.zeros(8),
        prior_covariance=_diagonal(
            [4.5, 3.2, 2.8, 2.6, 2.5, 2.3, 2.1, 2.0]
        ),
        component_means=np.asarray([
            [2.6, -0.8, 1.2, 0.0, 0.4, -0.3, 0.9, -0.6],
            [-2.1, 1.0, -0.9, 0.6, -0.5, 0.7, -1.1, 0.4],
        ]),
        component_covariances=np.asarray([
            _diagonal([0.18, 0.35, 0.26, 0.42, 0.38, 0.34, 0.29, 0.31]),
            _diagonal([0.45, 0.16, 0.37, 0.3, 0.28, 0.24, 0.2, 0.33]),
        ]),
        component_weights=np.asarray([0.55, 0.45]),
        curve_beta=0.14,
    ),
    "weak_curved_spike_slab10": GaussianReference(
        prior_mean=np.zeros(10),
        prior_covariance=_diagonal(
            [5.0, 3.8, 3.2, 2.9, 2.7, 2.6, 2.4, 2.2, 2.0, 1.8]
        ),
        component_means=np.asarray([
            [2.8, -0.9, 1.0, 0.5, -0.2, 0.7, 0.4, -0.5, 0.8, -0.3],
            [-2.5, 1.2, -1.1, -0.4, 0.6, -0.8, -0.3, 0.9, -0.7, 0.5],
        ]),
        component_covariances=np.asarray([
            _diagonal([0.14, 0.33, 0.22, 0.36, 0.31, 0.29, 0.35, 0.27, 0.25, 0.3]),
            _diagonal([0.41, 0.12, 0.3, 0.28, 0.26, 0.2, 0.24, 0.18, 0.23, 0.34]),
        ]),
        component_weights=np.asarray([0.48, 0.52]),
        curve_beta=0.12,
    ),
}


def _log_normal(
        values: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
) -> np.ndarray:
    delta = values - mean
    factor = np.linalg.cholesky(covariance)
    solved = np.linalg.solve(factor, np.moveaxis(delta, -1, 0))
    quadratic = np.sum(np.square(solved), axis=0)
    normalisation = (
        values.shape[-1] * np.log(2.0 * np.pi)
        + 2.0 * np.sum(np.log(np.diag(factor)))
    )
    return -0.5 * (normalisation + quadratic)


def _normalise_log_weights(log_weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError("Posterior weights contain no finite values.")
    maximum = np.max(log_weights[finite])
    weights = np.where(finite, np.exp(log_weights - maximum), 0.0)
    return weights / np.sum(weights)


def _latent_samples(
        samples: np.ndarray,
        reference: GaussianReference,
) -> np.ndarray:
    if reference.curve_beta is None:
        return samples
    latent = np.array(samples, copy=True)
    sigma0_squared = reference.prior_covariance[0, 0]
    shift = reference.curve_beta * (
        np.square(latent[..., 0]) - sigma0_squared
    )
    latent[..., 1] -= shift
    return latent


def posterior_diagnostics(
        case_name: str,
        samples: object,
        log_weights: object,
) -> dict[str, object]:
    """Compare weighted posterior samples with an analytic Gaussian reference."""
    reference = REFERENCES.get(case_name)
    if reference is None:
        return {}

    # V3 wraps named parameter pytrees in TreeField; v2 exposes the dictionary
    # directly. Unwrap only this stable container boundary before selecting x.
    if hasattr(samples, "tree"):
        samples = samples.tree
    if hasattr(samples, "to_dict"):
        samples = samples.to_dict()
    if isinstance(samples, dict):
        try:
            samples = samples["x"]
        except KeyError as error:
            raise ValueError(
                f"Posterior diagnostics for {case_name} require named 'x' samples."
            ) from error
    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    latent = _latent_samples(values, reference)
    weights = _normalise_log_weights(np.asarray(log_weights, dtype=float))
    if latent.shape[0] != weights.shape[0]:
        raise ValueError(
            "Posterior samples and weights must have the same leading size."
        )

    component_log_evidence = np.asarray([
        np.log(component_weight)
        + _log_normal(
            reference.prior_mean[None, :],
            component_mean,
            reference.prior_covariance + component_covariance,
        )[0]
        for component_weight, component_mean, component_covariance in zip(
            reference.component_weights,
            reference.component_means,
            reference.component_covariances,
            strict=True,
        )
    ])
    true_mode_weights = _normalise_log_weights(component_log_evidence)

    likelihood_terms = np.stack([
        np.log(component_weight)
        + _log_normal(latent, component_mean, component_covariance)
        for component_weight, component_mean, component_covariance in zip(
            reference.component_weights,
            reference.component_means,
            reference.component_covariances,
            strict=True,
        )
    ], axis=-1)
    maximum = np.max(likelihood_terms, axis=-1, keepdims=True)
    responsibilities = np.exp(likelihood_terms - maximum)
    responsibilities /= np.sum(responsibilities, axis=-1, keepdims=True)
    estimated_mode_weights = np.sum(
        weights[:, None] * responsibilities,
        axis=0,
    )

    component_posterior_means = []
    for component_mean, component_covariance in zip(
            reference.component_means,
            reference.component_covariances,
            strict=True,
    ):
        precision = (
            np.linalg.inv(reference.prior_covariance)
            + np.linalg.inv(component_covariance)
        )
        covariance = np.linalg.inv(precision)
        mean = covariance @ (
            np.linalg.solve(
                reference.prior_covariance,
                reference.prior_mean,
            )
            + np.linalg.solve(component_covariance, component_mean)
        )
        component_posterior_means.append(mean)
    true_mean = np.sum(
        true_mode_weights[:, None] * np.asarray(component_posterior_means),
        axis=0,
    )
    estimated_mean = np.sum(weights[:, None] * latent, axis=0)
    mode_errors = np.abs(estimated_mode_weights - true_mode_weights)
    return {
        "posterior_coordinate_system": "latent_gaussian",
        "posterior_mean_rmse": float(
            np.sqrt(np.mean(np.square(estimated_mean - true_mean)))
        ),
        "posterior_mode_weights": estimated_mode_weights.tolist(),
        "posterior_mode_weights_true": true_mode_weights.tolist(),
        "posterior_mode_weight_max_abs_error": float(np.max(mode_errors)),
        "posterior_missed_mode_count": int(np.sum(
            (true_mode_weights >= 0.05) & (estimated_mode_weights < 0.01)
        )),
        "posterior_incorrect_mode_weight_count": int(np.sum(
            mode_errors > 0.10
        )),
    }
