"""Experimental squared-exponential GP of stored log likelihoods.

Input whitening, output centring, hyperparameters and the kernel Cholesky are
fit together at a host boundary. Querying the predictive mean or its Hessian
does not refit anything or evaluate the true likelihood.
"""

from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import cho_solve
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


class GP(NamedTuple):
    input_mean: jax.Array  # [D]
    input_cholesky: jax.Array  # [D, D]
    transform: jax.Array  # [D, D], diag(1/length) @ inv(input_cholesky)
    locations: jax.Array  # [N, D], transformed training locations
    coefficients: jax.Array  # [N], amplitude and output scale absorbed
    output_mean: jax.Array  # []
    kernel_cholesky: jax.Array  # [N, N], unit-amplitude kernel, retained for audit
    log_hyperparameters: jax.Array  # [D+1]


def predictive_mean(gp: GP, point: jax.Array) -> jax.Array:
    delta = gp.locations - gp.transform @ (point - gp.input_mean)
    kernel = jnp.exp(-0.5 * jnp.sum(delta ** 2, axis=1))
    return gp.output_mean + jnp.sum(gp.coefficients * kernel)


def negative_predictive_hessian(gp: GP, point: jax.Array) -> jax.Array:
    """Closed-form RBF Hessian, mapped back to original cube coordinates."""
    delta = gp.locations - gp.transform @ (point - gp.input_mean)
    weights = gp.coefficients * jnp.exp(-0.5 * jnp.sum(delta ** 2, axis=1))
    # Matrix products avoid an [N,D,D] intermediate at every slice endpoint.
    negative_hessian = (
        jnp.sum(weights) * jnp.eye(point.shape[0], dtype=point.dtype)
        - (delta.T * weights) @ delta
    )
    result = gp.transform.T @ negative_hessian @ gp.transform
    return 0.5 * (result + result.T)


def _negative_log_marginal_likelihood(parameters, inputs, targets, valid, jitter):
    lengths = jnp.exp(parameters[:-1])
    amplitude = jnp.exp(2 * parameters[-1])
    delta = (inputs[:, None, :] - inputs[None, :, :]) / lengths
    kernel = jnp.exp(-0.5 * jnp.sum(delta ** 2, axis=-1))
    kernel *= valid[:, None] * valid[None, :]
    # Padded observations are independent unit-variance zero targets. They
    # contribute neither quadratic term nor log determinant nor derivatives.
    kernel += jnp.diag(jnp.where(valid, jitter, 1.0))
    cholesky = jnp.linalg.cholesky(kernel)
    alpha = cho_solve((cholesky, True), targets)
    # Factor the correlation matrix, not amplitude*K + absolute jitter:
    # a large fitted amplitude must not erase numerical regularisation.
    return (
        0.5 * targets @ alpha / amplitude + jnp.sum(jnp.log(jnp.diag(cholesky)))
        + jnp.sum(valid) * parameters[-1]
    )


_objective = jax.jit(jax.value_and_grad(_negative_log_marginal_likelihood))


def representative_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point coverage in the current whitened cloud."""
    count = min(len(points), count)
    chosen = np.empty(count, dtype=int)
    chosen[0] = np.argmin(np.sum(points ** 2, axis=1))
    distance = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for index in range(1, count):
        distance[chosen[:index]] = -np.inf
        chosen[index] = np.argmax(distance)
        distance = np.minimum(
            distance, np.sum((points - points[chosen[index]]) ** 2, axis=1),
        )
    return chosen


def fit_gp(
    points: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    max_points: int = 128,
    max_iterations: int = 100,
    relative_regularisation: float = 1e-8,
    jitter: float = 1e-10,
    previous: GP | None = None,
) -> tuple[GP, dict, np.ndarray]:
    """Fit an ARD RBF GP using only the supplied observations.

    The input mean/covariance use the entire supplied contour population.
    At most max_points representatives train the GP; omitted points can be
    used as a held-out diagnostic. There are no derivative observations or
    toy-specific assumptions in this fit.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(log_likelihoods, dtype=np.float64)
    if points.ndim != 2 or values.shape != (len(points),) or len(points) < 2:
        raise ValueError("GP needs at least two points and one scalar value per point.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("GP observations must be finite.")
    if max_points < 2 or max_iterations < 1 or jitter <= 0 or relative_regularisation <= 0:
        raise ValueError("GP sizes, iteration budget and regularisation must be positive.")
    dimension = points.shape[1]
    mean = points.mean(axis=0)
    centred = points - mean
    covariance = centred.T @ centred / (len(points) - 1)
    average_variance = float(np.trace(covariance) / dimension)
    if average_variance == 0.0:
        average_variance = 1 / 12
    covariance += relative_regularisation * average_variance * np.eye(dimension)
    cholesky = np.linalg.cholesky(covariance)
    inverse = solve_triangular(cholesky, np.eye(dimension), lower=True)
    whitened = centred @ inverse.T
    indices = representative_indices(whitened, max_points)
    count = len(indices)
    inputs = np.zeros((max_points, dimension))
    inputs[:count] = whitened[indices]
    output_mean = float(values[indices].mean())
    output_scale = max(float(values[indices].std()), 1e-12)
    targets = np.zeros(max_points)
    targets[:count] = (values[indices] - output_mean) / output_scale
    valid = np.arange(max_points) < count
    jax_inputs, jax_targets, jax_valid = map(jnp.asarray, (inputs, targets, valid))

    def objective(parameters):
        value, gradient = _objective(
            jnp.asarray(parameters), jax_inputs, jax_targets, jax_valid, jitter,
        )
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Non-finite GP marginal likelihood or gradient.")
        return float(value), np.asarray(gradient)

    initial = np.r_[np.full(dimension, np.log(np.sqrt(dimension))), 0.0]
    starts = [initial]
    if previous is not None:
        # Re-express the previous length metric in the new whitening frame.
        # The ARD family retains its diagonal; refitting can then adjust it.
        old_transform_in_new_frame = np.asarray(previous.transform) @ cholesky
        warm = initial.copy()
        warm[:-1] = -0.5 * np.log(np.sum(old_transform_in_new_frame ** 2, axis=0))
        warm[:-1] = np.clip(warm[:-1], np.log(0.1), np.log(1000.0))
        starts.append(warm)
    fits = [minimize(
        objective, start, jac=True, method="L-BFGS-B",
        bounds=[(np.log(0.1), np.log(1000.0))] * dimension + [(-7.0, 7.0)],
        options={"maxiter": max_iterations, "ftol": 1e-10, "gtol": 1e-6},
    ) for start in starts]
    # A warm start can inherit a poor local optimum. Compare it with a fresh
    # start using only GP marginal likelihood, never the analytic toy Hessian.
    fitted = min(fits, key=lambda fit: fit.fun)
    lengths = np.exp(fitted.x[:-1])
    amplitude = np.exp(2 * fitted.x[-1])
    locations = inputs / lengths
    distances = np.sum((locations[:, None] - locations[None, :]) ** 2, axis=-1)
    kernel = np.exp(-0.5 * distances)
    kernel *= valid[:, None] * valid[None, :]
    kernel += np.diag(np.where(valid, jitter, 1.0))
    kernel_cholesky = np.linalg.cholesky(kernel)
    alpha = solve_triangular(kernel_cholesky, targets, lower=True)
    alpha = solve_triangular(kernel_cholesky.T, alpha, lower=False)
    gp = GP(*map(jnp.asarray, (
        mean, cholesky, inverse / lengths[:, None], locations,
        alpha * output_scale, output_mean, kernel_cholesky, fitted.x,
    )))
    predictions = np.asarray(jax.jit(jax.vmap(predictive_mean, in_axes=(None, 0)))(
        gp, jnp.asarray(points),
    ))
    held_out = np.ones(len(points), dtype=bool)
    held_out[indices] = False
    metadata = {
        "population": len(points), "training_points": count,
        "iterations": int(fitted.nit), "optimizer_success": bool(fitted.success),
        "optimizer_message": str(fitted.message),
        "negative_log_marginal_likelihood": float(fitted.fun),
        "train_rmse": float(np.sqrt(np.mean((predictions[indices] - values[indices]) ** 2))),
        "held_out_rmse": (
            float(np.sqrt(np.mean((predictions[held_out] - values[held_out]) ** 2)))
            if np.any(held_out) else None
        ),
        "output_scale": output_scale, "lengths": lengths.tolist(),
        "amplitude": float(amplitude), "relative_jitter": jitter,
        "warm_start": previous is not None,
        "restart_objectives": [float(fit.fun) for fit in fits],
        "restart_iterations": [int(fit.nit) for fit in fits],
    }
    return gp, metadata, indices
