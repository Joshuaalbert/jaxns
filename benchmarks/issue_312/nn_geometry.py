"""A smooth learned residual representation for the toy's log likelihood.

The experiment models peak - log L as half a squared feature norm. Features
combine an affine map with a tanh network; even the linear part can represent
a Gaussian precision. Unlike the GP, this uses the known toy peak in fitting,
so it is a representation experiment, not a like-for-like regressor comparison.
"""

from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from benchmarks.issue_312.gp_geometry import representative_indices


class NN(NamedTuple):
    input_mean: jax.Array  # [D]
    input_cholesky: jax.Array  # [D, D]
    transform: jax.Array  # [D, D], inverse input Cholesky
    energy_scale: jax.Array  # []
    peak: jax.Array  # []
    linear: jax.Array  # [D, D]
    bias: jax.Array  # [D]
    hidden: jax.Array  # [D, H]
    hidden_bias: jax.Array  # [H]
    output: jax.Array  # [H, D]


def predictive_mean(network: NN, point: jax.Array) -> jax.Array:
    whitened = network.transform @ (point - network.input_mean)
    features = whitened @ network.linear + network.bias
    features += jnp.tanh(whitened @ network.hidden + network.hidden_bias) @ network.output
    return network.peak - 0.5 * network.energy_scale * jnp.sum(features ** 2)


def negative_predictive_hessian(network: NN, point: jax.Array) -> jax.Array:
    return -jax.hessian(predictive_mean, argnums=1)(network, point)


def rebase(
    network: NN, mean: np.ndarray, cholesky: np.ndarray, energy_scale: float,
) -> NN:
    """Change normalisation without changing the represented physical function.

    Updating input/output scales must not erase the learned curvature between
    fits. Transform both affine paths and rescale every feature output so the
    previous prediction and its derivatives remain exactly the same function.
    """
    dimension = len(mean)
    inverse = solve_triangular(cholesky, np.eye(dimension), lower=True)
    mapping = cholesky.T @ np.asarray(network.transform).T
    shift = (mean - np.asarray(network.input_mean)) @ np.asarray(network.transform).T
    scale = np.sqrt(float(network.energy_scale) / energy_scale)
    return NN(*map(jnp.asarray, (
        mean, cholesky, inverse, energy_scale, network.peak,
        mapping @ network.linear * scale,
        (network.bias + shift @ network.linear) * scale,
        mapping @ network.hidden, network.hidden_bias + shift @ network.hidden,
        network.output * scale,
    )))


def fit_nn(
    points: np.ndarray, log_likelihoods: np.ndarray, *, peak: float,
    max_points: int = 128, max_iterations: int = 1000, hidden_features: int = 32,
    previous: NN | None = None, relative_loss: bool = False,
) -> tuple[NN, dict, np.ndarray]:
    """Fit stored values, optionally refining an exactly rebased previous fit."""
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(log_likelihoods, dtype=np.float64)
    if points.ndim != 2 or values.shape != (len(points),) or len(points) < 2:
        raise ValueError("NN needs at least two points and one scalar value per point.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("NN observations must be finite.")
    if max_points < 2 or max_iterations < 1 or hidden_features < 0:
        raise ValueError("Invalid NN training budget or feature count.")
    if not np.isfinite(peak) or np.any(values > peak + 1e-4):
        raise ValueError("The energy representation needs a finite upper bound on log L.")
    dimension = points.shape[1]
    mean = points.mean(axis=0)
    centred = points - mean
    covariance = centred.T @ centred / (len(points) - 1)
    variance = max(float(np.trace(covariance) / dimension), 1e-15)
    cholesky = np.linalg.cholesky(covariance + 1e-8 * variance * np.eye(dimension))
    inverse = solve_triangular(cholesky, np.eye(dimension), lower=True)
    whitened = centred @ inverse.T
    indices = representative_indices(whitened, max_points)
    scale = max(float(np.mean(peak - values[indices])), 1e-12)
    inputs = jnp.asarray(whitened[indices])
    targets = jnp.asarray((peak - values[indices]) / scale)
    rng = np.random.default_rng(31203)
    fresh = NN(*map(jnp.asarray, (
        mean, cholesky, inverse, scale, peak,
        np.eye(dimension) * 0.1, np.zeros(dimension),
        rng.normal(size=(dimension, hidden_features)) / np.sqrt(dimension),
        rng.normal(size=hidden_features) * 0.1,
        rng.normal(size=(hidden_features, dimension)) * 0.02 / np.sqrt(max(hidden_features, 1)),
    )))
    starts = [fresh]
    if previous is not None:
        if previous.hidden.shape != fresh.hidden.shape:
            raise ValueError("Incremental fits must keep the same network architecture.")
        starts.append(rebase(previous, mean, cholesky, scale))
    initial, unravel = ravel_pytree(tuple(fresh[5:]))

    def loss(parameters):
        linear, bias, hidden, hidden_bias, output = unravel(parameters)
        features = inputs @ linear + bias + jnp.tanh(inputs @ hidden + hidden_bias) @ output
        energy = 0.5 * jnp.sum(features ** 2, axis=-1)
        residual = energy - targets
        if relative_loss:
            # Give nearby and far-off-contour probes comparable relative
            # weight; otherwise huge energies overwhelm likelihood accuracy
            # in the posterior. One nat floors the denominator near the peak.
            residual /= jnp.maximum(targets, 1.0 / scale)
        # Weak weight regularisation bounds interpolation freedom without
        # supplying derivative targets or information from the analytic ridge.
        return jnp.mean(residual ** 2) + 1e-10 * jnp.mean(parameters ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    def objective(parameters):
        value, gradient = value_and_grad(jnp.asarray(parameters))
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Non-finite NN objective or gradient.")
        return float(value), np.asarray(gradient)

    fits = [minimize(
        objective, np.asarray(ravel_pytree(tuple(start[5:]))[0]), jac=True, method="L-BFGS-B",
        options={"maxiter": max_iterations, "ftol": 1e-14, "gtol": 1e-9, "maxls": 40},
    ) for start in starts]
    selected = int(np.argmin([fit.fun for fit in fits]))
    fitted = fits[selected]
    network = NN(*fresh[:5], *unravel(jnp.asarray(fitted.x)))
    predictions = np.asarray(jax.jit(jax.vmap(predictive_mean, in_axes=(None, 0)))(
        network, jnp.asarray(points),
    ))
    held_out = np.ones(len(points), dtype=bool)
    held_out[indices] = False
    metadata = {
        "population": len(points), "training_points": len(indices),
        "hidden_features": hidden_features, "parameters": int(initial.size),
        "iterations": int(fitted.nit), "optimizer_success": bool(fitted.success),
        "optimizer_message": str(fitted.message), "loss": float(fitted.fun),
        "train_rmse": float(np.sqrt(np.mean((predictions[indices] - values[indices]) ** 2))),
        "held_out_rmse": (float(np.sqrt(np.mean((predictions[held_out] - values[held_out]) ** 2)))
                          if np.any(held_out) else None),
        "warm_start": previous is not None, "selected_start": selected,
        "relative_loss": relative_loss,
        "restart_losses": [float(fit.fun) for fit in fits],
        "restart_iterations": [int(fit.nit) for fit in fits],
        "energy_scale": scale,
    }
    return network, metadata, indices
