"""Explicit runtime, persistence and independent toy references for this benchmark."""

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import jax
import jaxctx
import numpy as np
from jax import numpy as jnp

import jaxns.model as model_module
from benchmarks.issue_312.run_ridge import prior_model
from jaxns.model import Model


SOURCE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path('/largedata/albert/jaxns-reflection-tuning-20261027')
BASELINE = '4e9cc2927b2d4ab4c50529af00201a4fd9960fe8'
# Frozen TFP Normal(0.5, 0.05) parameters are float32; coordinate arithmetic
# is float64. Exact pilot draws must use that effective scale, without changing
# the model or the independently reported mathematical reference logZ=0.
MODERATE_SCALE = float(np.float32(.05))


def make_model() -> Model:
    """Assert CPU and establish real cube precision before any model tracing."""
    if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only environment must be set before importing JAX.')
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'cpu' or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('A non-CPU backend was selected.')
    model_module.transform = partial(jaxctx.transform, base_dtype=jnp.float64)
    return Model(prior_model=partial(prior_model, 1e-5, 0.3))


def atomic_json(path: Path, payload: dict) -> None:
    """Publish a whole JSON record and its checksum together with no partial JSON."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    path.with_suffix(path.suffix + '.sha256').write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n',
    )


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Refuse replacement of published raw arrays; stage the compressed archive."""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    path.with_suffix(path.suffix + '.sha256').write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n',
    )


def numpy_log_likelihood(points: np.ndarray) -> np.ndarray:
    """Independent normalized Gaussian formula; no TFP/JAX implementation reuse."""
    residual = (points[:, 0] - .5 - .3 * np.sin(2 * np.pi * points[:, 1])) / 1e-5
    gaussian = (points[:, 2:16] - .5) / .05  # [N, 14]
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(.05 * np.sqrt(2 * np.pi))
    return peak - .5 * (residual**2 + np.sum(gaussian**2, axis=1))


def residuals(points: np.ndarray) -> np.ndarray:
    """Return 15 residual coordinates after the unit-Jacobian shear."""
    return np.column_stack([
        (points[:, 0] - .5 - .3 * np.sin(2 * np.pi * points[:, 1])) / 1e-5,
        (points[:, 2:16] - .5) / MODERATE_SCALE,
    ])


def exact_draws(rng: np.random.Generator, count: int, gap: float) -> tuple[np.ndarray, int]:
    """Independent constrained draws, including correct cube-truncated early contours.

    For gap<=100 reject cube-exiting points from the enclosing 15-ball. For
    larger finite gaps reject from an enclosing box: all moderate cube
    coordinates plus the maximum stiff residual interval. Both proposals have
    constant density on their enclosing domains. Whole prior is direct uniform.
    """
    if np.isposinf(gap):
        return rng.uniform(size=(count, 18)), count
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError('Exact finite draws require positive gap.')
    radius = np.sqrt(2 * gap)
    accepted = []
    remaining, proposed_count = count, 0
    while remaining:
        size = max(256, remaining)
        points = rng.uniform(size=(size, 18))
        if gap <= 100:
            ball = rng.normal(size=(size, 15))
            ball /= np.linalg.norm(ball, axis=1, keepdims=True)
            ball *= (radius * rng.uniform(size=size)**(1 / 15))[:, None]
            points[:, 2:16] = .5 + MODERATE_SCALE * ball[:, 1:]
            stiff = ball[:, 0]
        else:
            stiff = rng.uniform(-radius, radius, size=size)
        points[:, 0] = .5 + .3 * np.sin(2 * np.pi * points[:, 1]) + 1e-5 * stiff
        valid = np.all((points >= 0) & (points <= 1), axis=1)
        valid &= np.sum(residuals(points)**2, axis=1) < 2 * gap
        take = points[valid][:remaining]
        accepted.append(take)
        remaining -= len(take)
        proposed_count += size
        if proposed_count > max(count * 10000, 10000000):
            raise RuntimeError('Exact rejection budget exhausted; no approximate fallback.')
    return np.concatenate(accepted), proposed_count
