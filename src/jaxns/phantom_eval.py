from __future__ import annotations

import dataclasses
from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp, random
from jax.scipy import special as jsp

from jaxns.pytree import PureDataclassPytree
from jaxns.types import FloatArray, IntArray, BoolArray


@dataclasses.dataclass(slots=True, frozen=True)
class PhantomBootstrapContext(PureDataclassPytree):
    log_L_constraints: FloatArray
    valid_phantom: BoolArray
    log_L_phantom: FloatArray
    K_per_block: FloatArray
    eps_equal_prior: FloatArray


@dataclasses.dataclass(slots=True, frozen=True)
class PhantomEvaluation(PureDataclassPytree):
    alpha: FloatArray
    beta: FloatArray
    block_mass: FloatArray
    log_L_blocks: FloatArray
    rho_global: FloatArray
    bootstrap: PhantomBootstrapContext


@dataclasses.dataclass(slots=True, frozen=True)
class EvidenceSamples(PureDataclassPytree):
    log_Z_samples: FloatArray
    log_X_per_block: FloatArray
    log_L_per_block: FloatArray
    rho_samples: FloatArray


PhantomBootstrapContext.register_pytree()
PhantomEvaluation.register_pytree()
EvidenceSamples.register_pytree()


def _logsumexp(x: jax.Array, axis: Optional[int] = None) -> jax.Array:
    return jsp.logsumexp(x, axis=axis)


def _logdiffexp(log_a: jax.Array, log_b: jax.Array) -> jax.Array:
    return log_a + jnp.log1p(-jnp.exp(log_b - log_a))


def _rho_grid_default(
        grid_size: int = 200,
        rho_min: float = 1e-6,
        rho_max: float = 1.0,
        *,
        dtype: Optional[jnp.dtype] = None,
) -> jax.Array:
    start = jnp.log10(rho_min)
    stop = jnp.log10(rho_max)
    return jnp.logspace(start, stop, grid_size, dtype=dtype)


def _boundary_counts_from_clusters(
        log_L_blocks: jax.Array,
        log_L_constraints: jax.Array,
        log_L_phantom: jax.Array,
        valid_phantom: jax.Array,
        *,
        boot_idx: Optional[jax.Array] = None,
        boot_mask: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_clusters = log_L_constraints.shape[0]
    if boot_idx is None:
        boot_idx = jnp.arange(num_clusters)
    if boot_mask is None:
        boot_mask = jnp.ones((boot_idx.shape[0],), dtype=bool)

    c = log_L_constraints[boot_idx]
    ph = log_L_phantom[boot_idx]
    valid = valid_phantom[boot_idx]
    cluster_mask = boot_mask & valid

    dtype = log_L_blocks.dtype
    a = jnp.concatenate([jnp.full((1,), -jnp.inf, dtype=dtype), log_L_blocks[:-1]], axis=0)
    b = log_L_blocks

    eligible = cluster_mask[:, None] & (c[:, None] <= a[None, :])
    ph_expanded = ph[:, :, None]
    a_expanded = a[None, None, :]
    b_expanded = b[None, None, :]

    A = jnp.sum(eligible[:, None, :] & (ph_expanded > a_expanded), axis=(0, 1)).astype(dtype)
    B = jnp.sum(eligible[:, None, :] & (ph_expanded > b_expanded), axis=(0, 1)).astype(dtype)
    E = jnp.sum(eligible[:, None, :] & (ph_expanded == b_expanded), axis=(0, 1)).astype(dtype)
    return A, B, E


def _candlestick_d2_per_boundary(
        K_per_block: jax.Array,
        eps_equal_prior: FloatArray,
        A: jax.Array,
        B: jax.Array,
        E: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    dtype = K_per_block.dtype
    one = jnp.ones((), dtype=dtype)
    two = one + one
    eps = eps_equal_prior * one

    alpha0 = K_per_block + one
    alpha1 = K_per_block
    alpha2 = eps

    denom = alpha0 * alpha0 * (alpha0 + one)
    var1 = alpha1 * (alpha0 - alpha1) / denom
    var2 = alpha2 * (alpha0 - alpha2) / denom
    cov12 = -alpha1 * alpha2 / denom

    mask = A > 0
    A_safe = jnp.where(mask, A, one)
    p1 = B / A_safe
    p2 = E / A_safe

    mu1 = alpha1 / alpha0
    mu2 = alpha2 / alpha0
    u1 = p1 - mu1
    u2 = p2 - mu2

    det = var1 * var2 - cov12 * cov12
    det_safe = jnp.where(det > 0.0, det, one)
    d2_raw = (u1 * u1 * var2 + u2 * u2 * var1 - two * u1 * u2 * cov12) / det_safe
    d2 = jnp.where(mask & (det > 0.0), d2_raw, jnp.zeros_like(d2_raw))
    return d2, alpha0, A


def _candlestick_nll_grid(
        rho_grid: jax.Array,
        d2: jax.Array,
        alpha0: jax.Array,
        A: jax.Array,
        *,
        dim: int = 2,
) -> jax.Array:
    dtype = rho_grid.dtype
    one = jnp.ones((), dtype=dtype)
    two = one + one
    dim_over_two = (dim / 2.0) * one

    mask = A > 0
    A_safe = jnp.where(mask, A, one)
    kappa = one + alpha0[None, :] / (rho_grid[:, None] * A_safe[None, :])
    term = dim_over_two * jnp.log(kappa) + d2[None, :] / (two * kappa)
    term = jnp.where(mask[None, :], term, jnp.zeros_like(term))
    return jnp.sum(term, axis=1)


def _sample_rho_from_likelihood(
        key: jax.Array,
        d2: jax.Array,
        alpha0: jax.Array,
        A: jax.Array,
        rho_grid: jax.Array,
        *,
        dim: int = 2,
        rho_prior: str = "none",
) -> jax.Array:
    nll = _candlestick_nll_grid(rho_grid, d2, alpha0, A, dim=dim)
    logw = -nll
    if rho_prior == "log":
        logw = logw - jnp.log(rho_grid)
    logw = logw - jnp.max(logw)
    idx = random.categorical(key, logw)
    sample = rho_grid[idx]
    has_data = jnp.any(A > 0)
    return jnp.where(has_data, sample, rho_grid[-1])


def _fit_rho_mle(
        d2: jax.Array,
        alpha0: jax.Array,
        A: jax.Array,
        rho_grid: jax.Array,
        *,
        dim: int = 2,
) -> jax.Array:
    nll = _candlestick_nll_grid(rho_grid, d2, alpha0, A, dim=dim)
    idx = jnp.argmin(nll)
    best = rho_grid[idx]
    has_data = jnp.any(A > 0)
    return jnp.where(has_data, best, rho_grid[-1])


@partial(jax.jit, inline=True)
def evaluate_phantoms(
        log_L_blocks: FloatArray,
        log_L_constraints: FloatArray,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        *,
        eps_equal_prior: float = 1e-3,
        rho_grid: Optional[jax.Array] = None,
) -> PhantomEvaluation:
    first_idx = jnp.searchsorted(log_L_classic, log_L_blocks, side="left")
    K_per_block = K_classic[first_idx].astype(log_L_blocks.dtype)

    num_clusters = log_L_constraints.shape[0]
    boot_idx = jnp.arange(num_clusters)
    boot_mask = jnp.ones((num_clusters,), dtype=bool)

    A, B, E = _boundary_counts_from_clusters(
        log_L_blocks=log_L_blocks,
        log_L_constraints=log_L_constraints,
        log_L_phantom=log_L_phantom,
        valid_phantom=valid_phantom,
        boot_idx=boot_idx,
        boot_mask=boot_mask,
    )

    d2, alpha0, _ = _candlestick_d2_per_boundary(K_per_block, eps_equal_prior, A, B, E)
    if rho_grid is None:
        rho_grid = _rho_grid_default(dtype=log_L_blocks.dtype)
    rho_global = _fit_rho_mle(d2=d2, alpha0=alpha0, A=A, rho_grid=rho_grid, dim=2)

    one = jnp.ones((), dtype=log_L_blocks.dtype)
    alpha = jnp.where(A > 0, K_per_block + rho_global * B, K_per_block)
    beta = jnp.where(A > 0, one + rho_global * (A - B), one)

    r_mean = alpha / (alpha + beta)
    X = jnp.cumprod(r_mean)
    X_prev = jnp.concatenate([jnp.ones((1,), dtype=log_L_blocks.dtype), X[:-1]], axis=0)
    block_mass = jnp.exp(log_L_blocks) * (X_prev - X)

    bootstrap = PhantomBootstrapContext(
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        K_per_block=K_per_block,
        eps_equal_prior=eps_equal_prior,
    )

    return PhantomEvaluation(
        alpha=alpha,
        beta=beta,
        block_mass=block_mass,
        log_L_blocks=log_L_blocks,
        rho_global=rho_global,
        bootstrap=bootstrap,
    )


@partial(jax.jit, inline=True, static_argnames=["num_Z_samples", "batch_size"])
def compute_mc_shrinkage(
        seed: int,
        phantom_evaluation: PhantomEvaluation,
        num_Z_samples: int,
        *,
        batch_size: int | None = None,
) -> EvidenceSamples:
    key = random.PRNGKey(seed)
    alpha = phantom_evaluation.alpha
    beta = phantom_evaluation.beta
    log_L_blocks = phantom_evaluation.log_L_blocks
    G = log_L_blocks.shape[0]

    def single_sample(sample_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        r = random.beta(sample_key, a=alpha, b=beta, shape=(G,))
        r = jnp.clip(r, 1e-300, 1.0)
        log_r = jnp.log(r)
        log_X = jnp.cumsum(log_r)
        log_X_prev = jnp.concatenate([jnp.zeros((1,), dtype=log_r.dtype), log_X[:-1]], axis=0)
        log_dX = _logdiffexp(log_X_prev, log_X)
        log_terms = log_dX + log_L_blocks
        log_Z = _logsumexp(log_terms, axis=None)
        return log_Z, log_X

    keys = random.split(key, num_Z_samples)
    log_Z_samples, log_X_per_block = jax.lax.map(single_sample, keys, batch_size=batch_size)
    log_L_per_block = jnp.broadcast_to(log_L_blocks, (num_Z_samples, G))
    rho_samples = jnp.full((num_Z_samples,), phantom_evaluation.rho_global, dtype=log_L_blocks.dtype)
    return EvidenceSamples(
        log_Z_samples=log_Z_samples,
        log_X_per_block=log_X_per_block,
        log_L_per_block=log_L_per_block,
        rho_samples=rho_samples,
    )


@partial(jax.jit, inline=True, static_argnames=["num_Z_samples", "batch_size", "rho_prior"])
def compute_mc_shrinkage_v2(
        seed: int,
        phantom_evaluation: PhantomEvaluation,
        num_Z_samples: int,
        *,
        rho_grid: Optional[jax.Array] = None,
        rho_prior: str = "none",
        batch_size: int | None = None,
) -> EvidenceSamples:
    key = random.PRNGKey(seed)

    ctx = phantom_evaluation.bootstrap
    log_L_blocks = phantom_evaluation.log_L_blocks
    K_per_block = ctx.K_per_block
    eps_equal = ctx.eps_equal_prior
    log_L_constraints = ctx.log_L_constraints
    valid_phantom = ctx.valid_phantom
    log_L_phantom = ctx.log_L_phantom

    dtype = log_L_blocks.dtype
    if rho_grid is None:
        rho_grid = _rho_grid_default(dtype=dtype)

    num_clusters = log_L_constraints.shape[0]
    num_valid = jnp.sum(valid_phantom, dtype=jnp.int32)
    boot_mask = jnp.arange(num_clusters, dtype=jnp.int32) < num_valid

    zero = jnp.zeros((), dtype=dtype)
    neg_inf = jnp.full((), -jnp.inf, dtype=dtype)
    logits = jnp.where(valid_phantom, zero, neg_inf)
    has_valid = jnp.any(valid_phantom)
    logits = jnp.where(has_valid, logits, jnp.zeros_like(logits))

    def single_sample(sample_key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        key_boot, key_rho, key_r = random.split(sample_key, 3)
        boot_idx = random.categorical(key_boot, logits, shape=(num_clusters,))

        A, B, E = _boundary_counts_from_clusters(
            log_L_blocks=log_L_blocks,
            log_L_constraints=log_L_constraints,
            log_L_phantom=log_L_phantom,
            valid_phantom=valid_phantom,
            boot_idx=boot_idx,
            boot_mask=boot_mask,
        )

        d2, alpha0, _ = _candlestick_d2_per_boundary(K_per_block, eps_equal, A, B, E)
        rho = _sample_rho_from_likelihood(
            key_rho,
            d2=d2,
            alpha0=alpha0,
            A=A,
            rho_grid=rho_grid,
            dim=2,
            rho_prior=rho_prior,
        )

        one = jnp.ones((), dtype=dtype)
        alpha = jnp.where(A > 0, K_per_block + rho * B, K_per_block)
        beta = jnp.where(A > 0, one + rho * (A - B), one)

        r = random.beta(key_r, a=alpha, b=beta, shape=(log_L_blocks.shape[0],))
        r = jnp.clip(r, 1e-300, 1.0)
        log_r = jnp.log(r)
        log_X = jnp.cumsum(log_r)
        log_X_prev = jnp.concatenate([jnp.zeros((1,), dtype=log_r.dtype), log_X[:-1]], axis=0)
        log_dX = _logdiffexp(log_X_prev, log_X)
        log_terms = log_dX + log_L_blocks
        log_Z = _logsumexp(log_terms, axis=None)
        return log_Z, log_X, rho

    keys = random.split(key, num_Z_samples)
    log_Z_samples, log_X_per_block, rho_samples = jax.lax.map(single_sample, keys, batch_size=batch_size)
    log_L_per_block = jnp.broadcast_to(log_L_blocks, (num_Z_samples, log_L_blocks.shape[0]))
    return EvidenceSamples(
        log_Z_samples=log_Z_samples,
        log_X_per_block=log_X_per_block,
        log_L_per_block=log_L_per_block,
        rho_samples=rho_samples,
    )
