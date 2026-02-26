import dataclasses
from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp, random
from jax.scipy import special as jsp

from jaxns.log_semiring import LogSpace
from jaxns.pytree import PureDataclassPytree
from jaxns.types import FloatArray, IntArray, BoolArray


@dataclasses.dataclass(slots=True, frozen=True)
class EvidenceSamples(PureDataclassPytree):
    log_Z_samples: FloatArray  # [num_Z_samples] samples of the evidence log Z from the MC shrinkage sampling
    log_dZ_mean: FloatArray  # [num_blocks] L_{g} * (X_{g-1} - X_g) averaged over MC chains
    log_dZ_var: FloatArray  # [num_blocks] variance of L_{g} * (X_{g-1} - X_g) over MC chains
    rho_samples: FloatArray  # [num_Z_samples] samples of the global rho parameter used in the MC shrinkage sampling


EvidenceSamples.register_pytree()


def _logsumexp(x: FloatArray, axis: Optional[int] = None) -> FloatArray:
    return jsp.logsumexp(x, axis=axis)


def _logdiffexp(log_a: FloatArray, log_b: FloatArray) -> FloatArray:
    return log_a + jnp.log1p(-jnp.exp(log_b - log_a))


def _rho_grid_default(
        grid_size: int = 200,
        rho_min: float = 1e-6,
        rho_max: float = 1.0,
        *,
        dtype: Optional[jnp.dtype] = None,
) -> FloatArray:
    start = jnp.log10(rho_min)
    stop = jnp.log10(rho_max)
    return jnp.logspace(start, stop, grid_size, dtype=dtype)


def _boundary_counts_from_clusters(
        log_L_blocks: FloatArray,
        log_L_constraints: FloatArray,
        log_L_phantom: FloatArray,
        valid_phantom: FloatArray,
        *,
        boot_idx: Optional[IntArray] = None,
        boot_mask: Optional[BoolArray] = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
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
        K_per_block: FloatArray,
        eps_equal_prior: FloatArray,
        A: FloatArray,
        B: FloatArray,
        E: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
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
        rho_grid: FloatArray,
        d2: FloatArray,
        alpha0: FloatArray,
        A: FloatArray,
        *,
        dim: int = 2,
) -> FloatArray:
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
        key: FloatArray,
        d2: FloatArray,
        alpha0: FloatArray,
        A: FloatArray,
        rho_grid: FloatArray,
        *,
        dim: int = 2,
        rho_prior: str = "none",
) -> FloatArray:
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
        d2: FloatArray,
        alpha0: FloatArray,
        A: FloatArray,
        rho_grid: FloatArray,
        *,
        dim: int = 2,
) -> FloatArray:
    nll = _candlestick_nll_grid(rho_grid, d2, alpha0, A, dim=dim)
    idx = jnp.argmin(nll)
    best = rho_grid[idx]
    has_data = jnp.any(A > 0)
    return jnp.where(has_data, best, rho_grid[-1])


@partial(jax.jit, inline=True, static_argnames=["num_Z_samples", "batch_size", "rho_prior"])
def compute_mc_shrinkage_v2(
        seed: int,
        log_L_blocks: FloatArray,
        log_L_constraints: FloatArray,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_Z_samples: int,
        *,
        eps_equal_prior: float = 1e-3,
        rho_grid: Optional[FloatArray] = None,
        rho_prior: str = "none",
        batch_size: int | None = None,
) -> EvidenceSamples:
    """
    Monte-Carlo evidence sampling with phantom-aware shrinkage.

    Per Monte-Carlo draw this function:
    1) bootstraps valid phantom clusters,
    2) computes boundary counts ``(A, B, E)`` under the loose-cluster rule,
    3) samples a global ``rho`` from the candlestick likelihood,
    4) forms per-boundary Beta shrinkage posteriors,
    5) samples shrinkages and accumulates evidence contributions.

    Args:
        seed: PRNG seed for Monte-Carlo sampling.
        log_L_blocks: ``[num_blocks]`` strictly increasing block likelihood levels.
        log_L_constraints: ``[num_samples]`` cluster constraints for classic samples.
        log_L_classic: ``[num_samples]`` classic likelihood values.
        K_classic: ``[num_samples]`` classic live-point counts.
        valid_phantom: ``[num_samples]`` mask indicating which clusters have valid phantom draws.
        log_L_phantom: ``[num_samples, num_phantom]`` phantom likelihoods.
        num_Z_samples: Number of Monte-Carlo evidence samples.
        eps_equal_prior: Small equality-mass prior for candlestick calibration.
        rho_grid: Optional grid of rho values; if None a default log-spaced grid is used.
        rho_prior: Prior for rho grid sampling, either ``"none"`` or ``"log"``.
        batch_size: Optional batch size for ``jax.lax.map``.

    Returns:
        EvidenceSamples with:
          - ``log_Z_samples``: evidence samples ``[num_Z_samples]``;
          - ``log_dZ_mean``: mean per-block contribution in log-space ``[num_blocks]``;
          - ``log_dZ_var``: variance per-block contribution in log-space ``[num_blocks]``;
          - ``rho_samples``: sampled global rho values ``[num_Z_samples]``.
    """
    key = random.PRNGKey(seed)

    first_idx = jnp.searchsorted(log_L_classic, log_L_blocks, side="left")
    K_per_block = K_classic[first_idx].astype(log_L_blocks.dtype)
    eps_equal = jnp.asarray(eps_equal_prior, dtype=log_L_blocks.dtype)

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

    def single_sample(sample_key: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
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
        log_dZ = log_dX + log_L_blocks
        log_terms = log_dZ
        log_Z = _logsumexp(log_terms, axis=None)
        return log_Z, log_dZ, rho

    keys = random.split(key, num_Z_samples)
    log_Z_samples, log_dZ_samples, rho_samples = jax.lax.map(single_sample, keys, batch_size=batch_size)
    dZ_samples = LogSpace(log_dZ_samples)
    dZ_mean = dZ_samples.mean(axis=0)
    dZ_var = dZ_samples.var(axis=0)
    tiny = jnp.finfo(log_L_blocks.dtype).tiny
    return EvidenceSamples(
        log_Z_samples=log_Z_samples,
        log_dZ_mean=jnp.log(jnp.maximum(dZ_mean, tiny)),
        log_dZ_var=jnp.log(jnp.maximum(dZ_var, tiny)),
        rho_samples=rho_samples,
    )
