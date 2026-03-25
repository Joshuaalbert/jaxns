import dataclasses
from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp, random
from jax.scipy import special as jsp

from jaxns.log_semiring import LogSpace
from jaxns.pytree import PureDataclassPytree
from jaxns.types import FloatArray, IntArray, BoolArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class EvidenceSamples(PureDataclassPytree):
    log_Z_samples: FloatArray  # [num_Z_samples] samples of the evidence log Z from the MC shrinkage sampling
    H_samples: FloatArray  # [num_Z_samples] the information E[log_L - log_Z]
    log_dZ_mean: FloatArray  # [num_blocks] L_{g} * (X_{g-1} - X_g) averaged over MC chains
    log_dZ_var: FloatArray  # [num_blocks] variance of L_{g} * (X_{g-1} - X_g) over MC chains
    rho_samples: FloatArray  # [num_Z_samples] samples of the global rho parameter used in the MC shrinkage sampling
    eta_samples: FloatArray  # [num_Z_samples] estimated loose-reuse efficiency eta from phantom counts
    rho_eta_samples: FloatArray  # [num_Z_samples] sampled rho multiplied by estimated eta
    log_L_blocks: FloatArray  # [num_blocks] block levels derived from log_L_classic, padded with +inf
    block_first_idx: IntArray  # [num_blocks] first classic index per block, -1 for padded blocks

    def compute_phantom_efficiency(self, num_burn_in: int) -> FloatArray:
        """
        Determine the efficiency of adding more phantoms.
        Iff > 1 then for a given compute budget prefer decreasing number of live points, and increasing number of phantoms.
        """
        return _compute_phantom_efficiency(self, num_burn_in)


EvidenceSamples.register_pytree()


@partial(jax.jit, inline=True)
def _compute_phantom_efficiency(self: EvidenceSamples, num_burn_in: IntArray) -> FloatArray:
    # eta*rho*(num_burn_in - 1)
    return jnp.mean(self.rho_samples * self.eta_samples * (num_burn_in - 1) > 1)


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


def _boundary_counts_from_multiplicity(
        cluster_multiplicity: FloatArray,
        start_idx: IntArray,
        count_A_start_per_cluster: FloatArray,
        count_B_start_per_cluster: FloatArray,
        event_cluster_idx: IntArray,
        event_a_hi: IntArray,
        event_b_hi: IntArray,
        event_A_active: BoolArray,
        event_B_active: BoolArray,
        event_eq_idx: IntArray,
        event_eq_active: BoolArray,
        num_blocks: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    dtype = cluster_multiplicity.dtype

    dA_start = jnp.bincount(
        start_idx,
        weights=cluster_multiplicity * count_A_start_per_cluster,
        length=num_blocks + 1
    )
    dB_start = jnp.bincount(
        start_idx,
        weights=cluster_multiplicity * count_B_start_per_cluster,
        length=num_blocks + 1
    )

    event_weights = cluster_multiplicity[event_cluster_idx]
    dA_end = jnp.bincount(
        event_a_hi,
        weights=event_weights * jnp.asarray(event_A_active, dtype=dtype),
        length=num_blocks + 1
    )
    dB_end = jnp.bincount(
        event_b_hi,
        weights=event_weights * jnp.asarray(event_B_active, dtype=dtype),
        length=num_blocks + 1
    )

    dA = dA_start - dA_end
    dB = dB_start - dB_end
    A = jnp.cumsum(dA[:-1])
    B = jnp.cumsum(dB[:-1])

    eq_weights = jnp.where(event_eq_active, event_weights, jnp.zeros_like(event_weights))
    E = jnp.bincount(event_eq_idx, weights=eq_weights, length=num_blocks)
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


def _estimate_eta(
        K_per_block: FloatArray,
        A: FloatArray,
        num_phantom: int,
        block_valid_mask: BoolArray,
) -> FloatArray:
    dtype = K_per_block.dtype
    if num_phantom == 0:
        return jnp.zeros((), dtype=dtype)

    one = jnp.ones((), dtype=dtype)
    p = jnp.asarray(float(num_phantom), dtype=dtype)
    mask = jnp.logical_and(block_valid_mask, A > 0)

    K_safe = jnp.maximum(K_per_block, one)
    eta_min = one / (K_safe + one)
    eta_raw = A / (K_safe * p)
    eta_per_boundary = jnp.clip(eta_raw, eta_min, one)

    weights = jnp.where(mask, K_safe, jnp.zeros_like(K_safe))
    numer = jnp.sum(weights * eta_per_boundary)
    denom = jnp.sum(weights)
    return jnp.where(denom > 0, numer / denom, jnp.zeros_like(numer))


def _summarise_log_dz_samples(
        sample_fn,
        keys: PRNGKey,
        num_blocks: int,
        dtype: jnp.dtype,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    neg_inf = jnp.full((num_blocks,), -jnp.inf, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)

    def body(carry, sample_key):
        log_sum_dz, log_sum_dz_sq = carry
        log_Z, log_dZ, rho, eta, rho_eta, H = sample_fn(sample_key)
        log_sum_dz = jnp.logaddexp(log_sum_dz, log_dZ)
        log_sum_dz_sq = jnp.logaddexp(log_sum_dz_sq, two * log_dZ)
        return (log_sum_dz, log_sum_dz_sq), (log_Z, rho, eta, rho_eta, H)

    (log_sum_dz, log_sum_dz_sq), (log_Z_samples, rho_samples, eta_samples, rho_eta_samples, H_samples) = jax.lax.scan(
        body,
        init=(neg_inf, neg_inf),
        xs=keys,
    )

    log_num = jnp.log(jnp.asarray(keys.shape[0], dtype=dtype))
    log_dZ_mean = log_sum_dz - log_num
    log_dZ_second_moment = log_sum_dz_sq - log_num
    log_dZ_mean_sq = two * log_dZ_mean
    log_dZ_var = jnp.where(
        log_dZ_second_moment > log_dZ_mean_sq,
        _logdiffexp(log_dZ_second_moment, log_dZ_mean_sq),
        jnp.full_like(log_dZ_mean, -jnp.inf),
    )
    return log_Z_samples, rho_samples, eta_samples, rho_eta_samples, log_dZ_mean, log_dZ_var, H_samples


@partial(jax.jit, inline=True, static_argnames=["num_Z_samples", "rho_prior"])
def sample_mc_shrinkage(
        key: PRNGKey,
        log_L_constraints: FloatArray,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
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
        key: PRNGKey for Monte-Carlo sampling.
        log_L_constraints: ``[num_samples]`` cluster constraints for classic samples.
        log_L_classic: ``[num_samples]`` classic likelihood values.
        K_classic: ``[num_samples]`` classic live-point counts.
        valid_phantom: ``[num_samples]`` mask indicating which clusters have valid phantom draws.
        log_L_phantom: ``[num_samples, num_phantom]`` phantom likelihoods.
        num_samples: Number of valid leading entries in classic arrays.
        num_Z_samples: Number of Monte-Carlo evidence samples.
        eps_equal_prior: Small equality-mass prior for candlestick calibration.
        rho_grid: Optional grid of rho values; if None a default log-spaced grid is used.
        rho_prior: Prior for rho grid sampling, either ``"none"`` or ``"log"``.
        batch_size: Reserved for API compatibility; currently unused.

    Returns:
        EvidenceSamples with:
          - ``log_Z_samples``: evidence samples ``[num_Z_samples]``;
          - ``log_dZ_mean``: mean per-block contribution in log-space ``[num_blocks]``;
          - ``log_dZ_var``: variance per-block contribution in log-space ``[num_blocks]``;
          - ``rho_samples``: sampled global rho values ``[num_Z_samples]``;
          - ``eta_samples``: estimated loose-reuse efficiency values ``[num_Z_samples]``;
          - ``rho_eta_samples``: product ``rho * eta`` per sample ``[num_Z_samples]``;
          - ``log_L_blocks``: derived block levels padded with ``+inf``;
          - ``block_first_idx``: first classic index per block, ``-1`` for padded blocks.
    """
    N = log_L_classic.shape[0]
    sample_valid_mask = jnp.arange(N, dtype=jnp.int32) < num_samples
    positive_live_mask = K_classic > 0
    effective_sample_mask = jnp.logical_and(sample_valid_mask, positive_live_mask)
    has_phantom_dim = log_L_phantom.shape[1] > 0
    valid_classic = jnp.where(effective_sample_mask, log_L_classic, jnp.inf)
    if has_phantom_dim:
        log_L_blocks = jnp.unique(valid_classic, size=N, fill_value=jnp.inf)
        block_valid_mask = jnp.isfinite(log_L_blocks)
        first_idx_raw = jnp.searchsorted(valid_classic, log_L_blocks, side="left")
        first_idx_safe = jnp.clip(first_idx_raw, 0, jnp.maximum(N - 1, 0))
        block_first_idx = jnp.where(block_valid_mask, first_idx_raw.astype(jnp.int32), jnp.asarray(-1, jnp.int32))
        K_per_block = K_classic[first_idx_safe].astype(log_L_blocks.dtype)
        K_per_block = jnp.where(block_valid_mask, K_per_block, jnp.ones_like(K_per_block))
    else:
        log_L_blocks = valid_classic
        block_valid_mask = effective_sample_mask
        block_first_idx = jnp.where(block_valid_mask, jnp.arange(N, dtype=jnp.int32), jnp.asarray(-1, jnp.int32))
        K_per_block = jnp.where(block_valid_mask, K_classic.astype(log_L_blocks.dtype), jnp.ones_like(log_L_blocks))
    eps_equal = jnp.asarray(eps_equal_prior, dtype=log_L_blocks.dtype)

    dtype = log_L_blocks.dtype
    if rho_grid is None:
        rho_grid = _rho_grid_default(dtype=dtype)

    has_active_samples = jnp.any(effective_sample_mask)
    first_active_idx = jnp.argmax(effective_sample_mask.astype(jnp.int32))
    ref_log_L = log_L_classic[first_active_idx]
    constant_likelihood = jnp.logical_and(
        has_active_samples,
        jnp.all(jnp.where(effective_sample_mask, log_L_classic == ref_log_L, True))
    )

    num_blocks = log_L_blocks.shape[0]
    num_phantom = log_L_phantom.shape[1]
    num_valid_blocks = jnp.sum(block_valid_mask, dtype=jnp.int32)

    if num_phantom == 0:
        rho_fallback = rho_grid[-1]
        min_log_r = jnp.log(jnp.asarray(1e-300, dtype=dtype))

        def single_sample_no_phantom(sample_key: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
            ds = random.exponential(sample_key, shape=(num_blocks,)) / K_per_block
            log_r = jnp.maximum(-ds, min_log_r)
            log_X = jnp.cumsum(log_r)
            log_X_prev = jnp.concatenate([jnp.zeros((1,), dtype=log_r.dtype), log_X[:-1]], axis=0)
            log_dX = _logdiffexp(log_X_prev, log_X)
            log_dZ = jnp.where(block_valid_mask, log_dX + log_L_blocks, jnp.full_like(log_dX, -jnp.inf))
            log_Z = _logsumexp(log_dZ, axis=None)
            log_Z = jnp.where(constant_likelihood, ref_log_L, log_Z)
            eta = jnp.zeros((), dtype=dtype)
            rho_eta = jnp.zeros((), dtype=dtype)
            # H = E[log_L - log_Z]
            w = LogSpace(log_dZ) / LogSpace(log_Z)
            H = (w * LogSpace.from_signed_value(log_L_blocks - log_Z)).sum().value
            return log_Z, log_dZ, rho_fallback, eta, rho_eta, H

        keys = random.split(key, num_Z_samples)
        log_Z_samples, rho_samples, eta_samples, rho_eta_samples, log_dZ_mean, log_dZ_var, H_samples = _summarise_log_dz_samples(
            sample_fn=single_sample_no_phantom,
            keys=keys,
            num_blocks=num_blocks,
            dtype=dtype,
        )
        return EvidenceSamples(
            log_Z_samples=log_Z_samples,
            H_samples=H_samples,
            log_dZ_mean=jnp.where(block_valid_mask, log_dZ_mean, jnp.full_like(log_dZ_mean, -jnp.inf)),
            log_dZ_var=jnp.where(block_valid_mask, log_dZ_var, jnp.full_like(log_dZ_var, -jnp.inf)),
            rho_samples=rho_samples,
            eta_samples=eta_samples,
            rho_eta_samples=rho_eta_samples,
            log_L_blocks=log_L_blocks,
            block_first_idx=block_first_idx,
        )

    num_clusters = log_L_constraints.shape[0]
    effective_valid_phantom = valid_phantom & effective_sample_mask
    num_valid = jnp.sum(effective_valid_phantom, dtype=jnp.int32)
    valid_cluster_idx = jnp.nonzero(effective_valid_phantom, size=num_clusters, fill_value=0)[0]

    left_c = jnp.searchsorted(log_L_blocks, log_L_constraints, side='left')
    start_idx = jnp.where(jnp.isneginf(log_L_constraints), 0, left_c + 1)
    start_idx = jnp.minimum(start_idx, num_valid_blocks)
    start_idx = jnp.where(effective_valid_phantom, start_idx, 0)

    event_cluster_idx = jnp.repeat(jnp.arange(num_clusters, dtype=jnp.int32), repeats=num_phantom)
    event_start = start_idx[event_cluster_idx]
    event_logL = log_L_phantom.reshape((-1,))
    left_l = jnp.searchsorted(log_L_blocks, event_logL, side='left')
    event_a_hi = jnp.minimum(left_l + 1, num_valid_blocks)
    event_b_hi = jnp.minimum(left_l, num_valid_blocks)
    event_A_active = event_a_hi > event_start
    event_B_active = event_b_hi > event_start
    count_A_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_A_active, dtype=dtype),
        length=num_clusters
    )
    count_B_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_B_active, dtype=dtype),
        length=num_clusters
    )
    eq_ok = jnp.logical_and(left_l < num_valid_blocks, log_L_blocks[left_l] == event_logL)
    event_eq_idx = jnp.where(eq_ok, left_l, 0)
    event_eq_active = jnp.logical_and(eq_ok, event_eq_idx >= event_start)
    event_eq_active = jnp.logical_and(event_eq_active, effective_valid_phantom[event_cluster_idx])

    def single_sample(sample_key: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        key_boot, key_rho, key_r = random.split(sample_key, 3)

        safe_num_valid = jnp.maximum(num_valid, 1)
        boot_local_idx = random.randint(key_boot, shape=(num_clusters,), minval=0, maxval=safe_num_valid)
        boot_draw_mask = jnp.arange(num_clusters, dtype=jnp.int32) < num_valid
        boot_local_weights = jnp.where(boot_draw_mask, jnp.ones((num_clusters,), dtype=dtype), jnp.zeros((num_clusters,), dtype=dtype))
        local_multiplicity = jnp.bincount(boot_local_idx, weights=boot_local_weights, length=num_clusters)
        cluster_multiplicity = jnp.zeros((num_clusters,), dtype=dtype).at[valid_cluster_idx].add(local_multiplicity)

        A, B, E = _boundary_counts_from_multiplicity(
            cluster_multiplicity=cluster_multiplicity,
            start_idx=start_idx,
            count_A_start_per_cluster=count_A_start_per_cluster,
            count_B_start_per_cluster=count_B_start_per_cluster,
            event_cluster_idx=event_cluster_idx,
            event_a_hi=event_a_hi,
            event_b_hi=event_b_hi,
            event_A_active=event_A_active,
            event_B_active=event_B_active,
            event_eq_idx=event_eq_idx,
            event_eq_active=event_eq_active,
            num_blocks=num_blocks,
        )
        A = jnp.where(block_valid_mask, A, jnp.zeros_like(A))
        B = jnp.where(block_valid_mask, B, jnp.zeros_like(B))
        E = jnp.where(block_valid_mask, E, jnp.zeros_like(E))

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
        eta = _estimate_eta(K_per_block=K_per_block, A=A, num_phantom=num_phantom, block_valid_mask=block_valid_mask)
        rho_eta = rho * eta

        one = jnp.ones((), dtype=dtype)
        alpha = jnp.where(A > 0, K_per_block + rho * B, K_per_block)
        beta = jnp.where(A > 0, one + rho * (A - B), one)

        r = random.beta(key_r, a=alpha, b=beta, shape=(log_L_blocks.shape[0],))
        r = jnp.clip(r, 1e-300, 1.0)
        log_r = jnp.log(r)
        log_X = jnp.cumsum(log_r)
        log_X_prev = jnp.concatenate([jnp.zeros((1,), dtype=log_r.dtype), log_X[:-1]], axis=0)
        log_dX = _logdiffexp(log_X_prev, log_X)
        log_dZ = jnp.where(block_valid_mask, log_dX + log_L_blocks, jnp.full_like(log_dX, -jnp.inf))
        log_terms = log_dZ
        log_Z = _logsumexp(log_terms, axis=None)
        log_Z = jnp.where(constant_likelihood, ref_log_L, log_Z)
        w = LogSpace(log_dZ) / LogSpace(log_Z)
        H = (w * LogSpace.from_signed_value(log_L_blocks - log_Z)).sum().value
        return log_Z, log_dZ, rho, eta, rho_eta, H

    keys = random.split(key, num_Z_samples)
    log_Z_samples, rho_samples, eta_samples, rho_eta_samples, log_dZ_mean, log_dZ_var, H_samples = _summarise_log_dz_samples(
        sample_fn=single_sample,
        keys=keys,
        num_blocks=num_blocks,
        dtype=dtype,
    )
    return EvidenceSamples(
        log_Z_samples=log_Z_samples,
        log_dZ_mean=jnp.where(block_valid_mask, log_dZ_mean, jnp.full_like(log_dZ_mean, -jnp.inf)),
        log_dZ_var=jnp.where(block_valid_mask, log_dZ_var, jnp.full_like(log_dZ_var, -jnp.inf)),
        rho_samples=rho_samples,
        eta_samples=eta_samples,
        rho_eta_samples=rho_eta_samples,
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
        H_samples=H_samples
    )
