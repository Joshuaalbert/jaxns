"""
NumPy reference implementation: phantom-aware shrinkage evaluation and Monte Carlo evidence sampling.

This module is a **correctness reference** for later optimized implementations (e.g. JAX).
It prioritizes clarity, explicit assumptions, and careful documentation.

------------------------------------------------------------------------------
High-level summary
------------------------------------------------------------------------------
We represent nested sampling (NS) on a schedule of **strictly increasing** block log-likelihood values
    log_L_blocks[0] < log_L_blocks[1] < ... < log_L_blocks[G-1],
where each block corresponds to a distinct likelihood plateau level in the classic samples.

Between successive block levels a -> b, the *survival-mass shrinkage ratio* is
    r(a->b) = X(b)/X(a) = P_{theta ~ pi_a}( L(theta) > b ),
where pi_a is the constrained prior mu(. | L > a).

Classic NS supplies a process prior on block probabilities via the v3 Dirichlet:
    (p_>, p_=, p_<) ~ Dirichlet(K - m + 1, m + epsilon, 1 - epsilon),
where m is the likelihood plateau size.

Phantom samples are correlated states from constrained MCMC used to generate new live points.
They inform the three block probabilities, but correlation reduces effective information. We use:
  - per-block efficiency factors rho_g in (0,1] calibrated from cluster-bootstrap covariance
  - a conjugate Dirichlet update
        alpha_> += rho_g B_g
        alpha_= += rho_g E_g
        alpha_< += rho_g (A_g - B_g - E_g),
    where A = # {phantom L > a}, B = # {phantom L > b}, and E = # {phantom L == b}
    aggregated over *all loose clusters*
    (clusters generated under constraints c <= a).

------------------------------------------------------------------------------
New in v3
------------------------------------------------------------------------------
We add `sample_mc_shrinkage`, which:
  1) performs a **global cluster bootstrap** of phantom clusters for each evidence draw,
     preserving cross-boundary dependence induced by reusing loose samples; and
  2) samples rho from its **candlestick likelihood** on each bootstrap replicate (rather than fixing rho).

This improves uncertainty characterization over sampling each boundary independently from its
marginal Dirichlet posterior under a fixed rho.

------------------------------------------------------------------------------
Scope assumptions (per user spec)
------------------------------------------------------------------------------
- log_L_blocks is strictly increasing, already sorted, and does NOT include the root block.
  Root is implicitly log_L_root = -inf.
- Unaugmented schedule for now: log_L_blocks == np.unique(log_L_classic) exactly.
  (See design doc for what needs to change to support augmented schedules.)

------------------------------------------------------------------------------
Caution
------------------------------------------------------------------------------
- This module treats each boundary's block probability vector as independent given (rho_g, bootstrap replicate).
  Dependence is introduced through shared bootstrap resampling of clusters and shared rho sampling, but we do
  not build a full joint likelihood for all boundaries simultaneously.
- "candlestick likelihood" is an approximate calibration model (Gaussian + scaled chi-square).
  It is intended as a robust practical calibration, not a mechanistic model of MCMC autocorrelation.
"""

from typing import NamedTuple, Tuple, Optional

import numpy as np

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


def estimate_raw_rho_g_from_bootstrap_covariance(
        *,
        A: FloatArray,
        B: FloatArray,
        E: FloatArray,
        bootstrap_covariance: FloatArray,
        fallback_rho: float = 1.0,
) -> FloatArray:
    """Estimate raw per-block `rho_g` using the paper rank/trace formula."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    E = np.asarray(E, dtype=float)
    bootstrap_covariance = np.asarray(bootstrap_covariance, dtype=float)
    fallback = float(np.clip(fallback_rho, np.finfo(float).tiny, 1.0))

    out = np.full(A.shape, fallback, dtype=float)
    for idx in np.ndindex(A.shape):
        A_g = float(A[idx])
        if A_g <= 0.0:
            continue
        q_gt = float(B[idx]) / A_g
        q_eq = float(E[idx]) / A_g
        sigma = np.asarray(
            [
                [q_gt * (1.0 - q_gt) / A_g, -q_gt * q_eq / A_g],
                [-q_gt * q_eq / A_g, q_eq * (1.0 - q_eq) / A_g],
            ],
            dtype=float,
        )
        rank = float(np.linalg.matrix_rank(sigma))
        if rank <= 0.0:
            continue
        denominator = float(
            np.trace(np.linalg.pinv(sigma) @ bootstrap_covariance[idx])
        )
        raw = rank / denominator if denominator > 0.0 else np.nan
        if np.isfinite(raw) and raw > 0.0:
            out[idx] = raw
    return np.clip(out, np.finfo(float).tiny, 1.0)


def fit_low_order_rho_g_curve(
        *,
        raw_rho_g: FloatArray,
        race_time: FloatArray,
        valid_mask: BoolArray,
        polynomial_order: int = 2,
        fallback_rho: float = 1.0,
) -> FloatArray:
    """Fit a bounded low-order rho curve against normalized race time."""
    raw_rho_g = np.asarray(raw_rho_g, dtype=float)
    race_time = np.asarray(race_time, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    fallback = float(np.clip(fallback_rho, np.finfo(float).tiny, 1.0))

    fit_mask = (
        valid_mask
        & np.isfinite(raw_rho_g)
        & (raw_rho_g > 0.0)
        & np.isfinite(race_time)
    )
    if not np.any(fit_mask):
        return np.where(valid_mask, fallback, np.nan)

    max_time = float(np.max(np.where(fit_mask, race_time, 0.0)))
    normalized_time = np.where(max_time > 0.0, race_time / max_time, 0.0)
    normalized_time = np.where(np.isfinite(normalized_time), normalized_time, 0.0)

    powers = np.arange(int(polynomial_order) + 1, dtype=float)
    design = normalized_time[:, None] ** powers[None, :]
    weights = fit_mask.astype(float)
    weighted_design = design * weights[:, None]
    weighted_y = np.clip(raw_rho_g, np.finfo(float).tiny, 1.0) * weights
    coeffs = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    fitted = design @ coeffs
    fitted = np.where(np.isfinite(fitted), fitted, fallback)
    fitted = np.clip(fitted, np.finfo(float).tiny, 1.0)
    return np.where(valid_mask, fitted, np.nan)


class EvidenceSamples(NamedTuple):
    log_Z_samples: FloatArray  # [num_Z_samples] samples of the evidence log Z from the MC shrinkage sampling
    log_dZ_mean: FloatArray  # [num_blocks] L_{g} * (X_{g-1} - X_g) averaged over MC chains
    log_dZ_var: FloatArray  # [num_blocks] variance of L_{g} * (X_{g-1} - X_g) over MC chains
    rho_samples: FloatArray  # [num_Z_samples] samples of the global rho parameter used in the MC shrinkage sampling
    rho_values: FloatArray  # [num_blocks] raw per-block rho estimates aligned with log_L_blocks
    rho_fit: FloatArray  # [num_blocks] fitted rho curve aligned with log_L_blocks
    eta_samples: FloatArray  # [num_Z_samples] estimated loose-reuse efficiency eta from phantom counts
    rho_eta_samples: FloatArray  # [num_Z_samples] sampled rho multiplied by estimated eta
    log_L_blocks: FloatArray  # [num_blocks] block levels derived from log_L_classic, padded with +inf
    block_first_idx: IntArray  # [num_blocks] first classic index per block, -1 for padded blocks


# -----------------------------------------------------------------------------
# Numerics helpers
# -----------------------------------------------------------------------------

def _logsumexp(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Stable log-sum-exp."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    finite = np.isfinite(x_max)
    x_max_safe = np.where(finite, x_max, 0.0)
    out = x_max_safe + np.log(np.sum(np.exp(x - x_max_safe), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
        finite = np.squeeze(finite, axis=axis)
    else:
        out = out.item()
        finite = bool(finite.item())
    return np.where(finite, out, -np.inf)


def _logdiffexp(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """
    log(exp(log_a) - exp(log_b)), assuming log_a >= log_b elementwise.
    """
    log_a = np.asarray(log_a)
    log_b = np.asarray(log_b)
    diff = log_b - log_a  # <= 0
    return log_a + np.log1p(-np.exp(diff))


def _epsilon_for_block_size(block_size: np.ndarray) -> np.ndarray:
    """Paper default equality-atom prior policy."""
    block_size = np.asarray(block_size)
    return np.where(block_size == 1, 1e-6, 0.5).astype(float)


def _classic_dirichlet_parameters(
        K_per_block: np.ndarray,
        block_size: np.ndarray,
        block_valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classic v3 Dirichlet parameters and epsilon values."""
    K_per_block = np.asarray(K_per_block, dtype=float)
    block_size_float = np.asarray(block_size, dtype=float)
    block_valid_mask = np.asarray(block_valid_mask, dtype=bool)
    epsilon = _epsilon_for_block_size(block_size)
    alpha = np.stack(
        [
            K_per_block - block_size_float + 1.0,
            block_size_float + epsilon,
            1.0 - epsilon,
        ],
        axis=-1,
    )
    alpha = np.where(block_valid_mask[:, None], alpha, 0.0)
    return alpha, np.where(block_valid_mask, epsilon, 0.0)


def _phantom_conditioned_dirichlet_parameters(
        K_per_block: np.ndarray,
        block_size: np.ndarray,
        block_valid_mask: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
        rho_g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phantom-conditioned v3 Dirichlet parameters for `(p_>, p_=, p_<)`."""
    alpha, epsilon = _classic_dirichlet_parameters(
        K_per_block=K_per_block,
        block_size=block_size,
        block_valid_mask=block_valid_mask,
    )
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    E = np.asarray(E, dtype=float)
    rho_g = np.asarray(rho_g, dtype=float)
    has_negative_counts = (
        np.any(A[block_valid_mask] < 0.0)
        or np.any(B[block_valid_mask] < 0.0)
        or np.any(E[block_valid_mask] < 0.0)
    )
    if has_negative_counts:
        raise ValueError("Phantom Dirichlet counts must be non-negative.")
    if np.any(B[block_valid_mask] + E[block_valid_mask] > A[block_valid_mask]):
        raise ValueError(
            "Invalid Dirichlet phantom count relation: B_g + E_g must be <= A_g."
        )
    if (
            not np.all(np.isfinite(rho_g[block_valid_mask]))
            or np.any(rho_g[block_valid_mask] <= 0.0)
            or np.any(rho_g[block_valid_mask] > 1.0)
    ):
        raise ValueError("rho_g must be finite, positive, and <= 1 on valid blocks.")
    alpha = alpha.copy()
    alpha[:, 0] += rho_g * B
    alpha[:, 1] += rho_g * E
    alpha[:, 2] += rho_g * (A - B - E)
    alpha = np.where(block_valid_mask[:, None], alpha, 0.0)
    return alpha, epsilon


def _dirichlet_probability_means(alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha0 = np.sum(alpha, axis=-1)
    p_gt = np.where(alpha0 > 0.0, alpha[:, 0] / alpha0, 0.0)
    p_eq = np.where(alpha0 > 0.0, alpha[:, 1] / alpha0, 0.0)
    p_lt = np.where(alpha0 > 0.0, alpha[:, 2] / alpha0, 0.0)
    return p_gt, p_eq, p_lt


def _sample_dirichlet_probabilities(
        rng: np.random.Generator,
        alpha: np.ndarray,
        block_valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_alpha = np.where(alpha > 0.0, alpha, 1.0)
    gamma = rng.gamma(shape=safe_alpha)
    probs = gamma / np.sum(gamma, axis=-1, keepdims=True)
    probs = np.where(block_valid_mask[:, None], probs, 0.0)
    return probs[:, 0], probs[:, 1], probs[:, 2]


# -----------------------------------------------------------------------------
# Dirichlet candlestick pieces
# -----------------------------------------------------------------------------

def _dirichlet_mu_cov_2d(K: float, eps_equal: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return (mu2, Sigma2, alpha0) for (p_>, p_=) under Dirichlet(alpha_>, alpha_=, alpha_<) with:
        alpha_> = K
        alpha_= = eps_equal
        alpha_< = 1 - eps_equal

    alpha0 = K + 1.
    """
    if not (0.0 < eps_equal < 1.0):
        raise ValueError("eps_equal must be in (0,1).")
    alpha = np.array([float(K), float(eps_equal), float(1.0 - eps_equal)], dtype=float)
    alpha0 = float(np.sum(alpha))

    mu = alpha / alpha0

    denom = alpha0 ** 2 * (alpha0 + 1.0)
    cov_full = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                cov_full[i, j] = alpha[i] * (alpha0 - alpha[i]) / denom
            else:
                cov_full[i, j] = -alpha[i] * alpha[j] / denom

    mu2 = mu[:2].copy()
    Sigma2 = cov_full[:2, :2].copy()
    return mu2, Sigma2, alpha0


def _candlestick_nll(
        rho: float,
        d2: np.ndarray,
        alpha0: np.ndarray,
        A: np.ndarray,
        *,
        dim: int = 2,
) -> float:
    """
    Negative log-likelihood (up to constant) under the scaled-chi-square candlestick model:

        d2_g ~ kappa_g * ChiSquare(dim)
        kappa_g = 1 + alpha0_g / (rho * A_g)

    nll(rho) = sum_g [ (dim/2) * log(kappa_g) + d2_g / (2*kappa_g) ].

    Parameters
    ----------
    rho:
        Efficiency in (0,1].
    d2:
        Mahalanobis distances per boundary (>=0).
    alpha0:
        Dirichlet prior total concentration per boundary (K+1).
    A:
        Qualifying counts per boundary (A>0).
    """
    rho = float(rho)
    if rho <= 0.0:
        return np.inf
    kappa = 1.0 + alpha0 / (rho * A)
    return float(np.sum((dim / 2.0) * np.log(kappa) + d2 / (2.0 * kappa)))


def _rho_grid_default(grid_size: int = 200, rho_min: float = 1e-6, rho_max: float = 1.0) -> np.ndarray:
    """Log-spaced rho grid."""
    return np.logspace(np.log10(rho_min), np.log10(rho_max), int(grid_size))


def _sample_rho_from_likelihood(
        rng: np.random.Generator,
        d2: np.ndarray,
        alpha0: np.ndarray,
        A: np.ndarray,
        rho_grid: np.ndarray,
        *,
        dim: int = 2,
        prior: str = "none",
) -> float:
    """
    Sample rho from its (approximate) candlestick likelihood over a discrete grid.

    We treat:
        p(rho | data) ∝ L(data | rho) * prior(rho)

    Supported priors:
        - "none":   prior(rho) ∝ 1  (sample from likelihood only; user request)
        - "log":    prior(rho) ∝ 1/rho  (uniform in log rho)

    Returns a rho value from rho_grid.
    """
    rho_grid = np.asarray(rho_grid, dtype=float)
    if rho_grid.ndim != 1:
        raise ValueError("rho_grid must be 1D.")
    if d2.size == 0:
        return 1.0

    # Compute log weights = -nll + log prior
    logw = np.empty_like(rho_grid)
    for i, rho in enumerate(rho_grid):
        nll = _candlestick_nll(rho, d2=d2, alpha0=alpha0, A=A, dim=dim)
        logw[i] = -nll

    if prior == "none":
        pass
    elif prior == "log":
        logw = logw - np.log(rho_grid)
    else:
        raise ValueError(f"Unknown prior={prior!r}. Use 'none' or 'log'.")

    # Stabilize
    m = np.max(logw)
    w = np.exp(logw - m)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0.0:
        return float(rho_grid[-1])  # fallback ~ 1
    p = w / s
    idx = rng.choice(len(rho_grid), p=p)
    return float(rho_grid[idx])


def _fit_rho_mle(
        d2: np.ndarray,
        alpha0: np.ndarray,
        A: np.ndarray,
        rho_grid: np.ndarray,
        *,
        dim: int = 2,
) -> float:
    """MLE of rho over the grid."""
    best_rho = 1.0
    best_nll = np.inf
    for rho in rho_grid:
        nll = _candlestick_nll(rho, d2=d2, alpha0=alpha0, A=A, dim=dim)
        if nll < best_nll:
            best_nll = nll
            best_rho = float(rho)
    return float(np.clip(best_rho, np.min(rho_grid), np.max(rho_grid)))


def _estimate_eta(
        K_per_block: np.ndarray,
        A: np.ndarray,
        num_phantom: int,
        block_valid_mask: np.ndarray,
) -> float:
    if num_phantom == 0:
        return 0.0

    K_safe = np.maximum(np.asarray(K_per_block, dtype=float), 1.0)
    A = np.asarray(A, dtype=float)
    block_valid_mask = np.asarray(block_valid_mask, dtype=bool)
    mask = block_valid_mask & (A > 0.0)

    eta_min = 1.0 / (K_safe + 1.0)
    eta_raw = A / (K_safe * float(num_phantom))
    eta_per_boundary = np.clip(eta_raw, eta_min, 1.0)

    weights = np.where(mask, K_safe, 0.0)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(weights * eta_per_boundary) / denom)


# -----------------------------------------------------------------------------
# Boundary counting from phantom clusters (core "loose reuse" statistic)
# -----------------------------------------------------------------------------

def _boundary_counts_from_clusters(
        log_L_blocks: np.ndarray,
        log_L_constraints: np.ndarray,
        log_L_phantom: np.ndarray,
        *,
        cluster_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pooled phantom counts (A,B,E) per block boundary using *all loose clusters*.

    For boundary g:
        a = -inf if g==0 else log_L_blocks[g-1]
        b = log_L_blocks[g]

    Eligible clusters are those with constraint c <= a.

    Counts:
        A_g = #{phantom L > a} among eligible clusters
        B_g = #{phantom L > b} among eligible clusters
        E_g = #{phantom L == b} among eligible clusters   (for candlestick only)

    Parameters
    ----------
    log_L_blocks:
        Strictly increasing block levels, shape [G].
    log_L_constraints:
        Cluster constraints c_i, shape [N].
    log_L_phantom:
        Phantom likelihoods per cluster, shape [N, P].
    cluster_indices:
        Optional 1D array selecting clusters (with replacement allowed). If provided,
        counts are computed on the selected clusters and duplicates count multiple times.

    Returns
    -------
    (A, B, E): each shape [G], float counts.
    """
    log_L_blocks = np.asarray(log_L_blocks, dtype=float)
    c = np.asarray(log_L_constraints, dtype=float)
    ph = np.asarray(log_L_phantom, dtype=float)

    if cluster_indices is not None:
        cluster_indices = np.asarray(cluster_indices, dtype=int)
        c = c[cluster_indices]
        ph = ph[cluster_indices]

    G = log_L_blocks.size
    A = np.zeros(G, dtype=float)
    B = np.zeros(G, dtype=float)
    E = np.zeros(G, dtype=float)

    for g in range(G):
        a = -np.inf if g == 0 else log_L_blocks[g - 1]
        b = log_L_blocks[g]

        eligible = c <= a
        if not np.any(eligible):
            continue

        ph_e = ph[eligible]
        # A: qualifiers
        A_g = float(np.sum(ph_e > a))
        if A_g <= 0.0:
            continue
        B_g = float(np.sum(ph_e > b))
        E_g = float(np.sum(ph_e == b))

        A[g] = A_g
        B[g] = B_g
        E[g] = E_g

    return A, B, E


def _cluster_count_matrices(
        log_L_blocks: np.ndarray,
        log_L_constraints: np.ndarray,
        log_L_phantom: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_clusters = log_L_constraints.shape[0]
    num_blocks = log_L_blocks.shape[0]
    A_by_cluster = np.zeros((num_clusters, num_blocks), dtype=float)
    B_by_cluster = np.zeros((num_clusters, num_blocks), dtype=float)
    E_by_cluster = np.zeros((num_clusters, num_blocks), dtype=float)
    for cluster_idx in range(num_clusters):
        A, B, E = _boundary_counts_from_clusters(
            log_L_blocks=log_L_blocks,
            log_L_constraints=log_L_constraints[cluster_idx:cluster_idx + 1],
            log_L_phantom=log_L_phantom[cluster_idx:cluster_idx + 1],
        )
        A_by_cluster[cluster_idx] = A
        B_by_cluster[cluster_idx] = B
        E_by_cluster[cluster_idx] = E
    return A_by_cluster, B_by_cluster, E_by_cluster


def _bootstrap_covariance_from_cluster_counts(
        A_by_cluster: np.ndarray,
        B_by_cluster: np.ndarray,
        E_by_cluster: np.ndarray,
        valid_cluster_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact cluster-bootstrap covariance of q=(B/A, E/A)."""
    A_by_cluster = np.asarray(A_by_cluster, dtype=float)
    B_by_cluster = np.asarray(B_by_cluster, dtype=float)
    E_by_cluster = np.asarray(E_by_cluster, dtype=float)
    if A_by_cluster.ndim != 2:
        raise ValueError("A_by_cluster must have shape [num_clusters, num_blocks].")
    num_clusters, num_blocks = A_by_cluster.shape
    if num_clusters == 0:
        return np.zeros((num_blocks, 2, 2), dtype=float)

    if valid_cluster_mask is None:
        valid_cluster_mask = np.ones((num_clusters,), dtype=bool)
    valid_cluster_mask = np.asarray(valid_cluster_mask, dtype=bool)
    A_by_cluster = A_by_cluster[valid_cluster_mask]
    B_by_cluster = B_by_cluster[valid_cluster_mask]
    E_by_cluster = E_by_cluster[valid_cluster_mask]
    num_valid = A_by_cluster.shape[0]
    if num_valid == 0:
        return np.zeros((num_blocks, 2, 2), dtype=float)

    covariance = np.zeros((num_blocks, 2, 2), dtype=float)
    for block_idx in range(num_blocks):
        A_values = np.rint(A_by_cluster[:, block_idx]).astype(np.int64)
        if np.any(A_values < 0):
            raise ValueError("Bootstrap A counts must be non-negative.")
        max_total_A = int(num_valid * np.max(A_values, initial=0))
        if max_total_A == 0:
            continue

        B_values = B_by_cluster[:, block_idx]
        E_values = E_by_cluster[:, block_idx]
        probability = np.zeros((max_total_A + 1,), dtype=float)
        mean_B = np.zeros_like(probability)
        mean_E = np.zeros_like(probability)
        second_BB = np.zeros_like(probability)
        second_BE = np.zeros_like(probability)
        second_EE = np.zeros_like(probability)
        probability[0] = 1.0
        cluster_weight = 1.0 / float(num_valid)

        for _ in range(num_valid):
            next_probability = np.zeros_like(probability)
            next_mean_B = np.zeros_like(probability)
            next_mean_E = np.zeros_like(probability)
            next_second_BB = np.zeros_like(probability)
            next_second_BE = np.zeros_like(probability)
            next_second_EE = np.zeros_like(probability)
            for A_offset, B_value, E_value in zip(
                    A_values,
                    B_values,
                    E_values,
                    strict=True,
            ):
                stop = max_total_A + 1 - int(A_offset)
                target = slice(int(A_offset), max_total_A + 1)
                source = slice(0, stop)
                probability_shift = probability[source]
                mean_B_shift = mean_B[source]
                mean_E_shift = mean_E[source]
                second_BB_shift = second_BB[source]
                second_BE_shift = second_BE[source]
                second_EE_shift = second_EE[source]

                next_probability[target] += cluster_weight * probability_shift
                next_mean_B[target] += cluster_weight * (
                        mean_B_shift + B_value * probability_shift
                )
                next_mean_E[target] += cluster_weight * (
                        mean_E_shift + E_value * probability_shift
                )
                next_second_BB[target] += cluster_weight * (
                        second_BB_shift
                        + 2.0 * B_value * mean_B_shift
                        + B_value * B_value * probability_shift
                )
                next_second_BE[target] += cluster_weight * (
                        second_BE_shift
                        + B_value * mean_E_shift
                        + E_value * mean_B_shift
                        + B_value * E_value * probability_shift
                )
                next_second_EE[target] += cluster_weight * (
                        second_EE_shift
                        + 2.0 * E_value * mean_E_shift
                        + E_value * E_value * probability_shift
                )

            probability = next_probability
            mean_B = next_mean_B
            mean_E = next_mean_E
            second_BB = next_second_BB
            second_BE = next_second_BE
            second_EE = next_second_EE

        A_support = np.arange(max_total_A + 1, dtype=float)
        active = A_support > 0.0
        active_probability = float(np.sum(probability[active]))
        if active_probability <= 0.0:
            continue

        A_active = A_support[active]
        mean_gt = float(
            np.sum(mean_B[active] / A_active) / active_probability
        )
        mean_eq = float(
            np.sum(mean_E[active] / A_active) / active_probability
        )
        A_active_sq = A_active * A_active
        second_gt_gt = float(
            np.sum(second_BB[active] / A_active_sq) / active_probability
        )
        second_gt_eq = float(
            np.sum(second_BE[active] / A_active_sq) / active_probability
        )
        second_eq_eq = float(
            np.sum(second_EE[active] / A_active_sq) / active_probability
        )

        cov_gt_gt = max(second_gt_gt - mean_gt * mean_gt, 0.0)
        cov_gt_eq = second_gt_eq - mean_gt * mean_eq
        cov_eq_eq = max(second_eq_eq - mean_eq * mean_eq, 0.0)
        covariance[block_idx] = np.asarray(
            [
                [cov_gt_gt, cov_gt_eq],
                [cov_gt_eq, cov_eq_eq],
            ],
            dtype=float,
        )
    return covariance


def _candlestick_d2_per_boundary(
        K_per_block: np.ndarray,
        eps_equal_prior: float,
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Mahalanobis distances d2 for boundaries with A>0.

    Returns (d2, alpha0, A_used), each 1D arrays aligned over usable boundaries.
    """
    K_per_block = np.asarray(K_per_block, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    E = np.asarray(E, dtype=float)

    usable_idx = np.where(A > 0.0)[0]
    if usable_idx.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    d2_list = []
    alpha0_list = []
    A_list = []

    for g in usable_idx:
        K = float(K_per_block[g])
        A_g = float(A[g])
        p_hat = np.array([B[g] / A_g, E[g] / A_g], dtype=float)

        mu2, Sigma2, alpha0 = _dirichlet_mu_cov_2d(K=K, eps_equal=eps_equal_prior)

        # Mahalanobis
        u = p_hat - mu2
        try:
            inv = np.linalg.inv(Sigma2)
        except np.linalg.LinAlgError:
            # Should not happen with eps in (0,1), but be safe: skip.
            continue
        d2 = float(u.T @ inv @ u)
        d2_list.append(d2)
        alpha0_list.append(alpha0)
        A_list.append(A_g)

    return np.array(d2_list, dtype=float), np.array(alpha0_list, dtype=float), np.array(A_list, dtype=float)


def _validate_phantom_metadata(
        *,
        log_L_constraints: FloatArray,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
) -> None:
    if log_L_classic.ndim != 1:
        raise ValueError("log_L_classic must be one-dimensional.")
    num_clusters = log_L_classic.shape[0]
    if K_classic.shape != (num_clusters,):
        raise ValueError("K_classic shape must match log_L_classic.")
    if log_L_constraints.shape != (num_clusters,):
        raise ValueError("log_L_constraints shape must match the cluster axis.")
    if valid_phantom.ndim != 1:
        raise ValueError(
            "valid_phantom must be a one-dimensional per-cluster mask, not a "
            "per-phantom mask."
        )
    if valid_phantom.shape != (num_clusters,):
        raise ValueError("valid_phantom shape must match the cluster axis.")
    if log_L_phantom.ndim != 2:
        raise ValueError("log_L_phantom must be a two-dimensional array.")
    if log_L_phantom.shape[0] != num_clusters:
        raise ValueError("log_L_phantom shape must match the cluster axis.")

    n = int(np.asarray(num_samples).item())
    if n < 0 or n > num_clusters:
        raise ValueError("num_samples is outside the available cluster range.")
    if np.any(valid_phantom[n:]):
        raise ValueError(
            "valid_phantom contains a stale association beyond num_samples."
        )
    if n == 0:
        return
    active = K_classic[:n] > 0
    strict_violations = active & (log_L_classic[:n] <= log_L_constraints[:n])
    if np.any(strict_violations):
        bad = np.where(strict_violations)[0][0]
        raise ValueError(
            "Strict contour violation for active sample "
            f"{bad}: log_L_classic={log_L_classic[bad]} must be greater than "
            f"log_L_constraint={log_L_constraints[bad]}."
        )


def sample_mc_shrinkage(
        seed: int,
        log_L_constraints: FloatArray,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
        num_Z_samples: int,
        *,
        eps_equal_prior: float = 1e-3,
        rho_grid: Optional[np.ndarray] = None,
        rho_prior: str = "none",
) -> EvidenceSamples:
    """
    Improved Monte Carlo evidence sampling:
      - Global cluster bootstrap of phantom clusters (one bootstrap resample per evidence draw)
      - Sample rho from candlestick likelihood on the bootstrap replicate
      - Sample v3 Dirichlet block probabilities for the actual shrinkage path

    Algorithm (per evidence sample s)
    --------------------------------
    1) Resample phantom clusters with replacement from the set of clusters with valid_phantom=True.
       (Global bootstrap: the same resample is used for all boundaries, preserving cross-boundary dependence.)

    2) On the resampled cluster set, compute pooled counts (A_g, B_g, E_g) for every boundary g using the
       same "loose reuse" rule: clusters with constraint c <= a_g are eligible at boundary g.

    3) Compute candlestick Mahalanobis distances d2_g for boundaries with A_g>0, and sample rho from the
       discrete likelihood L(rho) ∝ exp(-nll(rho)) on a grid.

    4) Form per-boundary v3 Dirichlet posteriors:
          alpha_>,g = K_g - m_g + 1 + rho_g * B_g
          alpha_=,g = m_g + epsilon_g + rho_g * E_g
          alpha_<,g = 1 - epsilon_g + rho_g * (A_g - B_g - E_g).

    5) Sample p_>,g from the Dirichlet blocks, compute the X trajectory and Z.

    Notes
    -----
    - This procedure captures two effects that the baseline compute_mc_shrinkage misses:
        (i) cross-boundary dependence from reusing the same loose clusters; and
        (ii) uncertainty in rho (rho is sampled, not fixed).
    - It remains an approximation: it uses the candlestick likelihood and a Dirichlet conjugate update with
      effective counts rho_g*A_g. This is intended as a robust stochastic alternative to fixed-rho marginal
      sampling, without needing a full Markov time-series model.
    """
    if num_Z_samples < 1:
        raise ValueError("num_Z_samples must be >= 1.")
    rng = np.random.default_rng(int(seed))

    log_L_constraints = np.asarray(log_L_constraints, dtype=float)
    log_L_classic = np.asarray(log_L_classic, dtype=float)
    K_classic = np.asarray(K_classic, dtype=float)
    valid_phantom = np.asarray(valid_phantom, dtype=bool)
    log_L_phantom = np.asarray(log_L_phantom, dtype=float)
    _validate_phantom_metadata(
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
    )

    N = log_L_classic.shape[0]
    num_samples_int = int(np.asarray(num_samples).item())
    sample_valid_mask = np.arange(N, dtype=np.int32) < num_samples_int
    positive_live_mask = K_classic > 0
    effective_sample_mask = sample_valid_mask & positive_live_mask
    valid_classic = np.where(effective_sample_mask, log_L_classic, np.inf)
    sorted_order = np.argsort(valid_classic, kind="stable")
    sorted_log_L = valid_classic[sorted_order]
    sorted_K = K_classic[sorted_order]
    unique_valid, starts, block_counts = np.unique(
        sorted_log_L,
        return_index=True,
        return_counts=True,
    )
    log_L_blocks = np.full((N,), np.inf, dtype=float)
    log_L_blocks[:unique_valid.shape[0]] = unique_valid
    block_valid_mask = np.isfinite(log_L_blocks)
    block_first_idx = np.full((N,), -1, dtype=np.int32)
    block_first_idx[:unique_valid.shape[0]] = sorted_order[starts].astype(np.int32)
    block_first_idx = np.where(block_valid_mask, block_first_idx, np.int32(-1))
    block_size = np.zeros((N,), dtype=np.int32)
    block_size[:unique_valid.shape[0]] = np.where(
        np.isfinite(unique_valid),
        block_counts,
        0,
    ).astype(np.int32)
    first_idx_safe = np.clip(starts, 0, max(N - 1, 0))
    K_per_block = np.ones((N,), dtype=float)
    K_per_block[:unique_valid.shape[0]] = sorted_K[first_idx_safe].astype(float)
    K_per_block = np.where(block_valid_mask, K_per_block, np.ones_like(K_per_block))
    eps_equal = float(eps_equal_prior)
    if rho_grid is None:
        rho_grid = _rho_grid_default()

    G = log_L_blocks.size
    num_phantom = int(log_L_phantom.shape[1])
    if num_phantom == 0:
        rho_fallback = float(rho_grid[-1])
        zero_counts = np.zeros((G,), dtype=float)
        dirichlet_alpha, _ = _phantom_conditioned_dirichlet_parameters(
            K_per_block=K_per_block,
            block_size=block_size,
            block_valid_mask=block_valid_mask,
            A=zero_counts,
            B=zero_counts,
            E=zero_counts,
            rho_g=np.full((G,), rho_fallback, dtype=float),
        )
        log_dZ_out = np.empty((num_Z_samples, G), dtype=float)
        log_Z_out = np.empty((num_Z_samples,), dtype=float)
        for s in range(num_Z_samples):
            p_gt, _, _ = _sample_dirichlet_probabilities(
                rng,
                dirichlet_alpha,
                block_valid_mask,
            )
            r = np.where(block_valid_mask, p_gt, 1.0)
            r = np.clip(r, 1e-300, 1.0)
            log_X_prev = 0.0
            log_terms = np.empty(G, dtype=float)
            for g in range(G):
                log_X_curr = log_X_prev + np.log(r[g])
                log_dX = _logdiffexp(np.array(log_X_prev), np.array(log_X_curr)).item()
                if block_valid_mask[g]:
                    log_terms[g] = log_dX + log_L_blocks[g]
                else:
                    log_terms[g] = -np.inf
                log_dZ_out[s, g] = log_terms[g]
                log_X_prev = log_X_curr
            log_Z_out[s] = _logsumexp(log_terms, axis=None).item()
        dZ_samples = np.exp(log_dZ_out)
        dZ_mean = np.mean(dZ_samples, axis=0)
        dZ_var = np.var(dZ_samples, axis=0)
        tiny = np.finfo(float).tiny
        return EvidenceSamples(
            log_Z_samples=log_Z_out,
            log_dZ_mean=np.where(block_valid_mask, np.log(np.maximum(dZ_mean, tiny)), -np.inf),
            log_dZ_var=np.where(block_valid_mask, np.log(np.maximum(dZ_var, tiny)), -np.inf),
            rho_samples=np.full((num_Z_samples,), rho_fallback, dtype=float),
            rho_values=np.where(block_valid_mask, rho_fallback, np.nan),
            rho_fit=np.where(block_valid_mask, rho_fallback, np.nan),
            eta_samples=np.zeros((num_Z_samples,), dtype=float),
            rho_eta_samples=np.zeros((num_Z_samples,), dtype=float),
            log_L_blocks=log_L_blocks,
            block_first_idx=block_first_idx,
        )

    # Extract valid phantom clusters
    effective_valid_phantom = valid_phantom & effective_sample_mask
    valid_idx = np.where(effective_valid_phantom)[0]
    if valid_idx.size == 0:
        c_valid = log_L_constraints
        ph_valid = log_L_phantom
    else:
        c_valid = log_L_constraints[valid_idx]
        ph_valid = log_L_phantom[valid_idx]
    N_valid = valid_idx.size

    log_dZ_out = np.empty((num_Z_samples, G), dtype=float)
    log_Z_out = np.empty((num_Z_samples,), dtype=float)
    rho_samples = np.empty((num_Z_samples,), dtype=float)
    eta_samples = np.empty((num_Z_samples,), dtype=float)
    rho_eta_samples = np.empty((num_Z_samples,), dtype=float)
    if N_valid == 0:
        A_by_cluster = np.zeros((0, G), dtype=float)
        B_by_cluster = np.zeros((0, G), dtype=float)
        E_by_cluster = np.zeros((0, G), dtype=float)
    else:
        A_by_cluster, B_by_cluster, E_by_cluster = _cluster_count_matrices(
            log_L_blocks=log_L_blocks,
            log_L_constraints=c_valid,
            log_L_phantom=ph_valid,
        )
    A_full = np.sum(A_by_cluster, axis=0)
    B_full = np.sum(B_by_cluster, axis=0)
    E_full = np.sum(E_by_cluster, axis=0)
    A_full = np.where(block_valid_mask, A_full, 0.0)
    B_full = np.where(block_valid_mask, B_full, 0.0)
    E_full = np.where(block_valid_mask, E_full, 0.0)
    bootstrap_covariance = _bootstrap_covariance_from_cluster_counts(
        A_by_cluster=A_by_cluster,
        B_by_cluster=B_by_cluster,
        E_by_cluster=E_by_cluster,
    )
    rho_values_raw = estimate_raw_rho_g_from_bootstrap_covariance(
        A=A_full,
        B=B_full,
        E=E_full,
        bootstrap_covariance=bootstrap_covariance,
        fallback_rho=1.0,
    )
    race_time = np.cumsum(np.where(block_valid_mask, block_size, 0).astype(float))
    fit_mask = block_valid_mask & (A_full > 0.0)
    rho_fit_raw = fit_low_order_rho_g_curve(
        raw_rho_g=rho_values_raw,
        race_time=race_time,
        valid_mask=fit_mask,
        polynomial_order=2,
        fallback_rho=1.0,
    )
    rho_values = np.where(block_valid_mask, rho_values_raw, np.nan)
    rho_fit = np.where(
        block_valid_mask,
        np.where(np.isfinite(rho_fit_raw), rho_fit_raw, 1.0),
        np.nan,
    )

    # For each evidence sample: bootstrap clusters, sample rho, sample r trajectory, compute Z.
    for s in range(num_Z_samples):
        if N_valid == 0:
            A = np.zeros(G, dtype=float)
            B = np.zeros(G, dtype=float)
            E = np.zeros(G, dtype=float)
        else:
            boot_idx = rng.integers(0, N_valid, size=N_valid)  # cluster bootstrap indices into valid clusters
            A, B, E = _boundary_counts_from_clusters(
                log_L_blocks=log_L_blocks,
                log_L_constraints=c_valid,
                log_L_phantom=ph_valid,
                cluster_indices=boot_idx,
            )
        A = np.where(block_valid_mask, A, 0.0)
        B = np.where(block_valid_mask, B, 0.0)
        E = np.where(block_valid_mask, E, 0.0)

        d2, alpha0, A_used = _candlestick_d2_per_boundary(K_per_block, eps_equal, A, B, E)
        if d2.size == 0:
            rho = 1.0
        else:
            rho = _sample_rho_from_likelihood(
                rng,
                d2=d2,
                alpha0=alpha0,
                A=A_used,
                rho_grid=rho_grid,
                dim=2,
                prior=rho_prior,
            )
        rho_samples[s] = rho
        eta = _estimate_eta(
            K_per_block=K_per_block,
            A=A,
            num_phantom=int(log_L_phantom.shape[1]),
            block_valid_mask=block_valid_mask,
        )
        eta_samples[s] = eta
        rho_eta_samples[s] = rho * eta

        rho_for_shrinkage = np.where(np.isfinite(rho_fit), rho_fit, 1.0)
        dirichlet_alpha, _ = _phantom_conditioned_dirichlet_parameters(
            K_per_block=K_per_block,
            block_size=block_size,
            block_valid_mask=block_valid_mask,
            A=A,
            B=B,
            E=E,
            rho_g=rho_for_shrinkage,
        )

        # Sample p_> trajectory and compute log X, log Z for this sample.
        p_gt, _, _ = _sample_dirichlet_probabilities(
            rng,
            dirichlet_alpha,
            block_valid_mask,
        )
        r = np.where(block_valid_mask, p_gt, 1.0)
        r = np.clip(r, 1e-300, 1.0)

        log_X_prev = 0.0
        log_terms = np.empty(G, dtype=float)
        for g in range(G):
            log_X_curr = log_X_prev + np.log(r[g])
            log_dX = _logdiffexp(np.array(log_X_prev), np.array(log_X_curr)).item()
            if block_valid_mask[g]:
                log_terms[g] = log_dX + log_L_blocks[g]
            else:
                log_terms[g] = -np.inf
            log_dZ_out[s, g] = log_terms[g]
            log_X_prev = log_X_curr

        log_Z_out[s] = _logsumexp(log_terms, axis=None).item()

    dZ_samples = np.exp(log_dZ_out)
    dZ_mean = np.mean(dZ_samples, axis=0)
    dZ_var = np.var(dZ_samples, axis=0)
    tiny = np.finfo(float).tiny
    return EvidenceSamples(
        log_Z_samples=log_Z_out,
        log_dZ_mean=np.where(block_valid_mask, np.log(np.maximum(dZ_mean, tiny)), -np.inf),
        log_dZ_var=np.where(block_valid_mask, np.log(np.maximum(dZ_var, tiny)), -np.inf),
        rho_samples=rho_samples,
        rho_values=rho_values,
        rho_fit=rho_fit,
        eta_samples=eta_samples,
        rho_eta_samples=rho_eta_samples,
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
    )
