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

Classic NS supplies a process prior on r via the live count K at that boundary:
    r ~ Beta(K, 1)   (order-statistic model).

Phantom samples are correlated states from constrained MCMC used to generate new live points.
They inform r through exceedances, but correlation reduces effective information. We use:
  - a single global efficiency factor rho in (0,1] (calibrated from data)
  - a moment-matched conjugate update
        r | classic, phantoms  ~ Beta(K + rho*B, 1 + rho*(A-B)),
    where A = # {phantom L > a} and B = # {phantom L > b} aggregated over *all loose clusters*
    (clusters generated under constraints c <= a).

------------------------------------------------------------------------------
New in v3
------------------------------------------------------------------------------
We add `sample_mc_shrinkage`, which:
  1) performs a **global cluster bootstrap** of phantom clusters for each evidence draw,
     preserving cross-boundary dependence induced by reusing loose samples; and
  2) samples rho from its **candlestick likelihood** on each bootstrap replicate (rather than fixing rho).

This improves uncertainty characterization over sampling each boundary independently from its
marginal Beta posterior under a fixed rho.

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
- This module treats each boundary's shrinkage r_g as independent given (rho, bootstrap replicate).
  Dependence is introduced through shared bootstrap resampling of clusters and shared rho, but we do
  not build a full joint likelihood for all boundaries simultaneously.
- "candlestick likelihood" is an approximate calibration model (Gaussian + scaled chi-square).
  It is intended as a robust practical calibration, not a mechanistic model of MCMC autocorrelation.
"""

from typing import NamedTuple, Tuple, Optional

import numpy as np

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


class EvidenceSamples(NamedTuple):
    log_Z_samples: FloatArray  # [num_Z_samples] samples of the evidence log Z from the MC shrinkage sampling
    log_dZ_mean: FloatArray  # [num_blocks] L_{g} * (X_{g-1} - X_g) averaged over MC chains
    log_dZ_var: FloatArray  # [num_blocks] variance of L_{g} * (X_{g-1} - X_g) over MC chains
    rho_samples: FloatArray  # [num_Z_samples] samples of the global rho parameter used in the MC shrinkage sampling
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

    Algorithm (per evidence sample s)
    --------------------------------
    1) Resample phantom clusters with replacement from the set of clusters with valid_phantom=True.
       (Global bootstrap: the same resample is used for all boundaries, preserving cross-boundary dependence.)

    2) On the resampled cluster set, compute pooled counts (A_g, B_g, E_g) for every boundary g using the
       same "loose reuse" rule: clusters with constraint c <= a_g are eligible at boundary g.

    3) Compute candlestick Mahalanobis distances d2_g for boundaries with A_g>0, and sample rho from the
       discrete likelihood L(rho) ∝ exp(-nll(rho)) on a grid.

    4) Form per-boundary Beta posteriors:
          alpha_g = K_g + rho * B_g
          beta_g  = 1 + rho * (A_g - B_g)
       falling back to Beta(K_g,1) when A_g=0.

    5) Sample shrinkages r_g ~ Beta(alpha_g, beta_g), compute X trajectory and Z.

    Notes
    -----
    - This procedure captures two effects that the baseline compute_mc_shrinkage misses:
        (i) cross-boundary dependence from reusing the same loose clusters; and
        (ii) uncertainty in rho (rho is sampled, not fixed).
    - It remains an approximation: it uses the candlestick likelihood and a Beta conjugate update with
      effective counts rho*A. This is intended as a robust stochastic alternative to fixed-rho marginal
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

    N = log_L_classic.shape[0]
    num_samples_int = int(np.asarray(num_samples).item())
    sample_valid_mask = np.arange(N, dtype=np.int32) < num_samples_int
    valid_classic = np.where(sample_valid_mask, log_L_classic, np.inf)
    unique_valid = np.unique(valid_classic)
    log_L_blocks = np.full((N,), np.inf, dtype=float)
    log_L_blocks[:unique_valid.shape[0]] = unique_valid
    block_valid_mask = np.isfinite(log_L_blocks)

    first_idx_raw = np.searchsorted(valid_classic, log_L_blocks, side="left")
    first_idx_safe = np.clip(first_idx_raw, 0, max(N - 1, 0))
    block_first_idx = np.where(block_valid_mask, first_idx_raw.astype(np.int32), np.int32(-1))

    K_per_block = K_classic[first_idx_safe].astype(float)
    K_per_block = np.where(block_valid_mask, K_per_block, np.ones_like(K_per_block))
    eps_equal = float(eps_equal_prior)

    # Extract valid phantom clusters
    effective_valid_phantom = valid_phantom & sample_valid_mask
    valid_idx = np.where(effective_valid_phantom)[0]
    if valid_idx.size == 0:
        c_valid = log_L_constraints
        ph_valid = log_L_phantom
    else:
        c_valid = log_L_constraints[valid_idx]
        ph_valid = log_L_phantom[valid_idx]
    N_valid = valid_idx.size

    if rho_grid is None:
        rho_grid = _rho_grid_default()

    G = log_L_blocks.size
    log_dZ_out = np.empty((num_Z_samples, G), dtype=float)
    log_Z_out = np.empty((num_Z_samples,), dtype=float)
    rho_samples = np.empty((num_Z_samples,), dtype=float)

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

        # Build Beta params per boundary
        alpha = np.empty(G, dtype=float)
        beta = np.empty(G, dtype=float)
        for g in range(G):
            K = float(K_per_block[g])
            if A[g] <= 0.0:
                alpha[g] = K
                beta[g] = 1.0
            else:
                alpha[g] = K + rho * B[g]
                beta[g] = 1.0 + rho * (A[g] - B[g])

            # Numerical safety: keep strictly positive
            alpha[g] = max(alpha[g], 1e-12)
            beta[g] = max(beta[g], 1e-12)

        # Sample r trajectory and compute log X, log Z for this sample
        r = rng.beta(alpha, beta)
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
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
    )
