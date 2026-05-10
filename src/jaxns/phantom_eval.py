import dataclasses
from functools import partial
from typing import Optional

import jax
import numpy as np
from jax import numpy as jnp, random
from jax.scipy import special as jsp

from jaxns.log_semiring import LogSpace
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import BlockState
from jaxns.types import FloatArray, IntArray, BoolArray, PRNGKey
from jaxns.v3_shrinkage import (
    DirichletConcentrations,
    PhantomCountMatrices,
    classic_dirichlet_concentrations,
    compute_kish_participating_cluster_counts,
    compute_phantom_gate_active,
    dirichlet_probability_means,
    estimate_raw_rho_g_from_bootstrap_covariance,
    fit_low_order_rho_g_curve,
    gamma_weighted_phantom_probabilities_from_draws as _gamma_weighted_phantom_probabilities_from_draws,
    phantom_conditioned_dirichlet_concentrations,
    sample_gamma_weighted_phantom_probabilities,
    sample_dirichlet_probabilities,
    validate_lineage_capacity,
    validate_phantom_count_matrices,
)


@dataclasses.dataclass(slots=True, frozen=True)
class EvidenceSamples(PureDataclassPytree):
    log_Z_samples: FloatArray  # [num_Z_samples] samples of the evidence log Z from the MC shrinkage sampling
    H_samples: FloatArray  # [num_Z_samples] the information E[log_L - log_Z]
    log_dZ_mean: FloatArray  # [num_blocks] L_{g} * (X_{g-1} - X_g) averaged over MC chains
    log_dZ_var: FloatArray  # [num_blocks] variance of L_{g} * (X_{g-1} - X_g) over MC chains
    log_L_blocks: FloatArray  # [num_blocks] block levels derived from log_L_classic, padded with +inf
    block_first_idx: IntArray  # [num_blocks] first classic index per block, -1 for padded blocks
    block_size: IntArray  # [num_blocks] number of classic samples in each likelihood block
    incoming_K: IntArray  # [num_blocks] canonical incoming active lineage count per block
    kish_participating_cluster_counts: FloatArray  # [num_blocks] Kish participating-cluster count
    phantom_gate_active: BoolArray  # [num_blocks] active gamma phantom conditioning gate
    phantom_A: FloatArray | None = None  # [num_blocks] full-data phantom A_g counts
    phantom_B: FloatArray | None = None  # [num_blocks] full-data phantom B_g counts
    phantom_E: FloatArray | None = None  # [num_blocks] full-data phantom E_g counts
    phantom_R: FloatArray | None = None  # [num_blocks] full-data phantom R_g counts
    classic_alpha_gt: FloatArray | None = None  # [num_blocks] classic alpha for p_>
    classic_alpha_eq: FloatArray | None = None  # [num_blocks] classic alpha for p_=
    classic_alpha_lt: FloatArray | None = None  # [num_blocks] classic alpha for p_<
    conditioned_alpha_gt: FloatArray | None = None  # [num_blocks] phantom-conditioned alpha for p_>
    conditioned_alpha_eq: FloatArray | None = None  # [num_blocks] phantom-conditioned alpha for p_=
    conditioned_alpha_lt: FloatArray | None = None  # [num_blocks] phantom-conditioned alpha for p_<
    epsilon: FloatArray | None = None  # [num_blocks] equality-atom prior epsilon_g
    p_gt_samples: FloatArray | None = None  # [num_Z_samples, num_blocks] sampled strict endpoint probabilities
    p_eq_samples: FloatArray | None = None  # [num_Z_samples, num_blocks] sampled equality atom probabilities
    p_lt_samples: FloatArray | None = None  # [num_Z_samples, num_blocks] sampled open-interval probabilities
    p_gt_mean: FloatArray | None = None  # [num_blocks] posterior mean of p_>
    p_eq_mean: FloatArray | None = None  # [num_blocks] posterior mean of p_=
    p_lt_mean: FloatArray | None = None  # [num_blocks] posterior mean of p_<
    phantom_add_gt_samples: FloatArray | None = None  # [num_Z_samples, num_blocks]
    phantom_add_eq_samples: FloatArray | None = None  # [num_Z_samples, num_blocks]
    phantom_add_lt_samples: FloatArray | None = None  # [num_Z_samples, num_blocks]
    rho_samples: FloatArray | None = None
    rho_values: FloatArray | None = None
    rho_fit: FloatArray | None = None
    eta_samples: FloatArray | None = None
    rho_eta_samples: FloatArray | None = None

    @property
    def m_g(self) -> IntArray:
        """Alias for the v3 block sizes."""
        return self.block_size

    @property
    def K_g(self) -> IntArray:
        """Alias for the v3 incoming active lineage counts."""
        return self.incoming_K

    @property
    def L_blocks(self) -> FloatArray:
        """Likelihood-scale block levels aligned with `log_L_blocks`."""
        return jnp.exp(self.log_L_blocks)

    @property
    def A_g(self) -> FloatArray | None:
        """Alias for phantom `A_g` counts."""
        return self.phantom_A

    @property
    def B_g(self) -> FloatArray | None:
        """Alias for phantom `B_g` counts."""
        return self.phantom_B

    @property
    def E_g(self) -> FloatArray | None:
        """Alias for phantom `E_g` counts."""
        return self.phantom_E

    @property
    def R_g(self) -> FloatArray | None:
        """Alias for phantom `R_g` counts."""
        return self.phantom_R

    @property
    def deprecated_fields(self) -> tuple[str, ...]:
        """Compatibility names retained as explicit non-target diagnostics."""
        return ("rho_samples", "rho_values", "rho_fit", "rho_eta_samples")

    @property
    def classic_dirichlet_concentrations(self) -> DirichletConcentrations | None:
        """Classic v3 block Dirichlet concentrations, if returned."""
        if self.classic_alpha_gt is None:
            return None
        return DirichletConcentrations(
            alpha_gt=self.classic_alpha_gt,
            alpha_eq=self.classic_alpha_eq,
            alpha_lt=self.classic_alpha_lt,
            epsilon=self.epsilon,
        )

    @property
    def phantom_conditioned_dirichlet_concentrations(
            self,
    ) -> DirichletConcentrations | None:
        """Phantom-conditioned v3 block Dirichlet concentrations, if returned."""
        if self.conditioned_alpha_gt is None:
            return None
        return DirichletConcentrations(
            alpha_gt=self.conditioned_alpha_gt,
            alpha_eq=self.conditioned_alpha_eq,
            alpha_lt=self.conditioned_alpha_lt,
            epsilon=self.epsilon,
        )

    def compute_phantom_efficiency(self, num_burn_in: int) -> FloatArray:
        """
        Determine the efficiency of adding more phantoms.
        Iff > 1 then for a given compute budget prefer decreasing number of live points, and increasing number of phantoms.
        """
        if self.rho_samples is None or self.eta_samples is None:
            return jnp.asarray(0.0)
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


def _cluster_count_matrices_from_precompute(
        *,
        effective_valid_phantom: BoolArray,
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
        dtype: jnp.dtype,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    num_clusters = effective_valid_phantom.shape[0]
    if num_clusters == 0:
        empty = jnp.zeros((0, num_blocks), dtype=dtype)
        return empty, empty, empty

    cluster_eye = jnp.eye(num_clusters, dtype=dtype)

    def single_cluster_counts(cluster_multiplicity):
        return _boundary_counts_from_multiplicity(
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

    A_by_cluster, B_by_cluster, E_by_cluster = jax.vmap(single_cluster_counts)(cluster_eye)
    valid = effective_valid_phantom.astype(dtype)[:, None]
    return (
        A_by_cluster * valid,
        B_by_cluster * valid,
        E_by_cluster * valid,
    )


def _bootstrap_covariance_from_cluster_counts(
        A_by_cluster: FloatArray,
        B_by_cluster: FloatArray,
        E_by_cluster: FloatArray,
        valid_cluster_mask: BoolArray | None = None,
        max_count_per_cluster: int | None = None,
) -> FloatArray:
    """Exact cluster-bootstrap covariance of q=(B/A, E/A).

    The bootstrap denominator is resampled with the clusters.  Resamples with
    A=0 have undefined q and are omitted from the covariance; if all resamples
    are inactive, this returns zero covariance so the rho estimator can take its
    explicit fallback path.
    """
    num_clusters = A_by_cluster.shape[0]
    num_blocks = A_by_cluster.shape[1]
    dtype = A_by_cluster.dtype
    if num_clusters == 0:
        return jnp.zeros((num_blocks, 2, 2), dtype=dtype)

    if valid_cluster_mask is None:
        valid_cluster_mask = jnp.ones((num_clusters,), dtype=jnp.bool_)
    if max_count_per_cluster is None:
        try:
            max_count_per_cluster = int(np.max(np.asarray(A_by_cluster)))
        except Exception:
            max_count_per_cluster = num_blocks
    max_count_per_cluster = max(int(max_count_per_cluster), 0)
    max_total_A = num_clusters * max_count_per_cluster
    support_size = max_total_A + 1
    support_idx = jnp.arange(support_size, dtype=jnp.int32)

    A_int = jnp.rint(A_by_cluster).astype(jnp.int32)
    A_int = jnp.clip(A_int, 0, max_count_per_cluster)
    B_by_cluster = jnp.asarray(B_by_cluster, dtype=dtype)
    E_by_cluster = jnp.asarray(E_by_cluster, dtype=dtype)
    valid = jnp.asarray(valid_cluster_mask, dtype=jnp.bool_)
    num_valid = jnp.sum(valid, dtype=jnp.int32)
    safe_num_valid = jnp.maximum(num_valid, 1)
    cluster_weight = valid.astype(dtype) / safe_num_valid.astype(dtype)

    probability0 = jnp.zeros((num_blocks, support_size), dtype=dtype)
    probability0 = probability0.at[:, 0].set(jnp.ones((num_blocks,), dtype=dtype))
    zeros = jnp.zeros_like(probability0)
    state0 = (probability0, zeros, zeros, zeros, zeros, zeros)

    def shift_by_A(values: FloatArray, A_offset: IntArray) -> FloatArray:
        source_idx = support_idx[None, :] - A_offset[:, None]
        active = source_idx >= 0
        source_idx = jnp.clip(source_idx, 0, max_total_A)
        shifted = jnp.take_along_axis(values, source_idx, axis=1)
        return jnp.where(active, shifted, jnp.zeros_like(shifted))

    def draw_step(state, draw_idx):
        probability, mean_B, mean_E, second_BB, second_BE, second_EE = state

        def cluster_step(acc, cluster_values):
            (
                acc_probability,
                acc_mean_B,
                acc_mean_E,
                acc_second_BB,
                acc_second_BE,
                acc_second_EE,
            ) = acc
            A_offset, B_value, E_value, weight = cluster_values
            shifted_probability = shift_by_A(probability, A_offset)
            shifted_mean_B = shift_by_A(mean_B, A_offset)
            shifted_mean_E = shift_by_A(mean_E, A_offset)
            shifted_second_BB = shift_by_A(second_BB, A_offset)
            shifted_second_BE = shift_by_A(second_BE, A_offset)
            shifted_second_EE = shift_by_A(second_EE, A_offset)

            B_value = B_value[:, None]
            E_value = E_value[:, None]
            weight = weight.astype(dtype)
            acc_probability = acc_probability + weight * shifted_probability
            acc_mean_B = acc_mean_B + weight * (
                    shifted_mean_B + B_value * shifted_probability
            )
            acc_mean_E = acc_mean_E + weight * (
                    shifted_mean_E + E_value * shifted_probability
            )
            acc_second_BB = acc_second_BB + weight * (
                    shifted_second_BB
                    + 2.0 * B_value * shifted_mean_B
                    + B_value * B_value * shifted_probability
            )
            acc_second_BE = acc_second_BE + weight * (
                    shifted_second_BE
                    + B_value * shifted_mean_E
                    + E_value * shifted_mean_B
                    + B_value * E_value * shifted_probability
            )
            acc_second_EE = acc_second_EE + weight * (
                    shifted_second_EE
                    + 2.0 * E_value * shifted_mean_E
                    + E_value * E_value * shifted_probability
            )
            return (
                acc_probability,
                acc_mean_B,
                acc_mean_E,
                acc_second_BB,
                acc_second_BE,
                acc_second_EE,
            ), None

        acc0 = tuple(jnp.zeros_like(value) for value in state)
        updated, _ = jax.lax.scan(
            cluster_step,
            init=acc0,
            xs=(
                A_int,
                B_by_cluster,
                E_by_cluster,
                cluster_weight,
            ),
        )
        return jax.lax.cond(
            draw_idx < num_valid,
            lambda _: updated,
            lambda _: state,
            operand=None,
        ), None

    (
        probability,
        mean_B,
        mean_E,
        second_BB,
        second_BE,
        second_EE,
    ), _ = jax.lax.scan(
        draw_step,
        init=state0,
        xs=jnp.arange(num_clusters, dtype=jnp.int32),
    )

    active = support_idx > 0
    active_probability = jnp.sum(
        jnp.where(active[None, :], probability, jnp.zeros_like(probability)),
        axis=1,
    )
    safe_active_probability = jnp.where(
        active_probability > 0.0,
        active_probability,
        jnp.ones_like(active_probability),
    )
    A_support = jnp.asarray(support_idx, dtype=dtype)
    A_safe = jnp.where(active, A_support, jnp.ones_like(A_support))
    mean_gt = jnp.sum(
        jnp.where(active[None, :], mean_B / A_safe[None, :], 0.0),
        axis=1,
    ) / safe_active_probability
    mean_eq = jnp.sum(
        jnp.where(active[None, :], mean_E / A_safe[None, :], 0.0),
        axis=1,
    ) / safe_active_probability
    A_safe_sq = A_safe * A_safe
    second_gt_gt = jnp.sum(
        jnp.where(active[None, :], second_BB / A_safe_sq[None, :], 0.0),
        axis=1,
    ) / safe_active_probability
    second_gt_eq = jnp.sum(
        jnp.where(active[None, :], second_BE / A_safe_sq[None, :], 0.0),
        axis=1,
    ) / safe_active_probability
    second_eq_eq = jnp.sum(
        jnp.where(active[None, :], second_EE / A_safe_sq[None, :], 0.0),
        axis=1,
    ) / safe_active_probability

    cov_gt_gt = second_gt_gt - mean_gt * mean_gt
    cov_gt_eq = second_gt_eq - mean_gt * mean_eq
    cov_eq_eq = second_eq_eq - mean_eq * mean_eq
    zero = jnp.zeros((), dtype=dtype)
    cov_gt_gt = jnp.maximum(cov_gt_gt, zero)
    cov_eq_eq = jnp.maximum(cov_eq_eq, zero)
    covariance = jnp.stack(
        [
            jnp.stack([cov_gt_gt, cov_gt_eq], axis=-1),
            jnp.stack([cov_gt_eq, cov_eq_eq], axis=-1),
        ],
        axis=-2,
    )
    return jnp.where(
        (active_probability > 0.0)[:, None, None],
        covariance,
        jnp.zeros_like(covariance),
    )


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


def _legacy_compute_phantom_block_counts(
        *,
        log_L_blocks: FloatArray,
        block_valid_mask: BoolArray,
        log_L_constraints: FloatArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        sample_mask: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute public v3 phantom `A_g`, `B_g`, and `E_g` block counts.

    Counts are aligned with `log_L_blocks` and use only leading valid classic
    sample clusters whose phantom cluster is marked valid.
    """
    dtype = log_L_blocks.dtype
    num_blocks = log_L_blocks.shape[0]
    num_phantom = log_L_phantom.shape[1]
    if num_phantom == 0:
        empty = jnp.zeros((num_blocks,), dtype=dtype)
        return empty, empty, empty

    num_clusters = log_L_constraints.shape[0]
    effective_valid_phantom = valid_phantom & sample_mask
    num_valid_blocks = jnp.sum(block_valid_mask, dtype=jnp.int32)

    left_c = jnp.searchsorted(log_L_blocks, log_L_constraints, side="left")
    start_idx = jnp.where(jnp.isneginf(log_L_constraints), 0, left_c + 1)
    start_idx = jnp.minimum(start_idx, num_valid_blocks)
    start_idx = jnp.where(effective_valid_phantom, start_idx, 0)

    event_cluster_idx = jnp.repeat(
        jnp.arange(num_clusters, dtype=jnp.int32),
        repeats=num_phantom,
    )
    event_start = start_idx[event_cluster_idx]
    event_logL = log_L_phantom.reshape((-1,))
    left_l = jnp.searchsorted(log_L_blocks, event_logL, side="left")
    event_a_hi = jnp.minimum(left_l + 1, num_valid_blocks)
    event_b_hi = jnp.minimum(left_l, num_valid_blocks)
    event_A_active = event_a_hi > event_start
    event_B_active = event_b_hi > event_start
    count_A_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_A_active, dtype=dtype),
        length=num_clusters,
    )
    count_B_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_B_active, dtype=dtype),
        length=num_clusters,
    )
    eq_ok = jnp.logical_and(
        left_l < num_valid_blocks,
        log_L_blocks[left_l] == event_logL,
    )
    event_eq_idx = jnp.where(eq_ok, left_l, 0)
    event_eq_active = jnp.logical_and(eq_ok, event_eq_idx >= event_start)
    event_eq_active = jnp.logical_and(
        event_eq_active,
        effective_valid_phantom[event_cluster_idx],
    )
    cluster_multiplicity = effective_valid_phantom.astype(dtype)
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
    zeros = jnp.zeros_like(A)
    return (
        jnp.where(block_valid_mask, A, zeros),
        jnp.where(block_valid_mask, B, zeros),
        jnp.where(block_valid_mask, E, zeros),
    )


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
        log_Z = jnp.asarray(log_Z, dtype=dtype)
        log_dZ = jnp.asarray(log_dZ, dtype=dtype)
        rho = jnp.asarray(rho, dtype=dtype)
        eta = jnp.asarray(eta, dtype=dtype)
        rho_eta = jnp.asarray(rho_eta, dtype=dtype)
        H = jnp.asarray(H, dtype=dtype)
        log_sum_dz = jnp.logaddexp(log_sum_dz, log_dZ)
        log_sum_dz_sq = jnp.logaddexp(log_sum_dz_sq, two * log_dZ)
        return (log_sum_dz, log_sum_dz_sq), (log_Z, rho, eta, rho_eta, H)

    (
        log_sum_dz,
        log_sum_dz_sq,
    ), (
        log_Z_samples,
        rho_samples,
        eta_samples,
        rho_eta_samples,
        H_samples,
    ) = jax.lax.scan(
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
    return (
        log_Z_samples,
        rho_samples,
        eta_samples,
        rho_eta_samples,
        log_dZ_mean,
        log_dZ_var,
        H_samples,
    )


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
        block_state: BlockState | None = None,
        rho_grid: Optional[FloatArray] = None,
        rho_prior: str = "none",
        batch_size: int | None = None,
        C_min: float = 20,
) -> EvidenceSamples:
    """
    Monte-Carlo evidence sampling with gamma-weighted phantom shrinkage.

    Per Monte-Carlo draw this function samples independent race gammas and
    shared per-cluster ``Gamma(1, 1)`` phantom weights, applies the Kish
    participating-cluster gate, and accumulates evidence contributions from
    the resulting block probabilities.

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
        block_state: Optional canonical v3 block state. When supplied, its
            block likelihoods, membership sizes, and incoming lineage counts are
            used instead of reconstructing blocks from per-sample live counts.
        rho_grid: Deprecated compatibility argument; ignored by the
            gamma-weighted target.
        rho_prior: Deprecated compatibility argument; ignored by the
            gamma-weighted target.
        batch_size: Reserved for API compatibility; currently unused.
        C_min: Kish participating-cluster gate threshold. Defaults to 20.

    Returns:
        EvidenceSamples with:
          - ``log_Z_samples``: evidence samples ``[num_Z_samples]``;
          - ``log_dZ_mean``: mean per-block contribution in log-space ``[num_blocks]``;
          - ``log_dZ_var``: variance per-block contribution in log-space ``[num_blocks]``;
          - Kish/gate diagnostics and aggregate ``A_g/B_g/E_g/R_g`` counts;
          - sampled ``p_>``, ``p_=``, and ``p_<`` block probabilities;
          - sample means of the returned block probability draws;
          - ``log_L_blocks``: derived block levels padded with ``+inf``;
          - ``block_first_idx``: first classic index per block, ``-1`` for padded blocks.
    """
    validate_sample_mc_shrinkage_inputs(
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
        block_state=block_state,
    )
    return _sample_mc_shrinkage(
        key=key,
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
        num_Z_samples=num_Z_samples,
        eps_equal_prior=eps_equal_prior,
        block_state=block_state,
        rho_grid=rho_grid,
        rho_prior=rho_prior,
        batch_size=batch_size,
        C_min=C_min,
    )


def validate_sample_mc_shrinkage_inputs(
        *,
        log_L_constraints: FloatArray,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
        block_state: BlockState | None = None,
) -> None:
    """Validate MC shrinkage inputs at public Python API boundaries."""
    _validate_phantom_metadata(
        log_L_constraints=log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
    )
    _validate_mc_shrinkage_capacity(
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        num_samples=num_samples,
        block_state=block_state,
    )


def _validate_phantom_metadata(
        *,
        log_L_constraints: FloatArray,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
) -> None:
    try:
        log_l = np.asarray(log_L_classic)
        live_points = np.asarray(K_classic)
        constraints = np.asarray(log_L_constraints)
        cluster_valid = np.asarray(valid_phantom)
        phantom_l = np.asarray(log_L_phantom)
        n = int(np.asarray(num_samples))
    except Exception:
        return

    if log_l.ndim != 1:
        raise ValueError("log_L_classic must be one-dimensional.")
    num_clusters = log_l.shape[0]
    if live_points.shape != (num_clusters,):
        raise ValueError("K_classic shape must match log_L_classic.")
    if constraints.shape != (num_clusters,):
        raise ValueError("log_L_constraints shape must match the cluster axis.")
    if cluster_valid.ndim != 1:
        raise ValueError(
            "valid_phantom must be a one-dimensional per-cluster mask, not a "
            "per-phantom mask."
        )
    if cluster_valid.shape != (num_clusters,):
        raise ValueError("valid_phantom shape must match the cluster axis.")
    if phantom_l.ndim != 2:
        raise ValueError("log_L_phantom must be a two-dimensional array.")
    if phantom_l.shape[0] != num_clusters:
        raise ValueError("log_L_phantom shape must match the cluster axis.")
    if n < 0 or n > num_clusters:
        raise ValueError("num_samples is outside the available cluster range.")
    if np.any(cluster_valid[n:]):
        raise ValueError(
            "valid_phantom contains a stale association beyond num_samples."
        )
    if n == 0:
        return
    active = live_points[:n] > 0
    strict_violations = active & (log_l[:n] <= constraints[:n])
    if np.any(strict_violations):
        bad = np.where(strict_violations)[0][0]
        raise ValueError(
            "Strict contour violation for active sample "
            f"{bad}: log_L_classic={log_l[bad]} must be greater than "
            f"log_L_constraint={constraints[bad]}."
        )


def _validate_mc_shrinkage_capacity(
        *,
        log_L_classic: FloatArray,
        K_classic: IntArray,
        num_samples: IntArray,
        block_state: BlockState | None,
) -> None:
    if block_state is not None:
        _validate_block_state_shapes(block_state)
        _validate_block_state_matches_classic_samples(
            block_state=block_state,
            log_L_classic=log_L_classic,
            num_samples=num_samples,
        )
        validate_lineage_capacity(block_state)
        return
    try:
        n = int(np.asarray(num_samples))
        log_l = np.asarray(log_L_classic[:n])
        live_points = np.asarray(K_classic[:n])
    except Exception:
        return

    active = live_points > 0
    if not np.any(active):
        return
    order = np.argsort(log_l[active], kind="stable")
    log_l = log_l[active][order]
    live_points = live_points[active][order]
    _, starts, block_size = np.unique(
        log_l,
        return_index=True,
        return_counts=True,
    )
    incoming = live_points[starts]
    if np.any(incoming < block_size):
        bad = np.where(incoming < block_size)[0][0]
        raise ValueError(
            f"Invalid race block {bad}: incoming K_g={incoming[bad]} "
            f"is smaller than plateau size m_g={block_size[bad]}."
        )


def _block_state_alignment_error(field_name: str, detail: str) -> ValueError:
    return ValueError(
        f"Supplied block_state.{field_name} does not match "
        f"log_L_classic[:num_samples]: {detail}"
    )


def _validate_block_state_matches_classic_samples(
        *,
        block_state: BlockState,
        log_L_classic: FloatArray,
        num_samples: IntArray,
) -> None:
    try:
        n = int(np.asarray(num_samples))
        log_l = np.asarray(log_L_classic)
        log_l_blocks = np.asarray(block_state.log_L_blocks)
        valid = np.asarray(block_state.valid, dtype=bool)
        block_size = np.asarray(block_state.block_size)
        block_first_idx = np.asarray(block_state.block_first_idx)
        block_start = (
            None
            if block_state.block_start is None
            else np.asarray(block_state.block_start)
        )
        block_stop = (
            None
            if block_state.block_stop is None
            else np.asarray(block_state.block_stop)
        )
        block_sample_indices = (
            None
            if block_state.block_sample_indices is None
            else np.asarray(block_state.block_sample_indices)
        )
    except Exception:
        return

    sample_log_l = log_l[:n]
    sorted_order = np.argsort(sample_log_l, kind="stable")
    sorted_log_l = sample_log_l[sorted_order]
    expected_log_l, starts, expected_sizes = np.unique(
        sorted_log_l,
        return_index=True,
        return_counts=True,
    )
    expected_count = expected_log_l.shape[0]
    valid_positions = np.flatnonzero(valid)
    if valid_positions.shape[0] != expected_count:
        raise _block_state_alignment_error(
            "valid",
            f"got {valid_positions.shape[0]} valid blocks, expected "
            f"{expected_count} unique likelihood blocks.",
        )
    expected_positions = np.arange(expected_count)
    if not np.array_equal(valid_positions, expected_positions):
        raise _block_state_alignment_error(
            "valid",
            "valid blocks must occupy the leading block entries.",
        )

    if not np.array_equal(log_l_blocks[valid_positions], expected_log_l):
        mismatch = np.flatnonzero(
            log_l_blocks[valid_positions] != expected_log_l
        )
        bad = int(mismatch[0]) if mismatch.size else 0
        raise _block_state_alignment_error(
            "log_L_blocks",
            f"block {bad} has {log_l_blocks[valid_positions][bad]!r}, "
            f"expected {expected_log_l[bad]!r}.",
        )

    if not np.array_equal(block_size[valid_positions], expected_sizes):
        mismatch = np.flatnonzero(
            block_size[valid_positions] != expected_sizes
        )
        bad = int(mismatch[0]) if mismatch.size else 0
        raise _block_state_alignment_error(
            "block_size",
            f"block {bad} has size {block_size[valid_positions][bad]}, "
            f"expected {expected_sizes[bad]}.",
        )
    invalid_nonzero = np.flatnonzero((~valid) & (block_size != 0))
    if invalid_nonzero.size:
        bad = int(invalid_nonzero[0])
        raise _block_state_alignment_error(
            "block_size",
            f"padded block {bad} has non-zero size {block_size[bad]}.",
        )

    expected_members = [
        sorted_order[int(start):int(start + size)]
        for start, size in zip(starts, expected_sizes, strict=True)
    ]
    _validate_block_first_indices(
        block_first_idx=block_first_idx,
        valid=valid,
        sample_log_l=sample_log_l,
        expected_log_l=expected_log_l,
        valid_positions=valid_positions,
    )
    _validate_block_ranges(
        block_start=block_start,
        block_stop=block_stop,
        valid=valid,
        expected_sizes=expected_sizes,
        num_samples=n,
    )
    _validate_block_sample_indices(
        block_sample_indices=block_sample_indices,
        block_start=block_start,
        block_stop=block_stop,
        valid=valid,
        expected_members=expected_members,
        num_samples=n,
    )


def _validate_block_first_indices(
        *,
        block_first_idx: np.ndarray,
        valid: np.ndarray,
        sample_log_l: np.ndarray,
        expected_log_l: np.ndarray,
        valid_positions: np.ndarray,
) -> None:
    for expected_idx, block_idx in enumerate(valid_positions):
        first_idx = int(block_first_idx[block_idx])
        if first_idx < 0 or first_idx >= sample_log_l.shape[0]:
            raise _block_state_alignment_error(
                "block_first_idx",
                f"block {block_idx} points outside the leading sample range.",
            )
        if sample_log_l[first_idx] != expected_log_l[expected_idx]:
            raise _block_state_alignment_error(
                "block_first_idx",
                f"block {block_idx} points to sample {first_idx} with "
                f"log_L_classic={sample_log_l[first_idx]!r}, expected "
                f"{expected_log_l[expected_idx]!r}.",
            )

    invalid_bad = np.flatnonzero((~valid) & (block_first_idx != -1))
    if invalid_bad.size:
        bad = int(invalid_bad[0])
        raise _block_state_alignment_error(
            "block_first_idx",
            f"padded block {bad} has sample index {block_first_idx[bad]}, "
            "expected -1.",
        )


def _validate_block_ranges(
        *,
        block_start: np.ndarray | None,
        block_stop: np.ndarray | None,
        valid: np.ndarray,
        expected_sizes: np.ndarray,
        num_samples: int,
) -> None:
    if block_start is None and block_stop is None:
        return
    if block_start is None or block_stop is None:
        raise _block_state_alignment_error(
            "block_start",
            "block_start and block_stop must be supplied together.",
        )

    expected_starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(expected_sizes, dtype=np.int64)[:-1],
        ]
    )
    expected_stops = np.cumsum(expected_sizes, dtype=np.int64)
    for block_idx in range(expected_sizes.shape[0]):
        if int(block_start[block_idx]) != int(expected_starts[block_idx]):
            raise _block_state_alignment_error(
                "block_start",
                f"block {block_idx} has start {block_start[block_idx]}, "
                f"expected {expected_starts[block_idx]}.",
            )
        if int(block_stop[block_idx]) != int(expected_stops[block_idx]):
            raise _block_state_alignment_error(
                "block_stop",
                f"block {block_idx} has stop {block_stop[block_idx]}, "
                f"expected {expected_stops[block_idx]}.",
            )

    invalid_positions = np.flatnonzero(~valid)
    for block_idx in invalid_positions:
        if (
                int(block_start[block_idx]) != num_samples
                or int(block_stop[block_idx]) != num_samples
        ):
            raise _block_state_alignment_error(
                "block_start",
                f"padded block {block_idx} has range "
                f"[{block_start[block_idx]}, {block_stop[block_idx]}], "
                f"expected [{num_samples}, {num_samples}].",
            )


def _validate_block_sample_indices(
        *,
        block_sample_indices: np.ndarray | None,
        block_start: np.ndarray | None,
        block_stop: np.ndarray | None,
        valid: np.ndarray,
        expected_members: list[np.ndarray],
        num_samples: int,
) -> None:
    if block_sample_indices is None:
        return
    if block_sample_indices.ndim not in (1, 2):
        raise _block_state_alignment_error(
            "block_sample_indices",
            "expected a one- or two-dimensional membership array.",
        )
    if (
            block_sample_indices.ndim == 2
            and block_sample_indices.shape[0] != valid.shape[0]
    ):
        raise _block_state_alignment_error(
            "block_sample_indices",
            f"got first dimension {block_sample_indices.shape[0]}, expected "
            f"{valid.shape[0]}.",
        )

    for block_idx, expected in enumerate(expected_members):
        members = _members_for_block(
            block_sample_indices=block_sample_indices,
            block_idx=block_idx,
            block_start=block_start,
            block_stop=block_stop,
        )
        members = members[members >= 0]
        _validate_block_members(
            block_idx=block_idx,
            members=members,
            expected=expected,
            num_samples=num_samples,
        )

    _validate_padded_block_members(
        block_sample_indices=block_sample_indices,
        block_start=block_start,
        block_stop=block_stop,
        valid=valid,
    )


def _members_for_block(
        *,
        block_sample_indices: np.ndarray,
        block_idx: int,
        block_start: np.ndarray | None,
        block_stop: np.ndarray | None,
) -> np.ndarray:
    if block_sample_indices.ndim == 2:
        return block_sample_indices[block_idx]
    if block_start is None or block_stop is None:
        if block_idx >= block_sample_indices.shape[0]:
            return np.asarray([], dtype=block_sample_indices.dtype)
        return block_sample_indices[block_idx:block_idx + 1]
    start = int(block_start[block_idx])
    stop = int(block_stop[block_idx])
    return block_sample_indices[start:stop]


def _validate_block_members(
        *,
        block_idx: int,
        members: np.ndarray,
        expected: np.ndarray,
        num_samples: int,
) -> None:
    if np.any(members >= num_samples):
        bad_member = int(members[np.flatnonzero(members >= num_samples)[0]])
        raise _block_state_alignment_error(
            "block_sample_indices",
            f"block {block_idx} contains sample {bad_member}, outside "
            "the leading sample range.",
        )
    if np.unique(members).shape[0] != members.shape[0]:
        raise _block_state_alignment_error(
            "block_sample_indices",
            f"block {block_idx} contains duplicate sample memberships.",
        )
    if not np.array_equal(np.sort(members), np.sort(expected)):
        raise _block_state_alignment_error(
            "block_sample_indices",
            f"block {block_idx} has members {np.sort(members).tolist()}, "
            f"expected {np.sort(expected).tolist()}.",
        )


def _validate_padded_block_members(
        *,
        block_sample_indices: np.ndarray,
        block_start: np.ndarray | None,
        block_stop: np.ndarray | None,
        valid: np.ndarray,
) -> None:
    invalid_positions = np.flatnonzero(~valid)
    for block_idx in invalid_positions:
        if block_sample_indices.ndim == 2:
            members = block_sample_indices[block_idx]
        elif block_start is not None and block_stop is not None:
            start = int(block_start[block_idx])
            stop = int(block_stop[block_idx])
            members = block_sample_indices[start:stop]
        elif block_idx < block_sample_indices.shape[0]:
            members = block_sample_indices[block_idx:block_idx + 1]
        else:
            members = np.asarray([], dtype=block_sample_indices.dtype)
        if np.any(members >= 0):
            raise _block_state_alignment_error(
                "block_sample_indices",
                f"padded block {block_idx} contains sample memberships.",
            )


def _validate_block_state_shapes(block_state: BlockState) -> None:
    block_shape = jnp.shape(block_state.log_L_blocks)
    if len(block_shape) != 1:
        raise ValueError("block_state.log_L_blocks must be one-dimensional.")
    for name, value in (
            ("block_first_idx", block_state.block_first_idx),
            ("block_size", block_state.block_size),
            ("incoming_K", block_state.incoming_K),
            ("block_out_degree", block_state.block_out_degree),
            ("valid", block_state.valid),
    ):
        if jnp.shape(value) != block_shape:
            raise ValueError(
                f"block_state.{name} shape must align with "
                f"block_state.log_L_blocks; got {jnp.shape(value)}, "
                f"expected {block_shape}."
            )
    for name, value in (
            ("block_start", block_state.block_start),
            ("block_stop", block_state.block_stop),
    ):
        if value is not None and jnp.shape(value) != block_shape:
            raise ValueError(
                f"block_state.{name} shape must align with "
                f"block_state.log_L_blocks; got {jnp.shape(value)}, "
                f"expected {block_shape}."
            )
    if block_state.block_sample_indices is not None:
        membership_shape = jnp.shape(block_state.block_sample_indices)
        if len(membership_shape) not in (1, 2):
            raise ValueError(
                "block_state.block_sample_indices must be one- or "
                "two-dimensional."
            )
        if len(membership_shape) == 2 and membership_shape[0] != block_shape[0]:
            raise ValueError(
                "block_state.block_sample_indices first dimension must align "
                "with block_state.log_L_blocks."
            )


@partial(jax.jit, inline=True, static_argnames=["num_Z_samples", "rho_prior"])
def _legacy_sample_mc_shrinkage(
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
        block_state: BlockState | None = None,
        rho_grid: Optional[FloatArray] = None,
        rho_prior: str = "none",
        batch_size: int | None = None,
) -> EvidenceSamples:
    N = log_L_classic.shape[0]
    sample_valid_mask = jnp.arange(N, dtype=jnp.int32) < num_samples
    positive_live_mask = K_classic > 0
    effective_sample_mask = jnp.logical_and(sample_valid_mask, positive_live_mask)
    valid_classic = jnp.where(effective_sample_mask, log_L_classic, jnp.inf)
    sorted_order = jnp.argsort(valid_classic, stable=True)
    sorted_log_L = valid_classic[sorted_order]
    sorted_valid = effective_sample_mask[sorted_order]
    sorted_K = K_classic[sorted_order]
    if block_state is None:
        log_L_blocks = jnp.unique(sorted_log_L, size=N, fill_value=jnp.inf)
        block_valid_mask = log_L_blocks < jnp.inf
        first_idx_raw = jnp.searchsorted(sorted_log_L, log_L_blocks, side="left")
        first_idx_safe = jnp.clip(first_idx_raw, 0, jnp.maximum(N - 1, 0))
        block_first_idx = jnp.where(
            block_valid_mask,
            sorted_order[first_idx_safe].astype(jnp.int32),
            jnp.asarray(-1, jnp.int32),
        )
        block_ids = jnp.searchsorted(log_L_blocks, sorted_log_L, side="left")
        block_ids = jnp.clip(block_ids, 0, jnp.maximum(N - 1, 0))
        block_size = jnp.bincount(
            block_ids,
            weights=sorted_valid.astype(jnp.int32),
            length=N,
        ).astype(jnp.int32)
        incoming_K_int = sorted_K[first_idx_safe].astype(jnp.int32)
    else:
        log_L_blocks = block_state.log_L_blocks
        block_valid_mask = block_state.valid
        block_first_idx = block_state.block_first_idx.astype(jnp.int32)
        block_size = block_state.block_size.astype(jnp.int32)
        incoming_K_int = block_state.incoming_K.astype(jnp.int32)
    K_per_block = incoming_K_int.astype(log_L_blocks.dtype)
    K_per_block = jnp.where(block_valid_mask, K_per_block, jnp.ones_like(K_per_block))
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
    block_state_for_v3 = BlockState(
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
        block_size=block_size,
        incoming_K=incoming_K_int.astype(jnp.int32),
        block_out_degree=jnp.zeros((num_blocks,), dtype=jnp.int32),
        valid=block_valid_mask,
    )

    if num_phantom == 0:
        rho_fallback = rho_grid[-1]
        zero_counts = jnp.zeros((num_blocks,), dtype=dtype)
        rho_fallback_by_block = jnp.full((num_blocks,), rho_fallback, dtype=dtype)
        classic_concentrations = classic_dirichlet_concentrations(block_state_for_v3)
        conditioned_concentrations = phantom_conditioned_dirichlet_concentrations(
            block_state_for_v3,
            zero_counts,
            zero_counts,
            zero_counts,
            rho_fallback_by_block,
        )
        p_gt_mean, p_eq_mean, _ = dirichlet_probability_means(
            conditioned_concentrations
        )

        def single_sample_no_phantom(
                sample_key: FloatArray,
        ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
            p_gt_samples, _, _ = sample_dirichlet_probabilities(
                sample_key,
                conditioned_concentrations,
                num_samples=1,
            )
            p_gt = jnp.where(
                block_valid_mask,
                p_gt_samples[0],
                jnp.ones((num_blocks,), dtype=dtype),
            )
            p_gt = jnp.clip(p_gt, 1e-300, 1.0)
            log_r = jnp.log(p_gt)
            log_X = jnp.cumsum(log_r)
            log_X_prev = jnp.concatenate(
                [jnp.zeros((1,), dtype=log_r.dtype), log_X[:-1]],
                axis=0,
            )
            log_dX = _logdiffexp(log_X_prev, log_X)
            log_dZ = jnp.where(block_valid_mask, log_dX + log_L_blocks, jnp.full_like(log_dX, -jnp.inf))
            log_Z = _logsumexp(log_dZ, axis=None)
            log_Z = jnp.where(constant_likelihood, ref_log_L, log_Z)
            eta = jnp.zeros((), dtype=dtype)
            rho_eta = jnp.zeros((), dtype=dtype)
            # H = E[log_L - log_Z]
            w = LogSpace(log_dZ) / LogSpace(log_Z)
            entropy_terms = jnp.where(
                block_valid_mask,
                log_L_blocks - log_Z,
                jnp.zeros_like(log_L_blocks),
            )
            H = (w * LogSpace.from_signed_value(entropy_terms)).sum().value
            return log_Z, log_dZ, rho_fallback, eta, rho_eta, H

        keys = random.split(key, num_Z_samples)
        (
            log_Z_samples,
            rho_samples,
            eta_samples,
            rho_eta_samples,
            log_dZ_mean,
            log_dZ_var,
            H_samples,
        ) = _summarise_log_dz_samples(
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
            rho_values=jnp.where(block_valid_mask, rho_fallback, jnp.nan),
            rho_fit=jnp.where(block_valid_mask, rho_fallback, jnp.nan),
            eta_samples=eta_samples,
            rho_eta_samples=rho_eta_samples,
            log_L_blocks=log_L_blocks,
            block_first_idx=block_first_idx,
            block_size=block_size,
            incoming_K=incoming_K_int.astype(jnp.int32),
            phantom_A=zero_counts,
            phantom_B=zero_counts,
            phantom_E=zero_counts,
            classic_alpha_gt=classic_concentrations.alpha_gt,
            classic_alpha_eq=classic_concentrations.alpha_eq,
            classic_alpha_lt=classic_concentrations.alpha_lt,
            conditioned_alpha_gt=conditioned_concentrations.alpha_gt,
            conditioned_alpha_eq=conditioned_concentrations.alpha_eq,
            conditioned_alpha_lt=conditioned_concentrations.alpha_lt,
            epsilon=conditioned_concentrations.epsilon,
            p_gt_samples=None,
            p_eq_samples=None,
            p_gt_mean=jnp.where(block_valid_mask, p_gt_mean, jnp.nan),
            p_eq_mean=jnp.where(block_valid_mask, p_eq_mean, jnp.nan),
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

    A_by_cluster, B_by_cluster, E_by_cluster = _cluster_count_matrices_from_precompute(
        effective_valid_phantom=effective_valid_phantom,
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
        dtype=dtype,
    )
    A_full = jnp.sum(A_by_cluster, axis=0)
    B_full = jnp.sum(B_by_cluster, axis=0)
    E_full = jnp.sum(E_by_cluster, axis=0)
    A_full = jnp.where(block_valid_mask, A_full, jnp.zeros_like(A_full))
    B_full = jnp.where(block_valid_mask, B_full, jnp.zeros_like(B_full))
    E_full = jnp.where(block_valid_mask, E_full, jnp.zeros_like(E_full))
    bootstrap_covariance = _bootstrap_covariance_from_cluster_counts(
        A_by_cluster=A_by_cluster,
        B_by_cluster=B_by_cluster,
        E_by_cluster=E_by_cluster,
        valid_cluster_mask=effective_valid_phantom,
        max_count_per_cluster=num_phantom,
    )
    rho_values_raw = estimate_raw_rho_g_from_bootstrap_covariance(
        A=A_full,
        B=B_full,
        E=E_full,
        bootstrap_covariance=bootstrap_covariance,
        fallback_rho=1.0,
    ).astype(dtype)
    race_time = jnp.cumsum(jnp.where(block_valid_mask, block_size, 0).astype(dtype))
    fit_mask = jnp.logical_and(block_valid_mask, A_full > 0.0)
    rho_fit_raw = fit_low_order_rho_g_curve(
        raw_rho_g=rho_values_raw,
        race_time=race_time,
        valid_mask=fit_mask,
        polynomial_order=2,
        fallback_rho=1.0,
    ).astype(dtype)
    rho_values = jnp.where(
        block_valid_mask,
        rho_values_raw,
        jnp.nan,
    )
    rho_fit = jnp.where(
        block_valid_mask,
        jnp.where(jnp.isfinite(rho_fit_raw), rho_fit_raw, jnp.asarray(1.0, dtype=dtype)),
        jnp.nan,
    )
    classic_concentrations = classic_dirichlet_concentrations(block_state_for_v3)
    one = jnp.ones((), dtype=dtype)
    rho_for_full_conditioning = jnp.where(jnp.isfinite(rho_fit), rho_fit, one)
    conditioned_concentrations = phantom_conditioned_dirichlet_concentrations(
        block_state_for_v3,
        A_full,
        B_full,
        E_full,
        rho_for_full_conditioning,
    )
    p_gt_mean, p_eq_mean, _ = dirichlet_probability_means(
        conditioned_concentrations
    )

    def single_sample(
            sample_key: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        key_boot, key_rho, key_r = random.split(sample_key, 3)

        safe_num_valid = jnp.maximum(num_valid, 1)
        boot_local_idx = random.randint(
            key_boot,
            shape=(num_clusters,),
            minval=0,
            maxval=safe_num_valid,
        )
        boot_draw_mask = jnp.arange(num_clusters, dtype=jnp.int32) < num_valid
        boot_local_weights = jnp.where(
            boot_draw_mask,
            jnp.ones((num_clusters,), dtype=dtype),
            jnp.zeros((num_clusters,), dtype=dtype),
        )
        local_multiplicity = jnp.bincount(
            boot_local_idx,
            weights=boot_local_weights,
            length=num_clusters,
        )
        cluster_multiplicity = jnp.zeros((num_clusters,), dtype=dtype).at[
            valid_cluster_idx
        ].add(local_multiplicity)

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
        eta = _estimate_eta(
            K_per_block=K_per_block,
            A=A,
            num_phantom=num_phantom,
            block_valid_mask=block_valid_mask,
        )
        rho_eta = rho * eta

        rho_for_shrinkage = jnp.where(jnp.isfinite(rho_fit), rho_fit, one)
        sample_concentrations = phantom_conditioned_dirichlet_concentrations(
            block_state_for_v3,
            A,
            B,
            E,
            rho_for_shrinkage,
        )
        p_gt_samples, _, _ = sample_dirichlet_probabilities(
            key_r,
            sample_concentrations,
            num_samples=1,
        )
        p_gt = jnp.where(
            block_valid_mask,
            p_gt_samples[0],
            jnp.ones((num_blocks,), dtype=dtype),
        )
        p_gt = jnp.clip(p_gt, 1e-300, 1.0)
        log_r = jnp.log(p_gt)
        log_X = jnp.cumsum(log_r)
        log_X_prev = jnp.concatenate([jnp.zeros((1,), dtype=log_r.dtype), log_X[:-1]], axis=0)
        log_dX = _logdiffexp(log_X_prev, log_X)
        log_dZ = jnp.where(block_valid_mask, log_dX + log_L_blocks, jnp.full_like(log_dX, -jnp.inf))
        log_terms = log_dZ
        log_Z = _logsumexp(log_terms, axis=None)
        log_Z = jnp.where(constant_likelihood, ref_log_L, log_Z)
        w = LogSpace(log_dZ) / LogSpace(log_Z)
        entropy_terms = jnp.where(
            block_valid_mask,
            log_L_blocks - log_Z,
            jnp.zeros_like(log_L_blocks),
        )
        H = (w * LogSpace.from_signed_value(entropy_terms)).sum().value
        return log_Z, log_dZ, rho, eta, rho_eta, H

    keys = random.split(key, num_Z_samples)
    (
        log_Z_samples,
        rho_samples,
        eta_samples,
        rho_eta_samples,
        log_dZ_mean,
        log_dZ_var,
        H_samples,
    ) = _summarise_log_dz_samples(
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
        rho_values=rho_values,
        rho_fit=rho_fit,
        eta_samples=eta_samples,
        rho_eta_samples=rho_eta_samples,
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
        block_size=block_size,
        incoming_K=incoming_K_int.astype(jnp.int32),
        H_samples=H_samples,
        phantom_A=A_full,
        phantom_B=B_full,
        phantom_E=E_full,
        classic_alpha_gt=classic_concentrations.alpha_gt,
        classic_alpha_eq=classic_concentrations.alpha_eq,
        classic_alpha_lt=classic_concentrations.alpha_lt,
        conditioned_alpha_gt=conditioned_concentrations.alpha_gt,
        conditioned_alpha_eq=conditioned_concentrations.alpha_eq,
        conditioned_alpha_lt=conditioned_concentrations.alpha_lt,
        epsilon=conditioned_concentrations.epsilon,
        p_gt_samples=None,
        p_eq_samples=None,
        p_gt_mean=jnp.where(block_valid_mask, p_gt_mean, jnp.nan),
        p_eq_mean=jnp.where(block_valid_mask, p_eq_mean, jnp.nan),
    )


def _validate_phantom_count_inputs(
        *,
        log_L_blocks: FloatArray,
        block_valid_mask: BoolArray,
        log_L_constraints: FloatArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        sample_mask: BoolArray,
) -> None:
    try:
        blocks = np.asarray(log_L_blocks)
        valid_blocks = np.asarray(block_valid_mask)
        constraints = np.asarray(log_L_constraints)
        cluster_valid = np.asarray(valid_phantom)
        phantom_l = np.asarray(log_L_phantom)
        samples = np.asarray(sample_mask)
    except Exception:
        return

    if blocks.ndim != 1:
        raise ValueError("log_L_blocks must be one-dimensional.")
    if valid_blocks.shape != blocks.shape:
        raise ValueError("block_valid_mask shape must align with log_L_blocks.")
    if constraints.ndim != 1:
        raise ValueError("log_L_constraints must be one-dimensional.")
    num_clusters = constraints.shape[0]
    if cluster_valid.ndim != 1:
        raise ValueError(
            "valid_phantom must be a one-dimensional per-cluster mask, not a "
            "per-phantom mask."
        )
    if cluster_valid.shape != (num_clusters,):
        raise ValueError("valid_phantom shape must match the cluster axis.")
    if samples.shape != (num_clusters,):
        raise ValueError("sample_mask shape must match the cluster axis.")
    if phantom_l.ndim != 2:
        raise ValueError("log_L_phantom must be a two-dimensional array.")
    if phantom_l.shape[0] != num_clusters:
        raise ValueError("log_L_phantom shape must match the cluster axis.")
    if np.any(cluster_valid & ~samples):
        raise ValueError(
            "valid_phantom contains a stale sample_mask/num_samples "
            "association."
        )


def gamma_weighted_phantom_probabilities_from_draws(**kwargs):
    """Public JAX wrapper for explicit gamma-weighted phantom draws."""
    return _gamma_weighted_phantom_probabilities_from_draws(**kwargs)


def compute_phantom_count_matrices(
        *,
        log_L_blocks: FloatArray,
        block_valid_mask: BoolArray,
        log_L_constraints: FloatArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        sample_mask: BoolArray,
        C_min: float = 20,
) -> PhantomCountMatrices:
    """Compute parent-contour-gated per-cluster phantom count matrices."""
    _validate_phantom_count_inputs(
        log_L_blocks=log_L_blocks,
        block_valid_mask=block_valid_mask,
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=sample_mask,
    )
    log_L_blocks = jnp.asarray(log_L_blocks, dtype=mp_policy.measure_dtype)
    block_valid_mask = jnp.asarray(block_valid_mask, dtype=mp_policy.bool_dtype)
    log_L_constraints = jnp.asarray(
        log_L_constraints,
        dtype=log_L_blocks.dtype,
    )
    valid_phantom = jnp.asarray(valid_phantom, dtype=mp_policy.bool_dtype)
    log_L_phantom = jnp.asarray(log_L_phantom, dtype=log_L_blocks.dtype)
    sample_mask = jnp.asarray(sample_mask, dtype=mp_policy.bool_dtype)

    dtype = log_L_blocks.dtype
    num_blocks = log_L_blocks.shape[0]
    num_clusters = log_L_constraints.shape[0]
    num_phantom = log_L_phantom.shape[1]
    effective_valid_phantom = valid_phantom & sample_mask
    num_valid_blocks = jnp.sum(block_valid_mask, dtype=jnp.int32)

    left_c = jnp.searchsorted(log_L_blocks, log_L_constraints, side="left")
    start_idx = jnp.where(jnp.isneginf(log_L_constraints), 0, left_c + 1)
    start_idx = jnp.minimum(start_idx, num_valid_blocks)
    start_idx = jnp.where(effective_valid_phantom, start_idx, 0)

    event_cluster_idx = jnp.repeat(
        jnp.arange(num_clusters, dtype=jnp.int32),
        repeats=num_phantom,
    )
    event_start = start_idx[event_cluster_idx]
    event_logL = log_L_phantom.reshape((-1,))
    left_l = jnp.searchsorted(log_L_blocks, event_logL, side="left")
    event_a_hi = jnp.minimum(left_l + 1, num_valid_blocks)
    event_b_hi = jnp.minimum(left_l, num_valid_blocks)
    event_A_active = event_a_hi > event_start
    event_B_active = event_b_hi > event_start
    count_A_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_A_active, dtype=dtype),
        length=num_clusters,
    )
    count_B_start_per_cluster = jnp.bincount(
        event_cluster_idx,
        weights=jnp.asarray(event_B_active, dtype=dtype),
        length=num_clusters,
    )
    eq_ok = jnp.logical_and(
        left_l < num_valid_blocks,
        log_L_blocks[left_l] == event_logL,
    )
    event_eq_idx = jnp.where(eq_ok, left_l, 0)
    event_eq_active = jnp.logical_and(eq_ok, event_eq_idx >= event_start)
    event_eq_active = jnp.logical_and(
        event_eq_active,
        effective_valid_phantom[event_cluster_idx],
    )

    A_cg, B_cg, E_cg = _cluster_count_matrices_from_precompute(
        effective_valid_phantom=effective_valid_phantom,
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
        dtype=dtype,
    )
    valid_cols = block_valid_mask[None, :]
    zeros = jnp.zeros_like(A_cg)
    A_cg = jnp.where(valid_cols, A_cg, zeros)
    B_cg = jnp.where(valid_cols, B_cg, zeros)
    E_cg = jnp.where(valid_cols, E_cg, zeros)
    R_cg = A_cg - B_cg - E_cg
    validate_phantom_count_matrices(
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        block_valid_mask=block_valid_mask,
    )
    A_g = jnp.sum(A_cg, axis=0)
    B_g = jnp.sum(B_cg, axis=0)
    E_g = jnp.sum(E_cg, axis=0)
    R_g = jnp.sum(R_cg, axis=0)
    kish = compute_kish_participating_cluster_counts(A_cg)
    gate = compute_phantom_gate_active(A_cg, C_min=C_min) & block_valid_mask
    return PhantomCountMatrices(
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        R_cg=R_cg,
        A_g=jnp.where(block_valid_mask, A_g, jnp.zeros_like(A_g)),
        B_g=jnp.where(block_valid_mask, B_g, jnp.zeros_like(B_g)),
        E_g=jnp.where(block_valid_mask, E_g, jnp.zeros_like(E_g)),
        R_g=jnp.where(block_valid_mask, R_g, jnp.zeros_like(R_g)),
        kish_participating_cluster_counts=jnp.where(
            block_valid_mask,
            kish,
            jnp.zeros_like(kish),
        ),
        phantom_gate_active=gate,
    )


def compute_phantom_block_counts(
        *,
        log_L_blocks: FloatArray,
        block_valid_mask: BoolArray,
        log_L_constraints: FloatArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        sample_mask: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute aggregate public v3 phantom `A_g`, `B_g`, and `E_g` counts."""
    counts = compute_phantom_count_matrices(
        log_L_blocks=log_L_blocks,
        block_valid_mask=block_valid_mask,
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=sample_mask,
    )
    return counts.A_g, counts.B_g, counts.E_g


def _summarise_gamma_log_dz_samples(
        *,
        log_dZ: FloatArray,
        log_Z: FloatArray,
        block_valid_mask: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    dtype = log_dZ.dtype
    log_num = jnp.log(jnp.asarray(log_dZ.shape[0], dtype=dtype))
    log_dZ_mean = _logsumexp(log_dZ, axis=0) - log_num
    log_dZ_second = _logsumexp(2.0 * log_dZ, axis=0) - log_num
    log_dZ_mean_sq = 2.0 * log_dZ_mean
    log_dZ_var = jnp.where(
        log_dZ_second > log_dZ_mean_sq,
        _logdiffexp(log_dZ_second, log_dZ_mean_sq),
        jnp.full_like(log_dZ_mean, -jnp.inf),
    )
    weights = jnp.exp(log_dZ - log_Z[:, None])
    entropy_terms = jnp.where(
        block_valid_mask[None, :],
        log_dZ * 0.0 + 1.0,
        jnp.zeros_like(log_dZ),
    )
    entropy_terms = entropy_terms * (
        jnp.where(block_valid_mask, 1.0, 0.0)[None, :]
    )
    del entropy_terms
    return log_dZ_mean, log_dZ_var, weights


def _sample_mc_shrinkage(
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
        block_state: BlockState | None = None,
        rho_grid: Optional[FloatArray] = None,
        rho_prior: str = "none",
        batch_size: int | None = None,
        C_min: float = 20,
) -> EvidenceSamples:
    del eps_equal_prior, rho_grid, rho_prior, batch_size
    N = log_L_classic.shape[0]
    sample_valid_mask = jnp.arange(N, dtype=jnp.int32) < num_samples
    positive_live_mask = K_classic > 0
    effective_sample_mask = jnp.logical_and(sample_valid_mask, positive_live_mask)
    if block_state is None:
        n = int(np.asarray(num_samples))
        log_l_np = np.asarray(log_L_classic, dtype=float)
        live_np = np.asarray(K_classic, dtype=np.int32)
        sample_mask_np = np.arange(log_l_np.shape[0]) < n
        active_np = sample_mask_np & (live_np > 0)
        valid_classic_np = np.where(active_np, log_l_np, np.inf)
        sorted_order_np = np.argsort(valid_classic_np, kind="stable")
        sorted_log_np = valid_classic_np[sorted_order_np]
        sorted_k_np = live_np[sorted_order_np]
        unique_log_np, starts_np, counts_np = np.unique(
            sorted_log_np[np.isfinite(sorted_log_np)],
            return_index=True,
            return_counts=True,
        )
        block_count = unique_log_np.shape[0]
        log_L_blocks = jnp.asarray(unique_log_np, dtype=log_L_classic.dtype)
        block_valid_mask = jnp.ones((block_count,), dtype=jnp.bool_)
        block_first_idx = jnp.asarray(
            sorted_order_np[starts_np],
            dtype=jnp.int32,
        )
        block_size = jnp.asarray(counts_np, dtype=jnp.int32)
        incoming_K_int = jnp.asarray(
            sorted_k_np[starts_np],
            dtype=jnp.int32,
        )
        block_out_degree = jnp.zeros((block_count,), dtype=jnp.int32)
        block_start = None
        block_stop = None
        block_sample_indices = None
    else:
        log_L_blocks = block_state.log_L_blocks
        block_valid_mask = block_state.valid
        block_first_idx = block_state.block_first_idx.astype(jnp.int32)
        block_size = block_state.block_size.astype(jnp.int32)
        incoming_K_int = block_state.incoming_K.astype(jnp.int32)
        block_out_degree = block_state.block_out_degree.astype(jnp.int32)
        block_start = block_state.block_start
        block_stop = block_state.block_stop
        block_sample_indices = block_state.block_sample_indices

    num_blocks = log_L_blocks.shape[0]
    block_state_for_v3 = BlockState(
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
        block_size=block_size,
        incoming_K=incoming_K_int,
        block_out_degree=block_out_degree,
        valid=block_valid_mask,
        block_start=block_start,
        block_stop=block_stop,
        block_sample_indices=block_sample_indices,
    )
    classic_concentrations = classic_dirichlet_concentrations(block_state_for_v3)
    counts = compute_phantom_count_matrices(
        log_L_blocks=log_L_blocks,
        block_valid_mask=block_valid_mask,
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=effective_sample_mask,
        C_min=C_min,
    )
    probability_samples = sample_gamma_weighted_phantom_probabilities(
        key=key,
        block_state=block_state_for_v3,
        A_cg=counts.A_cg,
        B_cg=counts.B_cg,
        E_cg=counts.E_cg,
        num_samples=num_Z_samples,
        C_min=C_min,
    )
    p_gt_for_path = jnp.where(
        block_valid_mask[None, :],
        probability_samples.p_gt_samples,
        jnp.ones((num_Z_samples, num_blocks), dtype=log_L_blocks.dtype),
    )
    p_gt_for_path = jnp.clip(p_gt_for_path, 1e-300, 1.0)
    log_X = jnp.cumsum(jnp.log(p_gt_for_path), axis=-1)
    log_X_prev = jnp.concatenate(
        [
            jnp.zeros((num_Z_samples, 1), dtype=log_X.dtype),
            log_X[:, :-1],
        ],
        axis=-1,
    )
    log_dX = _logdiffexp(log_X_prev, log_X)
    log_dZ = log_dX + log_L_blocks[None, :]
    log_dZ = jnp.where(block_valid_mask[None, :], log_dZ, -jnp.inf)
    log_Z_samples = _logsumexp(log_dZ, axis=-1)
    log_dZ_mean, log_dZ_var, weights = _summarise_gamma_log_dz_samples(
        log_dZ=log_dZ,
        log_Z=log_Z_samples,
        block_valid_mask=block_valid_mask,
    )
    entropy_terms = jnp.where(
        block_valid_mask[None, :],
        log_L_blocks[None, :] - log_Z_samples[:, None],
        jnp.zeros_like(log_dZ),
    )
    H_samples = jnp.sum(weights * entropy_terms, axis=-1)
    p_gt_mean = jnp.mean(probability_samples.p_gt_samples, axis=0)
    p_eq_mean = jnp.mean(probability_samples.p_eq_samples, axis=0)
    p_lt_mean = jnp.mean(probability_samples.p_lt_samples, axis=0)
    return EvidenceSamples(
        log_Z_samples=log_Z_samples,
        H_samples=H_samples,
        log_dZ_mean=jnp.where(
            block_valid_mask,
            log_dZ_mean,
            jnp.full_like(log_dZ_mean, -jnp.inf),
        ),
        log_dZ_var=jnp.where(
            block_valid_mask,
            log_dZ_var,
            jnp.full_like(log_dZ_var, -jnp.inf),
        ),
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
        block_size=block_size,
        incoming_K=incoming_K_int,
        kish_participating_cluster_counts=counts.kish_participating_cluster_counts,
        phantom_gate_active=counts.phantom_gate_active,
        phantom_A=counts.A_g,
        phantom_B=counts.B_g,
        phantom_E=counts.E_g,
        phantom_R=counts.R_g,
        classic_alpha_gt=classic_concentrations.alpha_gt,
        classic_alpha_eq=classic_concentrations.alpha_eq,
        classic_alpha_lt=classic_concentrations.alpha_lt,
        conditioned_alpha_gt=None,
        conditioned_alpha_eq=None,
        conditioned_alpha_lt=None,
        epsilon=classic_concentrations.epsilon,
        p_gt_samples=probability_samples.p_gt_samples,
        p_eq_samples=probability_samples.p_eq_samples,
        p_lt_samples=probability_samples.p_lt_samples,
        p_gt_mean=jnp.where(block_valid_mask, p_gt_mean, jnp.nan),
        p_eq_mean=jnp.where(block_valid_mask, p_eq_mean, jnp.nan),
        p_lt_mean=jnp.where(block_valid_mask, p_lt_mean, jnp.nan),
        phantom_add_gt_samples=probability_samples.phantom_add_gt_samples,
        phantom_add_eq_samples=probability_samples.phantom_add_eq_samples,
        phantom_add_lt_samples=probability_samples.phantom_add_lt_samples,
        rho_samples=None,
        rho_values=None,
        rho_fit=None,
        eta_samples=None,
        rho_eta_samples=None,
    )
