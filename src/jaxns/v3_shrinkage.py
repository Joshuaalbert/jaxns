import dataclasses

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from jaxns.log_semiring import LogSpace, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import BlockState
from jaxns.types import BoolArray, FloatArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class DirichletConcentrations(PureDataclassPytree):
    alpha_gt: FloatArray
    alpha_eq: FloatArray
    alpha_lt: FloatArray
    epsilon: FloatArray


DirichletConcentrations.register_pytree()


def dirichlet_probability_means(
        concentrations: DirichletConcentrations,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Posterior means for v3 `(p_>, p_=, p_<)` block probabilities."""
    alpha0 = (
            concentrations.alpha_gt
            + concentrations.alpha_eq
            + concentrations.alpha_lt
    )
    p_gt = jnp.where(alpha0 > 0.0, concentrations.alpha_gt / alpha0, 0.0)
    p_eq = jnp.where(alpha0 > 0.0, concentrations.alpha_eq / alpha0, 0.0)
    p_lt = jnp.where(alpha0 > 0.0, concentrations.alpha_lt / alpha0, 0.0)
    return p_gt, p_eq, p_lt


@dataclasses.dataclass(slots=True, frozen=True)
class V3EvidenceSamples(PureDataclassPytree):
    log_Z_samples: FloatArray
    log_dZ_samples: FloatArray
    p_gt_samples: FloatArray
    p_eq_samples: FloatArray


V3EvidenceSamples.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class V3EvidenceSummary(PureDataclassPytree):
    log_Z_mean: FloatArray
    log_Z_uncert: FloatArray
    log_Z_linear_mean: FloatArray
    log_Z2_linear_mean: FloatArray
    log_dZ_mean: FloatArray
    log_dZ2_mean: FloatArray
    log_dZ2_sum: FloatArray
    log_X_mean: FloatArray


V3EvidenceSummary.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class PhantomCountMatrices(PureDataclassPytree):
    A_cg: FloatArray
    B_cg: FloatArray
    E_cg: FloatArray
    R_cg: FloatArray
    A_g: FloatArray
    B_g: FloatArray
    E_g: FloatArray
    R_g: FloatArray
    kish_participating_cluster_counts: FloatArray
    phantom_gate_active: BoolArray


PhantomCountMatrices.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class GammaWeightedPhantomDraws(PureDataclassPytree):
    race_gamma_gt: FloatArray
    race_gamma_eq: FloatArray
    race_gamma_lt: FloatArray
    cluster_weights: FloatArray


GammaWeightedPhantomDraws.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class GammaWeightedPhantomProbabilities(PureDataclassPytree):
    p_gt: FloatArray
    p_eq: FloatArray
    p_lt: FloatArray
    phantom_add_gt: FloatArray
    phantom_add_eq: FloatArray
    phantom_add_lt: FloatArray
    kish_participating_cluster_counts: FloatArray
    phantom_gate_active: BoolArray


GammaWeightedPhantomProbabilities.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class GammaWeightedPhantomProbabilitySamples(PureDataclassPytree):
    p_gt_samples: FloatArray
    p_eq_samples: FloatArray
    p_lt_samples: FloatArray
    phantom_add_gt_samples: FloatArray
    phantom_add_eq_samples: FloatArray
    phantom_add_lt_samples: FloatArray
    kish_participating_cluster_counts: FloatArray
    phantom_gate_active: BoolArray
    race_gamma_gt: FloatArray
    race_gamma_eq: FloatArray
    race_gamma_lt: FloatArray
    cluster_weights: FloatArray


GammaWeightedPhantomProbabilitySamples.register_pytree()


def epsilon_for_block_size(block_size: FloatArray) -> FloatArray:
    """Paper default equality-atom prior policy."""
    block_size = jnp.asarray(block_size)
    return jnp.where(
        block_size == 1,
        jnp.asarray(1e-6, dtype=mp_policy.measure_dtype),
        jnp.asarray(0.5, dtype=mp_policy.measure_dtype),
    )


def classic_dirichlet_concentrations(block_state: BlockState) -> DirichletConcentrations:
    """Classic v3 Dirichlet concentrations for `(p_>, p_=, p_<)`."""
    validate_lineage_capacity(block_state)
    dtype = mp_policy.measure_dtype
    k_g = block_state.incoming_K.astype(dtype)
    m_g = block_state.block_size.astype(dtype)
    eps = epsilon_for_block_size(m_g).astype(dtype)
    alpha_gt = k_g - m_g + 1.0
    alpha_eq = m_g + eps
    alpha_lt = 1.0 - eps
    zeros = jnp.zeros_like(alpha_gt)
    return DirichletConcentrations(
        alpha_gt=jnp.where(block_state.valid, alpha_gt, zeros),
        alpha_eq=jnp.where(block_state.valid, alpha_eq, zeros),
        alpha_lt=jnp.where(block_state.valid, alpha_lt, zeros),
        epsilon=jnp.where(block_state.valid, eps, zeros),
    )


def phantom_conditioned_dirichlet_concentrations(
        block_state: BlockState,
        phantom_A: FloatArray,
        phantom_B: FloatArray,
        phantom_E: FloatArray,
        rho_g: FloatArray,
) -> DirichletConcentrations:
    """Phantom-conditioned Dirichlet concentrations for v3 blocks.

    Blocks with `A_g = 0` naturally reduce to the classic concentrations when
    `B_g = E_g = 0`. This is the explicit no-phantom-information convention for
    the paper's otherwise undefined `hat p_g` case.
    """
    _validate_phantom_conditioned_inputs(
        block_state=block_state,
        phantom_A=phantom_A,
        phantom_B=phantom_B,
        phantom_E=phantom_E,
        rho_g=rho_g,
    )
    classic = classic_dirichlet_concentrations(block_state)
    dtype = classic.alpha_gt.dtype
    A = jnp.asarray(phantom_A, dtype=dtype)
    B = jnp.asarray(phantom_B, dtype=dtype)
    E = jnp.asarray(phantom_E, dtype=dtype)
    rho = jnp.asarray(rho_g, dtype=dtype)
    alpha_gt = classic.alpha_gt + rho * B
    alpha_eq = classic.alpha_eq + rho * E
    alpha_lt = classic.alpha_lt + rho * (A - B - E)
    zeros = jnp.zeros_like(alpha_gt)
    return DirichletConcentrations(
        alpha_gt=jnp.where(block_state.valid, alpha_gt, zeros),
        alpha_eq=jnp.where(block_state.valid, alpha_eq, zeros),
        alpha_lt=jnp.where(block_state.valid, alpha_lt, zeros),
        epsilon=classic.epsilon,
    )


def _validate_phantom_conditioned_inputs(
        *,
        block_state: BlockState,
        phantom_A: FloatArray,
        phantom_B: FloatArray,
        phantom_E: FloatArray,
        rho_g: FloatArray,
) -> None:
    expected_shape = block_state.log_L_blocks.shape
    for name, value in (
            ("phantom_A", phantom_A),
            ("phantom_B", phantom_B),
            ("phantom_E", phantom_E),
            ("rho_g", rho_g),
    ):
        if jnp.shape(value) != expected_shape:
            raise ValueError(
                f"{name} shape must align with block_state.log_L_blocks; "
                f"got {jnp.shape(value)}, expected {expected_shape}."
            )

    try:
        valid = np.asarray(block_state.valid, dtype=bool)
        A = np.asarray(phantom_A, dtype=float)
        B = np.asarray(phantom_B, dtype=float)
        E = np.asarray(phantom_E, dtype=float)
        rho = np.asarray(rho_g, dtype=float)
    except Exception:
        return

    if not np.all(np.isfinite(A[valid])):
        raise ValueError("phantom_A counts must be finite on valid blocks.")
    if not np.all(np.isfinite(B[valid])):
        raise ValueError("phantom_B counts must be finite on valid blocks.")
    if not np.all(np.isfinite(E[valid])):
        raise ValueError("phantom_E counts must be finite on valid blocks.")
    if np.any(A[valid] < 0.0) or np.any(B[valid] < 0.0) or np.any(E[valid] < 0.0):
        raise ValueError("Phantom Dirichlet counts must be non-negative.")
    if np.any(B[valid] + E[valid] > A[valid]):
        raise ValueError(
            "Invalid Dirichlet phantom count relation: B_g + E_g must be <= A_g."
        )
    if (
            not np.all(np.isfinite(rho[valid]))
            or np.any(rho[valid] <= 0.0)
            or np.any(rho[valid] > 1.0)
    ):
        raise ValueError("rho_g must be finite, positive, and <= 1 on valid blocks.")


def estimate_raw_rho_g_from_bootstrap_covariance(
        *,
        A: FloatArray,
        B: FloatArray,
        E: FloatArray,
        bootstrap_covariance: FloatArray,
        fallback_rho: float = 1.0,
) -> FloatArray:
    """Estimate raw per-block `rho_g` using the paper's rank/trace formula."""
    A = jnp.asarray(A, dtype=mp_policy.measure_dtype)
    B = jnp.asarray(B, dtype=mp_policy.measure_dtype)
    E = jnp.asarray(E, dtype=mp_policy.measure_dtype)
    bootstrap_covariance = jnp.asarray(
        bootstrap_covariance,
        dtype=mp_policy.measure_dtype,
    )
    fallback = jnp.asarray(fallback_rho, dtype=A.dtype)
    fallback = jnp.clip(fallback, jnp.finfo(A.dtype).tiny, 1.0)

    A_safe = jnp.where(A > 0.0, A, 1.0)
    q_gt = B / A_safe
    q_eq = E / A_safe
    sigma = jnp.stack(
        [
            jnp.stack(
                [
                    q_gt * (1.0 - q_gt) / A_safe,
                    -q_gt * q_eq / A_safe,
                ],
                axis=-1,
            ),
            jnp.stack(
                [
                    -q_gt * q_eq / A_safe,
                    q_eq * (1.0 - q_eq) / A_safe,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )

    sigma_pinv = jnp.linalg.pinv(sigma)
    rank = jnp.linalg.matrix_rank(sigma).astype(A.dtype)
    denominator = jnp.trace(
        jnp.matmul(sigma_pinv, bootstrap_covariance),
        axis1=-2,
        axis2=-1,
    )
    raw = rank / denominator
    usable = (
            (A > 0.0)
            & (rank > 0.0)
            & jnp.isfinite(denominator)
            & (denominator > 0.0)
            & jnp.isfinite(raw)
            & (raw > 0.0)
    )
    raw = jnp.where(usable, raw, fallback)
    return jnp.clip(raw, jnp.finfo(A.dtype).tiny, 1.0)


def fit_low_order_rho_g_curve(
        *,
        raw_rho_g: FloatArray,
        race_time: FloatArray,
        valid_mask: BoolArray,
        polynomial_order: int = 2,
        fallback_rho: float = 1.0,
) -> FloatArray:
    """Fit a bounded low-order rho curve against normalized race time."""
    raw_rho_g = jnp.asarray(raw_rho_g, dtype=mp_policy.measure_dtype)
    race_time = jnp.asarray(race_time, dtype=mp_policy.measure_dtype)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)
    fallback = jnp.asarray(fallback_rho, dtype=raw_rho_g.dtype)
    fallback = jnp.clip(fallback, jnp.finfo(raw_rho_g.dtype).tiny, 1.0)

    fit_mask = (
            valid_mask
            & jnp.isfinite(raw_rho_g)
            & (raw_rho_g > 0.0)
            & jnp.isfinite(race_time)
    )
    valid_count = jnp.sum(fit_mask.astype(mp_policy.count_dtype))
    max_time = jnp.max(
        jnp.where(fit_mask, race_time, jnp.asarray(0.0, dtype=race_time.dtype))
    )
    normalized_time = jnp.where(max_time > 0.0, race_time / max_time, 0.0)
    normalized_time = jnp.where(jnp.isfinite(normalized_time), normalized_time, 0.0)

    powers = jnp.arange(polynomial_order + 1, dtype=mp_policy.count_dtype)
    design = normalized_time[:, None] ** powers[None, :]
    weights = fit_mask.astype(raw_rho_g.dtype)
    weighted_design = design * weights[:, None]
    rho_floor = jnp.asarray(jnp.finfo(raw_rho_g.dtype).tiny, dtype=raw_rho_g.dtype)
    weighted_y = jnp.clip(raw_rho_g, rho_floor, 1.0) * weights
    coeffs = jnp.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    fitted = design @ coeffs
    fitted = jnp.where(valid_count > 0, fitted, fallback)
    fitted = jnp.where(jnp.isfinite(fitted), fitted, fallback)
    fitted = jnp.clip(fitted, jnp.finfo(raw_rho_g.dtype).tiny, 1.0)
    return jnp.where(valid_mask, fitted, jnp.nan)


def validate_phantom_count_matrices(
        *,
        A_cg: FloatArray,
        B_cg: FloatArray,
        E_cg: FloatArray,
        block_valid_mask: BoolArray,
) -> None:
    """Validate per-cluster phantom count matrices for v3 conditioning."""
    expected_shape = jnp.shape(A_cg)
    if len(expected_shape) != 2:
        raise ValueError("A_cg must have shape [num_clusters, num_blocks].")
    for name, value in (("B_cg", B_cg), ("E_cg", E_cg)):
        if jnp.shape(value) != expected_shape:
            raise ValueError(
                f"{name} shape must align with A_cg; got {jnp.shape(value)}, "
                f"expected {expected_shape}."
            )
    if jnp.shape(block_valid_mask) != (expected_shape[1],):
        raise ValueError(
            "block_valid_mask shape must align with the count matrix block "
            f"axis; got {jnp.shape(block_valid_mask)}, expected "
            f"{(expected_shape[1],)}."
        )

    try:
        valid = np.asarray(block_valid_mask, dtype=bool)
        A = np.asarray(A_cg, dtype=float)
        B = np.asarray(B_cg, dtype=float)
        E = np.asarray(E_cg, dtype=float)
    except Exception:
        return

    if not np.all(np.isfinite(A[:, valid])):
        raise ValueError("A_cg counts must be finite on valid blocks.")
    if not np.all(np.isfinite(B[:, valid])):
        raise ValueError("B_cg counts must be finite on valid blocks.")
    if not np.all(np.isfinite(E[:, valid])):
        raise ValueError("E_cg counts must be finite on valid blocks.")
    if (
            np.any(A[:, valid] < 0.0)
            or np.any(B[:, valid] < 0.0)
            or np.any(E[:, valid] < 0.0)
    ):
        raise ValueError("Phantom count matrices must be non-negative.")
    if np.any(B[:, valid] + E[:, valid] > A[:, valid]):
        raise ValueError(
            "Invalid phantom count relation: B_cg + E_cg must be <= A_cg."
        )


def compute_kish_participating_cluster_counts(
        A_cg: FloatArray,
) -> FloatArray:
    """Kish participating-cluster count per block from `A_cg`."""
    A = jnp.asarray(A_cg, dtype=mp_policy.measure_dtype)
    numerator = jnp.square(jnp.sum(A, axis=0))
    denominator = jnp.sum(jnp.square(A), axis=0)
    return jnp.where(
        denominator > 0.0,
        numerator / denominator,
        jnp.zeros_like(numerator),
    )


def compute_phantom_gate_active(
        A_cg: FloatArray,
        *,
        C_min: float = 20,
) -> BoolArray:
    """Return the canonical Kish gate mask, defaulting to `C_min=20`."""
    A = jnp.asarray(A_cg, dtype=mp_policy.measure_dtype)
    participation = jnp.sum(A, axis=0) > 0.0
    denominator = jnp.sum(jnp.square(A), axis=0)
    kish = compute_kish_participating_cluster_counts(A)
    return (denominator > 0.0) & participation & (
            kish >= jnp.asarray(C_min, dtype=kish.dtype)
    )


def gamma_weighted_phantom_probabilities_from_draws(
        *,
        block_state: BlockState,
        A_cg: FloatArray,
        B_cg: FloatArray,
        E_cg: FloatArray,
        race_gamma_gt: FloatArray,
        race_gamma_eq: FloatArray,
        race_gamma_lt: FloatArray,
        cluster_weights: FloatArray,
        C_min: float = 20,
) -> GammaWeightedPhantomProbabilities:
    """Apply explicit race gammas and shared per-cluster gamma weights."""
    validate_phantom_count_matrices(
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        block_valid_mask=block_state.valid,
    )
    A = jnp.asarray(A_cg, dtype=mp_policy.measure_dtype)
    B = jnp.asarray(B_cg, dtype=A.dtype)
    E = jnp.asarray(E_cg, dtype=A.dtype)
    R = A - B - E
    weights = jnp.asarray(cluster_weights, dtype=A.dtype)
    if jnp.shape(weights) != (A.shape[0],):
        raise ValueError(
            "cluster_weights shape must align with the count matrix cluster "
            f"axis; got {jnp.shape(weights)}, expected {(A.shape[0],)}."
        )
    expected_block_shape = block_state.log_L_blocks.shape
    for name, value in (
            ("race_gamma_gt", race_gamma_gt),
            ("race_gamma_eq", race_gamma_eq),
            ("race_gamma_lt", race_gamma_lt),
    ):
        if jnp.shape(value) != expected_block_shape:
            raise ValueError(
                f"{name} shape must align with block_state.log_L_blocks; "
                f"got {jnp.shape(value)}, expected {expected_block_shape}."
            )

    kish = compute_kish_participating_cluster_counts(A)
    gate = compute_phantom_gate_active(A, C_min=C_min) & block_state.valid
    gate_float = gate.astype(A.dtype)
    phantom_add_gt = (weights @ B) * gate_float
    phantom_add_eq = (weights @ E) * gate_float
    phantom_add_lt = (weights @ R) * gate_float
    m_gt = jnp.asarray(race_gamma_gt, dtype=A.dtype) + phantom_add_gt
    m_eq = jnp.asarray(race_gamma_eq, dtype=A.dtype) + phantom_add_eq
    m_lt = jnp.asarray(race_gamma_lt, dtype=A.dtype) + phantom_add_lt
    total = m_gt + m_eq + m_lt
    safe_total = jnp.where(total > 0.0, total, jnp.ones_like(total))
    p_gt = jnp.where(block_state.valid, m_gt / safe_total, 0.0)
    p_eq = jnp.where(block_state.valid, m_eq / safe_total, 0.0)
    p_lt = jnp.where(block_state.valid, m_lt / safe_total, 0.0)
    return GammaWeightedPhantomProbabilities(
        p_gt=p_gt,
        p_eq=p_eq,
        p_lt=p_lt,
        phantom_add_gt=phantom_add_gt,
        phantom_add_eq=phantom_add_eq,
        phantom_add_lt=phantom_add_lt,
        kish_participating_cluster_counts=jnp.where(
            block_state.valid,
            kish,
            jnp.zeros_like(kish),
        ),
        phantom_gate_active=gate,
    )


def sample_gamma_weighted_phantom_draws(
        *,
        key: PRNGKey,
        block_state: BlockState,
        num_clusters: int,
        num_samples: int,
) -> GammaWeightedPhantomDraws:
    """Draw independent race gammas and shared `v_c ~ Gamma(1, 1)` weights."""
    concentrations = classic_dirichlet_concentrations(block_state)
    alpha_gt = jnp.where(block_state.valid, concentrations.alpha_gt, 1.0)
    alpha_eq = jnp.where(block_state.valid, concentrations.alpha_eq, 1.0)
    alpha_lt = jnp.where(block_state.valid, concentrations.alpha_lt, 1.0)
    key_gt, key_eq, key_lt, key_v = jax.random.split(key, 4)
    sample_shape = (int(num_samples),) + block_state.log_L_blocks.shape
    race_gamma_gt = jax.random.gamma(key_gt, alpha_gt, shape=sample_shape)
    race_gamma_eq = jax.random.gamma(key_eq, alpha_eq, shape=sample_shape)
    race_gamma_lt = jax.random.gamma(key_lt, alpha_lt, shape=sample_shape)
    race_gamma_gt = jnp.where(block_state.valid[None, :], race_gamma_gt, 0.0)
    race_gamma_eq = jnp.where(block_state.valid[None, :], race_gamma_eq, 0.0)
    race_gamma_lt = jnp.where(block_state.valid[None, :], race_gamma_lt, 0.0)
    cluster_shape = (int(num_samples), int(num_clusters))
    cluster_weights = jax.random.gamma(
        key_v,
        jnp.ones((int(num_clusters),), dtype=mp_policy.measure_dtype),
        shape=cluster_shape,
    )
    return GammaWeightedPhantomDraws(
        race_gamma_gt=race_gamma_gt,
        race_gamma_eq=race_gamma_eq,
        race_gamma_lt=race_gamma_lt,
        cluster_weights=cluster_weights,
    )


def sample_gamma_weighted_phantom_probabilities(
        *,
        key: PRNGKey,
        block_state: BlockState,
        A_cg: FloatArray,
        B_cg: FloatArray,
        E_cg: FloatArray,
        num_samples: int,
        C_min: float = 20,
) -> GammaWeightedPhantomProbabilitySamples:
    """Sample gamma-weighted phantom-conditioned block probabilities."""
    validate_phantom_count_matrices(
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        block_valid_mask=block_state.valid,
    )
    A = jnp.asarray(A_cg, dtype=mp_policy.measure_dtype)
    B = jnp.asarray(B_cg, dtype=A.dtype)
    E = jnp.asarray(E_cg, dtype=A.dtype)
    R = A - B - E
    draws = sample_gamma_weighted_phantom_draws(
        key=key,
        block_state=block_state,
        num_clusters=A.shape[0],
        num_samples=int(num_samples),
    )
    kish = compute_kish_participating_cluster_counts(A)
    gate = compute_phantom_gate_active(A, C_min=C_min) & block_state.valid
    gate_float = gate.astype(A.dtype)[None, :]
    phantom_add_gt = (draws.cluster_weights @ B) * gate_float
    phantom_add_eq = (draws.cluster_weights @ E) * gate_float
    phantom_add_lt = (draws.cluster_weights @ R) * gate_float
    m_gt = draws.race_gamma_gt + phantom_add_gt
    m_eq = draws.race_gamma_eq + phantom_add_eq
    m_lt = draws.race_gamma_lt + phantom_add_lt
    total = m_gt + m_eq + m_lt
    safe_total = jnp.where(total > 0.0, total, jnp.ones_like(total))
    valid = block_state.valid[None, :]
    p_gt = jnp.where(valid, m_gt / safe_total, 0.0)
    p_eq = jnp.where(valid, m_eq / safe_total, 0.0)
    p_lt = jnp.where(valid, m_lt / safe_total, 0.0)
    return GammaWeightedPhantomProbabilitySamples(
        p_gt_samples=p_gt,
        p_eq_samples=p_eq,
        p_lt_samples=p_lt,
        phantom_add_gt_samples=phantom_add_gt,
        phantom_add_eq_samples=phantom_add_eq,
        phantom_add_lt_samples=phantom_add_lt,
        kish_participating_cluster_counts=jnp.where(
            block_state.valid,
            kish,
            jnp.zeros_like(kish),
        ),
        phantom_gate_active=gate,
        race_gamma_gt=draws.race_gamma_gt,
        race_gamma_eq=draws.race_gamma_eq,
        race_gamma_lt=draws.race_gamma_lt,
        cluster_weights=draws.cluster_weights,
    )


def validate_lineage_capacity(block_state: BlockState) -> None:
    """Raise if any valid block has fewer incoming lineages than members."""
    try:
        valid = np.asarray(block_state.valid, dtype=bool)
        incoming = np.asarray(block_state.incoming_K)
        sizes = np.asarray(block_state.block_size)
    except Exception:
        return
    if np.any(valid & (incoming < sizes)):
        bad = np.where(valid & (incoming < sizes))[0][0]
        raise ValueError(
            f"Invalid race block {bad}: incoming K_g={incoming[bad]} "
            f"is smaller than plateau size m_g={sizes[bad]}."
        )


def sample_dirichlet_probabilities(
        key: PRNGKey,
        concentrations: DirichletConcentrations,
        *,
        num_samples: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sample block probabilities from v3 Dirichlet concentrations."""
    alpha = jnp.stack(
        [
            concentrations.alpha_gt,
            concentrations.alpha_eq,
            concentrations.alpha_lt,
        ],
        axis=-1,
    )
    safe_alpha = jnp.where(alpha > 0.0, alpha, 1.0)
    gamma = jax.random.gamma(key, safe_alpha, shape=(num_samples,) + safe_alpha.shape)
    probs = gamma / jnp.sum(gamma, axis=-1, keepdims=True)
    valid = concentrations.alpha_gt > 0.0
    probs = jnp.where(valid[None, :, None], probs, 0.0)
    return probs[..., 0], probs[..., 1], probs[..., 2]


def _logdiffexp(log_a: FloatArray, log_b: FloatArray) -> FloatArray:
    return log_a + jnp.log1p(-jnp.exp(log_b - log_a))


def expected_v3_evidence_summary(
        block_state: BlockState,
        concentrations: DirichletConcentrations,
) -> V3EvidenceSummary:
    """Analytic v3 block evidence moments under independent Dirichlet blocks."""
    alpha_gt = concentrations.alpha_gt
    alpha_not_gt = concentrations.alpha_eq + concentrations.alpha_lt
    alpha0 = alpha_gt + alpha_not_gt
    valid = block_state.valid & (alpha0 > 0.0)
    one = jnp.asarray(1.0, dtype=alpha0.dtype)

    log_p_gt_mean = jnp.log(alpha_gt) - jnp.log(alpha0)
    log_p_gt_second = (
            jnp.log(alpha_gt)
            + jnp.log(alpha_gt + one)
            - jnp.log(alpha0)
            - jnp.log(alpha0 + one)
    )
    log_not_gt_mean = jnp.log(alpha_not_gt) - jnp.log(alpha0)
    log_not_gt_second = (
            jnp.log(alpha_not_gt)
            + jnp.log(alpha_not_gt + one)
            - jnp.log(alpha0)
            - jnp.log(alpha0 + one)
    )
    log_gt_not_gt = (
            jnp.log(alpha_gt)
            + jnp.log(alpha_not_gt)
            - jnp.log(alpha0)
            - jnp.log(alpha0 + one)
    )

    log_p_gt_prefix = jnp.cumsum(jnp.where(valid, log_p_gt_mean, 0.0))
    log_p_gt_prefix_with_zero = jnp.concatenate(
        [jnp.zeros((1,), dtype=log_p_gt_prefix.dtype), log_p_gt_prefix],
        axis=0,
    )
    log_p_gt_prev = jnp.concatenate(
        [jnp.zeros((1,), dtype=log_p_gt_prefix.dtype), log_p_gt_prefix[:-1]],
        axis=0,
    )
    log_p_gt_second_prefix = jnp.cumsum(jnp.where(valid, log_p_gt_second, 0.0))
    log_p_gt_second_prev = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=log_p_gt_second_prefix.dtype),
            log_p_gt_second_prefix[:-1],
        ],
        axis=0,
    )

    log_dX_mean = log_p_gt_prev + log_not_gt_mean
    log_dX_second = log_p_gt_second_prev + log_not_gt_second
    log_dZ_mean = block_state.log_L_blocks + log_dX_mean
    log_dZ2_mean = 2.0 * block_state.log_L_blocks + log_dX_second
    log_dZ_mean = jnp.where(valid, log_dZ_mean, -jnp.inf)
    log_dZ2_mean = jnp.where(valid, log_dZ2_mean, -jnp.inf)

    log_Z_linear_mean = logsumexp(log_dZ_mean, axis=0)
    log_dZ2_sum = logsumexp(log_dZ2_mean, axis=0)

    num_blocks = block_state.log_L_blocks.shape[0]
    lower_idx = jnp.arange(num_blocks)[:, None]
    upper_idx = jnp.arange(num_blocks)[None, :]
    cross_mask = lower_idx < upper_idx
    safe_gap_mean = (
            log_p_gt_prefix_with_zero[upper_idx]
            - log_p_gt_prefix_with_zero[lower_idx + 1]
    )
    log_dX_cross = (
            log_p_gt_second_prev[:, None]
            + log_gt_not_gt[:, None]
            + safe_gap_mean
            + log_not_gt_mean[None, :]
    )
    log_dZ_cross = (
            jnp.log(jnp.asarray(2.0, dtype=alpha0.dtype))
            + block_state.log_L_blocks[:, None]
            + block_state.log_L_blocks[None, :]
            + log_dX_cross
    )
    valid_cross = valid[:, None] & valid[None, :] & cross_mask
    log_dZ_cross = jnp.where(valid_cross, log_dZ_cross, -jnp.inf)
    log_Z2_linear_mean = jnp.logaddexp(log_dZ2_sum, logsumexp(log_dZ_cross))
    log_Z_mean = 2.0 * log_Z_linear_mean - 0.5 * log_Z2_linear_mean
    log_Z_var = jnp.maximum(
        log_Z2_linear_mean - 2.0 * log_Z_linear_mean,
        jnp.finfo(mp_policy.measure_dtype).eps,
    )
    log_Z_uncert = jnp.sqrt(log_Z_var)

    log_X_mean = scatter_v3_block_values_to_samples(block_state, log_p_gt_prefix)
    return V3EvidenceSummary(
        log_Z_mean=log_Z_mean,
        log_Z_uncert=log_Z_uncert,
        log_Z_linear_mean=log_Z_linear_mean,
        log_Z2_linear_mean=log_Z2_linear_mean,
        log_dZ_mean=log_dZ_mean,
        log_dZ2_mean=log_dZ2_mean,
        log_dZ2_sum=log_dZ2_sum,
        log_X_mean=log_X_mean,
    )


def sample_v3_evidence(
        key: PRNGKey,
        block_state: BlockState,
        concentrations: DirichletConcentrations,
        *,
        num_samples: int,
) -> V3EvidenceSamples:
    """Sample evidence using `X_g = X_{g-1} p_{>g}` block shrinkage."""
    p_gt, p_eq, _ = sample_dirichlet_probabilities(
        key,
        concentrations,
        num_samples=num_samples,
    )
    p_gt = jnp.clip(p_gt, 1e-300, 1.0)
    log_X = jnp.cumsum(jnp.log(p_gt), axis=-1)
    log_X_prev = jnp.concatenate(
        [jnp.zeros((num_samples, 1), dtype=log_X.dtype), log_X[:, :-1]],
        axis=-1,
    )
    log_dX = _logdiffexp(log_X_prev, log_X)
    log_dZ = log_dX + block_state.log_L_blocks[None, :]
    log_dZ = jnp.where(block_state.valid[None, :], log_dZ, -jnp.inf)
    log_Z = logsumexp(log_dZ, axis=-1)
    return V3EvidenceSamples(
        log_Z_samples=log_Z,
        log_dZ_samples=log_dZ,
        p_gt_samples=p_gt,
        p_eq_samples=p_eq,
    )


def expected_v3_log_posterior_weights(
        block_state: BlockState,
        concentrations: DirichletConcentrations,
) -> FloatArray:
    """Expected per-classic-sample log posterior weights under v3 block rules."""
    alpha0 = concentrations.alpha_gt + concentrations.alpha_eq + concentrations.alpha_lt
    p_gt = jnp.where(alpha0 > 0.0, concentrations.alpha_gt / alpha0, 1.0)
    p_eq = jnp.where(alpha0 > 0.0, concentrations.alpha_eq / alpha0, 0.0)
    p_gt = jnp.clip(p_gt, 1e-300, 1.0)
    log_X = jnp.cumsum(jnp.log(p_gt))
    log_X_prev = jnp.concatenate([jnp.zeros((1,), dtype=log_X.dtype), log_X[:-1]], axis=0)

    plateau_mass = (
            block_state.log_L_blocks
            + log_X_prev
            + jnp.log(jnp.clip(p_eq, 1e-300, 1.0))
            - jnp.log(jnp.maximum(block_state.block_size, 1).astype(mp_policy.measure_dtype))
    )
    non_plateau_mass = (
            block_state.log_L_blocks
            + log_X_prev
            + jnp.log(jnp.clip(1.0 - p_gt, 1e-300, 1.0))
    )
    log_block_weight = jnp.where(block_state.block_size > 1, plateau_mass, non_plateau_mass)
    log_block_weight = jnp.where(block_state.valid, log_block_weight, -jnp.inf)
    if block_state.block_sample_indices is None:
        return normalise_log_space(LogSpace(log_block_weight), norm_type="sum").log_abs_val

    try:
        block_indices = np.asarray(block_state.block_sample_indices)
        valid_blocks = np.asarray(block_state.valid, dtype=bool)
    except Exception:
        return _scatter_block_log_weights_jax(block_state, log_block_weight)

    valid_members = block_indices >= 0
    max_idx = int(np.max(block_indices[valid_members])) if np.any(valid_members) else -1
    output_size = max_idx + 1
    if output_size == 0:
        return jnp.zeros((0,), dtype=log_block_weight.dtype)
    sample_log_weights = jnp.full((output_size,), -jnp.inf, dtype=log_block_weight.dtype)
    if block_indices.ndim == 1:
        if block_state.block_start is None or block_state.block_stop is None:
            members_by_block = [
                np.asarray([member])
                for member in block_indices
            ]
        else:
            starts = np.asarray(block_state.block_start)
            stops = np.asarray(block_state.block_stop)
            members_by_block = [
                block_indices[int(start):int(stop)]
                for start, stop in zip(starts, stops, strict=True)
            ]
    else:
        members_by_block = [block_indices[block_idx] for block_idx in range(block_indices.shape[0])]
    for block_idx, members in enumerate(members_by_block):
        if block_idx >= valid_blocks.shape[0] or not bool(valid_blocks[block_idx]):
            continue
        members = members[members >= 0]
        if members.size == 0:
            continue
        sample_log_weights = sample_log_weights.at[
            jnp.asarray(members, dtype=mp_policy.index_dtype)
        ].set(log_block_weight[block_idx])
    return normalise_log_space(LogSpace(sample_log_weights), norm_type="sum").log_abs_val


def scatter_v3_block_values_to_samples(
        block_state: BlockState,
        block_values: FloatArray,
) -> FloatArray:
    """Scatter per-block values to classic sample slots using block membership."""
    if block_state.block_sample_indices is None:
        return block_values

    try:
        block_indices = np.asarray(block_state.block_sample_indices)
        valid_blocks = np.asarray(block_state.valid, dtype=bool)
    except Exception:
        return _scatter_block_values_to_samples_jax(block_state, block_values)

    valid_members = block_indices >= 0
    max_idx = int(np.max(block_indices[valid_members])) if np.any(valid_members) else -1
    output_size = max_idx + 1
    if output_size == 0:
        return jnp.zeros((0,), dtype=block_values.dtype)
    sample_values = jnp.full((output_size,), -jnp.inf, dtype=block_values.dtype)
    if block_indices.ndim == 1:
        if block_state.block_start is None or block_state.block_stop is None:
            members_by_block = [
                np.asarray([member])
                for member in block_indices
            ]
        else:
            starts = np.asarray(block_state.block_start)
            stops = np.asarray(block_state.block_stop)
            members_by_block = [
                block_indices[int(start):int(stop)]
                for start, stop in zip(starts, stops, strict=True)
            ]
    else:
        members_by_block = [block_indices[block_idx] for block_idx in range(block_indices.shape[0])]
    for block_idx, members in enumerate(members_by_block):
        if block_idx >= valid_blocks.shape[0] or not bool(valid_blocks[block_idx]):
            continue
        members = members[members >= 0]
        if members.size == 0:
            continue
        sample_values = sample_values.at[
            jnp.asarray(members, dtype=mp_policy.index_dtype)
        ].set(block_values[block_idx])
    return sample_values


def _scatter_block_values_to_samples_jax(
        block_state: BlockState,
        block_values: FloatArray,
) -> FloatArray:
    block_indices = block_state.block_sample_indices
    output_size = block_state.log_L_blocks.shape[0]
    if len(block_indices.shape) == 1:
        flat_members = block_indices
        if block_state.block_stop is None:
            flat_block_ids = jnp.arange(block_indices.shape[0], dtype=mp_policy.index_dtype)
        else:
            sorted_position = jnp.arange(block_indices.shape[0], dtype=mp_policy.index_dtype)
            flat_block_ids = jnp.searchsorted(
                block_state.block_stop.astype(mp_policy.index_dtype),
                sorted_position,
                side="right",
            ).astype(mp_policy.index_dtype)
            flat_block_ids = jnp.clip(flat_block_ids, 0, output_size - 1)
    else:
        num_blocks, members_per_block = block_indices.shape
        flat_members = jnp.reshape(block_indices, (-1,))
        flat_block_ids = jnp.repeat(
            jnp.arange(num_blocks, dtype=mp_policy.index_dtype),
            members_per_block,
        )
    in_bounds = (flat_members >= 0) & (flat_members < output_size)
    valid_members = in_bounds & block_state.valid[flat_block_ids]
    safe_members = jnp.where(valid_members, flat_members, -1)
    flat_values = jnp.where(
        valid_members,
        block_values[flat_block_ids],
        jnp.asarray(-jnp.inf, dtype=block_values.dtype),
    )
    sample_values = jnp.full(
        (output_size,),
        -jnp.inf,
        dtype=block_values.dtype,
    )
    return sample_values.at[safe_members].set(flat_values, mode="drop")


def _scatter_block_log_weights_jax(
        block_state: BlockState,
        log_block_weight: FloatArray,
) -> FloatArray:
    """Scatter block weights to default sample indices inside JAX transforms."""
    block_indices = block_state.block_sample_indices
    output_size = block_state.log_L_blocks.shape[0]
    if len(block_indices.shape) == 1:
        flat_members = block_indices
        if block_state.block_stop is None:
            flat_block_ids = jnp.arange(block_indices.shape[0], dtype=mp_policy.index_dtype)
        else:
            sorted_position = jnp.arange(block_indices.shape[0], dtype=mp_policy.index_dtype)
            flat_block_ids = jnp.searchsorted(
                block_state.block_stop.astype(mp_policy.index_dtype),
                sorted_position,
                side="right",
            ).astype(mp_policy.index_dtype)
            flat_block_ids = jnp.clip(flat_block_ids, 0, output_size - 1)
    else:
        num_blocks, members_per_block = block_indices.shape
        flat_members = jnp.reshape(block_indices, (-1,))
        flat_block_ids = jnp.repeat(
            jnp.arange(num_blocks, dtype=mp_policy.index_dtype),
            members_per_block,
        )
    in_bounds = (flat_members >= 0) & (flat_members < output_size)
    valid_members = in_bounds & block_state.valid[flat_block_ids]
    safe_members = jnp.where(valid_members, flat_members, -1)
    flat_weights = jnp.where(
        valid_members,
        log_block_weight[flat_block_ids],
        jnp.asarray(-jnp.inf, dtype=log_block_weight.dtype),
    )
    sample_log_weights = jnp.full(
        (output_size,),
        -jnp.inf,
        dtype=log_block_weight.dtype,
    )
    sample_log_weights = sample_log_weights.at[safe_members].set(flat_weights, mode="drop")
    return normalise_log_space(LogSpace(sample_log_weights), norm_type="sum").log_abs_val
