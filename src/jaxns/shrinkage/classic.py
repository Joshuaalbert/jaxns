import dataclasses

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from jaxns.algorithm.race_tree import BlockState
from jaxns.log_semiring import LogSpace, normalise_log_space
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.types import BoolArray, FloatArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class DirichletConcentrations(PureDataclassPytree):
    alpha_gt: FloatArray  # [G]
    alpha_eq: FloatArray  # [G]
    alpha_lt: FloatArray  # [G]
    epsilon: FloatArray  # [G]


DirichletConcentrations.register_pytree()


def dirichlet_probability_means(
        concentrations: DirichletConcentrations,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Posterior means for `(p_>, p_=, p_<)` block probabilities."""
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
class EvidenceSamples(PureDataclassPytree):
    log_Z_samples: FloatArray  # [M]
    log_dZ_samples: FloatArray  # [M, G]
    p_gt_samples: FloatArray  # [M, G]
    p_eq_samples: FloatArray  # [M, G]


EvidenceSamples.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class EvidenceSummary(PureDataclassPytree):
    log_Z_mean: FloatArray  # []
    log_Z_uncert: FloatArray  # []
    log_Z_linear_mean: FloatArray  # [] log(E[Z])
    log_Z2_linear_mean: FloatArray  # [] log(E[Z^2])
    log_dZ_mean: FloatArray  # [G]
    log_dZ2_mean: FloatArray  # [G]
    log_dZ2_sum: FloatArray  # []
    log_X_mean: FloatArray  # [G]


EvidenceSummary.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class PhantomCountMatrices(PureDataclassPytree):
    A_cg: FloatArray  # [C, G]
    B_cg: FloatArray  # [C, G]
    E_cg: FloatArray  # [C, G]
    R_cg: FloatArray  # [C, G]
    A_g: FloatArray  # [G]
    B_g: FloatArray  # [G]
    E_g: FloatArray  # [G]
    R_g: FloatArray  # [G]
    kish_participating_cluster_counts: FloatArray  # [G]
    phantom_gate_active: BoolArray  # [G]


PhantomCountMatrices.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class GammaWeightedPhantomDraws(PureDataclassPytree):
    race_gamma_gt: FloatArray  # [M, G]
    race_gamma_eq: FloatArray  # [M, G]
    race_gamma_lt: FloatArray  # [M, G]
    cluster_weights: FloatArray  # [M, C]


GammaWeightedPhantomDraws.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class GammaWeightedPhantomProbabilities(PureDataclassPytree):
    p_gt: FloatArray  # [G]
    p_eq: FloatArray  # [G]
    p_lt: FloatArray  # [G]
    phantom_add_gt: FloatArray  # [G]
    phantom_add_eq: FloatArray  # [G]
    phantom_add_lt: FloatArray  # [G]
    kish_participating_cluster_counts: FloatArray  # [G]
    phantom_gate_active: BoolArray  # [G]


GammaWeightedPhantomProbabilities.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class GammaWeightedPhantomProbabilitySamples(PureDataclassPytree):
    p_gt_samples: FloatArray  # [M, G]
    p_eq_samples: FloatArray  # [M, G]
    p_lt_samples: FloatArray  # [M, G]
    phantom_add_gt_samples: FloatArray  # [M, G]
    phantom_add_eq_samples: FloatArray  # [M, G]
    phantom_add_lt_samples: FloatArray  # [M, G]
    kish_participating_cluster_counts: FloatArray  # [G]
    phantom_gate_active: BoolArray  # [G]
    race_gamma_gt: FloatArray  # [M, G]
    race_gamma_eq: FloatArray  # [M, G]
    race_gamma_lt: FloatArray  # [M, G]
    cluster_weights: FloatArray  # [M, C]


GammaWeightedPhantomProbabilitySamples.register_pytree()


def epsilon_for_block_size(block_size: FloatArray) -> FloatArray:
    """Paper default equality-atom prior policy."""
    block_size = jnp.asarray(block_size)
    return jnp.where(
        block_size == 1,
        jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        jnp.asarray(0.5, dtype=mp_policy.measure_dtype),
    )


def classic_dirichlet_concentrations(block_state: BlockState) -> DirichletConcentrations:
    """Classic Dirichlet concentrations for `(p_>, p_=, p_<)`."""
    validate_lineage_capacity(block_state)
    dtype = mp_policy.measure_dtype
    k_g = block_state.incoming_K.astype(dtype)
    m_g = block_state.block_size.astype(dtype)
    eps = epsilon_for_block_size(m_g).astype(dtype)
    alpha_gt = k_g - m_g + 1.0
    # A singleton is not evidence for an atom: the paper requires the exact
    # two-class Beta(K, 1) race. Only a genuine plateau has an equality class.
    alpha_eq = jnp.where(m_g > 1.0, m_g + eps, 0.0)
    alpha_lt = 1.0 - eps
    zeros = jnp.zeros_like(alpha_gt)
    return DirichletConcentrations(
        alpha_gt=jnp.where(block_state.valid, alpha_gt, zeros),
        alpha_eq=jnp.where(block_state.valid, alpha_eq, zeros),
        alpha_lt=jnp.where(block_state.valid, alpha_lt, zeros),
        epsilon=jnp.where(block_state.valid, eps, zeros),
    )


def validate_phantom_count_matrices(
        *,
        A_cg: FloatArray,
        B_cg: FloatArray,
        E_cg: FloatArray,
        block_valid_mask: BoolArray,
) -> None:
    """Validate per-cluster phantom count matrices for conditioning."""
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
    except (TypeError, ValueError):
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
    # A singleton block has no evidence for an equality atom. In that
    # two-class model every eligible phantom is either above the strict
    # endpoint (B) or in its complement (A - B); only plateau blocks split
    # equality observations out as a third category.
    atom_present = block_state.valid & (block_state.block_size > 1)
    model_E = jnp.where(atom_present[None, :], E, 0.0)
    model_R = A - B - model_E
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
    phantom_add_eq = (weights @ model_E) * gate_float
    phantom_add_lt = (weights @ model_R) * gate_float
    m_gt = jnp.asarray(race_gamma_gt, dtype=A.dtype) + phantom_add_gt
    race_eq = jnp.where(
        atom_present,
        jnp.asarray(race_gamma_eq, dtype=A.dtype),
        0.0,
    )
    m_eq = race_eq + phantom_add_eq
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
    key_gt, key_eq, key_lt, key_v = jax.random.split(key, 4)
    sample_shape = (int(num_samples),) + block_state.log_L_blocks.shape
    race_gamma_gt = jax.random.gamma(key_gt, alpha_gt, shape=sample_shape)

    def sample_equality_gamma(eq_key):
        positive = block_state.valid & (concentrations.alpha_eq > 0.0)
        safe_alpha = jnp.where(positive, concentrations.alpha_eq, 1.0)
        draws = jax.random.gamma(eq_key, safe_alpha, shape=sample_shape)
        return jnp.where(positive[None, :], draws, 0.0)

    race_gamma_eq = jax.lax.cond(
        jnp.any(block_state.valid & (concentrations.alpha_eq > 0.0)),
        sample_equality_gamma,
        lambda _: jnp.zeros(sample_shape, dtype=alpha_gt.dtype),
        key_eq,
    )

    def sample_open_interval_gamma(lt_key):
        positive = block_state.valid & (concentrations.alpha_lt > 0.0)
        safe_alpha = jnp.where(positive, concentrations.alpha_lt, 1.0)
        draws = jax.random.gamma(lt_key, safe_alpha, shape=sample_shape)
        return jnp.where(positive[None, :], draws, 0.0)

    # Singleton races have alpha_< = 1 exactly, so an exponential draw is
    # the same Gamma(1, 1) law without invoking the iterative Gamma kernel.
    all_valid_lt_are_exponential = jnp.all(
        ~block_state.valid | (concentrations.alpha_lt == 1.0)
    )
    race_gamma_lt = jax.lax.cond(
        all_valid_lt_are_exponential,
        lambda lt_key: jnp.where(
            block_state.valid[None, :],
            jax.random.exponential(
                lt_key,
                shape=sample_shape,
                dtype=alpha_gt.dtype,
            ),
            0.0,
        ),
        sample_open_interval_gamma,
        key_lt,
    )
    race_gamma_gt = jnp.where(block_state.valid[None, :], race_gamma_gt, 0.0)
    race_gamma_eq = jnp.where(block_state.valid[None, :], race_gamma_eq, 0.0)
    race_gamma_lt = jnp.where(block_state.valid[None, :], race_gamma_lt, 0.0)
    cluster_shape = (int(num_samples), int(num_clusters))
    cluster_weights = jax.random.exponential(
        key_v,
        shape=cluster_shape,
        dtype=mp_policy.measure_dtype,
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
    # Keep phantom observations in the same category model as the classic
    # block: singleton blocks are two-class, plateau blocks are three-class.
    atom_present = block_state.valid & (block_state.block_size > 1)
    model_E = jnp.where(atom_present[None, :], E, 0.0)
    model_R = A - B - model_E
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
    phantom_add_eq = (draws.cluster_weights @ model_E) * gate_float
    phantom_add_lt = (draws.cluster_weights @ model_R) * gate_float
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
    except (TypeError, ValueError):
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
    """Sample block probabilities from Dirichlet concentrations."""
    valid = concentrations.alpha_gt > 0.0
    singleton = (
        valid
        & (concentrations.alpha_eq == 0.0)
        & (concentrations.alpha_lt == 1.0)
    )

    def sample_singleton_races(singleton_key):
        # For the overwhelmingly common non-plateau case the paper's
        # two-class race is Beta(K, 1). Its inverse CDF is exact and avoids
        # materialising three expensive Gamma fields merely to normalise them.
        uniform = jax.random.uniform(
            singleton_key,
            shape=(num_samples,) + concentrations.alpha_gt.shape,
            minval=jnp.finfo(concentrations.alpha_gt.dtype).tiny,
            maxval=1.0,
            dtype=concentrations.alpha_gt.dtype,
        )
        safe_alpha_gt = jnp.where(valid, concentrations.alpha_gt, 1.0)
        p_gt = jnp.exp(jnp.log(uniform) / safe_alpha_gt[None, :])
        p_gt = jnp.where(valid[None, :], p_gt, 0.0)
        p_eq = jnp.zeros_like(p_gt)
        p_lt = jnp.where(valid[None, :], 1.0 - p_gt, 0.0)
        return p_gt, p_eq, p_lt

    def sample_three_class_races(plateau_key):
        alpha = jnp.stack(
            [
                concentrations.alpha_gt,
                concentrations.alpha_eq,
                concentrations.alpha_lt,
            ],
            axis=-1,
        )
        positive = alpha > 0.0
        safe_alpha = jnp.where(positive, alpha, 1.0)
        gamma = jax.random.gamma(
            plateau_key,
            safe_alpha,
            shape=(num_samples,) + safe_alpha.shape,
        )
        # Zero-concentration components are absent categories, rather than
        # Gamma(1, 1) compatibility placeholders.
        gamma = jnp.where(positive[None, :, :], gamma, 0.0)
        probs = gamma / jnp.sum(gamma, axis=-1, keepdims=True)
        probs = jnp.where(valid[None, :, None], probs, 0.0)
        return probs[..., 0], probs[..., 1], probs[..., 2]

    # Most scientific runs have no plateaus at all. Keep the exact fast path
    # as one runtime branch so XLA does not execute the unused Gamma sampler.
    all_valid_blocks_are_singletons = jnp.all(singleton | ~valid)
    return jax.lax.cond(
        all_valid_blocks_are_singletons,
        sample_singleton_races,
        sample_three_class_races,
        key,
    )


def _logdiffexp(log_a: FloatArray, log_b: FloatArray) -> FloatArray:
    return log_a + jnp.log1p(-jnp.exp(log_b - log_a))


def expected_evidence_summary(
        block_state: BlockState,
        concentrations: DirichletConcentrations,
) -> EvidenceSummary:
    """Analytic block evidence moments under independent Dirichlet blocks."""
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

    # The pairwise second-moment term factorises into a lower-block term
    # times an upper-block term. An exclusive cumulative log-sum therefore
    # evaluates every g<h pair in linear memory, avoiding the former B-by-B
    # matrix in user-facing result construction.
    lower_term = (
        block_state.log_L_blocks
        + log_p_gt_second_prev
        + log_gt_not_gt
        - log_p_gt_prefix_with_zero[1:]
    )
    upper_term = (
        block_state.log_L_blocks
        + log_p_gt_prev
        + log_not_gt_mean
    )
    lower_term = jnp.where(valid, lower_term, -jnp.inf)
    upper_term = jnp.where(valid, upper_term, -jnp.inf)
    inclusive_lower_sum = jax.lax.associative_scan(jnp.logaddexp, lower_term)
    exclusive_lower_sum = jnp.concatenate(
        [
            jnp.full((1,), -jnp.inf, dtype=alpha0.dtype),
            inclusive_lower_sum[:-1],
        ],
        axis=0,
    )
    log_cross_sum = logsumexp(
        jnp.log(jnp.asarray(2.0, dtype=alpha0.dtype))
        + upper_term
        + exclusive_lower_sum
    )
    log_Z2_linear_mean = jnp.logaddexp(log_dZ2_sum, log_cross_sum)
    log_Z_mean = 2.0 * log_Z_linear_mean - 0.5 * log_Z2_linear_mean
    log_Z_var = jnp.maximum(
        log_Z2_linear_mean - 2.0 * log_Z_linear_mean,
        jnp.finfo(mp_policy.measure_dtype).eps,
    )
    log_Z_uncert = jnp.sqrt(log_Z_var)

    log_X_mean = scatter_block_values_to_samples(block_state, log_p_gt_prefix)
    return EvidenceSummary(
        log_Z_mean=log_Z_mean,
        log_Z_uncert=log_Z_uncert,
        log_Z_linear_mean=log_Z_linear_mean,
        log_Z2_linear_mean=log_Z2_linear_mean,
        log_dZ_mean=log_dZ_mean,
        log_dZ2_mean=log_dZ2_mean,
        log_dZ2_sum=log_dZ2_sum,
        log_X_mean=log_X_mean,
    )


def sample_evidence(
        key: PRNGKey,
        block_state: BlockState,
        concentrations: DirichletConcentrations,
        *,
        num_samples: int,
) -> EvidenceSamples:
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
    return EvidenceSamples(
        log_Z_samples=log_Z,
        log_dZ_samples=log_dZ,
        p_gt_samples=p_gt,
        p_eq_samples=p_eq,
    )


def expected_log_posterior_weights(
        block_state: BlockState,
        concentrations: DirichletConcentrations,
) -> FloatArray:
    """Expected per-classic-sample log posterior weights under block rules."""
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
    except (TypeError, ValueError):
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


def scatter_block_values_to_samples(
        block_state: BlockState,
        block_values: FloatArray,
) -> FloatArray:
    """Scatter per-block values to classic sample slots using block membership."""
    if block_state.block_sample_indices is None:
        return block_values

    try:
        block_indices = np.asarray(block_state.block_sample_indices)
        valid_blocks = np.asarray(block_state.valid, dtype=bool)
    except (TypeError, ValueError):
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
