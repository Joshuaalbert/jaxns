"""NumPy reference for the paper's gamma-weighted evidence model.

This module deliberately favours linear loops and explicit arrays over JAX
optimisations. It is the correctness oracle for phantom counts, Kish gating,
gamma cluster weighting, and final Monte Carlo evidence sampling.
"""

import dataclasses

import numpy as np

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


def _logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    maximum = np.max(x, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    total = np.sum(np.exp(x - safe_maximum), axis=axis, keepdims=True)
    output = safe_maximum + np.log(total)
    if axis is None:
        return np.asarray(output).reshape(())
    return np.squeeze(output, axis=axis)


def _logdiffexp(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Return log(exp(log_a) - exp(log_b)) for log_a >= log_b."""
    log_a = np.asarray(log_a)
    log_b = np.asarray(log_b)
    return log_a + np.log1p(-np.exp(log_b - log_a))


def _epsilon_for_block_size(block_size: np.ndarray) -> np.ndarray:
    """Use no equality class for singletons and a neutral plateau prior."""
    block_size = np.asarray(block_size)
    return np.where(block_size == 1, 0.0, 0.5).astype(float)


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
    if valid_phantom.shape != (num_clusters,):
        raise ValueError("valid_phantom shape must match the cluster axis.")
    if log_L_phantom.ndim != 2 or log_L_phantom.shape[0] != num_clusters:
        raise ValueError("log_L_phantom shape must match the cluster axis.")
    n = int(np.asarray(num_samples).item())
    if n < 0 or n > num_clusters:
        raise ValueError("num_samples is outside the available cluster range.")
    if np.any(valid_phantom[n:]):
        raise ValueError(
            "valid_phantom contains a stale association beyond num_samples."
        )
    active = K_classic[:n] > 0
    strict_violations = active & (
        log_L_classic[:n] <= log_L_constraints[:n]
    )
    if np.any(strict_violations):
        bad = np.where(strict_violations)[0][0]
        raise ValueError(
            "Strict contour violation for active sample "
            f"{bad}: log_L_classic={log_L_classic[bad]} must be greater than "
            f"log_L_constraint={log_L_constraints[bad]}."
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PhantomCountMatrices:
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


@dataclasses.dataclass(frozen=True, slots=True)
class GammaWeightedPhantomProbabilities:
    p_gt: FloatArray  # [G]
    p_eq: FloatArray  # [G]
    p_lt: FloatArray  # [G]
    phantom_add_gt: FloatArray  # [G]
    phantom_add_eq: FloatArray  # [G]
    phantom_add_lt: FloatArray  # [G]
    kish_participating_cluster_counts: FloatArray  # [G]
    phantom_gate_active: BoolArray  # [G]


@dataclasses.dataclass(frozen=True, slots=True)
class GammaWeightedPhantomSamples:
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


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceSamples:
    log_Z_samples: FloatArray  # [M]
    H_samples: FloatArray  # [M]
    log_dZ_mean: FloatArray  # [G]
    log_dZ_var: FloatArray  # [G]
    log_L_blocks: FloatArray  # [G]
    block_first_idx: IntArray  # [G]
    block_size: IntArray  # [G]
    incoming_K: IntArray  # [G]
    kish_participating_cluster_counts: FloatArray  # [G]
    phantom_gate_active: BoolArray  # [G]
    phantom_A: FloatArray  # [G]
    phantom_B: FloatArray  # [G]
    phantom_E: FloatArray  # [G]
    phantom_R: FloatArray  # [G]
    classic_alpha_gt: FloatArray  # [G]
    classic_alpha_eq: FloatArray  # [G]
    classic_alpha_lt: FloatArray  # [G]
    epsilon: FloatArray  # [G]
    p_gt_samples: FloatArray  # [M, G]
    p_eq_samples: FloatArray  # [M, G]
    p_lt_samples: FloatArray  # [M, G]
    p_gt_mean: FloatArray  # [G]
    p_eq_mean: FloatArray  # [G]
    p_lt_mean: FloatArray  # [G]
    phantom_add_gt_samples: FloatArray  # [M, G]
    phantom_add_eq_samples: FloatArray  # [M, G]
    phantom_add_lt_samples: FloatArray  # [M, G]

    @property
    def A_g(self) -> FloatArray:
        return self.phantom_A

    @property
    def B_g(self) -> FloatArray:
        return self.phantom_B

    @property
    def E_g(self) -> FloatArray:
        return self.phantom_E

    @property
    def R_g(self) -> FloatArray:
        return self.phantom_R


def _validate_count_inputs(
        *,
        log_L_blocks: FloatArray,
        block_valid_mask: BoolArray,
        log_L_constraints: FloatArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        sample_mask: BoolArray,
) -> None:
    if log_L_blocks.ndim != 1:
        raise ValueError("log_L_blocks must be one-dimensional.")
    if block_valid_mask.shape != log_L_blocks.shape:
        raise ValueError("block_valid_mask shape must align with log_L_blocks.")
    if log_L_constraints.ndim != 1:
        raise ValueError("log_L_constraints must be one-dimensional.")
    num_clusters = log_L_constraints.shape[0]
    if valid_phantom.ndim != 1:
        raise ValueError(
            "valid_phantom must be a one-dimensional per-cluster mask, not a "
            "per-phantom mask."
        )
    if valid_phantom.shape != (num_clusters,):
        raise ValueError("valid_phantom shape must match the cluster axis.")
    if sample_mask.shape != (num_clusters,):
        raise ValueError("sample_mask shape must match the cluster axis.")
    if log_L_phantom.ndim != 2:
        raise ValueError("log_L_phantom must be a two-dimensional array.")
    if log_L_phantom.shape[0] != num_clusters:
        raise ValueError("log_L_phantom shape must match the cluster axis.")
    if np.any(valid_phantom & ~sample_mask):
        raise ValueError(
            "valid_phantom contains a stale sample_mask/num_samples "
            "association."
        )


def validate_phantom_count_matrices(
        *,
        A_cg: FloatArray,
        B_cg: FloatArray,
        E_cg: FloatArray,
        block_valid_mask: BoolArray,
) -> None:
    if A_cg.ndim != 2:
        raise ValueError("A_cg must have shape [num_clusters, num_blocks].")
    if B_cg.shape != A_cg.shape or E_cg.shape != A_cg.shape:
        raise ValueError("B_cg and E_cg shape must align with A_cg.")
    if block_valid_mask.shape != (A_cg.shape[1],):
        raise ValueError("block_valid_mask shape must align with block axis.")
    valid = np.asarray(block_valid_mask, dtype=bool)
    if not np.all(np.isfinite(A_cg[:, valid])):
        raise ValueError("A_cg counts must be finite on valid blocks.")
    if not np.all(np.isfinite(B_cg[:, valid])):
        raise ValueError("B_cg counts must be finite on valid blocks.")
    if not np.all(np.isfinite(E_cg[:, valid])):
        raise ValueError("E_cg counts must be finite on valid blocks.")
    if (
            np.any(A_cg[:, valid] < 0.0)
            or np.any(B_cg[:, valid] < 0.0)
            or np.any(E_cg[:, valid] < 0.0)
    ):
        raise ValueError("Phantom count matrices must be non-negative.")
    if np.any(B_cg[:, valid] + E_cg[:, valid] > A_cg[:, valid]):
        raise ValueError(
            "Invalid phantom count relation: B_cg + E_cg must be <= A_cg."
        )


def compute_kish_participating_cluster_counts(A_cg: FloatArray) -> FloatArray:
    A = np.asarray(A_cg, dtype=float)
    denominator = np.sum(np.square(A), axis=0)
    numerator = np.square(np.sum(A, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def compute_phantom_gate_active(
        A_cg: FloatArray,
        *,
        C_min: float = 20,
) -> BoolArray:
    A = np.asarray(A_cg, dtype=float)
    denominator = np.sum(np.square(A), axis=0)
    participation = np.sum(A, axis=0) > 0.0
    kish = compute_kish_participating_cluster_counts(A)
    return (denominator > 0.0) & participation & (kish >= float(C_min))


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
    log_L_blocks = np.asarray(log_L_blocks, dtype=float)
    block_valid_mask = np.asarray(block_valid_mask, dtype=bool)
    log_L_constraints = np.asarray(log_L_constraints, dtype=float)
    valid_phantom = np.asarray(valid_phantom, dtype=bool)
    log_L_phantom = np.asarray(log_L_phantom, dtype=float)
    sample_mask = np.asarray(sample_mask, dtype=bool)
    _validate_count_inputs(
        log_L_blocks=log_L_blocks,
        block_valid_mask=block_valid_mask,
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=sample_mask,
    )
    num_clusters = log_L_constraints.shape[0]
    num_blocks = log_L_blocks.shape[0]
    A_cg = np.zeros((num_clusters, num_blocks), dtype=float)
    B_cg = np.zeros((num_clusters, num_blocks), dtype=float)
    E_cg = np.zeros((num_clusters, num_blocks), dtype=float)
    effective_valid = valid_phantom & sample_mask
    valid_count = int(np.sum(block_valid_mask))
    for cluster_idx in range(num_clusters):
        if not effective_valid[cluster_idx]:
            continue
        constraint = log_L_constraints[cluster_idx]
        for block_idx in range(valid_count):
            parent = -np.inf if block_idx == 0 else log_L_blocks[block_idx - 1]
            endpoint = log_L_blocks[block_idx]
            if constraint > parent:
                continue
            values = log_L_phantom[cluster_idx]
            A_cg[cluster_idx, block_idx] = np.sum(values > parent)
            B_cg[cluster_idx, block_idx] = np.sum(values > endpoint)
            E_cg[cluster_idx, block_idx] = np.sum(values == endpoint)
    R_cg = A_cg - B_cg - E_cg
    validate_phantom_count_matrices(
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        block_valid_mask=block_valid_mask,
    )
    kish = compute_kish_participating_cluster_counts(A_cg)
    gate = compute_phantom_gate_active(A_cg, C_min=C_min) & block_valid_mask
    return PhantomCountMatrices(
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        R_cg=R_cg,
        A_g=np.where(block_valid_mask, np.sum(A_cg, axis=0), 0.0),
        B_g=np.where(block_valid_mask, np.sum(B_cg, axis=0), 0.0),
        E_g=np.where(block_valid_mask, np.sum(E_cg, axis=0), 0.0),
        R_g=np.where(block_valid_mask, np.sum(R_cg, axis=0), 0.0),
        kish_participating_cluster_counts=np.where(block_valid_mask, kish, 0.0),
        phantom_gate_active=gate,
    )


def gamma_weighted_phantom_probabilities_from_draws(
        *,
        block_state,
        A_cg: FloatArray,
        B_cg: FloatArray,
        E_cg: FloatArray,
        race_gamma_gt: FloatArray,
        race_gamma_eq: FloatArray,
        race_gamma_lt: FloatArray,
        cluster_weights: FloatArray,
        C_min: float = 20,
) -> GammaWeightedPhantomProbabilities:
    A = np.asarray(A_cg, dtype=float)
    B = np.asarray(B_cg, dtype=float)
    E = np.asarray(E_cg, dtype=float)
    valid = np.asarray(block_state.valid, dtype=bool)
    validate_phantom_count_matrices(
        A_cg=A,
        B_cg=B,
        E_cg=E,
        block_valid_mask=valid,
    )
    weights = np.asarray(cluster_weights, dtype=float)
    if weights.shape != (A.shape[0],):
        raise ValueError("cluster_weights shape must align with cluster axis.")
    # Mirror the production model exactly: singleton blocks have no equality
    # category, so their complement count is A-B rather than A-B-E.
    atom_present = valid & (np.asarray(block_state.block_size) > 1)
    model_E = np.where(atom_present[None, :], E, 0.0)
    model_R = A - B - model_E
    kish = compute_kish_participating_cluster_counts(A)
    gate = compute_phantom_gate_active(A, C_min=C_min) & valid
    gate_float = gate.astype(float)
    add_gt = (weights @ B) * gate_float
    add_eq = (weights @ model_E) * gate_float
    add_lt = (weights @ model_R) * gate_float
    gt = np.asarray(race_gamma_gt, dtype=float) + add_gt
    eq = np.where(
        atom_present,
        np.asarray(race_gamma_eq, dtype=float),
        0.0,
    ) + add_eq
    lt = np.asarray(race_gamma_lt, dtype=float) + add_lt
    total = gt + eq + lt
    total = np.where(total > 0.0, total, 1.0)
    return GammaWeightedPhantomProbabilities(
        p_gt=np.where(valid, gt / total, 0.0),
        p_eq=np.where(valid, eq / total, 0.0),
        p_lt=np.where(valid, lt / total, 0.0),
        phantom_add_gt=add_gt,
        phantom_add_eq=add_eq,
        phantom_add_lt=add_lt,
        kish_participating_cluster_counts=np.where(valid, kish, 0.0),
        phantom_gate_active=gate,
    )


def _classic_alpha_from_block_state(block_state) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(block_state.valid, dtype=bool)
    block_size = np.asarray(block_state.block_size, dtype=float)
    incoming = np.asarray(block_state.incoming_K, dtype=float)
    epsilon = _epsilon_for_block_size(block_size)
    alpha = np.stack(
        [
            incoming - block_size + 1.0,
            np.where(block_size > 1.0, block_size + epsilon, 0.0),
            1.0 - epsilon,
        ],
        axis=-1,
    )
    return np.where(valid[:, None], alpha, 0.0), np.where(valid, epsilon, 0.0)


def sample_gamma_weighted_phantom_probabilities(
        *,
        rng: np.random.Generator,
        block_state,
        A_cg: FloatArray,
        B_cg: FloatArray,
        E_cg: FloatArray,
        num_samples: int,
        C_min: float = 20,
) -> GammaWeightedPhantomSamples:
    A = np.asarray(A_cg, dtype=float)
    B = np.asarray(B_cg, dtype=float)
    E = np.asarray(E_cg, dtype=float)
    valid = np.asarray(block_state.valid, dtype=bool)
    validate_phantom_count_matrices(
        A_cg=A,
        B_cg=B,
        E_cg=E,
        block_valid_mask=valid,
    )
    alpha, _ = _classic_alpha_from_block_state(block_state)
    atom_present = valid & (alpha[:, 1] > 0.0)
    model_E = np.where(atom_present[None, :], E, 0.0)
    model_R = A - B - model_E
    safe_alpha = np.where(alpha > 0.0, alpha, 1.0)
    race_gt = rng.gamma(shape=safe_alpha[:, 0], size=(int(num_samples), alpha.shape[0]))
    race_eq = rng.gamma(shape=safe_alpha[:, 1], size=(int(num_samples), alpha.shape[0]))
    race_lt = rng.gamma(shape=safe_alpha[:, 2], size=(int(num_samples), alpha.shape[0]))
    race_eq = np.where(alpha[None, :, 1] > 0.0, race_eq, 0.0)
    race_lt = np.where(alpha[None, :, 2] > 0.0, race_lt, 0.0)
    race_gt = np.where(valid[None, :], race_gt, 0.0)
    race_eq = np.where(valid[None, :], race_eq, 0.0)
    race_lt = np.where(valid[None, :], race_lt, 0.0)
    cluster_weights = rng.gamma(
        shape=1.0,
        scale=1.0,
        size=(int(num_samples), A.shape[0]),
    )
    kish = compute_kish_participating_cluster_counts(A)
    gate = compute_phantom_gate_active(A, C_min=C_min) & valid
    gate_float = gate.astype(float)[None, :]
    add_gt = (cluster_weights @ B) * gate_float
    add_eq = (cluster_weights @ model_E) * gate_float
    add_lt = (cluster_weights @ model_R) * gate_float
    gt = race_gt + add_gt
    eq = race_eq + add_eq
    lt = race_lt + add_lt
    total = gt + eq + lt
    total = np.where(total > 0.0, total, 1.0)
    p_gt = np.where(valid[None, :], gt / total, 0.0)
    p_eq = np.where(valid[None, :], eq / total, 0.0)
    p_lt = np.where(valid[None, :], lt / total, 0.0)
    return GammaWeightedPhantomSamples(
        p_gt_samples=p_gt,
        p_eq_samples=p_eq,
        p_lt_samples=p_lt,
        phantom_add_gt_samples=add_gt,
        phantom_add_eq_samples=add_eq,
        phantom_add_lt_samples=add_lt,
        kish_participating_cluster_counts=np.where(valid, kish, 0.0),
        phantom_gate_active=gate,
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=cluster_weights,
    )


def _block_state_from_arrays(
        log_L_classic: FloatArray,
        K_classic: IntArray,
        num_samples: IntArray,
):
    from jaxns.algorithm.race_tree import BlockState

    N = log_L_classic.shape[0]
    n = int(np.asarray(num_samples).item())
    sample_mask = np.arange(N) < n
    active = sample_mask & (K_classic > 0)
    valid_classic = np.where(active, log_L_classic, np.inf)
    order = np.argsort(valid_classic, kind="stable")
    sorted_log = valid_classic[order]
    sorted_K = K_classic[order]
    finite_sorted = sorted_log[np.isfinite(sorted_log)]
    unique_valid, starts, counts = np.unique(
        finite_sorted,
        return_index=True,
        return_counts=True,
    )
    block_count = unique_valid.shape[0]
    log_L_blocks = unique_valid.astype(float)
    block_valid = np.isfinite(log_L_blocks)
    block_first_idx = order[starts].astype(np.int32)
    block_first_idx = np.where(block_valid, block_first_idx, np.int32(-1))
    block_size = counts.astype(np.int32)
    first_idx_safe = np.clip(starts, 0, max(N - 1, 0))
    incoming = sorted_K[first_idx_safe].astype(np.int32)
    incoming = np.where(block_valid, incoming, 0).astype(np.int32)
    return BlockState(
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx,
        block_size=block_size,
        incoming_K=incoming,
        block_out_degree=np.zeros((block_count,), dtype=np.int32),
        valid=block_valid,
    ), active


def _log_dz_from_probabilities(
        p_gt_samples: FloatArray,
        log_L_blocks: FloatArray,
        block_valid_mask: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    p_gt = np.where(block_valid_mask[None, :], p_gt_samples, 1.0)
    p_gt = np.clip(p_gt, 1e-300, 1.0)
    log_X = np.cumsum(np.log(p_gt), axis=-1)
    log_X_prev = np.concatenate(
        [np.zeros((p_gt.shape[0], 1)), log_X[:, :-1]],
        axis=-1,
    )
    log_dX = _logdiffexp(log_X_prev, log_X)
    log_dZ = np.where(
        block_valid_mask[None, :],
        log_dX + log_L_blocks[None, :],
        -np.inf,
    )
    log_Z = _logsumexp(log_dZ, axis=-1)
    weights = np.exp(log_dZ - log_Z[:, None])
    H = np.sum(
        weights
        * np.where(
            block_valid_mask[None, :],
            log_L_blocks[None, :] - log_Z[:, None],
            0.0,
        ),
        axis=-1,
    )
    return log_Z, log_dZ, H


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
        C_min: float = 20,
) -> EvidenceSamples:
    if num_Z_samples < 1:
        raise ValueError("num_Z_samples must be >= 1.")
    rng = np.random.default_rng(int(seed))
    log_L_constraints = np.asarray(log_L_constraints, dtype=float)
    log_L_classic = np.asarray(log_L_classic, dtype=float)
    K_classic = np.asarray(K_classic, dtype=np.int32)
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
    block_state, sample_mask = _block_state_from_arrays(
        log_L_classic,
        K_classic,
        num_samples,
    )
    counts = compute_phantom_count_matrices(
        log_L_blocks=np.asarray(block_state.log_L_blocks),
        block_valid_mask=np.asarray(block_state.valid, dtype=bool),
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=sample_mask,
        C_min=C_min,
    )
    probability_samples = sample_gamma_weighted_phantom_probabilities(
        rng=rng,
        block_state=block_state,
        A_cg=counts.A_cg,
        B_cg=counts.B_cg,
        E_cg=counts.E_cg,
        num_samples=num_Z_samples,
        C_min=C_min,
    )
    log_Z, log_dZ, H = _log_dz_from_probabilities(
        probability_samples.p_gt_samples,
        np.asarray(block_state.log_L_blocks, dtype=float),
        np.asarray(block_state.valid, dtype=bool),
    )
    dZ = np.exp(log_dZ)
    dZ_mean = np.mean(dZ, axis=0)
    dZ_var = np.var(dZ, axis=0)
    tiny = np.finfo(float).tiny
    alpha, epsilon = _classic_alpha_from_block_state(block_state)
    p_gt_mean = np.mean(probability_samples.p_gt_samples, axis=0)
    p_eq_mean = np.mean(probability_samples.p_eq_samples, axis=0)
    p_lt_mean = np.mean(probability_samples.p_lt_samples, axis=0)
    valid = np.asarray(block_state.valid, dtype=bool)
    return EvidenceSamples(
        log_Z_samples=log_Z,
        H_samples=H,
        log_dZ_mean=np.where(valid, np.log(np.maximum(dZ_mean, tiny)), -np.inf),
        log_dZ_var=np.where(valid, np.log(np.maximum(dZ_var, tiny)), -np.inf),
        log_L_blocks=np.asarray(block_state.log_L_blocks, dtype=float),
        block_first_idx=np.asarray(block_state.block_first_idx, dtype=np.int32),
        block_size=np.asarray(block_state.block_size, dtype=np.int32),
        incoming_K=np.asarray(block_state.incoming_K, dtype=np.int32),
        kish_participating_cluster_counts=counts.kish_participating_cluster_counts,
        phantom_gate_active=counts.phantom_gate_active,
        phantom_A=counts.A_g,
        phantom_B=counts.B_g,
        phantom_E=counts.E_g,
        phantom_R=counts.R_g,
        classic_alpha_gt=alpha[:, 0],
        classic_alpha_eq=alpha[:, 1],
        classic_alpha_lt=alpha[:, 2],
        epsilon=epsilon,
        p_gt_samples=probability_samples.p_gt_samples,
        p_eq_samples=probability_samples.p_eq_samples,
        p_lt_samples=probability_samples.p_lt_samples,
        p_gt_mean=np.where(valid, p_gt_mean, np.nan),
        p_eq_mean=np.where(valid, p_eq_mean, np.nan),
        p_lt_mean=np.where(valid, p_lt_mean, np.nan),
        phantom_add_gt_samples=probability_samples.phantom_add_gt_samples,
        phantom_add_eq_samples=probability_samples.phantom_add_eq_samples,
        phantom_add_lt_samples=probability_samples.phantom_add_lt_samples,
    )
