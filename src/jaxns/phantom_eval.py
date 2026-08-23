import dataclasses

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy import special as jsp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import BlockState
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey
from jaxns.v3_shrinkage import (
    DirichletConcentrations,
    GammaWeightedPhantomProbabilitySamples,
    PhantomCountMatrices,
    classic_dirichlet_concentrations,
    compute_kish_participating_cluster_counts,
    compute_phantom_gate_active,
    sample_dirichlet_probabilities,
    validate_lineage_capacity,
    validate_phantom_count_matrices,
)
from jaxns.v3_shrinkage import (
    gamma_weighted_phantom_probabilities_from_draws as _gamma_weighted_phantom_probabilities_from_draws,
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

    @property
    def log_Z_mean(self) -> FloatArray:
        """Monte Carlo mean of the sampled log-evidence."""
        return jnp.mean(self.log_Z_samples)

    @property
    def log_Z_uncert(self) -> FloatArray:
        """Monte Carlo standard deviation of the sampled log-evidence."""
        return jnp.std(self.log_Z_samples)

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


EvidenceSamples.register_pytree()


def _logsumexp(x: FloatArray, axis: int | None = None) -> FloatArray:
    return jsp.logsumexp(x, axis=axis)


def _logdiffexp(log_a: FloatArray, log_b: FloatArray) -> FloatArray:
    return log_a + jnp.log1p(-jnp.exp(log_b - log_a))


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
    """Aggregate cluster interval events without a cluster-by-block matrix."""
    dtype = cluster_multiplicity.dtype
    dA = jnp.bincount(
        start_idx,
        weights=cluster_multiplicity * count_A_start_per_cluster,
        length=num_blocks + 1,
    )
    dB = jnp.bincount(
        start_idx,
        weights=cluster_multiplicity * count_B_start_per_cluster,
        length=num_blocks + 1,
    )
    event_weights = cluster_multiplicity[event_cluster_idx]
    dA = dA - jnp.bincount(
        event_a_hi,
        weights=event_weights * event_A_active.astype(dtype),
        length=num_blocks + 1,
    )
    dB = dB - jnp.bincount(
        event_b_hi,
        weights=event_weights * event_B_active.astype(dtype),
        length=num_blocks + 1,
    )
    E = jnp.bincount(
        event_eq_idx,
        weights=event_weights * event_eq_active.astype(dtype),
        length=num_blocks,
    )
    return jnp.cumsum(dA[:-1]), jnp.cumsum(dB[:-1]), E


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
        dtype,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Materialise per-cluster diagnostics from sparse interval events."""
    num_clusters = effective_valid_phantom.shape[0]
    cluster_idx = jnp.arange(num_clusters, dtype=jnp.int32)
    event_cluster_valid = effective_valid_phantom[event_cluster_idx]

    dA = jnp.zeros((num_clusters, num_blocks + 1), dtype=dtype)
    dA = dA.at[cluster_idx, start_idx].add(
        jnp.where(
            effective_valid_phantom,
            count_A_start_per_cluster,
            0.0,
        )
    )
    dA = dA.at[event_cluster_idx, event_a_hi].add(
        jnp.where(
            event_cluster_valid & event_A_active,
            -jnp.ones_like(event_a_hi, dtype=dtype),
            0.0,
        )
    )

    dB = jnp.zeros((num_clusters, num_blocks + 1), dtype=dtype)
    dB = dB.at[cluster_idx, start_idx].add(
        jnp.where(
            effective_valid_phantom,
            count_B_start_per_cluster,
            0.0,
        )
    )
    dB = dB.at[event_cluster_idx, event_b_hi].add(
        jnp.where(
            event_cluster_valid & event_B_active,
            -jnp.ones_like(event_b_hi, dtype=dtype),
            0.0,
        )
    )

    E = jnp.zeros((num_clusters, num_blocks), dtype=dtype)
    E = E.at[event_cluster_idx, event_eq_idx].add(
        event_eq_active.astype(dtype)
    )
    return jnp.cumsum(dA[:, :-1], axis=1), jnp.cumsum(
        dB[:, :-1], axis=1
    ), E


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
        block_state: BlockState | None = None,
        batch_size: int | None = None,
        C_min: float = 20,
        phantom_group_idx: IntArray | None = None,
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
        block_state: Optional canonical v3 block state. When supplied, its
            block likelihoods, membership sizes, and incoming lineage counts are
            used instead of reconstructing blocks from per-sample live counts.
        batch_size: Reserved for API compatibility; currently unused.
        C_min: Kish participating-cluster gate threshold. Defaults to 20.
        phantom_group_idx: Optional independence-group identity for each
            stored cluster. Clusters with the same identity share one gamma
            weight; omitted identities treat every cluster as independent.

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
    if phantom_group_idx is not None:
        if jnp.shape(phantom_group_idx) != jnp.shape(log_L_classic):
            raise ValueError(
                "phantom_group_idx shape must match log_L_classic."
            )
        try:
            groups = np.asarray(phantom_group_idx)
            cluster_valid = np.asarray(valid_phantom, dtype=bool)
            n = int(np.asarray(num_samples))
        except (TypeError, ValueError):
            pass
        else:
            if np.any(groups[:n][cluster_valid[:n]] < 0):
                raise ValueError(
                    "Valid phantom clusters require non-negative group identities."
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
        block_state=block_state,
        batch_size=batch_size,
        C_min=C_min,
        phantom_group_idx=phantom_group_idx,
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
    except (TypeError, ValueError):
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
    except (TypeError, ValueError):
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
    except (TypeError, ValueError):
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
    except (TypeError, ValueError):
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
    return log_dZ_mean, log_dZ_var, weights


def _sample_gamma_weighted_probabilities_from_events(
        *,
        key: PRNGKey,
        block_state: BlockState,
        log_L_constraints: FloatArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        sample_mask: BoolArray,
        num_Z_samples: int,
        C_min: float,
        phantom_group_idx: IntArray,
):
    """Sample phantom-conditioned races without a clusters-by-block matrix.

    Each phantom contributes interval boundaries in block index. Weighted
    cluster counts are therefore accumulated with difference arrays and
    cumulative sums. This is algebraically identical to multiplying the dense
    A/B/E matrices by shared cluster gamma weights, but it keeps the MC path's
    memory linear in clusters, events, blocks, and requested MC draws.
    """
    log_L_blocks = block_state.log_L_blocks
    block_valid_mask = block_state.valid
    dtype = log_L_blocks.dtype
    num_blocks = log_L_blocks.shape[0]
    num_clusters = log_L_constraints.shape[0]
    num_phantom = log_L_phantom.shape[1]
    effective_valid = valid_phantom & sample_mask
    phantom_group_idx = jnp.clip(
        phantom_group_idx.astype(jnp.int32),
        0,
        num_clusters - 1,
    )
    num_valid_blocks = jnp.sum(block_valid_mask, dtype=jnp.int32)

    left_constraint = jnp.searchsorted(
        log_L_blocks,
        log_L_constraints,
        side="left",
    )
    start_idx = jnp.where(
        jnp.isneginf(log_L_constraints),
        0,
        left_constraint + 1,
    )
    start_idx = jnp.minimum(start_idx, num_valid_blocks)
    start_idx = jnp.where(effective_valid, start_idx, 0)

    event_cluster_idx = jnp.repeat(
        jnp.arange(num_clusters, dtype=jnp.int32),
        repeats=num_phantom,
    )
    event_start = start_idx[event_cluster_idx]
    event_log_L = log_L_phantom.reshape((-1,))
    left_likelihood = jnp.searchsorted(
        log_L_blocks,
        event_log_L,
        side="left",
    )
    event_a_hi = jnp.minimum(left_likelihood + 1, num_valid_blocks)
    event_b_hi = jnp.minimum(left_likelihood, num_valid_blocks)
    event_A_active = event_a_hi > event_start
    event_B_active = event_b_hi > event_start
    count_A_start = jnp.bincount(
        event_cluster_idx,
        weights=event_A_active.astype(dtype),
        length=num_clusters,
    )
    count_B_start = jnp.bincount(
        event_cluster_idx,
        weights=event_B_active.astype(dtype),
        length=num_clusters,
    )
    eq_ok = (
        (left_likelihood < num_valid_blocks)
        & (log_L_blocks[left_likelihood] == event_log_L)
    )
    event_eq_idx = jnp.where(eq_ok, left_likelihood, 0)
    event_eq_active = (
        eq_ok
        & (event_eq_idx >= event_start)
        & effective_valid[event_cluster_idx]
    )

    def aggregate(cluster_multiplicity):
        return _boundary_counts_from_multiplicity(
            cluster_multiplicity=cluster_multiplicity,
            start_idx=start_idx,
            count_A_start_per_cluster=count_A_start,
            count_B_start_per_cluster=count_B_start,
            event_cluster_idx=event_cluster_idx,
            event_a_hi=event_a_hi,
            event_b_hi=event_b_hi,
            event_A_active=event_A_active,
            event_B_active=event_B_active,
            event_eq_idx=event_eq_idx,
            event_eq_active=event_eq_active,
            num_blocks=num_blocks,
        )

    cluster_presence = effective_valid.astype(dtype)
    A_g, B_g, E_g = aggregate(cluster_presence)
    R_g = A_g - B_g - E_g

    # Chains assigned the same independence identity are one correlated
    # cluster for weighting. Sweep sparse start/end events in (block, group)
    # order to obtain the exact grouped Kish denominator without an N-by-N
    # count matrix.
    start_group = phantom_group_idx
    start_block = start_idx
    start_delta = jnp.where(
        effective_valid,
        count_A_start,
        jnp.zeros_like(count_A_start),
    )
    end_group = phantom_group_idx[event_cluster_idx]
    end_block = event_a_hi
    end_delta = jnp.where(
        effective_valid[event_cluster_idx] & event_A_active,
        -jnp.ones_like(event_a_hi, dtype=dtype),
        jnp.zeros_like(event_a_hi, dtype=dtype),
    )
    event_group = jnp.concatenate([start_group, end_group])
    event_block = jnp.concatenate([start_block, end_block])
    event_delta = jnp.concatenate([start_delta, end_delta])
    composite_key = (
        event_block.astype(jnp.int64) * jnp.asarray(num_clusters, jnp.int64)
        + event_group.astype(jnp.int64)
    )
    event_order = jnp.argsort(composite_key, stable=True)
    sorted_group = event_group[event_order]
    sorted_block = event_block[event_order]
    sorted_delta = event_delta[event_order]

    def grouped_kish_step(carry, event):
        group_counts, sum_squares = carry
        group, delta = event
        old_count = group_counts[group]
        new_count = old_count + delta
        next_sum_squares = (
            sum_squares + jnp.square(new_count) - jnp.square(old_count)
        )
        return (
            group_counts.at[group].set(new_count),
            next_sum_squares,
        ), next_sum_squares - sum_squares

    (_, _), sum_square_delta = jax.lax.scan(
        grouped_kish_step,
        (
            jnp.zeros((num_clusters,), dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        ),
        (sorted_group, sorted_delta),
    )
    sum_A2 = jnp.cumsum(
        jnp.bincount(
            sorted_block,
            weights=sum_square_delta,
            length=num_blocks + 1,
        )
    )[:-1]
    kish = jnp.where(sum_A2 > 0, jnp.square(A_g) / sum_A2, 0.0)
    gate = (
        (sum_A2 > 0)
        & (A_g > 0)
        & (kish >= jnp.asarray(C_min, dtype=dtype))
        & block_valid_mask
    )

    key_gt, key_eq, key_lt, key_cluster = jax.random.split(key, 4)
    concentrations = classic_dirichlet_concentrations(block_state)
    safe_gt = jnp.where(block_valid_mask, concentrations.alpha_gt, 1.0)
    draw_shape = (num_Z_samples, num_blocks)
    race_gt = jax.random.gamma(key_gt, safe_gt, shape=draw_shape)

    def sample_equality_gamma(eq_key):
        positive = block_valid_mask & (concentrations.alpha_eq > 0.0)
        safe_alpha = jnp.where(positive, concentrations.alpha_eq, 1.0)
        draws = jax.random.gamma(eq_key, safe_alpha, shape=draw_shape)
        return jnp.where(positive[None, :], draws, 0.0)

    race_eq = jax.lax.cond(
        jnp.any(block_valid_mask & (concentrations.alpha_eq > 0.0)),
        sample_equality_gamma,
        lambda _: jnp.zeros(draw_shape, dtype=dtype),
        key_eq,
    )

    def sample_open_interval_gamma(lt_key):
        positive = block_valid_mask & (concentrations.alpha_lt > 0.0)
        safe_alpha = jnp.where(positive, concentrations.alpha_lt, 1.0)
        draws = jax.random.gamma(lt_key, safe_alpha, shape=draw_shape)
        return jnp.where(positive[None, :], draws, 0.0)

    all_valid_lt_are_exponential = jnp.all(
        ~block_valid_mask | (concentrations.alpha_lt == 1.0)
    )
    race_lt = jax.lax.cond(
        all_valid_lt_are_exponential,
        lambda lt_key: jnp.where(
            block_valid_mask[None, :],
            jax.random.exponential(
                lt_key,
                shape=draw_shape,
                dtype=dtype,
            ),
            0.0,
        ),
        sample_open_interval_gamma,
        key_lt,
    )
    # Gamma(1, 1) is exactly Exponential(1); the direct primitive is much
    # cheaper for the large [MC draw, cluster] field.
    group_weights = jax.random.exponential(
        key_cluster,
        shape=(num_Z_samples, num_clusters),
        dtype=dtype,
    )
    weighted_A, weighted_B, weighted_E = jax.vmap(aggregate)(
        group_weights[:, phantom_group_idx] * cluster_presence[None, :]
    )
    gate_value = gate.astype(dtype)[None, :]
    phantom_add_gt = weighted_B * gate_value
    phantom_add_eq = weighted_E * gate_value
    phantom_add_lt = (weighted_A - weighted_B - weighted_E) * gate_value
    mass_gt = race_gt + phantom_add_gt
    mass_eq = race_eq + phantom_add_eq
    mass_lt = race_lt + phantom_add_lt
    total = mass_gt + mass_eq + mass_lt
    p_gt = jnp.where(block_valid_mask[None, :], mass_gt / total, 0.0)
    p_eq = jnp.where(block_valid_mask[None, :], mass_eq / total, 0.0)
    p_lt = jnp.where(block_valid_mask[None, :], mass_lt / total, 0.0)
    probabilities = GammaWeightedPhantomProbabilitySamples(
        p_gt_samples=p_gt,
        p_eq_samples=p_eq,
        p_lt_samples=p_lt,
        phantom_add_gt_samples=phantom_add_gt,
        phantom_add_eq_samples=phantom_add_eq,
        phantom_add_lt_samples=phantom_add_lt,
        kish_participating_cluster_counts=jnp.where(
            block_valid_mask,
            kish,
            0.0,
        ),
        phantom_gate_active=gate,
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=group_weights,
    )
    return probabilities, A_g, B_g, E_g, R_g, kish, gate


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
        block_state: BlockState | None = None,
        batch_size: int | None = None,
        C_min: float = 20,
        phantom_group_idx: IntArray | None = None,
) -> EvidenceSamples:
    del batch_size
    N = log_L_classic.shape[0]
    sample_valid_mask = jnp.arange(N, dtype=jnp.int32) < num_samples
    positive_live_mask = K_classic > 0
    effective_sample_mask = jnp.logical_and(sample_valid_mask, positive_live_mask)
    if phantom_group_idx is None:
        phantom_group_idx = jnp.arange(N, dtype=jnp.int32)
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
    if log_L_phantom.shape[1] == 0:
        # With phantoms disabled there is no cluster-conditioning problem to
        # build. Bypass the clusters-by-block count matrices entirely and draw
        # the independent classic race posterior directly.
        p_gt_samples, p_eq_samples, p_lt_samples = sample_dirichlet_probabilities(
            key,
            classic_concentrations,
            num_samples=num_Z_samples,
        )
        p_gt_for_path = jnp.where(
            block_valid_mask[None, :],
            p_gt_samples,
            jnp.ones_like(p_gt_samples),
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
        log_dZ = jnp.where(
            block_valid_mask[None, :],
            log_dX + log_L_blocks[None, :],
            -jnp.inf,
        )
        log_Z_samples = _logsumexp(log_dZ, axis=-1)
        log_dZ_mean, log_dZ_var, weights = _summarise_gamma_log_dz_samples(
            log_dZ=log_dZ,
            log_Z=log_Z_samples,
            block_valid_mask=block_valid_mask,
        )
        entropy_terms = jnp.where(
            block_valid_mask[None, :],
            log_L_blocks[None, :] - log_Z_samples[:, None],
            0.0,
        )
        H_samples = jnp.sum(weights * entropy_terms, axis=-1)
        zeros = jnp.zeros((num_blocks,), dtype=log_L_blocks.dtype)
        zero_additions = jnp.zeros(
            (num_Z_samples, num_blocks),
            dtype=log_L_blocks.dtype,
        )
        return EvidenceSamples(
            log_Z_samples=log_Z_samples,
            H_samples=H_samples,
            log_dZ_mean=jnp.where(block_valid_mask, log_dZ_mean, -jnp.inf),
            log_dZ_var=jnp.where(block_valid_mask, log_dZ_var, -jnp.inf),
            log_L_blocks=log_L_blocks,
            block_first_idx=block_first_idx,
            block_size=block_size,
            incoming_K=incoming_K_int,
            kish_participating_cluster_counts=zeros,
            phantom_gate_active=jnp.zeros(
                (num_blocks,),
                dtype=mp_policy.bool_dtype,
            ),
            phantom_A=zeros,
            phantom_B=zeros,
            phantom_E=zeros,
            phantom_R=zeros,
            classic_alpha_gt=classic_concentrations.alpha_gt,
            classic_alpha_eq=classic_concentrations.alpha_eq,
            classic_alpha_lt=classic_concentrations.alpha_lt,
            epsilon=classic_concentrations.epsilon,
            p_gt_samples=p_gt_samples,
            p_eq_samples=p_eq_samples,
            p_lt_samples=p_lt_samples,
            p_gt_mean=jnp.mean(p_gt_samples, axis=0),
            p_eq_mean=jnp.mean(p_eq_samples, axis=0),
            p_lt_mean=jnp.mean(p_lt_samples, axis=0),
            phantom_add_gt_samples=zero_additions,
            phantom_add_eq_samples=zero_additions,
            phantom_add_lt_samples=zero_additions,
        )
    (
        probability_samples,
        phantom_A,
        phantom_B,
        phantom_E,
        phantom_R,
        kish_participating_cluster_counts,
        phantom_gate_active,
    ) = _sample_gamma_weighted_probabilities_from_events(
        key=key,
        block_state=block_state_for_v3,
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=effective_sample_mask,
        num_Z_samples=num_Z_samples,
        C_min=C_min,
        phantom_group_idx=phantom_group_idx,
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
        kish_participating_cluster_counts=kish_participating_cluster_counts,
        phantom_gate_active=phantom_gate_active,
        phantom_A=phantom_A,
        phantom_B=phantom_B,
        phantom_E=phantom_E,
        phantom_R=phantom_R,
        classic_alpha_gt=classic_concentrations.alpha_gt,
        classic_alpha_eq=classic_concentrations.alpha_eq,
        classic_alpha_lt=classic_concentrations.alpha_lt,
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
    )
