import dataclasses

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import Samples
from jaxns.types import BoolArray, FloatArray, IntArray


@dataclasses.dataclass(slots=True, frozen=True)
class BlockState(PureDataclassPytree):
    """Canonical v3 block view of race-tree samples."""

    log_L_blocks: FloatArray
    block_first_idx: IntArray
    block_size: IntArray
    incoming_K: IntArray
    block_out_degree: IntArray
    valid: BoolArray
    block_start: IntArray | None = None
    block_stop: IntArray | None = None
    block_sample_indices: IntArray | None = None


BlockState.register_pytree()


def build_block_state(
        samples: Samples,
        root_out_degree: IntArray,
        num_samples: IntArray | None = None,
        *,
        sample_indices: IntArray | None = None,
        validate: bool = False,
) -> BlockState:
    """Derive likelihood blocks and incoming active lineage counts.

    The returned arrays are padded to the sample array length. Padded entries have
    `valid=False`, `log_L_blocks=inf`, `block_first_idx=-1`, and zero counts.
    """
    if num_samples is None:
        num_samples = samples.log_likelihoods.shape[0]
    if validate:
        return _build_block_state_python(
            samples=samples,
            root_out_degree=root_out_degree,
            num_samples=num_samples,
            sample_indices=sample_indices,
        )
    num_samples = jnp.asarray(num_samples, dtype=mp_policy.count_dtype)
    max_samples = samples.log_likelihoods.shape[0]
    sample_idx = jnp.arange(max_samples, dtype=mp_policy.index_dtype)
    sample_valid = sample_idx < num_samples
    if sample_indices is None:
        labels = sample_idx
    else:
        labels = jnp.asarray(sample_indices, dtype=mp_policy.index_dtype)
        if labels.shape[0] < max_samples:
            labels = jnp.concatenate(
                [
                    labels,
                    sample_idx[labels.shape[0]:],
                ],
                axis=0,
            )

    log_L_valid = jnp.where(sample_valid, samples.log_likelihoods, jnp.inf)
    sorted_order = jnp.argsort(log_L_valid, stable=True)
    sorted_log_L = log_L_valid[sorted_order]
    sorted_out_degree = samples.out_degree[sorted_order]
    sorted_valid = sample_valid[sorted_order]
    sorted_labels = labels[sorted_order]

    log_L_blocks = jnp.unique(sorted_log_L, size=max_samples, fill_value=jnp.inf)
    block_valid = log_L_blocks < jnp.inf
    block_first_idx_raw = jnp.searchsorted(sorted_log_L, log_L_blocks, side="left")
    block_first_idx_safe = jnp.clip(block_first_idx_raw, 0, max(max_samples - 1, 0))
    block_first_idx = jnp.where(
        block_valid,
        sorted_labels[block_first_idx_safe],
        jnp.asarray(-1, dtype=mp_policy.index_dtype),
    )

    block_ids = jnp.searchsorted(log_L_blocks, sorted_log_L, side="left")
    block_ids = jnp.clip(block_ids, 0, max(max_samples - 1, 0))
    block_size = jnp.bincount(
        block_ids,
        weights=sorted_valid.astype(mp_policy.count_dtype),
        length=max_samples,
    ).astype(mp_policy.count_dtype)
    block_out_degree = jnp.bincount(
        block_ids,
        weights=jnp.where(sorted_valid, sorted_out_degree, 0).astype(mp_policy.count_dtype),
        length=max_samples,
    ).astype(mp_policy.count_dtype)

    def scan_fn(k_in, block_values):
        m_g, d_g, is_valid = block_values
        k_out = jnp.where(is_valid, k_in - m_g + d_g, k_in)
        return k_out, jnp.where(is_valid, k_in, jnp.asarray(0, dtype=k_in.dtype))

    _, incoming_K = jax.lax.scan(
        scan_fn,
        jnp.asarray(root_out_degree, dtype=mp_policy.count_dtype),
        (block_size, block_out_degree, block_valid),
    )

    block_start = jnp.cumsum(block_size) - block_size
    block_stop = jnp.cumsum(block_size)
    no_sample = jnp.asarray(-1, dtype=mp_policy.index_dtype)
    block_sample_indices = jnp.where(sorted_valid, sorted_labels, no_sample)

    state = BlockState(
        log_L_blocks=log_L_blocks,
        block_first_idx=block_first_idx.astype(mp_policy.index_dtype),
        block_size=block_size,
        incoming_K=incoming_K.astype(mp_policy.count_dtype),
        block_out_degree=block_out_degree,
        valid=block_valid,
        block_start=block_start,
        block_stop=block_stop,
        block_sample_indices=block_sample_indices,
    )
    return state


def _build_block_state_python(
        samples: Samples,
        root_out_degree: IntArray,
        num_samples: IntArray,
        sample_indices: IntArray | None,
) -> BlockState:
    n = int(np.asarray(num_samples))
    log_l = np.asarray(samples.log_likelihoods[:n])
    out_degree = np.asarray(samples.out_degree[:n], dtype=np.int64)
    constraints = np.asarray(samples.log_L_constraints[:n])
    root = int(np.asarray(root_out_degree))
    _validate_race_tree_inputs(
        log_likelihoods=log_l,
        log_L_constraints=constraints,
        out_degree=out_degree,
        root_out_degree=root,
        num_samples=n,
        max_samples=samples.log_likelihoods.shape[0],
    )
    if sample_indices is None:
        labels = np.arange(n, dtype=np.int64)
    else:
        labels = np.asarray(sample_indices[:n], dtype=np.int64)

    order = np.lexsort((labels, log_l))
    sorted_log_l = log_l[order]
    sorted_out_degree = out_degree[order]
    sorted_labels = labels[order]
    unique_log_l, starts, sizes = np.unique(
        sorted_log_l,
        return_index=True,
        return_counts=True,
    )
    stops = starts + sizes
    block_out_degree = np.asarray([
        np.sum(sorted_out_degree[start:stop])
        for start, stop in zip(starts, stops, strict=True)
    ], dtype=np.int64)

    incoming = []
    k = int(np.asarray(root_out_degree))
    for size, degree_sum in zip(sizes, block_out_degree, strict=True):
        incoming.append(k)
        k = k - int(size) + int(degree_sum)
    max_block_size = int(np.max(sizes)) if sizes.size else 0
    block_sample_indices = np.full(
        (unique_log_l.shape[0], max_block_size),
        -1,
        dtype=np.int64,
    )
    for block_idx, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        block_labels = sorted_labels[start:stop]
        block_sample_indices[block_idx, :block_labels.shape[0]] = block_labels

    state = BlockState(
        log_L_blocks=jnp.asarray(unique_log_l, dtype=samples.log_likelihoods.dtype),
        block_first_idx=jnp.asarray(sorted_labels[starts], dtype=mp_policy.index_dtype),
        block_size=jnp.asarray(sizes, dtype=mp_policy.count_dtype),
        incoming_K=jnp.asarray(incoming, dtype=mp_policy.count_dtype),
        block_out_degree=jnp.asarray(block_out_degree, dtype=mp_policy.count_dtype),
        valid=jnp.ones(unique_log_l.shape, dtype=mp_policy.bool_dtype),
        block_start=jnp.asarray(starts, dtype=mp_policy.index_dtype),
        block_stop=jnp.asarray(stops, dtype=mp_policy.index_dtype),
        block_sample_indices=jnp.asarray(block_sample_indices, dtype=mp_policy.index_dtype),
    )
    validate_block_state(state)
    return state


def _validate_race_tree_inputs(
        *,
        log_likelihoods: np.ndarray,
        log_L_constraints: np.ndarray,
        out_degree: np.ndarray,
        root_out_degree: int,
        num_samples: int,
        max_samples: int,
) -> None:
    if num_samples < 0 or num_samples > max_samples:
        raise ValueError(
            f"num_samples={num_samples} is outside the available sample "
            f"range [0, {max_samples}]."
        )
    if num_samples == 0:
        return
    if root_out_degree <= 0:
        raise ValueError("root_out_degree must be positive for a non-empty race tree.")
    if np.any(out_degree < 0):
        bad = np.where(out_degree < 0)[0][0]
        raise ValueError(f"Invalid race tree: out_degree[{bad}] is negative.")
    if np.any(log_likelihoods <= log_L_constraints):
        bad = np.where(log_likelihoods <= log_L_constraints)[0][0]
        raise ValueError(
            "Strict contour violation for sample "
            f"{bad}: log_likelihood={log_likelihoods[bad]} must be greater "
            f"than log_L_constraint={log_L_constraints[bad]}."
        )
    total_children = root_out_degree + int(np.sum(out_degree))
    if total_children != num_samples:
        raise ValueError(
            "Invalid race tree out-degree total: "
            f"root_out_degree + sum(out_degree) = {total_children}, "
            f"expected num_samples={num_samples}."
        )


def validate_block_state(block_state: BlockState) -> None:
    """Validate block-level race-tree invariants with Python exceptions."""
    valid = np.asarray(block_state.valid, dtype=bool)
    incoming = np.asarray(block_state.incoming_K)
    sizes = np.asarray(block_state.block_size)
    out_degree = np.asarray(block_state.block_out_degree)
    log_L = np.asarray(block_state.log_L_blocks)
    if np.any(valid & (incoming < sizes)):
        bad = np.where(valid & (incoming < sizes))[0][0]
        raise ValueError(
            f"Invalid race block {bad}: incoming K_g={incoming[bad]} "
            f"is smaller than plateau size m_g={sizes[bad]}."
        )
    valid_log_L = log_L[valid]
    if valid_log_L.size > 1 and np.any(np.diff(valid_log_L) <= 0):
        raise ValueError("Block likelihoods must be strictly increasing.")
    valid_incoming = incoming[valid]
    valid_sizes = sizes[valid]
    valid_out_degree = out_degree[valid]
    if valid_incoming.size == 0:
        return
    outgoing = valid_incoming - valid_sizes + valid_out_degree
    if np.any(outgoing[:-1] != valid_incoming[1:]):
        bad = np.where(outgoing[:-1] != valid_incoming[1:])[0][0]
        raise ValueError(
            f"Invalid race block transition {bad}: outgoing K={outgoing[bad]} "
            f"does not match next incoming K={valid_incoming[bad + 1]}."
        )
    if outgoing[-1] != 0:
        raise ValueError(
            f"Invalid race tree ending: final active lineage count is {outgoing[-1]}, "
            "expected 0."
        )
