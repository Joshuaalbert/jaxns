"""Efficient paired Monte Carlo reduction for phantom-prefix experiments."""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from scipy import sparse
from scipy.special import logsumexp

from jaxns.algorithm.race_tree import BlockState
from jaxns.shrinkage.classic import classic_dirichlet_concentrations
from jaxns.shrinkage.phantom import (
    _logdiffexp,
    _logsumexp,
    _prepare_phantom_events,
)
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class _SparsePrefixPlan:
    """Host sparse intervals and exact prefix-specific Kish gates."""

    A_boundary: sparse.csr_matrix
    B_boundary: sparse.csr_matrix
    gates: np.ndarray  # [Q, G]
    log_L_blocks: np.ndarray  # [G]
    alpha_gt: np.ndarray  # [G_capacity]
    alpha_not_gt: np.ndarray  # [G_capacity], equality plus open interval
    valid_blocks: np.ndarray  # [G_capacity]
    num_blocks: int
    num_clusters: int


def _weighted_prefix_counts(
        cluster_weights: FloatArray,
        event_cluster: IntArray,
        event_group: IntArray,
        event_start: IntArray,
        event_a_stop: IntArray,
        event_b_stop: IntArray,
        event_a_active: BoolArray,
        event_b_active: BoolArray,
        cluster_presence: FloatArray,
        *,
        num_groups: int,
        num_blocks: int,
) -> tuple[FloatArray, FloatArray]:
    """Aggregate every event once, then cumulatively expose each prefix."""
    dtype = cluster_weights.dtype
    weights = (
        cluster_weights[event_cluster]
        * cluster_presence[event_cluster]
    )

    dA = jnp.zeros((num_groups, num_blocks + 1), dtype=dtype)  # [Q, G+1]
    dA = dA.at[event_group, event_start].add(
        weights * event_a_active.astype(dtype)
    )
    dA = dA.at[event_group, event_a_stop].add(
        -weights * event_a_active.astype(dtype)
    )

    dB = jnp.zeros((num_groups, num_blocks + 1), dtype=dtype)  # [Q, G+1]
    dB = dB.at[event_group, event_start].add(
        weights * event_b_active.astype(dtype)
    )
    dB = dB.at[event_group, event_b_stop].add(
        -weights * event_b_active.astype(dtype)
    )

    # The first cumulative sum turns boundary events into the contribution
    # from one newly retained D-sized group. The second forms D, 2D, ..., 9D.
    delta_A = jnp.cumsum(dA[:, :-1], axis=-1)  # [Q, G]
    delta_B = jnp.cumsum(dB[:, :-1], axis=-1)  # [Q, G]
    return jnp.cumsum(delta_A, axis=0), jnp.cumsum(delta_B, axis=0)


def _log_Z_from_p_gt(
        p_gt: FloatArray,
        block_state: BlockState,
) -> FloatArray:
    """Reduce strict shrinkage probabilities to one log-evidence per row."""
    valid = block_state.valid
    path_probability = jnp.where(valid[None, :], p_gt, 1.0)
    path_probability = jnp.clip(path_probability, 1e-300, 1.0)
    log_X = jnp.cumsum(jnp.log(path_probability), axis=-1)
    log_X_previous = jnp.concatenate(
        [
            jnp.zeros((p_gt.shape[0], 1), dtype=p_gt.dtype),
            log_X[:, :-1],
        ],
        axis=-1,
    )
    log_dX = _logdiffexp(log_X_previous, log_X)
    log_dZ = jnp.where(
        valid[None, :],
        log_dX + block_state.log_L_blocks[None, :],
        -jnp.inf,
    )
    return _logsumexp(log_dZ, axis=-1)


@partial(
    jax.jit,
    inline=True,
    static_argnames=(
        "dimension",
        "num_draws",
        "batch_size",
        "num_groups",
    ),
)
def sample_phantom_prefix_sweep(
        *,
        key: PRNGKey,
        log_L_constraints: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
        block_state: BlockState,
        dimension: int,
        num_draws: int,
        batch_size: int,
        num_groups: int = 9,
        C_min: float = 20.0,
) -> tuple[FloatArray, BoolArray]:
    """Draw classic and D-through-9D evidence on one shared random field.

    This kernel is intentionally specialised to the continuous paper cases.
    The caller validates that every valid block is a singleton. Sharing race
    gammas and cluster weights changes only coupling between sweep entries;
    every column retains its exact marginal phantom-conditioning law.

    Returns:
        Log-evidence draws with shape ``[num_draws, 1 + num_groups]`` and
        Kish-gate masks with shape ``[num_groups, num_blocks]``.
    """
    num_blocks = block_state.log_L_blocks.shape[0]
    num_clusters = log_L_constraints.shape[0]
    sample_mask = (
        jnp.arange(num_clusters, dtype=jnp.int32) < num_samples
    ) & (K_classic > 0)

    # The full event plan supplies the boundary locations. Prefix-specific
    # plans are evaluated once to retain the exact Kish gate for each kD.
    events = _prepare_phantom_events(
        block_state=block_state,
        log_L_constraints=log_L_constraints,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        sample_mask=sample_mask,
        C_min=C_min,
    )
    gates = []
    for group in range(1, num_groups + 1):
        prefix_events = _prepare_phantom_events(
            block_state=block_state,
            log_L_constraints=log_L_constraints,
            valid_phantom=valid_phantom,
            log_L_phantom=log_L_phantom[:, :group * dimension],
            sample_mask=sample_mask,
            C_min=C_min,
        )
        gates.append(prefix_events.gate)
    gates = jnp.stack(gates)  # [Q, G]

    num_phantom = log_L_phantom.shape[1]
    event_position = jnp.tile(  # [E]
        jnp.arange(num_phantom, dtype=jnp.int32),
        num_clusters,
    )
    event_group = event_position // dimension  # [E]
    event_start = events.start_idx[events.event_cluster_idx]  # [E]
    concentrations = classic_dirichlet_concentrations(block_state)
    valid = block_state.valid
    safe_gt = jnp.where(valid, concentrations.alpha_gt, 1.0)
    draw_index = jnp.arange(batch_size, dtype=jnp.int32)
    num_batches = (num_draws + batch_size - 1) // batch_size

    def draw_batch(_, batch_index):
        batch_key = (
            key
            if num_batches == 1
            else jax.random.fold_in(key, batch_index)
        )
        key_gt, _, key_lt, key_cluster = jax.random.split(batch_key, 4)
        draw_shape = (batch_size, num_blocks)
        race_gt = jax.random.gamma(key_gt, safe_gt, shape=draw_shape)
        race_lt = jnp.where(
            valid[None, :],
            jax.random.exponential(
                key_lt,
                shape=draw_shape,
                dtype=block_state.log_L_blocks.dtype,
            ),
            0.0,
        )
        cluster_weights = jax.random.exponential(
            key_cluster,
            shape=(batch_size, num_clusters),
            dtype=block_state.log_L_blocks.dtype,
        )
        weighted_A, weighted_B = jax.vmap(
            lambda weights: _weighted_prefix_counts(
                weights,
                events.event_cluster_idx,
                event_group,
                event_start,
                events.event_a_hi,
                events.event_b_hi,
                events.event_A_active,
                events.event_B_active,
                events.cluster_presence,
                num_groups=num_groups,
                num_blocks=num_blocks,
            )
        )(cluster_weights)

        gate_value = gates.astype(race_gt.dtype)[None, :, :]  # [1, Q, G]
        mass_gt = race_gt[:, None, :] + weighted_B * gate_value
        mass_lt = (
            race_lt[:, None, :]
            + (weighted_A - weighted_B) * gate_value
        )
        p_gt_phantom = mass_gt / (mass_gt + mass_lt)  # [M, Q, G]
        p_gt_classic = race_gt / (race_gt + race_lt)  # [M, G]
        p_gt = jnp.concatenate(  # [M, 1+Q, G]
            [p_gt_classic[:, None, :], p_gt_phantom],
            axis=1,
        )
        log_Z = _log_Z_from_p_gt(
            p_gt.reshape((batch_size * (num_groups + 1), num_blocks)),
            block_state,
        ).reshape((batch_size, num_groups + 1))
        active = batch_index * batch_size + draw_index < num_draws
        return None, jnp.where(active[:, None], log_Z, 0.0)

    _, log_Z_batches = jax.lax.scan(
        draw_batch,
        None,
        jnp.arange(num_batches, dtype=jnp.int32),
    )
    log_Z = log_Z_batches.reshape((-1, num_groups + 1))[:num_draws]
    return log_Z, gates


def _boundary_matrix(
        *,
        start: np.ndarray,
        stop: np.ndarray,
        cluster: np.ndarray,
        group: np.ndarray,
        active: np.ndarray,
        num_groups: int,
        num_blocks: int,
        num_clusters: int,
) -> sparse.csr_matrix:
    """Encode grouped half-open intervals as a sparse difference matrix."""
    selected = np.flatnonzero(active)
    selected_cluster = cluster[selected]
    selected_group = group[selected]
    stride = num_blocks + 1
    positive_row = selected_group * stride + start[selected_cluster]
    negative_row = selected_group * stride + stop[selected]
    rows = np.concatenate((positive_row, negative_row))
    columns = np.concatenate((selected_cluster, selected_cluster))
    values = np.concatenate((
        np.ones(selected.size, dtype=np.float64),
        -np.ones(selected.size, dtype=np.float64),
    ))
    # Converting to CSR combines repeated boundaries from the same chain.
    # Each MC draw can then use one sparse matrix multiplication rather than
    # millions of CPU XLA scatter updates.
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(num_groups * stride, num_clusters),
    ).tocsr()


def _prefix_sum_squares(
        *,
        start: np.ndarray,
        likelihood_stop: np.ndarray,
        valid_clusters: np.ndarray,
        dimension: int,
        num_groups: int,
        num_blocks: int,
) -> np.ndarray:
    """Compute ``sum_c A_cg^2`` without a dense cluster-by-block matrix.

    Squaring a count is equivalent to one copy of every phantom interval plus
    two copies of every pairwise interval intersection. Assigning a pair to
    its later phantom makes cumulative D-sized groups exactly equal to the
    desired D, 2D, ..., 9D prefix counts.
    """
    num_phantom = likelihood_stop.shape[1]
    boundary = np.zeros(
        (num_groups, num_blocks + 1),
        dtype=np.float64,
    )
    for phantom_index in range(num_phantom):
        group = phantom_index // dimension
        stop = likelihood_stop[:, phantom_index]
        diagonal = valid_clusters & (stop > start)
        start_weight = diagonal.astype(np.float64)
        if phantom_index:
            pair_stop = np.minimum(
                stop[:, None],
                likelihood_stop[:, :phantom_index],
            )
            pair_active = valid_clusters[:, None] & (
                pair_stop > start[:, None]
            )
            start_weight = start_weight + 2.0 * np.sum(
                pair_active,
                axis=1,
            )
            boundary[group] -= np.bincount(
                pair_stop[pair_active],
                weights=np.full(
                    np.count_nonzero(pair_active),
                    2.0,
                    dtype=np.float64,
                ),
                minlength=num_blocks + 1,
            )
        boundary[group] += np.bincount(
            start,
            weights=start_weight,
            minlength=num_blocks + 1,
        )
        boundary[group] -= np.bincount(
            stop[diagonal],
            minlength=num_blocks + 1,
        )
    group_squares = np.cumsum(boundary[:, :-1], axis=1)  # [Q, G]
    return np.cumsum(group_squares, axis=0)  # [Q, G]


def _prepare_sparse_prefix_plan(
        *,
        log_L_constraints: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
        block_state: BlockState,
        dimension: int,
        num_groups: int,
        C_min: float,
) -> _SparsePrefixPlan:
    """Build exact interval events for singleton and plateau blocks."""
    valid_blocks = np.asarray(jax.device_get(block_state.valid), dtype=bool)
    num_blocks = int(np.sum(valid_blocks))
    if not np.all(valid_blocks[:num_blocks]) or np.any(
            valid_blocks[num_blocks:]
    ):
        raise ValueError("Valid likelihood blocks must form one prefix.")

    all_levels = np.asarray(jax.device_get(block_state.log_L_blocks))
    levels = all_levels[:num_blocks]
    constraints = np.asarray(jax.device_get(log_L_constraints))
    lineage_counts = np.asarray(jax.device_get(K_classic))
    cluster_valid = np.asarray(jax.device_get(valid_phantom), dtype=bool)
    num_clusters = constraints.shape[0]
    valid_clusters = (
        cluster_valid
        & (np.arange(num_clusters) < int(np.asarray(num_samples)))
        & (lineage_counts > 0)
    )

    start = np.searchsorted(levels, constraints, side="left") + 1
    start[np.isneginf(constraints)] = 0
    start = np.minimum(start, num_blocks).astype(np.int64)
    phantom = np.asarray(jax.device_get(log_L_phantom))
    num_phantom = phantom.shape[1]
    if num_phantom != num_groups * dimension:
        raise ValueError("Retained phantoms do not fill the requested groups.")
    left = np.searchsorted(
        levels,
        phantom.reshape((-1,)),
        side="left",
    ).reshape((num_clusters, num_phantom))
    A_stop = np.minimum(left + 1, num_blocks).astype(np.int64)
    B_stop = np.minimum(left, num_blocks).astype(np.int64)
    cluster = np.repeat(np.arange(num_clusters), num_phantom)
    position = np.tile(np.arange(num_phantom), num_clusters)
    group = position // dimension
    event_valid = valid_clusters[cluster]
    flat_A_stop = A_stop.reshape((-1,))
    flat_B_stop = B_stop.reshape((-1,))
    A_active = event_valid & (flat_A_stop > start[cluster])
    B_active = event_valid & (flat_B_stop > start[cluster])
    A_boundary = _boundary_matrix(
        start=start,
        stop=flat_A_stop,
        cluster=cluster,
        group=group,
        active=A_active,
        num_groups=num_groups,
        num_blocks=num_blocks,
        num_clusters=num_clusters,
    )
    B_boundary = _boundary_matrix(
        start=start,
        stop=flat_B_stop,
        cluster=cluster,
        group=group,
        active=B_active,
        num_groups=num_groups,
        num_blocks=num_blocks,
        num_clusters=num_clusters,
    )

    group_A = np.cumsum(
        np.asarray(A_boundary @ valid_clusters.astype(np.float64)).reshape(
            (num_groups, num_blocks + 1)
        )[:, :-1],
        axis=1,
    )
    prefix_A = np.cumsum(group_A, axis=0)
    prefix_A2 = _prefix_sum_squares(
        start=start,
        likelihood_stop=A_stop,
        valid_clusters=valid_clusters,
        dimension=dimension,
        num_groups=num_groups,
        num_blocks=num_blocks,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        kish = np.where(
            prefix_A2 > 0.0,
            np.square(prefix_A) / prefix_A2,
            0.0,
        )
    gates = (
        (prefix_A2 > 0.0)
        & (prefix_A > 0.0)
        & (kish >= C_min)
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    return _SparsePrefixPlan(
        A_boundary=A_boundary,
        B_boundary=B_boundary,
        gates=gates,
        log_L_blocks=levels,
        alpha_gt=np.asarray(jax.device_get(concentrations.alpha_gt)),
        alpha_not_gt=np.asarray(jax.device_get(
            concentrations.alpha_eq + concentrations.alpha_lt,
        )),
        valid_blocks=valid_blocks,
        num_blocks=num_blocks,
        num_clusters=num_clusters,
    )


def _log_Z_numpy(
        p_gt: np.ndarray,
        log_L_blocks: np.ndarray,
) -> np.ndarray:
    """Reduce strict shrinkage probabilities using stable NumPy arithmetic."""
    probability = np.clip(p_gt, 1e-300, 1.0)
    log_probability = np.log(probability)
    log_X_previous = np.cumsum(log_probability, axis=-1)
    # Subtracting the current increment converts the inclusive cumulative
    # volume to X_{g-1}. Using p_g directly for log(1-p_g) both removes two
    # full-size temporaries and avoids recovering a small log(p_g) by
    # subtracting adjacent, deeply compressed cumulative volumes.
    log_X_previous -= log_probability
    with np.errstate(divide="ignore", invalid="ignore"):
        log_dX = log_X_previous + np.log1p(-probability)
    return logsumexp(log_dX + log_L_blocks, axis=-1)


def _numpy_generator(key: PRNGKey) -> np.random.Generator:
    """Create one reproducible host generator from a JAX subkey.

    The large paper states make CPU XLA spend minutes compiling and copying
    each dynamically shaped gamma field. NumPy draws the same gamma and
    exponential laws directly in compiled host code; separate JAX subkeys
    keep the race and cluster fields independent and reproducible.
    """
    words = np.asarray(jax.device_get(key), dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(words))


def sample_phantom_prefix_sweep_reference(
        *,
        key: PRNGKey,
        log_L_constraints: FloatArray,
        K_classic: IntArray,
        valid_phantom: BoolArray,
        log_L_phantom: FloatArray,
        num_samples: IntArray,
        block_state: BlockState,
        dimension: int,
        num_draws: int,
        batch_size: int,
        num_workers: int = 1,
        num_groups: int = 9,
        C_min: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the exact prefix sweep with host sparse interval reductions.

    This is a benchmark-only reference path for large CPU experiments. It
    preserves the JAX kernel's random-key schedule and all marginal laws, but
    SciPy CSR multiplication is materially faster than CPU XLA scatter for the
    millions of phantom interval boundaries in the paper cases. Singleton
    and plateau blocks use the same marginal strict-shrinkage law as the
    core; the equality/open-interval split cancels from evidence. Worker
    threads share the immutable sparse plan. ``Executor.map`` preserves batch
    order, so changing the worker count does not change any random field or
    output row.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")
    if num_workers > 1 and batch_size != 1:
        raise ValueError(
            "Parallel prefix sampling requires batch_size=1; combining "
            "thread and array batching materially increases peak memory."
        )
    plan = _prepare_sparse_prefix_plan(
        log_L_constraints=log_L_constraints,
        K_classic=K_classic,
        valid_phantom=valid_phantom,
        log_L_phantom=log_L_phantom,
        num_samples=num_samples,
        block_state=block_state,
        dimension=dimension,
        num_groups=num_groups,
        C_min=C_min,
    )
    num_batches = (num_draws + batch_size - 1) // batch_size
    # Invalid storage padding is not part of the race. Drawing only the valid
    # block prefix avoids paying for up to one extra capacity quantum per MC
    # field and keeps memory proportional to scientific samples.
    safe_gt = plan.alpha_gt[:plan.num_blocks]
    not_gt_shape = plan.alpha_not_gt[:plan.num_blocks]  # [G]
    plateau = not_gt_shape != 1.0  # [G], singleton complement is Exp(1)
    # The gate is immutable across draws and large at paper scale. Materialise
    # its arithmetic dtype once so all workers share it instead of copying one
    # Q-by-G array for every scalar MC task.
    gate = plan.gates.astype(safe_gt.dtype)[None, :, :]  # [1, Q, G]

    def draw_batch(batch_index: int) -> np.ndarray:
        """Draw one independently keyed batch against the shared plan."""
        batch_key = (
            key
            if num_batches == 1
            else jax.random.fold_in(key, batch_index)
        )
        key_gt, key_eq, key_lt, key_cluster = jax.random.split(batch_key, 4)
        shape = (batch_size, plan.num_blocks)
        race_gt = _numpy_generator(key_gt).gamma(
            shape=safe_gt,
            scale=1.0,
            size=shape,
        )
        race_lt = _numpy_generator(key_lt).exponential(
            scale=1.0,
            size=shape,
        )
        if np.any(plateau):
            # Evidence uses only p_>. The independent equality and open-
            # interval race gammas therefore collapse to Gamma(alpha_= +
            # alpha_<, 1), and their phantom additions sum to A-B. This is
            # the core's three-class plateau law marginalized over the split.
            # Use the otherwise-unused equality key for these complements;
            # every singleton random field remains bitwise unchanged.
            race_lt[:, plateau] = _numpy_generator(key_eq).gamma(
                shape=not_gt_shape[plateau], scale=1.0,
                size=(batch_size, int(np.sum(plateau))),
            )
        cluster_weights = _numpy_generator(key_cluster).exponential(
            scale=1.0,
            size=(batch_size, plan.num_clusters),
        )
        weighted_A = np.asarray(
            plan.A_boundary @ cluster_weights.T
        ).T.reshape((batch_size, num_groups, plan.num_blocks + 1))
        weighted_B = np.asarray(
            plan.B_boundary @ cluster_weights.T
        ).T.reshape((batch_size, num_groups, plan.num_blocks + 1))
        weighted_A = np.cumsum(
            np.cumsum(weighted_A[..., :-1], axis=-1),
            axis=1,
        )
        weighted_B = np.cumsum(
            np.cumsum(weighted_B[..., :-1], axis=-1),
            axis=1,
        )
        mass_gt = race_gt[:, None, :] + weighted_B * gate
        mass_lt = (
            race_lt[:, None, :]
            + (weighted_A - weighted_B) * gate
        )
        p_gt_phantom = mass_gt / (mass_gt + mass_lt)
        p_gt_classic = race_gt / (race_gt + race_lt)
        p_gt = np.concatenate(
            (p_gt_classic[:, None, :], p_gt_phantom),
            axis=1,
        )
        return _log_Z_numpy(p_gt, plan.log_L_blocks)

    batch_indices = range(num_batches)
    if num_workers == 1:
        output = list(map(draw_batch, batch_indices))
    else:
        # NumPy random draws and SciPy sparse products release the GIL. Scalar
        # batches let independent draws occupy CPU cores while keeping one
        # read-only ~O(CD) sparse plan and O(G) temporary storage per worker.
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            output = list(executor.map(draw_batch, batch_indices))
    log_Z = np.concatenate(output, axis=0)[:num_draws]
    padded_gates = np.zeros(
        (num_groups, plan.valid_blocks.size),
        dtype=bool,
    )
    padded_gates[:, :plan.num_blocks] = plan.gates
    return log_Z, padded_gates
