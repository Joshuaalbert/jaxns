from __future__ import annotations

import dataclasses
from typing import Literal
from typing import cast
from typing import get_args

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import build_block_state
from jaxns.samples import Samples
from jaxns.state import State
from jaxns.types import BoolArray, FloatArray, IntArray
from jaxns.v3_shrinkage import classic_dirichlet_concentrations
from jaxns.v3_shrinkage import DirichletConcentrations


AllocationTarget = Literal[
    "uniform",
    "evidence_improving",
    "posterior_improving",
]
SUPPORTED_ALLOCATION_TARGETS = frozenset(get_args(AllocationTarget))


@dataclasses.dataclass(slots=True, frozen=True)
class VolumePath(PureDataclassPytree):
    X_prev: FloatArray
    X: FloatArray
    shell_mass: FloatArray


VolumePath.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class AllocationPlan(PureDataclassPytree):
    target_K: IntArray
    current_K: IntArray
    unit_peak_utility: FloatArray
    log_L_blocks: FloatArray
    valid: BoolArray
    volume_path: VolumePath


AllocationPlan.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class ParentWork(PureDataclassPytree):
    parent_idxs: IntArray
    parent_log_L_constraints: FloatArray
    target_block_idxs: IntArray
    parent_block_idxs: IntArray
    fallback_to_root: BoolArray


ParentWork.register_pytree()


def validate_allocation_target(allocation_target: str) -> AllocationTarget:
    if allocation_target not in SUPPORTED_ALLOCATION_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_ALLOCATION_TARGETS))
        raise ValueError(
            f"Unknown allocation_target={allocation_target!r}. "
            f"Expected one of: {supported}."
        )
    return cast(AllocationTarget, allocation_target)


def _validate_delta_K(delta_K: IntArray) -> None:
    try:
        if int(np.asarray(delta_K)) <= 0:
            raise ValueError("delta_K must be positive.")
    except TypeError:
        return


def _stable_likelihood_from_log_L(
        log_L_blocks: FloatArray,
        valid: BoolArray | None = None,
) -> FloatArray:
    """Exponentiate log-likelihoods after removing a finite valid offset."""
    log_L_blocks = jnp.asarray(log_L_blocks, dtype=mp_policy.measure_dtype)
    finite = jnp.isfinite(log_L_blocks)
    if valid is not None:
        finite = finite & jnp.asarray(valid, dtype=mp_policy.bool_dtype)
    max_log_L = jnp.max(
        jnp.where(finite, log_L_blocks, -jnp.inf),
        initial=jnp.asarray(-jnp.inf, dtype=log_L_blocks.dtype),
    )
    max_log_L = jnp.where(jnp.isfinite(max_log_L), max_log_L, 0.0)
    return jnp.where(finite, jnp.exp(log_L_blocks - max_log_L), 0.0)


def expected_volume_path(
        concentrations: DirichletConcentrations,
        valid: BoolArray | None = None,
) -> VolumePath:
    """Expected nested volume path from v3 Dirichlet block summaries."""
    alpha0 = (
            concentrations.alpha_gt
            + concentrations.alpha_eq
            + concentrations.alpha_lt
    )
    p_gt = jnp.where(alpha0 > 0.0, concentrations.alpha_gt / alpha0, 1.0)
    if valid is not None:
        p_gt = jnp.where(valid, p_gt, 1.0)
    p_gt = jnp.clip(p_gt, 0.0, 1.0)
    X = jnp.cumprod(p_gt)
    X_prev = jnp.concatenate([jnp.ones((1,), dtype=X.dtype), X[:-1]], axis=0)
    shell_mass = jnp.maximum(X_prev - X, 0.0)
    if valid is not None:
        X_prev = jnp.where(valid, X_prev, 0.0)
        X = jnp.where(valid, X, 0.0)
        shell_mass = jnp.where(valid, shell_mass, 0.0)
    return VolumePath(X_prev=X_prev, X=X, shell_mass=shell_mass)


def evidence_improvement_utility(
        log_L_blocks: FloatArray,
        volume_path: VolumePath,
        alpha_gt: FloatArray,
        beta_gt: FloatArray,
        valid: BoolArray | None = None,
) -> FloatArray:
    """Compute the paper's evidence-improving utility `U^Z_g`.

    Args:
        log_L_blocks: Block log-likelihoods `log(L_g)`.
        volume_path: Expected or sampled nested volumes for the same blocks.
        alpha_gt: Beta alpha parameter for `p_{>g}`.
        beta_gt: Beta beta parameter for `p_{>g}`.
        valid: Optional block mask.

        Returns:
        Per-block allocation utility on an arbitrary nonnegative scale.
    """
    likelihood = _stable_likelihood_from_log_L(log_L_blocks, valid=valid)
    shell_dZ = likelihood * volume_path.shell_mass
    if valid is not None:
        shell_dZ = jnp.where(valid, shell_dZ, 0.0)
    Z = jnp.sum(shell_dZ)

    downstream = jnp.cumsum(shell_dZ[::-1])[::-1] - shell_dZ
    B_h = jnp.where(
        Z > 0.0,
        (likelihood * volume_path.X - downstream) / Z,
        0.0,
    )
    beta_var_delta = jnp.where(
        (alpha_gt > 0.0) & (beta_gt > 0.0),
        1.0 / jnp.square(alpha_gt)
        - 1.0 / jnp.square(alpha_gt + beta_gt),
        0.0,
    )
    reduction = jnp.square(B_h) * jnp.maximum(beta_var_delta, 0.0)
    if valid is not None:
        reduction = jnp.where(valid, reduction, 0.0)

    downstream_weighted = (
            jnp.cumsum((volume_path.X * reduction)[::-1])[::-1]
            - volume_path.X * reduction
    )
    utility = jnp.where(
        volume_path.X > 0.0,
        downstream_weighted / volume_path.X,
        0.0,
    )
    utility = jnp.where(jnp.isfinite(utility), utility, 0.0)
    if valid is not None:
        utility = jnp.where(valid, utility, 0.0)
    return jnp.maximum(utility, 0.0)


def posterior_improvement_utility(
        volume_path: VolumePath,
        shell_weights: FloatArray,
        valid: BoolArray | None = None,
        *,
        conservative: bool = False,
) -> FloatArray:
    """Compute the paper's posterior-improving utility `U^P_g`."""
    weights = jnp.where(jnp.isfinite(shell_weights), shell_weights, 0.0)
    weights = jnp.maximum(weights, 0.0)
    if valid is not None:
        weights = jnp.where(valid, weights, 0.0)
    W = jnp.sum(weights)
    Q = jnp.sum(jnp.square(weights))

    if conservative:
        delta = jnp.where(
            (Q > 0.0) & (weights > 0.0),
            jnp.square(W) / (Q - jnp.square(weights) / 3.0)
            - jnp.square(W) / Q,
            0.0,
        )
    else:
        w2 = jnp.square(weights)
        denominator_term = 2.0 * Q - w2
        ratio = jnp.where(denominator_term > 0.0, w2 / denominator_term, 0.0)
        denominator = jnp.sqrt((Q - w2 / 2.0) * (w2 / 2.0))
        exact = jnp.arctan(jnp.sqrt(ratio)) / denominator
        delta = jnp.where(
            (Q > 0.0) & (weights > 0.0) & jnp.isfinite(exact),
            jnp.square(W) * (exact - 1.0 / Q),
            0.0,
        )

    delta = jnp.where(jnp.isfinite(delta), delta, 0.0)
    delta = jnp.maximum(delta, 0.0)
    downstream_weighted = (
            jnp.cumsum((volume_path.shell_mass * delta)[::-1])[::-1]
            - volume_path.shell_mass * delta
    )
    utility = jnp.where(
        volume_path.X > 0.0,
        downstream_weighted / volume_path.X,
        0.0,
    )
    utility = jnp.where(jnp.isfinite(utility), utility, 0.0)
    if valid is not None:
        utility = jnp.where(valid, utility, 0.0)
    return jnp.maximum(utility, 0.0)


def normalise_allocation_utility(
        utility: FloatArray,
        valid: BoolArray | None = None,
) -> FloatArray:
    """Convert arbitrary utility values to a deterministic unit-peak target."""
    utility = jnp.asarray(utility, dtype=mp_policy.measure_dtype)
    clean = jnp.where(jnp.isfinite(utility), utility, 0.0)
    clean = jnp.maximum(clean, 0.0)
    if valid is None:
        valid = jnp.ones(clean.shape, dtype=mp_policy.bool_dtype)
    clean = jnp.where(valid, clean, 0.0)
    peak = jnp.max(clean, initial=jnp.asarray(0.0, dtype=clean.dtype))
    fallback = valid.astype(clean.dtype)
    return jnp.where(peak > 0.0, clean / peak, fallback)


def integer_allocation_targets(
        root_out_degree: IntArray,
        iteration: IntArray,
        delta_K: IntArray,
        unit_peak_utility: FloatArray,
        valid: BoolArray | None = None,
        max_target: IntArray | None = None,
) -> IntArray:
    """Map a unit-peak allocation curve to integer active-lineage targets.

    The rounding policy is ceil-after-scaling so any positive utility gets a
    concrete lineage target. Optional `max_target` clips targets to the
    current allocation capacity used by the caller.
    """
    _validate_delta_K(delta_K)
    if valid is None:
        valid = jnp.ones_like(unit_peak_utility, dtype=mp_policy.bool_dtype)
    scaled = (
            jnp.asarray(root_out_degree, dtype=mp_policy.measure_dtype)
            + jnp.asarray(iteration, dtype=mp_policy.measure_dtype)
            * jnp.asarray(delta_K, dtype=mp_policy.measure_dtype)
            * jnp.asarray(unit_peak_utility, dtype=mp_policy.measure_dtype)
    )
    targets = jnp.ceil(scaled).astype(mp_policy.count_dtype)
    if max_target is not None:
        targets = jnp.minimum(
            targets,
            jnp.asarray(max_target, dtype=targets.dtype),
        )
    return jnp.where(valid, targets, jnp.asarray(0, dtype=targets.dtype))


def build_allocation_plan(
        state: State,
        allocation_target: AllocationTarget,
        iteration: IntArray,
        delta_K: IntArray,
        root_out_degree: IntArray | None = None,
        posterior_conservative: bool = False,
        posterior_utility: Literal["exact", "conservative"] = "exact",
) -> AllocationPlan:
    """Build the block-level v3 allocation target for one outer iteration."""
    allocation_target = validate_allocation_target(allocation_target)
    _validate_delta_K(delta_K)
    if posterior_utility not in ("exact", "conservative"):
        raise ValueError(
            "posterior_utility must be 'exact' or 'conservative'."
        )
    posterior_conservative = (
            posterior_conservative
            or posterior_utility == "conservative"
    )
    if root_out_degree is None:
        root_out_degree = state.root_out_degree
    block_state = build_block_state(
        state.samples,
        root_out_degree=state.root_out_degree,
        num_samples=state.num_samples,
    )
    concentrations = classic_dirichlet_concentrations(block_state)
    volume_path = expected_volume_path(
        concentrations,
        valid=block_state.valid,
    )

    if allocation_target == "uniform":
        unit_peak_utility = jnp.where(
            block_state.valid,
            jnp.ones_like(
                block_state.log_L_blocks,
                dtype=mp_policy.measure_dtype,
            ),
            jnp.zeros_like(
                block_state.log_L_blocks,
                dtype=mp_policy.measure_dtype,
            ),
        )
    elif allocation_target == "evidence_improving":
        utility = evidence_improvement_utility(
            log_L_blocks=block_state.log_L_blocks,
            volume_path=volume_path,
            alpha_gt=concentrations.alpha_gt,
            beta_gt=concentrations.alpha_eq + concentrations.alpha_lt,
            valid=block_state.valid,
        )
        unit_peak_utility = normalise_allocation_utility(
            utility,
            valid=block_state.valid,
        )
    else:
        shell_weights = (
                _stable_likelihood_from_log_L(
                    block_state.log_L_blocks,
                    valid=block_state.valid,
                )
                * volume_path.shell_mass
        )
        utility = posterior_improvement_utility(
            volume_path=volume_path,
            shell_weights=shell_weights,
            valid=block_state.valid,
            conservative=posterior_conservative,
        )
        unit_peak_utility = normalise_allocation_utility(
            utility,
            valid=block_state.valid,
        )

    target_K = integer_allocation_targets(
        root_out_degree=root_out_degree,
        iteration=iteration,
        delta_K=delta_K,
        unit_peak_utility=unit_peak_utility,
        valid=block_state.valid,
    )
    return AllocationPlan(
        target_K=target_K,
        current_K=block_state.incoming_K,
        unit_peak_utility=unit_peak_utility,
        log_L_blocks=block_state.log_L_blocks,
        valid=block_state.valid,
        volume_path=volume_path,
    )


def _empty_parent_work() -> ParentWork:
    return ParentWork(
        parent_idxs=jnp.asarray([], dtype=mp_policy.index_dtype),
        parent_log_L_constraints=jnp.asarray(
            [],
            dtype=mp_policy.measure_dtype,
        ),
        target_block_idxs=jnp.asarray([], dtype=mp_policy.index_dtype),
        parent_block_idxs=jnp.asarray([], dtype=mp_policy.index_dtype),
        fallback_to_root=jnp.asarray([], dtype=mp_policy.bool_dtype),
    )


def _choose_index_from_weights(
        key,
        candidate_indices: np.ndarray,
        weights: np.ndarray,
) -> int:
    if candidate_indices.size == 1:
        return int(candidate_indices[0])
    finite_positive_weights = np.where(
        np.isfinite(weights) & (weights > 0.0),
        weights,
        0.0,
    )
    total = float(np.sum(finite_positive_weights))
    if total <= 0.0:
        return int(candidate_indices[0])
    quantile = float(jax.random.uniform(key, dtype=jnp.float32))
    cdf = np.cumsum(finite_positive_weights / total)
    selected_offset = min(
        int(np.searchsorted(cdf, quantile, side="right")),
        candidate_indices.size - 1,
    )
    return int(candidate_indices[selected_offset])


def select_parent_work(
        key,
        state: State,
        plan: AllocationPlan,
        num_parents: int,
) -> ParentWork:
    """Select concrete in-flight parent indices for under-allocated blocks."""
    if num_parents <= 0:
        return _empty_parent_work()

    try:
        block_state = build_block_state(
            state.samples,
            root_out_degree=state.root_out_degree,
            num_samples=state.num_samples,
            validate=True,
        )
    except ValueError:
        block_state = build_block_state(
            state.samples,
            root_out_degree=state.root_out_degree,
            num_samples=state.num_samples,
        )
    target_K = np.asarray(plan.target_K, dtype=np.int64)
    current_K = np.asarray(plan.current_K, dtype=np.int64).copy()
    valid = np.asarray(plan.valid, dtype=bool)
    X = np.asarray(plan.volume_path.X, dtype=float)
    log_likelihoods = np.asarray(state.samples.log_likelihoods)
    block_sample_indices = np.asarray(block_state.block_sample_indices)

    parent_idxs = []
    parent_log_L_constraints = []
    target_block_idxs = []
    parent_block_idxs = []
    fallback_to_root = []

    for _ in range(num_parents):
        deficits = target_K - current_K
        target_candidates = np.where(valid & (deficits > 0))[0]
        if target_candidates.size == 0:
            break
        has_strict_parent = np.asarray([
            np.any(valid & (X[target_block_idx] < X))
            for target_block_idx in target_candidates
        ], dtype=bool)
        if (
                np.any(has_strict_parent)
                and int(np.asarray(state.num_samples))
                == int(np.asarray(state.root_out_degree))
        ):
            target_candidates = target_candidates[has_strict_parent]

        target_weights = deficits[target_candidates].astype(float)
        target_weights = np.maximum(target_weights, 0.0)
        target_weights = np.where(
            np.isfinite(target_weights),
            target_weights,
            0.0,
        )
        if np.sum(target_weights) <= 0.0:
            target_weights = deficits[target_candidates].astype(float)
            target_weights = np.maximum(target_weights, 0.0)
        if np.sum(target_weights) <= 0.0:
            break
        target_block_idx = _choose_index_from_weights(
            key,
            target_candidates,
            target_weights,
        )
        if target_candidates.size > 1:
            key, _ = jax.random.split(key)

        X_target = X[target_block_idx]
        strict_parent_mask = valid & (X_target < X)
        parent_candidates = np.where(strict_parent_mask)[0]
        if parent_candidates.size == 0:
            parent_idxs.append(-1)
            parent_log_L_constraints.append(-np.inf)
            target_block_idxs.append(target_block_idx)
            parent_block_idxs.append(-1)
            fallback_to_root.append(True)
            current_K[target_block_idx] += 1
            continue

        parent_weights = X_target / X[parent_candidates]
        parent_weights = np.where(
            np.isfinite(parent_weights),
            parent_weights,
            0.0,
        )
        parent_weights = np.maximum(parent_weights, 0.0)
        parent_block_idx = _choose_index_from_weights(
            key,
            parent_candidates,
            parent_weights,
        )
        if parent_candidates.size > 1:
            key, _ = jax.random.split(key)

        if block_sample_indices.ndim == 1:
            start = int(np.asarray(block_state.block_start[parent_block_idx]))
            stop = int(np.asarray(block_state.block_stop[parent_block_idx]))
            sample_candidates = block_sample_indices[start:stop]
        else:
            sample_candidates = block_sample_indices[parent_block_idx]
        sample_candidates = sample_candidates[sample_candidates >= 0]
        sample_candidates = sample_candidates.astype(np.int64)
        if sample_candidates.size == 0:
            parent_idx = -1
            parent_constraint = -np.inf
            parent_block_idx = -1
            did_fallback = True
        else:
            parent_idx = _choose_index_from_weights(
                key,
                sample_candidates,
                np.ones(sample_candidates.shape, dtype=float),
            )
            if sample_candidates.size > 1:
                key, _ = jax.random.split(key)
            parent_constraint = float(log_likelihoods[parent_idx])
            did_fallback = False

        parent_idxs.append(parent_idx)
        parent_log_L_constraints.append(parent_constraint)
        target_block_idxs.append(target_block_idx)
        parent_block_idxs.append(parent_block_idx)
        fallback_to_root.append(did_fallback)
        current_K[target_block_idx] += 1

    if not parent_idxs:
        return _empty_parent_work()
    return ParentWork(
        parent_idxs=jnp.asarray(parent_idxs, dtype=mp_policy.index_dtype),
        parent_log_L_constraints=jnp.asarray(
            parent_log_L_constraints,
            dtype=mp_policy.measure_dtype,
        ),
        target_block_idxs=jnp.asarray(
            target_block_idxs,
            dtype=mp_policy.index_dtype,
        ),
        parent_block_idxs=jnp.asarray(
            parent_block_idxs,
            dtype=mp_policy.index_dtype,
        ),
        fallback_to_root=jnp.asarray(
            fallback_to_root,
            dtype=mp_policy.bool_dtype,
        ),
    )


def accept_parent_work(
        state: State,
        parent_work: ParentWork,
        new_samples: Samples,
) -> State:
    """Accept completed parent work using the preserved in-flight parents."""
    num_new = int(parent_work.parent_idxs.shape[0])
    if num_new == 0:
        return state

    constraints = parent_work.parent_log_L_constraints
    new_samples = dataclasses.replace(
        new_samples,
        log_L_constraints=constraints,
        out_degree=jnp.zeros((num_new,), dtype=state.samples.out_degree.dtype),
        num_likelihood_evaluations=(
            new_samples.num_likelihood_evaluations.astype(
                state.samples.num_likelihood_evaluations.dtype
            )
        ),
    )
    fallback_to_root = parent_work.fallback_to_root
    parent_idxs = jnp.where(
        fallback_to_root,
        jnp.asarray(0, dtype=mp_policy.index_dtype),
        parent_work.parent_idxs,
    )
    delta_parent_out_degree = jnp.where(
        fallback_to_root,
        jnp.asarray(0, dtype=state.samples.out_degree.dtype),
        jnp.asarray(1, dtype=state.samples.out_degree.dtype),
    )
    root_delta = jnp.sum(
        fallback_to_root.astype(state.root_out_degree.dtype),
        dtype=state.root_out_degree.dtype,
    )
    candidate_idx = jnp.argmax(new_samples.log_likelihoods)
    candidate_log_L_supremum = new_samples.log_likelihoods[candidate_idx]
    candidate_U_supremum = jax.tree.map(
        lambda u: u[candidate_idx],
        new_samples.U_samples,
    )
    improves_supremum = candidate_log_L_supremum > state.log_L_supremum

    samples = state.samples.append_samples(
        insert_idx=jnp.asarray(state.num_samples, dtype=jnp.int64),
        parent_idxs=parent_idxs,
        samples=new_samples,
        delta_parent_out_degree=delta_parent_out_degree,
    ).sort()
    return State(
        root_out_degree=state.root_out_degree + root_delta,
        samples=samples,
        num_samples=(
                state.num_samples
                + jnp.asarray(num_new, dtype=state.num_samples.dtype)
        ),
        log_L_supremum=jnp.where(
            improves_supremum,
            candidate_log_L_supremum,
            state.log_L_supremum,
        ),
        U_supremum=jax.tree.map(
            lambda u_new, u_old: jnp.where(improves_supremum, u_new, u_old),
            candidate_U_supremum,
            state.U_supremum,
        ),
        model=state.model,
        args=state.args,
        params=state.params,
        termination_reason=state.termination_reason,
    )
