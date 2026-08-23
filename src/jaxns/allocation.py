from __future__ import annotations

import dataclasses
from typing import Literal, cast, get_args

import numpy as np
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.race_tree import BlockState, build_block_state
from jaxns.samples import Samples
from jaxns.state import State
from jaxns.types import BoolArray, FloatArray, IntArray
from jaxns.v3_shrinkage import DirichletConcentrations, classic_dirichlet_concentrations

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

    def allocation_gap(self) -> IntArray:
        """Return the positive lineage deficit for every valid block."""
        return jnp.where(
            self.valid,
            jnp.maximum(self.target_K - self.current_K, 0),
            0,
        )

    def under_allocated(self, relevant: BoolArray | None = None) -> BoolArray:
        """Mask valid blocks with a positive current allocation gap."""
        mask = self.valid if relevant is None else self.valid & relevant
        return mask & (self.current_K < self.target_K)


AllocationPlan.register_pytree()


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
        block_state: BlockState | None = None,
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
    if block_state is None:
        block_state = build_block_state(
            state.samples,
            root_out_degree=state.root_out_degree,
            num_samples=state.num_samples,
            likelihood_order=state.likelihood_order,
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


def stationary_seed_indices_python(
        samples: Samples,
        num_samples: int,
        log_L_constraint: float,
        *,
        from_root: bool,
) -> np.ndarray:
    """Pure-Python reference for exact stationary seed eligibility."""
    constraints = np.asarray(samples.log_L_constraints[:num_samples])
    likelihoods = np.asarray(samples.log_likelihoods[:num_samples])
    if from_root:
        if samples.parent_idx is None:
            eligible = np.isneginf(constraints)
        else:
            eligible = np.asarray(samples.parent_idx[:num_samples]) == -1
    else:
        eligible = (
            (constraints <= log_L_constraint)
            & (log_L_constraint < likelihoods)
        )
    return np.flatnonzero(eligible).astype(np.int64)


def closest_seedable_parent_block_python(
        state: State,
        block_state: BlockState,
        requested_block_idx: int,
) -> int:
    """Pure-Python reference for closest-shallower seed reparenting."""
    num_samples = int(np.asarray(state.num_samples))
    valid = np.asarray(block_state.valid, dtype=bool)
    log_l = np.asarray(block_state.log_L_blocks)
    candidate = requested_block_idx
    while candidate >= 0:
        if valid[candidate]:
            seeds = stationary_seed_indices_python(
                state.samples,
                num_samples,
                float(log_l[candidate]),
                from_root=False,
            )
            if seeds.size:
                return candidate
        candidate -= 1
    root_seeds = stationary_seed_indices_python(
        state.samples,
        num_samples,
        -np.inf,
        from_root=True,
    )
    return -1 if root_seeds.size else -2
