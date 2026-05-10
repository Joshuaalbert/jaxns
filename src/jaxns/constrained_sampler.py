import dataclasses
import operator
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, NamedTuple, Any

import jax
import numpy as np
from jax import numpy as jnp, random

from jaxns.cumulative_ops import cumulative_op_static
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.multi_ellipsoid_utils import ellipsoid_params, log_ellipsoid_volume
from jaxns.pytree import PureDataclassPytree, TreeField, pytree_ravel
from jaxns.random_utils import sample_uniformly_masked
from jaxns.samples import Samples, SeedPoint, PhantomSamples
from jaxns.types import FloatArray, IntArray, PRNGKey, UType, BoolArray


ISOTROPIC_DIRECTION_KERNELS = frozenset(("isotropic", "isotropic_gaussian"))
ELLIPSOIDAL_DIRECTION_KERNELS = frozenset(("ellipsoidal", "ellipsoidal_gaussian"))
GMM_DIRECTION_KERNELS = frozenset(("gmm", "non_isotropic", "non-isotropic"))
STRAIGHT_LINE_TRAJECTORIES = frozenset(("straight_line", "straight_line_perfect"))
GALILEAN_TRAJECTORIES = frozenset(("galilean",))
UNSUPPORTED_TRAJECTORIES = frozenset(("gradient_guided",))


class AbstractSampler(ABC):
    """
    Performs constrained-prior sampling for nested sampling.

    Public sampler calls use the v3 strict parent-contour contract: the
    supplied seed must satisfy ``seed_point.log_L0 > log_L_constraint``, and the
    returned classic sample is drawn within ``L(U) > log_L_constraint``.
    Equality with the contour is invalid. Execution code is responsible for
    choosing a valid seed or falling back to the sentinel parent before calling
    the sampler.

    V3 phantom samples are likelihood-only diagnostics from post-burn-in Markov
    chain states. Sampler outputs do not retain phantom coordinates, and
    phantoms must not be promoted to posterior or classic race samples.

    The sampler is assumed to be stateless and pure.
    """

    @abstractmethod
    def num_phantom(self) -> int:
        """
        Get the number of post-burn-in phantom likelihoods per classic sample.

        The count is validated to be non-negative. ``phantom_burn_in`` reserves
        early chain states for convergence before diagnostic likelihoods are
        retained; it is not a behavioral switch for accepting invalid states.

        Returns:
            the number of phantom likelihood diagnostics to produce per real
            sample.
        """
        ...

    @abstractmethod
    def get_sample(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            args=(),
            params=None,
            adaptation_context=None,
    ) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
        """
        Produce one classic sample inside the strict likelihood contour.

        The seed and every accepted sampler state must satisfy
        ``log_L > log_L_constraint``. Seeds equal to the contour are rejected
        because v3 constrained priors are strict. Returned phantom samples, when
        collected, contain only likelihood values and validity structure for
        diagnostics; their coordinates are discarded and do not feed posterior
        samples.

        Args:
            key: PRNGkey
            log_L_constraint: the strict parent contour to sample within
            seed_point: a seed point to begin sampling from

        Returns:
            U_sample: an i.i.d. sample within the strict constraint
            log_L: the log-likelihood of the sample
            num_likelihood_evaluations: number of likelihood evaluations used to produce the sample
            phantom_samples: likelihood-only post-burn-in diagnostic states
                from the same strict contour.
        """
        ...

@partial(jax.jit, inline=True)
def _sample_direction(key: PRNGKey, u0: TreeField[UType], radii: TreeField[UType] | None = None, rotation: UType | None = None) -> TreeField[UType]:
    """
    Choose a direction randomly from S^(D-1).

    Args:
        key: PRNG key
        u0: a point in the sample space, used to determine the shape of the direction.

    Returns:
        direction: [D] direction from S^(D-1)
    """
    ndim = u0.ndim()
    if ndim == 1:
        leaf_dtype = jax.tree.leaves(u0)[0].dtype
        sign = jnp.where(
            random.bernoulli(key),
            jnp.ones((), leaf_dtype),
            -jnp.ones((), leaf_dtype),
        )
        return u0.ones_like() * sign
    direction = u0.random_normal_like(key)
    if radii is not None:
        direction = radii * direction
    if rotation is not None:
        direction = rotation @ direction
    eps = jnp.asarray(1e-6, direction.norm().dtype)
    norm = jnp.maximum(eps, direction.norm())
    return direction / norm

@partial(jax.jit, inline=True)
def _slice_bounds(point_U0: TreeField[UType], direction: TreeField[UType]) -> tuple[FloatArray, FloatArray]:
    """
    Compute the slice bounds, t, where point_U0 + direction * t intersects uit cube boundary.

    Args:
        point_U0: starting point of slice
        direction: direction of slice

    Returns:
        left_bound: left most point (<= 0).
        right_bound: right most point (>= 0).
    """
    leaf_dtype = jax.tree.leaves(point_U0)[0].dtype
    zero = jnp.zeros((), leaf_dtype)
    one = jnp.ones((), leaf_dtype)
    inf = jnp.full((), jnp.inf, leaf_dtype)
    t1 = (one - point_U0) / direction
    t1_right = jax.tree.reduce(lambda x, y: jnp.minimum(jnp.min(jnp.where(x >= zero, x, inf)), jnp.min(jnp.where(y >= zero, y, inf))), t1,
                               initializer=jnp.inf)
    t1_left = jax.tree.reduce(lambda x, y: jnp.maximum(jnp.max(jnp.where(x <= zero, x, -inf)), jnp.max(jnp.where(y <= zero, y, -inf))), t1,
                              initializer=-jnp.inf)
    t0 = -point_U0 / direction
    t0_right = jax.tree.reduce(lambda x, y: jnp.minimum(jnp.min(jnp.where(x >= zero, x, inf)), jnp.min(jnp.where(y >= zero, y, inf))), t0,
                               initializer=jnp.inf)
    t0_left = jax.tree.reduce(lambda x, y: jnp.maximum(jnp.max(jnp.where(x <= zero, x, -inf)), jnp.max(jnp.where(y <= zero, y, -inf))), t0,
                              initializer=-jnp.inf)
    right_bound = jnp.minimum(t0_right, t1_right)
    left_bound = jnp.maximum(t0_left, t1_left)
    return left_bound, right_bound

@partial(jax.jit, inline=True)
def _pick_point_in_interval(key: PRNGKey, point_U0: TreeField[UType], direction: TreeField[UType], left: FloatArray,
                            right: FloatArray) -> tuple[TreeField[UType], FloatArray]:
    """
    Select a point along slice in [point_U0 + direction * left, point_U0 + direction * right]

    Args:
        key: PRNG key
        point_U0: [D]
        direction: [D]
        left: left most point (<= 0).
        right: right most point (>= 0).

    Returns:
        point_U: [D]
        t: selection point between [left, right]
    """
    leaf_dtype = jax.tree.leaves(point_U0)[0].dtype
    t = left + random.uniform(key, dtype=leaf_dtype) * (right - left)
    point_U = point_U0 + direction * t
    return point_U, t

@partial(jax.jit, inline=True)
def _shrink_interval(t: FloatArray, left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Not successful proposal, so shrink, optionally apply exponential shrinkage.
    """
    zero = jnp.zeros_like(t)
    left = jnp.where(t < zero, t, left)
    right = jnp.where(t > zero, t, right)

    return left, right


@dataclasses.dataclass(frozen=True, slots=True)
class EllipsoidalGaussianDirectionKernel:
    component_means: jax.Array
    component_radii: jax.Array
    component_rotations: jax.Array
    component_probabilities: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class GreedyShrinkageResult:
    proposal: TreeField[UType]
    log_likelihood: FloatArray
    num_likelihood_evaluations: IntArray
    used_current_point_fallback: BoolArray
    num_shrinkage_steps: IntArray
    accepted_t: FloatArray
    rejected_t: FloatArray
    left_bounds: FloatArray
    right_bounds: FloatArray


@dataclasses.dataclass(frozen=True, slots=True)
class GalileanSideTrajectory:
    points: FloatArray
    directions: FloatArray
    terminal_direction: FloatArray
    num_likelihood_evaluations: IntArray


@dataclasses.dataclass(frozen=True, slots=True)
class GalileanTrajectory:
    points: FloatArray
    segment_lengths: FloatArray
    num_likelihood_evaluations: IntArray


@dataclasses.dataclass(frozen=True, slots=True)
class GalileanSample:
    point: FloatArray
    segment_index: IntArray
    alpha: FloatArray


def _as_mode_name(value: object) -> str | None:
    if isinstance(value, str):
        return value.lower()
    for attr_name in ("mode", "name", "value"):
        if hasattr(value, attr_name):
            attr_value = getattr(value, attr_name)
            if isinstance(attr_value, str):
                return attr_value.lower()
            if hasattr(attr_value, "value"):
                return str(attr_value.value).lower()
            return str(attr_value).lower()
    return None


def _validate_trajectory_mode(trajectory: object) -> None:
    trajectory_mode = _as_mode_name(trajectory)
    if trajectory_mode in STRAIGHT_LINE_TRAJECTORIES:
        return
    if trajectory_mode in GALILEAN_TRAJECTORIES:
        return
    if trajectory_mode in UNSUPPORTED_TRAJECTORIES:
        raise NotImplementedError(
            f"Trajectory mode {trajectory_mode!r} is unsupported and "
            "ambiguous. Use explicit trajectory='galilean' for the v3 "
            "gradient-informed sampler."
        )
    raise ValueError(
        f"Unsupported trajectory mode {trajectory!r}. Supported trajectory "
        "modes are 'straight_line' and 'galilean'."
    )


def _validate_direction_kernel_mode(direction_kernel: object) -> None:
    if isinstance(direction_kernel, EllipsoidalGaussianDirectionKernel):
        return
    direction_mode = _as_mode_name(direction_kernel)
    if direction_mode in ISOTROPIC_DIRECTION_KERNELS:
        return
    if direction_mode in ELLIPSOIDAL_DIRECTION_KERNELS | GMM_DIRECTION_KERNELS:
        return
    if hasattr(direction_kernel, "component_probabilities"):
        return
    raise ValueError(
        f"Unsupported direction kernel {direction_kernel!r}. Supported "
        "baseline kernels are 'isotropic' and 'ellipsoidal'."
    )


def _build_ellipsoidal_direction_kernel(
        *,
        adaptation_context,
) -> EllipsoidalGaussianDirectionKernel:
    """
    Freeze an ellipsoidal Gaussian direction kernel for one sampler chain.

    Component selection probabilities are normalized integrated volumes. The
    component means are snapshotted for diagnostics/configuration completeness;
    direction draws use each component covariance around zero so the kernel is
    symmetric and independent of the current chain point.
    """
    if adaptation_context is None:
        raise ValueError(
            "Ellipsoidal direction kernels require an adaptation_context."
        )

    if all(
            _context_has(adaptation_context, attr_name)
            for attr_name in (
                "component_means",
                "component_radii",
                "component_rotations",
            )
    ):
        component_means = jnp.asarray(
            _context_get(adaptation_context, "component_means")
        )
        component_radii = jnp.asarray(
            _context_get(adaptation_context, "component_radii")
        )
        component_rotations = jnp.asarray(
            _context_get(adaptation_context, "component_rotations")
        )
        if _context_has(adaptation_context, "component_integrated_volumes"):
            integrated_volumes = jnp.asarray(
                _context_get(
                    adaptation_context,
                    "component_integrated_volumes",
                ),
                dtype=component_radii.dtype,
            )
        else:
            integrated_volumes = jnp.exp(
                jax.vmap(log_ellipsoid_volume)(component_radii)
            )
    else:
        component_means, component_radii, component_rotations, integrated_volumes = (
            _build_ellipsoidal_components_from_history(adaptation_context)
        )

    clipped_volumes = jnp.where(
        integrated_volumes > jnp.zeros((), integrated_volumes.dtype),
        integrated_volumes,
        jnp.zeros_like(integrated_volumes),
    )
    total_volume = jnp.sum(clipped_volumes)
    try:
        has_positive_volume = bool(np.asarray(total_volume > 0.0))
    except Exception:
        has_positive_volume = True
    if not has_positive_volume:
        raise ValueError(
            "Ellipsoidal direction kernel requires at least one positive "
            "component integrated volume."
        )
    component_probabilities = clipped_volumes / total_volume

    return EllipsoidalGaussianDirectionKernel(
        component_means=component_means,
        component_radii=component_radii,
        component_rotations=component_rotations,
        component_probabilities=component_probabilities,
    )


def _context_has(context, name: str) -> bool:
    if isinstance(context, dict):
        return name in context
    return hasattr(context, name)


def _context_get(context, name: str, default=None):
    if isinstance(context, dict):
        return context.get(name, default)
    return getattr(context, name, default)


def _adaptation_samples_to_matrix(samples_U) -> FloatArray:
    if hasattr(samples_U, "tree"):
        samples_U = samples_U.tree
    leaves = jax.tree.leaves(samples_U)
    if not leaves:
        raise ValueError("Ellipsoidal adaptation_context samples_U is empty.")
    first_leaf = jnp.asarray(leaves[0])
    if first_leaf.ndim == 0:
        raise ValueError(
            "Ellipsoidal adaptation_context samples_U must include a sample "
            "axis."
        )
    num_samples = first_leaf.shape[0]
    flat_leaves = [
        jnp.reshape(jnp.asarray(leaf), (num_samples, -1))
        for leaf in leaves
    ]
    return jnp.concatenate(flat_leaves, axis=1)


def _build_ellipsoidal_components_from_history(
        adaptation_context,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Build a frozen ellipsoidal mixture from execution-supplied sample history.

    Execution supplies the sample history at chain start. The resulting
    component fields are snapshotted here and reused for every direction in that
    chain; later adaptation-context updates only affect future chains.
    """
    if not _context_has(adaptation_context, "samples_U"):
        raise ValueError(
            "Ellipsoidal adaptation_context must provide either precomputed "
            "component fields or samples_U history."
        )

    points = _adaptation_samples_to_matrix(
        _context_get(adaptation_context, "samples_U")
    )
    num_points = points.shape[0]
    valid_mask = _context_get(adaptation_context, "valid_mask", None)
    if valid_mask is None:
        valid_mask = jnp.ones((num_points,), dtype=mp_policy.bool_dtype)
    else:
        valid_mask = jnp.asarray(valid_mask, dtype=mp_policy.bool_dtype)

    try:
        num_valid = int(np.sum(np.asarray(valid_mask)))
    except Exception:
        num_valid = num_points
    if num_valid < 1:
        raise ValueError(
            "Ellipsoidal adaptation_context samples_U has no valid samples."
        )

    # Keep the ticket-0008 path minimal: sample history is converted into one
    # bounding ellipsoid and frozen for the chain. Future tickets can split this
    # into multiple components without changing the sampler contract.
    params = ellipsoid_params(points=points, mask=valid_mask)
    component_means = params.mu[None, :]
    component_radii = params.radii[None, :]
    component_rotations = params.rotation[None, :, :]
    integrated_volumes = jnp.exp(log_ellipsoid_volume(params.radii))[None]
    return (
        component_means,
        component_radii,
        component_rotations,
        integrated_volumes,
    )


def _freeze_direction_kernel(
        direction_kernel: object,
        adaptation_context,
):
    direction_mode = _as_mode_name(direction_kernel)
    if isinstance(direction_kernel, EllipsoidalGaussianDirectionKernel):
        return direction_kernel
    if direction_mode in ISOTROPIC_DIRECTION_KERNELS:
        return "isotropic"
    if direction_mode in ELLIPSOIDAL_DIRECTION_KERNELS | GMM_DIRECTION_KERNELS:
        return _build_ellipsoidal_direction_kernel(
            adaptation_context=adaptation_context,
        )
    if hasattr(direction_kernel, "component_probabilities"):
        return direction_kernel
    raise ValueError(
        f"Unsupported direction kernel {direction_kernel!r}. Supported "
        "baseline kernels are 'isotropic' and 'ellipsoidal'."
    )


def _sample_direction_component(
        *,
        key: PRNGKey,
        direction_kernel,
) -> IntArray:
    probabilities = jnp.asarray(direction_kernel.component_probabilities)
    log_probabilities = jnp.log(probabilities)
    return random.categorical(key, logits=log_probabilities)


def _flat_direction_to_tree(
        current_point: TreeField[UType],
        flat_direction: FloatArray,
) -> TreeField[UType]:
    leaves, tree_def = jax.tree.flatten(current_point.tree)
    if not leaves:
        return current_point.zeros_like()

    split_indices = np.cumsum([np.size(leaf) for leaf in leaves[:-1]])
    flat_leaves = jnp.split(flat_direction, split_indices)
    direction_leaves = [
        jnp.reshape(flat_leaf, leaf.shape)
        for flat_leaf, leaf in zip(flat_leaves, leaves)
    ]
    return TreeField(jax.tree.unflatten(tree_def, direction_leaves))


def _sample_direction_from_kernel(
        *,
        key: PRNGKey,
        direction_kernel,
        current_point: TreeField[UType],
) -> TreeField[UType]:
    direction_mode = _as_mode_name(direction_kernel)
    if direction_mode in ISOTROPIC_DIRECTION_KERNELS:
        return _sample_direction(key, current_point)

    component_key, sample_key = random.split(key, 2)
    component_idx = _sample_direction_component(
        key=component_key,
        direction_kernel=direction_kernel,
    )
    component_radii = direction_kernel.component_radii[component_idx]
    component_rotation = direction_kernel.component_rotations[component_idx]

    raw_direction = random.normal(
        sample_key,
        shape=component_radii.shape,
        dtype=component_radii.dtype,
    )
    flat_direction = component_rotation @ (component_radii * raw_direction)
    eps = jnp.asarray(1e-6, flat_direction.dtype)
    norm = jnp.maximum(eps, jnp.linalg.norm(flat_direction))
    return _flat_direction_to_tree(current_point, flat_direction / norm)


def _resolve_positive_limit(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool.")
    try:
        resolved = operator.index(value)
    except TypeError as e:
        raise ValueError(f"{name} must be an integer.") from e
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return resolved


def _point_tree(value):
    if hasattr(value, "tree"):
        return value.tree
    return value


class _StaticLogLikelihoodFn:
    """Stable static JIT argument for repeated sampler calls on one problem."""

    __slots__ = ("model", "args", "params", "_hash", "__weakref__")

    def __init__(self, model: Model, args: tuple, params):
        self.model = model
        self.args = args
        self.params = params
        self._hash = hash((id(model), id(args), id(params)))

    def __call__(self, U):
        return self.model.log_likelihood(
            U,
            args=self.args,
            params=self.params,
            allow_nan=False,
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return (
                isinstance(other, _StaticLogLikelihoodFn)
                and self.model is other.model
                and self.args is other.args
                and self.params is other.params
        )


def _ravel_point(value) -> tuple[FloatArray, Callable[[FloatArray], UType]]:
    return pytree_ravel(_point_tree(value))


def _normalize_galilean_vector(vector: FloatArray, name: str) -> FloatArray:
    vector = jnp.ravel(jnp.asarray(vector))
    norm = jnp.linalg.norm(vector)
    if not bool(np.asarray(jnp.isfinite(norm))) or not bool(np.asarray(norm > 0.0)):
        raise ValueError(f"Galilean {name} is degenerate or zero.")
    return vector / norm


def _flat_point_in_unit_cube(point: FloatArray) -> bool:
    point = jnp.ravel(jnp.asarray(point))
    inside = jnp.all((point >= 0.0) & (point <= 1.0))
    return bool(np.asarray(inside))


def _reflect_galilean_direction(
        direction: FloatArray,
        gradient_normal: FloatArray,
) -> FloatArray:
    """
    Reflect a Galilean direction through a likelihood-gradient normal.

    The normal is normalized before applying the Householder reflection. This
    keeps the helper stable for callers that pass either a raw gradient or an
    already-unit boundary normal.
    """
    unit_direction = _normalize_galilean_vector(direction, "direction")
    unit_normal = _normalize_galilean_vector(gradient_normal, "gradient normal")
    reflected = unit_direction - 2.0 * jnp.vdot(unit_direction, unit_normal) * unit_normal
    return _normalize_galilean_vector(reflected, "reflected direction")


def _build_galilean_side(
        *,
        U0,
        direction,
        log_L_constraint: FloatArray,
        initial_step_size: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        grad_log_likelihood_fn: Callable[[UType], UType],
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> GalileanSideTrajectory:
    """
    Build one side of the paper Galilean trajectory until the U-turn criterion.
    """
    max_reflections = _resolve_positive_limit(
        max_reflections,
        "max_reflections",
    )
    max_step_halvings = _resolve_positive_limit(
        max_step_halvings,
        "max_step_halvings",
    )
    max_step_doublings = _resolve_positive_limit(
        max_step_doublings,
        "max_step_doublings",
    )

    current_point, unravel_fn = _ravel_point(U0)
    current_direction = _normalize_galilean_vector(direction, "direction")
    initial_direction = current_direction
    step_size = jnp.asarray(initial_step_size, dtype=current_point.dtype)
    if not bool(np.asarray(jnp.isfinite(step_size))) or not bool(np.asarray(step_size > 0.0)):
        raise ValueError("Galilean initial_step_size must be positive and finite.")

    constraint = jnp.asarray(log_L_constraint)
    num_likelihood_evaluations = jnp.asarray(0, mp_policy.count_dtype)

    def _is_inside(point: FloatArray) -> bool:
        nonlocal num_likelihood_evaluations
        if not _flat_point_in_unit_cube(point):
            return False
        log_likelihood = jnp.asarray(
            log_likelihood_fn(unravel_fn(point)),
            dtype=constraint.dtype,
        )
        num_likelihood_evaluations += jnp.ones(
            (),
            mp_policy.count_dtype,
        )
        return bool(np.asarray(log_likelihood > constraint))

    if not _is_inside(current_point):
        raise ValueError("Galilean seed point must satisfy the strict contour.")

    points = [current_point]
    directions = [current_direction]

    for _ in range(max_reflections):
        alignment = jnp.vdot(current_direction, initial_direction)
        if bool(np.asarray(alignment < 0.0)):
            return GalileanSideTrajectory(
                points=jnp.stack(points, axis=0),
                directions=jnp.stack(directions, axis=0),
                terminal_direction=current_direction,
                num_likelihood_evaluations=num_likelihood_evaluations,
            )

        proposal = current_point + step_size * current_direction
        if _is_inside(proposal):
            last_inside = proposal
            search_step_size = step_size
            found_boundary = False
            for _ in range(max_step_doublings):
                candidate_step_size = 2.0 * search_step_size
                candidate = current_point + candidate_step_size * current_direction
                if _is_inside(candidate):
                    last_inside = candidate
                    search_step_size = candidate_step_size
                else:
                    found_boundary = True
                    break
            if not found_boundary:
                raise RuntimeError(
                    "Galilean boundary search exceeded max_step_doublings "
                    "without leaving the contour."
                )
            current_point = last_inside
            step_size = search_step_size
        else:
            search_step_size = step_size
            found_inside = False
            for _ in range(max_step_halvings):
                search_step_size = 0.5 * search_step_size
                candidate = current_point + search_step_size * current_direction
                if _is_inside(candidate):
                    current_point = candidate
                    step_size = search_step_size
                    found_inside = True
                    break
            if not found_inside:
                raise RuntimeError(
                    "Galilean boundary search exceeded max_step_halvings "
                    "without returning inside the contour."
                )

        gradient, _ = _ravel_point(grad_log_likelihood_fn(unravel_fn(current_point)))
        current_direction = _reflect_galilean_direction(
            current_direction,
            gradient,
        )
        points.append(current_point)
        directions.append(current_direction)

    raise RuntimeError(
        "Galilean trajectory hit max_reflections before satisfying the "
        "U-turn reflection limit."
    )


def _build_galilean_trajectory(
        *,
        U0,
        direction,
        log_L_constraint: FloatArray,
        initial_step_size: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        grad_log_likelihood_fn: Callable[[UType], UType],
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> GalileanTrajectory:
    positive_side = _build_galilean_side(
        U0=U0,
        direction=direction,
        log_L_constraint=log_L_constraint,
        initial_step_size=initial_step_size,
        log_likelihood_fn=log_likelihood_fn,
        grad_log_likelihood_fn=grad_log_likelihood_fn,
        max_reflections=max_reflections,
        max_step_halvings=max_step_halvings,
        max_step_doublings=max_step_doublings,
    )
    negative_side = _build_galilean_side(
        U0=U0,
        direction=-jnp.asarray(direction),
        log_L_constraint=log_L_constraint,
        initial_step_size=initial_step_size,
        log_likelihood_fn=log_likelihood_fn,
        grad_log_likelihood_fn=grad_log_likelihood_fn,
        max_reflections=max_reflections,
        max_step_halvings=max_step_halvings,
        max_step_doublings=max_step_doublings,
    )
    points = jnp.concatenate(
        [
            negative_side.points[:0:-1],
            positive_side.points,
        ],
        axis=0,
    )
    segment_lengths = jnp.linalg.norm(jnp.diff(points, axis=0), axis=1)
    if not bool(np.asarray(jnp.all(segment_lengths > 0.0))):
        raise RuntimeError("Galilean trajectory contains a degenerate segment.")
    return GalileanTrajectory(
        points=points,
        segment_lengths=segment_lengths,
        num_likelihood_evaluations=(
            positive_side.num_likelihood_evaluations
            + negative_side.num_likelihood_evaluations
        ),
    )


def _sample_galilean_trajectory(
        *,
        key: PRNGKey,
        points: FloatArray,
        segment_lengths: FloatArray,
) -> GalileanSample:
    points = jnp.asarray(points)
    segment_lengths = jnp.asarray(segment_lengths, dtype=points.dtype)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("Galilean trajectory points must have shape (N, D), N >= 2.")
    if segment_lengths.shape != (points.shape[0] - 1,):
        raise ValueError("Galilean segment_lengths must match trajectory segments.")
    if not bool(np.asarray(jnp.all(segment_lengths >= 0.0))):
        raise ValueError("Galilean segment lengths must be non-negative.")
    total_length = jnp.sum(segment_lengths)
    if not bool(np.asarray(total_length > 0.0)):
        raise ValueError("Galilean trajectory has zero total path length.")

    segment_key, alpha_key = random.split(key, 2)
    path_position = random.uniform(
        segment_key,
        dtype=points.dtype,
        minval=jnp.zeros((), points.dtype),
        maxval=total_length,
    )
    cumulative_lengths = jnp.cumsum(segment_lengths)
    segment_index = jnp.searchsorted(
        cumulative_lengths,
        path_position,
        side="right",
    )
    segment_index = jnp.minimum(segment_index, segment_lengths.shape[0] - 1)
    alpha = random.uniform(alpha_key, dtype=points.dtype)
    point = (
        points[segment_index]
        + alpha * (points[segment_index + 1] - points[segment_index])
    )
    return GalileanSample(
        point=point,
        segment_index=segment_index,
        alpha=alpha,
    )


def _sampled_galilean_point(sample) -> FloatArray:
    for attr_name in ("point", "point_U", "U", "U_sample"):
        if hasattr(sample, attr_name):
            return jnp.asarray(getattr(sample, attr_name))
    if isinstance(sample, dict):
        for key in ("point", "point_U", "U", "U_sample"):
            if key in sample:
                return jnp.asarray(sample[key])
    if isinstance(sample, tuple):
        return jnp.asarray(sample[0])
    return jnp.asarray(sample)


def _sample_galilean_markov_transition(
        *,
        key: PRNGKey,
        U0: TreeField[UType],
        direction: TreeField[UType],
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        grad_log_likelihood_fn: Callable[[UType], UType],
        initial_step_size: FloatArray,
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> tuple[TreeField[UType], FloatArray, IntArray]:
    build_key, sample_key = random.split(key, 2)
    del build_key
    direction_flat, _ = _ravel_point(direction)
    _, unravel_fn = _ravel_point(U0)
    trajectory = _build_galilean_trajectory(
        U0=U0.tree,
        direction=direction_flat,
        log_L_constraint=log_L_constraint,
        initial_step_size=initial_step_size,
        log_likelihood_fn=log_likelihood_fn,
        grad_log_likelihood_fn=grad_log_likelihood_fn,
        max_reflections=max_reflections,
        max_step_halvings=max_step_halvings,
        max_step_doublings=max_step_doublings,
    )
    sample = _sample_galilean_trajectory(
        key=sample_key,
        points=trajectory.points,
        segment_lengths=trajectory.segment_lengths,
    )
    point_tree = unravel_fn(_sampled_galilean_point(sample))
    log_likelihood = jnp.asarray(
        log_likelihood_fn(point_tree),
        dtype=jnp.asarray(log_L_constraint).dtype,
    )
    if (
            not _flat_point_in_unit_cube(_sampled_galilean_point(sample))
            or not bool(np.asarray(log_likelihood > log_L_constraint))
    ):
        raise RuntimeError(
            "Galilean trajectory sample did not satisfy the strict contour "
            "and unit-cube support."
        )
    trajectory_evaluations = getattr(
        trajectory,
        "num_likelihood_evaluations",
        jnp.asarray(0, mp_policy.count_dtype),
    )
    num_likelihood_evaluations = (
        jnp.asarray(trajectory_evaluations, dtype=mp_policy.count_dtype)
        + jnp.ones((), mp_policy.count_dtype)
    )
    return TreeField(point_tree), log_likelihood, num_likelihood_evaluations


def _greedy_shrink_to_strict_contour(
        *,
        key: PRNGKey,
        U0: TreeField[UType],
        direction: TreeField[UType],
        left: FloatArray,
        right: FloatArray,
        log_L0: FloatArray,
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        max_steps: int,
        proposal_t_sequence: FloatArray | None = None,
) -> GreedyShrinkageResult:
    """
    Greedily shrink a straight-line interval until a strict-contour point lands.

    Each rejected proposal becomes the nearest bracket boundary on its side of
    the current point. The current point remains inside the bracket and is
    returned as a strict-contour fallback if the retry budget is exhausted.
    """
    rejected_t = []
    left_bounds = []
    right_bounds = []
    num_likelihood_evaluations = jnp.asarray(0, mp_policy.count_dtype)
    current_left = left
    current_right = right
    run_key = key

    for step in range(operator.index(max_steps)):
        if proposal_t_sequence is None:
            run_key, proposal_key = random.split(run_key, 2)
            point_U, proposal_t = _pick_point_in_interval(
                key=proposal_key,
                point_U0=U0,
                direction=direction,
                left=current_left,
                right=current_right,
            )
        else:
            proposal_t = proposal_t_sequence[step]
            point_U = U0 + direction * proposal_t

        log_L = log_likelihood_fn(point_U.tree).astype(log_L_constraint.dtype)
        num_likelihood_evaluations += jnp.ones(
            (),
            mp_policy.count_dtype,
        )
        if bool(np.asarray(log_L > log_L_constraint)):
            return GreedyShrinkageResult(
                proposal=point_U,
                log_likelihood=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations,
                used_current_point_fallback=jnp.asarray(
                    False,
                    mp_policy.bool_dtype,
                ),
                num_shrinkage_steps=num_likelihood_evaluations,
                accepted_t=proposal_t,
                rejected_t=jnp.asarray(rejected_t, dtype=proposal_t.dtype),
                left_bounds=jnp.asarray(left_bounds, dtype=current_left.dtype),
                right_bounds=jnp.asarray(
                    right_bounds,
                    dtype=current_right.dtype,
                ),
            )

        current_left, current_right = _shrink_interval(
            t=proposal_t,
            left=current_left,
            right=current_right,
        )
        rejected_t.append(proposal_t)
        left_bounds.append(current_left)
        right_bounds.append(current_right)

    return GreedyShrinkageResult(
        proposal=U0,
        log_likelihood=log_L0,
        num_likelihood_evaluations=num_likelihood_evaluations,
        used_current_point_fallback=jnp.asarray(True, mp_policy.bool_dtype),
        num_shrinkage_steps=num_likelihood_evaluations,
        accepted_t=jnp.zeros_like(left),
        rejected_t=jnp.asarray(rejected_t, dtype=left.dtype),
        left_bounds=jnp.asarray(left_bounds, dtype=left.dtype),
        right_bounds=jnp.asarray(right_bounds, dtype=right.dtype),
    )


@partial(jax.jit, inline=True, static_argnames=["no_step_out", "gradient_guided", "log_likelihood_fn"])
def _new_proposal(
        key: PRNGKey,
        U0: TreeField[UType],
        direction: TreeField[UType],
        slice_width: FloatArray,
        no_step_out: bool,
        gradient_guided: bool,
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        log_L0: FloatArray | None = None,
        max_shrinkage_steps: int = 32,
) -> tuple[TreeField[UType], FloatArray, IntArray, TreeField[UType], FloatArray]:
    """
    Sample from a slice about a seed point.

    Args:
        key: PRNG key
        direction: the direction to sample along
        no_step_out: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
        gradient_guided: if true then do householder reflections
        log_L_constraint: the constraint to sample within
        log_likelihood_fn: the log-likelihood function

    Returns:
        point_U: the new sample
        log_L: the log-likelihood of the new sample
        num_likelihood_evaluations: the number of likelihood evaluations performed
    """

    class Carry(NamedTuple):
        key: PRNGKey
        direction: TreeField[UType]
        left: FloatArray
        right: FloatArray
        t: FloatArray
        point_U: TreeField[UType]
        log_L: FloatArray
        num_likelihood_evaluations: IntArray
        num_shrinkage_steps: IntArray

    def cond(carry: Carry) -> BoolArray:
        satisfaction = jnp.all(carry.log_L > log_L_constraint)
        has_budget = carry.num_shrinkage_steps < jnp.asarray(
            max_shrinkage_steps,
            mp_policy.count_dtype,
        )
        return jnp.bitwise_and(jnp.bitwise_not(satisfaction), has_budget)

    def body(carry: Carry) -> Carry:
        key, t_key = random.split(carry.key, 2)
        left, right = _shrink_interval(
            t=carry.t,
            left=carry.left,
            right=carry.right
        )
        point_U, t = _pick_point_in_interval(
            key=t_key,
            point_U0=U0,
            direction=carry.direction,
            left=left,
            right=right
        )
        log_L = log_likelihood_fn(point_U.tree).astype(log_L_constraint.dtype)
        num_likelihood_evaluations = carry.num_likelihood_evaluations + jnp.ones_like(carry.num_likelihood_evaluations)
        num_shrinkage_steps = carry.num_shrinkage_steps + jnp.ones_like(carry.num_shrinkage_steps)
        return Carry(
            key=key,
            t=t,
            left=left,
            right=right,
            point_U=point_U,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
            num_shrinkage_steps=num_shrinkage_steps,
            direction=carry.direction
        )

    # Chose the direction to go
    num_likelihood_evaluations = jnp.full((), 0, mp_policy.count_dtype)

    run_key, t_key, step_key, after_key = random.split(key, 4)

    (left_bound, right_bound) = _slice_bounds(
        point_U0=U0,
        direction=direction
    )
    slice_width = jnp.asarray(slice_width, left_bound.dtype)

    if no_step_out:
        left, right = left_bound, right_bound
        step_out_num_likelihood_evaluations = jnp.asarray(0, mp_policy.count_dtype)
    else:
        class StepOutCarry(NamedTuple):
            key: PRNGKey
            left: FloatArray
            right: FloatArray
            left_outside_slice: BoolArray
            right_outside_slice: BoolArray
            num_likelihood_evaluations: IntArray

        eps = jnp.asarray(1e-12, left_bound.dtype)
        use_full_slice = jnp.isinf(slice_width)
        effective_slice_width = jnp.maximum(slice_width, eps)
        place_key, step_key = random.split(step_key)
        uniform_origin = random.uniform(place_key, dtype=left_bound.dtype)
        initial_left = -uniform_origin * effective_slice_width
        initial_right = initial_left + effective_slice_width

        left = jnp.where(use_full_slice, left_bound, jnp.maximum(left_bound, initial_left))
        right = jnp.where(use_full_slice, right_bound, jnp.minimum(right_bound, initial_right))

        def _point_at_t(t: FloatArray) -> TreeField[UType]:
            return U0 + direction * t

        def step_out_cond(carry: StepOutCarry) -> BoolArray:
            can_expand_left = carry.left > left_bound
            can_expand_right = carry.right < right_bound
            both_outside_box = jnp.bitwise_not(jnp.bitwise_or(can_expand_left, can_expand_right))
            both_outside_slice = jnp.bitwise_and(carry.left_outside_slice, carry.right_outside_slice)
            return jnp.bitwise_not(jnp.bitwise_or(both_outside_box, both_outside_slice))

        def step_out_body(carry: StepOutCarry) -> StepOutCarry:
            key, choose_key = random.split(carry.key, 2)

            can_expand_left = carry.left > left_bound
            can_expand_right = carry.right < right_bound

            choose_left_random = random.uniform(choose_key, dtype=left_bound.dtype) < jnp.where(
                can_expand_left & can_expand_right, 0.5, jnp.where(can_expand_left, 1., 0.))

            current_width = jnp.maximum(carry.right - carry.left, eps)
            candidate_left = jnp.maximum(left_bound, carry.left - current_width)
            candidate_right = jnp.minimum(right_bound, carry.right + current_width)

            t_eval = jnp.where(choose_left_random, candidate_left, candidate_right)

            next_left = jnp.where(choose_left_random, candidate_left, carry.left)
            next_right = jnp.where(choose_left_random, carry.right, candidate_right)

            log_L_eval = log_likelihood_fn(_point_at_t(t_eval).tree)
            outside_slice_eval = log_L_eval <= log_L_constraint
            next_left_outside_slice = jnp.where(choose_left_random, outside_slice_eval, carry.left_outside_slice)
            next_right_outside_slice = jnp.where(choose_left_random, carry.right_outside_slice, outside_slice_eval)
            num_likelihood_evaluations = carry.num_likelihood_evaluations + jnp.ones_like(carry.num_likelihood_evaluations)

            return StepOutCarry(
                key=key,
                left=next_left,
                right=next_right,
                left_outside_slice=next_left_outside_slice,
                right_outside_slice=next_right_outside_slice,
                num_likelihood_evaluations=num_likelihood_evaluations
            )

        left_outside_slice = log_likelihood_fn(_point_at_t(left).tree) <= log_L_constraint
        right_outside_slice = log_likelihood_fn(_point_at_t(right).tree) <= log_L_constraint
        step_out_init = StepOutCarry(
            key=step_key,
            left=left,
            right=right,
            left_outside_slice=left_outside_slice,
            right_outside_slice=right_outside_slice,
            num_likelihood_evaluations=jnp.asarray(2, mp_policy.count_dtype)
        )

        step_out_carry = jax.lax.while_loop(
            cond_fun=lambda c: jnp.bitwise_and(jnp.bitwise_not(use_full_slice), step_out_cond(c)),
            body_fun=step_out_body,
            init_val=step_out_init
        )
        left, right = step_out_carry.left, step_out_carry.right
        step_out_num_likelihood_evaluations = step_out_carry.num_likelihood_evaluations

    point_U, t = _pick_point_in_interval(
        key=t_key,
        point_U0=U0,
        direction=direction,
        left=left,
        right=right
    )
    log_L = log_likelihood_fn(point_U.tree).astype(log_L_constraint.dtype)
    num_likelihood_evaluations += jnp.ones_like(num_likelihood_evaluations)
    num_likelihood_evaluations += step_out_num_likelihood_evaluations
    init_carry = Carry(
        key=run_key,
        direction=direction,
        left=left,
        right=right,
        t=t,
        point_U=point_U,
        log_L=log_L,
        num_likelihood_evaluations=num_likelihood_evaluations,
        num_shrinkage_steps=jnp.asarray(0, mp_policy.count_dtype),
    )

    carry = jax.lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=init_carry
    )

    # Update direction
    proposal_satisfied = jnp.all(carry.log_L > log_L_constraint)
    if log_L0 is None:
        fallback_log_L = log_likelihood_fn(U0.tree).astype(log_L_constraint.dtype)
        fallback_num_evaluations = jnp.ones(
            (),
            mp_policy.count_dtype,
        )
    else:
        fallback_log_L = jnp.asarray(log_L0, dtype=log_L_constraint.dtype)
        fallback_num_evaluations = jnp.asarray(0, mp_policy.count_dtype)

    point_U = jax.tree.map(
        lambda proposal_leaf, current_leaf: jnp.where(
            proposal_satisfied,
            proposal_leaf,
            current_leaf,
        ),
        carry.point_U,
        U0,
    )
    log_L = jnp.where(proposal_satisfied, carry.log_L, fallback_log_L)
    direction = carry.direction
    num_likelihood_evaluations = carry.num_likelihood_evaluations + jnp.where(
        proposal_satisfied,
        jnp.asarray(0, mp_policy.count_dtype),
        fallback_num_evaluations,
    )
    if gradient_guided:
        # Perform HMC with Householder reflections
        raise NotImplementedError("Gradient guided slice sampler not implemented.")
    else:
        # Randomly choose a new direction
        direction = _sample_direction(after_key, direction)
    next_slice_width = 2 * (carry.right - carry.left)
    return point_U, log_L, num_likelihood_evaluations, direction, next_slice_width


@partial(jax.jit, inline=True)
def get_seed_point(key: PRNGKey, samples: Samples, log_L_constraint: FloatArray) -> SeedPoint:
    """
    Get a seed point from samples that satisfies the likelihood constraint. This is done by masking samples that do not satisfy the constraint, and sampling uniformly from the remaining samples.

    Args:
        key: PRNGKey
        samples: samples to select from.
        log_L_constraint: the constraint to sample within.

    Returns:
        seed point that satisfy the constraint.
    """
    select_mask = samples.log_likelihoods > log_L_constraint
    seed_point = sample_uniformly_masked(
        key=key,
        v=SeedPoint(U0=samples.U_samples, log_L0=samples.log_likelihoods),
        select_mask=select_mask,
        num_samples=1,
        squeeze=True
    )
    return seed_point


def _resolve_num_slices(num_slices: int) -> int:
    if isinstance(num_slices, bool):
        raise ValueError("num_slices must be an integer, not bool.")
    try:
        resolved = operator.index(num_slices)
    except TypeError as e:
        raise ValueError("num_slices must be an integer.") from e
    if resolved < 1:
        raise ValueError(f"num_slices should be >= 1, got {num_slices}.")
    return resolved


def _resolve_phantom_burn_in(
        num_slices: int,
        phantom_burn_in: int | None,
) -> int:
    """Validate burn-in leaves a non-negative post-burn-in phantom capacity."""
    resolved_num_slices = _resolve_num_slices(num_slices)
    if phantom_burn_in is None:
        burn_in = int(resolved_num_slices * 0.1)
    else:
        if isinstance(phantom_burn_in, bool):
            raise ValueError("phantom_burn_in must be an integer, not bool.")
        try:
            burn_in = operator.index(phantom_burn_in)
        except TypeError as e:
            raise ValueError("phantom_burn_in must be an integer.") from e
    if burn_in < 0 or burn_in > resolved_num_slices - 1:
        raise ValueError(
            f"phantom_burn_in must satisfy 0 <= burn_in <= num_slices - 1, "
            f"got burn_in={burn_in}, num_slices={resolved_num_slices}."
        )
    return burn_in


def _resolve_max_shrinkage_steps(max_shrinkage_steps: int) -> int:
    if isinstance(max_shrinkage_steps, bool):
        raise ValueError("max_shrinkage_steps must be an integer, not bool.")
    try:
        resolved = operator.index(max_shrinkage_steps)
    except TypeError as e:
        raise ValueError("max_shrinkage_steps must be an integer.") from e
    if resolved < 0:
        raise ValueError(
            "max_shrinkage_steps must be non-negative, got "
            f"{max_shrinkage_steps}."
        )
    return resolved


@dataclasses.dataclass(slots=True, frozen=True)
class UniDimSliceSampler(AbstractSampler, PureDataclassPytree):
    """
    One-dimensional slice sampler for strict constrained priors.

    The public sampler contract is strict: valid seeds and accepted samples must
    satisfy ``log_L > log_L_constraint``. A seed with likelihood equal to the
    contour is rejected, matching the v3 parent-contour semantics.

    When phantom collection is enabled, retained phantoms are post-burn-in
    chain-state likelihoods only. Their coordinates are intentionally discarded
    and cannot feed posterior samples, MAP/supremum estimates, resampling, or
    classic race-sample counts.

    Args:
        model: AbstractModel
        num_slices: number of slices between acceptance. Note: some other software use units of prior dimension.
        no_step_out: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
            Otherwise, uses a doubling procedure (exponentially finding bracket).
            Note: Perfect is a misnomer, as perfection also depends on the number of slices between acceptance.
        gradient_guided: if true then do HMC with householder reflections.
        collect_phantom_samples: if true, then collect phantom samples
        phantom_burn_in: number of initial chain states to exclude before
            retaining phantom likelihood diagnostics. Must satisfy
            ``0 <= burn_in <= num_slices - 1`` so ``num_phantom()`` is never
            negative.
    """

    model: Model
    num_slices: int
    no_step_out: bool = True
    gradient_guided: bool = False
    collect_phantom_samples: bool = False
    phantom_burn_in: int | None = None
    direction_kernel: object = "isotropic"
    trajectory: object = "straight_line"
    max_shrinkage_steps: int = 32
    galilean_initial_step_size: float = 0.05
    max_galilean_reflections: int = 64
    max_galilean_step_halvings: int = 32
    max_galilean_step_doublings: int = 32

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(
            this,
            [
                'num_slices',
                'no_step_out',
                'gradient_guided',
                'collect_phantom_samples',
                'phantom_burn_in',
                'direction_kernel',
                'trajectory',
                'max_shrinkage_steps',
                'galilean_initial_step_size',
                'max_galilean_reflections',
                'max_galilean_step_halvings',
                'max_galilean_step_doublings',
            ],
        )


    def _check(self):
        _resolve_phantom_burn_in(self.num_slices, self.phantom_burn_in)
        _resolve_max_shrinkage_steps(self.max_shrinkage_steps)
        _resolve_positive_limit(
            self.max_galilean_reflections,
            "max_galilean_reflections",
        )
        _resolve_positive_limit(
            self.max_galilean_step_halvings,
            "max_galilean_step_halvings",
        )
        _resolve_positive_limit(
            self.max_galilean_step_doublings,
            "max_galilean_step_doublings",
        )
        _validate_direction_kernel_mode(self.direction_kernel)
        _validate_trajectory_mode(self.trajectory)
        if self.gradient_guided:
            raise NotImplementedError(
                "The legacy gradient_guided flag is ambiguous with explicit "
                "trajectory modes. Use the Ticket 0012 Galilean trajectory "
                "once it is implemented."
            )

    def __post_init__(self):
        self._check()

    def num_phantom(self) -> int:
        if self.collect_phantom_samples:
            burn_in = _resolve_phantom_burn_in(
                self.num_slices,
                self.phantom_burn_in,
            )
            return _resolve_num_slices(self.num_slices) - 1 - burn_in
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            args=(),
            params=None,
            adaptation_context=None,
    ) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
        try:
            valid_seed = bool(np.asarray(seed_point.log_L0 > log_L_constraint))
        except Exception:
            valid_seed = True
        if not valid_seed:
            raise ValueError("Seed point must satisfy the strict likelihood constraint.")

        class XType(NamedTuple):
            key: jax.Array
            direction_key: jax.Array
            alpha: jax.Array

        num_slices = _resolve_num_slices(self.num_slices)
        log_likelihood_fn = _StaticLogLikelihoodFn(self.model, args, params)
        force_python_loop = bool(
            _context_get(adaptation_context, "force_python_loop", False)
        )
        direction_adaptation_context = _context_get(
            adaptation_context,
            "direction_adaptation_context",
            adaptation_context,
        )
        direction_kernel = _freeze_direction_kernel(
            self.direction_kernel,
            direction_adaptation_context,
        )
        direction_template = TreeField(seed_point.U0)
        trajectory_mode = _as_mode_name(self.trajectory)

        if trajectory_mode in GALILEAN_TRAJECTORIES:
            grad_log_likelihood_fn = jax.grad(log_likelihood_fn)
            all_samples = []
            all_log_likelihoods = []
            num_likelihood_evaluations = jnp.asarray(
                0,
                mp_policy.count_dtype,
            )
            current_point = direction_template
            current_log_likelihood = jnp.asarray(
                seed_point.log_L0,
                dtype=log_L_constraint.dtype,
            )
            proposal_keys = random.split(key, num_slices)
            direction_keys = random.split(random.fold_in(key, 1), num_slices)

            for proposal_key, direction_key in zip(
                    proposal_keys,
                    direction_keys,
            ):
                direction = _sample_direction_from_kernel(
                    key=direction_key,
                    direction_kernel=direction_kernel,
                    current_point=direction_template,
                )
                current_point, current_log_likelihood, delta_evaluations = (
                    _sample_galilean_markov_transition(
                        key=proposal_key,
                        U0=current_point,
                        direction=direction,
                        log_L_constraint=log_L_constraint,
                        log_likelihood_fn=log_likelihood_fn,
                        grad_log_likelihood_fn=grad_log_likelihood_fn,
                        initial_step_size=jnp.asarray(
                            self.galilean_initial_step_size,
                            dtype=log_L_constraint.dtype,
                        ),
                        max_reflections=self.max_galilean_reflections,
                        max_step_halvings=self.max_galilean_step_halvings,
                        max_step_doublings=self.max_galilean_step_doublings,
                    )
                )
                all_samples.append(current_point)
                all_log_likelihoods.append(current_log_likelihood)
                num_likelihood_evaluations += delta_evaluations

            num_phantom = self.num_phantom()
            if num_phantom == 0:
                phantom_log_likelihoods = jnp.zeros(
                    (0,),
                    dtype=log_L_constraint.dtype,
                )
                phantom_valid_mask = jnp.zeros(
                    (0,),
                    dtype=mp_policy.bool_dtype,
                )
            else:
                phantom_start = num_slices - 1 - num_phantom
                phantom_log_likelihoods = jnp.stack(
                    all_log_likelihoods[phantom_start:-1],
                    axis=0,
                )
                phantom_valid_mask = jnp.ones(
                    (num_phantom,),
                    dtype=mp_policy.bool_dtype,
                )
            phantom_samples = PhantomSamples(
                U_samples=None,
                log_L=phantom_log_likelihoods,
                valid_mask=phantom_valid_mask,
            )
            return (
                all_samples[-1].tree,
                all_log_likelihoods[-1],
                num_likelihood_evaluations,
                phantom_samples,
            )

        class Carry(NamedTuple):
            U_sample: TreeField[UType]
            log_L_constraint: FloatArray
            log_L: FloatArray
            num_likelihood_evaluations: IntArray
            direction: TreeField[UType]
            slice_width: FloatArray

        def propose_op(carry: Carry, x: XType) -> Carry:
            U_sample, log_L, num_likelihood_evaluations, direction, slice_width = _new_proposal(
                key=x.key,
                U0=carry.U_sample,
                direction=carry.direction,
                slice_width=carry.slice_width,
                no_step_out=self.no_step_out,
                gradient_guided=self.gradient_guided,
                log_L_constraint=carry.log_L_constraint,
                log_likelihood_fn=log_likelihood_fn,
                log_L0=carry.log_L,
                max_shrinkage_steps=self.max_shrinkage_steps,
            )
            direction = _sample_direction_from_kernel(
                key=x.direction_key,
                direction_kernel=direction_kernel,
                current_point=direction_template,
            )

            carry = Carry(
                U_sample=U_sample,
                log_L_constraint=carry.log_L_constraint,
                log_L=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations + carry.num_likelihood_evaluations,
                direction=direction,
                slice_width=slice_width
            )
            return carry

        direction_key, sample_key = jax.random.split(key, 2)

        init_direction = _sample_direction_from_kernel(
            key=direction_key,
            direction_kernel=direction_kernel,
            current_point=direction_template,
        )
        slice_width_dtype = jax.tree.leaves(seed_point.U0)[0].dtype

        #### initial proposal to get slice width for cumulative op with perfect stepout
        sample_key, init_sample_key, init_direction_key = random.split(
            sample_key,
            3,
        )

        U_sample, log_L, num_likelihood_evaluations, _, slice_width = _new_proposal(
            key=init_sample_key,
            U0=direction_template,
            direction=init_direction,
            slice_width=jnp.asarray(jnp.inf, slice_width_dtype),
            no_step_out=True,
            gradient_guided=self.gradient_guided,
            log_L_constraint=log_L_constraint,
            log_likelihood_fn=log_likelihood_fn,
            log_L0=seed_point.log_L0,
            max_shrinkage_steps=self.max_shrinkage_steps,
        )
        init_direction = _sample_direction_from_kernel(
            key=init_direction_key,
            direction_kernel=direction_kernel,
            current_point=direction_template,
        )

        init_carry = Carry(
            U_sample=U_sample,
            log_L_constraint=log_L_constraint,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
            direction=init_direction,
            slice_width=slice_width
        )

        proposal_key, direction_scan_key = random.split(sample_key, 2)
        xs = XType(
            key=random.split(proposal_key, num_slices - 1),
            direction_key=random.split(direction_scan_key, num_slices - 1),
            alpha=jnp.linspace(0.5, 1., num_slices - 1)
        )
        if force_python_loop:
            carry = init_carry
            samples = []
            for i in range(num_slices - 1):
                carry = propose_op(
                    carry,
                    XType(
                        key=xs.key[i],
                        direction_key=xs.direction_key[i],
                        alpha=xs.alpha[i],
                    ),
                )
                samples.append(carry)
            final_carry = carry
            if samples:
                cumulative_samples = jax.tree.map(
                    lambda *values: jnp.stack(values, axis=0),
                    *samples,
                )
            else:
                cumulative_samples = jax.tree.map(
                    lambda value: jnp.zeros(
                        (0,) + jnp.shape(value),
                        dtype=jnp.asarray(value).dtype,
                    ),
                    init_carry,
                )
        else:
            final_carry, cumulative_samples = cumulative_op_static(
                op=propose_op,
                init=init_carry,
                xs=xs
            )

        # concat initial sample to cumulative samples
        cumulative_samples = jax.tree.map(
            lambda x, y: jnp.concatenate([x[None], y], axis=0),
            init_carry,
            cumulative_samples
        )

        # Last sample is the final classic sample. Earlier post-burn-in states
        # contribute only likelihood diagnostics; their coordinates are not
        # stored and cannot become posterior or classic race samples.
        assert self.num_phantom() <= num_slices - 1, "num_phantom() should be in [0, num_slices - 1]"

        phantom_fraction = jax.tree.map(lambda x: x[num_slices - 1 - self.num_phantom():-1], cumulative_samples)
        phantom_samples = PhantomSamples(
            U_samples=None,
            log_L=phantom_fraction.log_L,
            valid_mask=jnp.ones(phantom_fraction.log_L.shape, mp_policy.bool_dtype)
        )

        U_sample = final_carry.U_sample.tree
        log_L_sample = final_carry.log_L
        num_likelihood_evaluations = final_carry.num_likelihood_evaluations

        return U_sample, log_L_sample, num_likelihood_evaluations, phantom_samples


UniDimSliceSampler.register_pytree()
