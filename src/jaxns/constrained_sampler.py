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
_MISSING = object()


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


@dataclasses.dataclass(frozen=True, slots=True)
class GalileanBracketSample:
    point: FloatArray
    outside_point: FloatArray


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


def _context_any(context, names: tuple[str, ...], default=None):
    for name in names:
        value = _context_get(context, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


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


def _has_traced_leaf(value) -> bool:
    """Return whether a pytree contains a JAX tracer leaf."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _normalize_galilean_vector(vector: FloatArray, name: str) -> FloatArray:
    vector = jnp.ravel(jnp.asarray(vector))
    norm = jnp.linalg.norm(vector)
    if not bool(np.asarray(jnp.isfinite(norm))) or not bool(np.asarray(norm > 0.0)):
        raise ValueError(f"Galilean {name} is degenerate or zero.")
    return vector / norm


def _normalize_galilean_vector_jax(vector: FloatArray) -> FloatArray:
    """Normalize a flat vector without Python validation for traced calls."""
    vector = jnp.ravel(jnp.asarray(vector))
    eps = jnp.asarray(1e-12, vector.dtype)
    norm = jnp.maximum(jnp.linalg.norm(vector), eps)
    return vector / norm


def _flat_point_in_unit_cube(point: FloatArray) -> bool:
    point = jnp.ravel(jnp.asarray(point))
    inside = jnp.all((point >= 0.0) & (point <= 1.0))
    return bool(np.asarray(inside))


def _flat_point_in_unit_cube_jax(point: FloatArray) -> BoolArray:
    """Return a traced-compatible unit-cube support predicate."""
    point = jnp.ravel(jnp.asarray(point))
    return jnp.all((point >= 0.0) & (point <= 1.0))


def _unit_cube_forward_step_limit_jax(
        point: FloatArray,
        direction: FloatArray,
) -> FloatArray:
    """Return the first non-negative step from ``point`` to the unit cube wall."""
    # point, direction: [D]
    point = jnp.ravel(jnp.asarray(point))
    direction = jnp.ravel(jnp.asarray(direction))
    inf = jnp.full((), jnp.inf, dtype=point.dtype)
    upper_steps = jnp.where(direction > 0.0, (1.0 - point) / direction, inf)
    lower_steps = jnp.where(direction < 0.0, -point / direction, inf)
    step_limit = jnp.minimum(jnp.min(upper_steps), jnp.min(lower_steps))
    return jnp.maximum(step_limit, jnp.asarray(0.0, point.dtype))


def _clip_galilean_step_to_unit_cube_jax(
        *,
        point: FloatArray,
        direction: FloatArray,
        step_size: FloatArray,
) -> tuple[FloatArray, BoolArray]:
    """Clip a forward Galilean step at the unit cube support boundary."""
    step_limit = _unit_cube_forward_step_limit_jax(point, direction)
    clipped_step_size = jnp.minimum(step_size, step_limit)
    hit_support_boundary = step_size >= step_limit
    return clipped_step_size, hit_support_boundary


def _unit_cube_support_normal_jax(
        point: FloatArray,
        direction: FloatArray | None = None,
) -> FloatArray:
    """Return an outward support normal for a unit-cube boundary point."""
    # point, direction, normal: [D]
    point = jnp.ravel(jnp.asarray(point))
    lower_violation = jnp.where(point < 0.0, point, 0.0)
    upper_violation = jnp.where(point > 1.0, point - 1.0, 0.0)
    violation_normal = lower_violation + upper_violation
    if direction is None:
        return violation_normal

    direction = jnp.ravel(jnp.asarray(direction))
    eps = jnp.asarray(1e-10, point.dtype)
    lower_hit = (point <= eps) & (direction < 0.0)
    upper_hit = (point >= (1.0 - eps)) & (direction > 0.0)
    boundary_normal = jnp.where(
        lower_hit,
        -jnp.ones_like(point),
        jnp.zeros_like(point),
    )
    boundary_normal = jnp.where(
        upper_hit,
        jnp.ones_like(point),
        boundary_normal,
    )
    has_violation = jnp.linalg.norm(violation_normal) > eps
    return jnp.where(has_violation, violation_normal, boundary_normal)


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


def _reflect_galilean_direction_jax(
        direction: FloatArray,
        gradient_normal: FloatArray,
        support_normal: FloatArray,
) -> FloatArray:
    """Reflect a direction using a traced-compatible boundary normal."""
    # direction, gradient_normal, support_normal: [D]
    unit_direction = _normalize_galilean_vector_jax(direction)
    gradient_normal = jnp.ravel(jnp.asarray(gradient_normal))
    support_normal = jnp.ravel(jnp.asarray(support_normal))
    eps = jnp.asarray(1e-12, gradient_normal.dtype)
    normal_norm = jnp.linalg.norm(gradient_normal)
    support_norm = jnp.linalg.norm(support_normal)
    gradient_or_fallback = jnp.where(
        normal_norm > eps,
        gradient_normal,
        unit_direction,
    )
    gradient_or_fallback_norm = jnp.linalg.norm(gradient_or_fallback)
    unit_normal = jnp.where(
        support_norm > eps,
        support_normal / jnp.maximum(support_norm, eps),
        gradient_or_fallback / jnp.maximum(gradient_or_fallback_norm, eps),
    )
    reflected = (
        unit_direction
        - 2.0 * jnp.vdot(unit_direction, unit_normal) * unit_normal
    )
    return _normalize_galilean_vector_jax(reflected)


def _sample_galilean_reflection_point_from_bracket(
        *,
        key: PRNGKey,
        inside_point: FloatArray,
        outside_point: FloatArray,
        is_inside_fn: Callable[[FloatArray], bool],
) -> GalileanBracketSample:
    """Sample the Galilean reflection point from an inside/outside bracket.

    Args:
        key: PRNG key for the stochastic bracket shrink.
        inside_point: [D] point known to be inside the strict contour.
        outside_point: [D] point known to be outside the strict contour.
        is_inside_fn: predicate that checks strict contour membership and
            records any likelihood-evaluation accounting owned by the caller.

    Returns:
        The first uniformly sampled bracket point that falls inside, plus the
        last outside endpoint after any failed draws.
    """
    # inside_point, outside_point, candidate: [D]
    inside_point = jnp.ravel(jnp.asarray(inside_point))
    outside_point = jnp.ravel(jnp.asarray(outside_point))
    shrink_key = key
    current_outside = outside_point
    while True:
        shrink_key, sample_key = random.split(shrink_key)
        alpha = random.uniform(sample_key, dtype=inside_point.dtype)
        candidate = inside_point + alpha * (current_outside - inside_point)
        if is_inside_fn(candidate):
            return GalileanBracketSample(
                point=candidate,
                outside_point=current_outside,
            )
        current_outside = candidate


class _GalileanJaxBracketSample(NamedTuple):
    point: FloatArray
    outside_point: FloatArray
    num_likelihood_evaluations: IntArray


def _sample_galilean_reflection_point_from_bracket_jax(
        *,
        key: PRNGKey,
        inside_point: FloatArray,
        outside_point: FloatArray,
        strict_inside_fn: Callable[
            [FloatArray],
            tuple[BoolArray, FloatArray, IntArray],
        ],
) -> _GalileanJaxBracketSample:
    """Traceable stochastic shrink from a Galilean contour bracket.

    Args:
        key: PRNG key for the stochastic bracket shrink.
        inside_point: [D] point known to be inside the strict contour.
        outside_point: [D] point known to be outside the strict contour.
        strict_inside_fn: traced predicate returning strict-contour membership,
            log-likelihood, and likelihood-evaluation count for a point.

    Returns:
        The first uniformly sampled bracket point that falls inside, plus the
        last outside endpoint and additional likelihood-evaluation count.
    """
    # inside_point, outside_point, candidate: [D]
    inside_point = jnp.ravel(jnp.asarray(inside_point))
    outside_point = jnp.ravel(jnp.asarray(outside_point))

    class BracketCarry(NamedTuple):
        outside_point: FloatArray
        key: PRNGKey
        found_inside: BoolArray
        point: FloatArray
        num_likelihood_evaluations: IntArray

    def cond(carry: BracketCarry) -> BoolArray:
        return jnp.logical_not(carry.found_inside)

    def body(carry: BracketCarry) -> BracketCarry:
        sample_key, next_key = random.split(carry.key)
        alpha = random.uniform(sample_key, dtype=inside_point.dtype)
        candidate = inside_point + alpha * (
            carry.outside_point - inside_point
        )
        candidate_inside, _, candidate_evaluations = strict_inside_fn(
            candidate
        )
        return BracketCarry(
            outside_point=jnp.where(
                candidate_inside,
                carry.outside_point,
                candidate,
            ),
            key=next_key,
            found_inside=candidate_inside,
            point=jnp.where(candidate_inside, candidate, carry.point),
            num_likelihood_evaluations=(
                carry.num_likelihood_evaluations + candidate_evaluations
            ),
        )

    final_carry = jax.lax.while_loop(
        cond,
        body,
        BracketCarry(
            outside_point=outside_point,
            key=key,
            found_inside=jnp.asarray(False, mp_policy.bool_dtype),
            point=inside_point,
            num_likelihood_evaluations=jnp.zeros(
                (),
                mp_policy.count_dtype,
            ),
        ),
    )
    return _GalileanJaxBracketSample(
        point=final_carry.point,
        outside_point=final_carry.outside_point,
        num_likelihood_evaluations=final_carry.num_likelihood_evaluations,
    )


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
        key: PRNGKey | None = None,
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

        proposal_step_size, proposal_hit_support = (
            _clip_galilean_step_to_unit_cube_jax(
                point=current_point,
                direction=current_direction,
                step_size=step_size,
            )
        )
        proposal = current_point + proposal_step_size * current_direction
        proposal_inside = (
                not bool(np.asarray(proposal_hit_support))
                and _is_inside(proposal)
        )
        if proposal_inside:
            last_inside = proposal
            search_step_size = step_size
            found_boundary = False
            first_outside = proposal
            for _ in range(max_step_doublings):
                candidate_step_size = 2.0 * search_step_size
                clipped_step_size, candidate_hit_support = (
                    _clip_galilean_step_to_unit_cube_jax(
                        point=current_point,
                        direction=current_direction,
                        step_size=candidate_step_size,
                    )
                )
                candidate = current_point + clipped_step_size * current_direction
                candidate_inside = (
                        not bool(np.asarray(candidate_hit_support))
                        and _is_inside(candidate)
                )
                if candidate_inside:
                    last_inside = candidate
                    search_step_size = candidate_step_size
                else:
                    first_outside = candidate
                    found_boundary = True
                    break
            if not found_boundary:
                raise RuntimeError(
                    "Galilean boundary search exceeded max_step_doublings "
                    "without leaving the contour."
                )
            if key is None:
                current_point = last_inside
                step_size = search_step_size
            else:
                key, bracket_key = random.split(key)
                bracket_sample = _sample_galilean_reflection_point_from_bracket(
                    key=bracket_key,
                    inside_point=last_inside,
                    outside_point=first_outside,
                    is_inside_fn=_is_inside,
                )
                current_point = bracket_sample.point
                step_size = search_step_size
                first_outside = bracket_sample.outside_point
            current_outside_point = first_outside
        else:
            search_step_size = step_size
            found_inside = False
            found_inside_point = current_point
            at_support_boundary = bool(
                np.asarray(
                    _unit_cube_forward_step_limit_jax(
                        current_point,
                        current_direction,
                    )
                    <= jnp.asarray(1e-12, current_point.dtype)
                )
            )
            if at_support_boundary:
                current_outside_point = current_point
            else:
                for _ in range(max_step_halvings):
                    search_step_size = 0.5 * search_step_size
                    clipped_step_size, candidate_hit_support = (
                        _clip_galilean_step_to_unit_cube_jax(
                            point=current_point,
                            direction=current_direction,
                            step_size=search_step_size,
                        )
                    )
                    candidate = current_point + clipped_step_size * current_direction
                    candidate_inside = (
                            not bool(np.asarray(candidate_hit_support))
                            and _is_inside(candidate)
                    )
                    if candidate_inside:
                        found_inside_point = candidate
                        found_inside = True
                        break
                if not found_inside:
                    raise RuntimeError(
                        "Galilean boundary search exceeded max_step_halvings "
                        "without returning inside the contour."
                    )
                if key is None:
                    current_point = found_inside_point
                    step_size = search_step_size
                    current_outside_point = proposal
                else:
                    key, bracket_key = random.split(key)
                    bracket_sample = _sample_galilean_reflection_point_from_bracket(
                        key=bracket_key,
                        inside_point=found_inside_point,
                        outside_point=proposal,
                        is_inside_fn=_is_inside,
                    )
                    current_point = bracket_sample.point
                    step_size = search_step_size
                    current_outside_point = bracket_sample.outside_point

        gradient, _ = _ravel_point(grad_log_likelihood_fn(unravel_fn(current_point)))
        support_normal = _unit_cube_support_normal_jax(
            current_outside_point,
            current_direction,
        )
        current_direction = _reflect_galilean_direction_jax(
            current_direction,
            gradient,
            support_normal,
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
        key: PRNGKey | None = None,
) -> GalileanTrajectory:
    positive_key = None
    negative_key = None
    if key is not None:
        positive_key, negative_key = random.split(key)

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
        key=positive_key,
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
        key=negative_key,
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
        key=build_key,
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


class _GalileanFixedSide(NamedTuple):
    segment_starts: FloatArray
    segment_ends: FloatArray
    segment_lengths: FloatArray
    step_size: FloatArray
    num_likelihood_evaluations: IntArray


def _sample_galilean_markov_transition_jax_with_step(
        *,
        key: PRNGKey,
        U0: TreeField[UType],
        log_L0: FloatArray,
        direction: TreeField[UType],
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        grad_log_likelihood_fn: Callable[[UType], UType],
        initial_step_size: FloatArray,
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> tuple[TreeField[UType], FloatArray, IntArray, FloatArray]:
    """Sample a Galilean transition using JAX control flow.

    The two trajectory sides are materialized into fixed-size segment buffers
    of length ``max_reflections``. Valid segment lengths define the explicit
    path-length CDF used for the final uniform draw.
    """
    # flat_u0, flat_direction: [D]
    flat_u0, unravel_fn = _ravel_point(U0)
    flat_direction, _ = _ravel_point(direction)
    flat_direction = _normalize_galilean_vector_jax(flat_direction)
    constraint = jnp.asarray(log_L_constraint)
    step_size0 = jnp.asarray(initial_step_size, dtype=flat_u0.dtype)
    step_size0 = jnp.maximum(step_size0, jnp.asarray(1e-12, flat_u0.dtype))
    del max_step_halvings, max_step_doublings

    def log_likelihood_flat(point: FloatArray) -> FloatArray:
        return jnp.asarray(
            log_likelihood_fn(unravel_fn(point)),
            dtype=constraint.dtype,
        )

    def strict_inside_and_log_likelihood(
            point: FloatArray,
    ) -> tuple[BoolArray, FloatArray, IntArray]:
        in_unit_cube = _flat_point_in_unit_cube_jax(point)

        def evaluate_inside(_):
            def accept_root_contour(__):
                return (
                    jnp.asarray(True, mp_policy.bool_dtype),
                    jnp.asarray(-jnp.inf, dtype=constraint.dtype),
                    jnp.zeros((), mp_policy.count_dtype),
                )

            def evaluate_likelihood(__):
                log_likelihood = log_likelihood_flat(point)
                return (
                    log_likelihood > constraint,
                    log_likelihood,
                    jnp.ones((), mp_policy.count_dtype),
                )

            return jax.lax.cond(
                jnp.isneginf(constraint),
                accept_root_contour,
                evaluate_likelihood,
                operand=None,
            )

        def reject_outside(_):
            return (
                jnp.asarray(False, mp_policy.bool_dtype),
                jnp.asarray(-jnp.inf, dtype=constraint.dtype),
                jnp.zeros((), mp_policy.count_dtype),
            )

        return jax.lax.cond(
            in_unit_cube,
            evaluate_inside,
            reject_outside,
            operand=None,
        )

    class SideCarry(NamedTuple):
        current_point: FloatArray
        current_direction: FloatArray
        step_size: FloatArray
        key: PRNGKey
        done: BoolArray
        num_likelihood_evaluations: IntArray

    def build_side(initial_direction: FloatArray, side_key) -> _GalileanFixedSide:
        # initial_direction: [D]
        initial_direction = _normalize_galilean_vector_jax(initial_direction)

        class BoundaryResult(NamedTuple):
            next_point: FloatArray
            next_step_size: FloatArray
            outside_point: FloatArray
            num_likelihood_evaluations: IntArray

        def build_boundary(carry: SideCarry, boundary_key) -> BoundaryResult:
            old_point = carry.current_point
            proposal_step_size, proposal_hit_support = (
                _clip_galilean_step_to_unit_cube_jax(
                    point=old_point,
                    direction=carry.current_direction,
                    step_size=carry.step_size,
                )
            )
            proposal = old_point + proposal_step_size * carry.current_direction

            (
                proposal_inside,
                _,
                proposal_evaluations,
            ) = jax.lax.cond(
                proposal_hit_support,
                lambda _: (
                    jnp.asarray(False, mp_policy.bool_dtype),
                    jnp.asarray(-jnp.inf, dtype=constraint.dtype),
                    jnp.zeros((), mp_policy.count_dtype),
                ),
                lambda _: strict_inside_and_log_likelihood(proposal),
                operand=None,
            )

            class GrowCarry(NamedTuple):
                search_step_size: FloatArray
                last_inside: FloatArray
                first_outside: FloatArray
                candidate_inside: BoolArray
                step_count: IntArray
                num_likelihood_evaluations: IntArray

            def grow_cond(grow: GrowCarry) -> BoolArray:
                return grow.candidate_inside

            def grow_body(grow: GrowCarry) -> GrowCarry:
                candidate_step_size = (
                    jnp.asarray(2.0, grow.search_step_size.dtype)
                    * grow.search_step_size
                )
                clipped_step_size, hit_support_boundary = (
                    _clip_galilean_step_to_unit_cube_jax(
                        point=old_point,
                        direction=carry.current_direction,
                        step_size=candidate_step_size,
                    )
                )
                candidate = (
                    old_point
                    + clipped_step_size * carry.current_direction
                )
                (
                    candidate_inside,
                    _,
                    candidate_evaluations,
                ) = jax.lax.cond(
                    hit_support_boundary,
                    lambda _: (
                        jnp.asarray(False, mp_policy.bool_dtype),
                        jnp.asarray(-jnp.inf, dtype=constraint.dtype),
                        jnp.zeros((), mp_policy.count_dtype),
                    ),
                    lambda _: strict_inside_and_log_likelihood(candidate),
                    operand=None,
                )
                return GrowCarry(
                    search_step_size=jnp.where(
                        candidate_inside,
                        candidate_step_size,
                        grow.search_step_size,
                    ),
                    last_inside=jnp.where(
                        candidate_inside,
                        candidate,
                        grow.last_inside,
                    ),
                    first_outside=jnp.where(
                        candidate_inside,
                        grow.first_outside,
                        candidate,
                    ),
                    candidate_inside=candidate_inside,
                    step_count=(
                        grow.step_count
                        + jnp.ones((), mp_policy.count_dtype)
                    ),
                    num_likelihood_evaluations=(
                        grow.num_likelihood_evaluations
                        + candidate_evaluations
                        ),
                )

            def grow_from_inside(_: None) -> BoundaryResult:
                grown = jax.lax.while_loop(
                    grow_cond,
                    grow_body,
                    GrowCarry(
                        search_step_size=carry.step_size,
                        last_inside=proposal,
                        first_outside=proposal,
                        candidate_inside=jnp.asarray(True, mp_policy.bool_dtype),
                        step_count=jnp.zeros((), mp_policy.count_dtype),
                        num_likelihood_evaluations=(
                            carry.num_likelihood_evaluations
                            + proposal_evaluations
                        ),
                    ),
                )
                bracket_sample = (
                    _sample_galilean_reflection_point_from_bracket_jax(
                        key=boundary_key,
                        inside_point=grown.last_inside,
                        outside_point=grown.first_outside,
                        strict_inside_fn=strict_inside_and_log_likelihood,
                    )
                )
                return BoundaryResult(
                    next_point=bracket_sample.point,
                    next_step_size=grown.search_step_size,
                    outside_point=bracket_sample.outside_point,
                    num_likelihood_evaluations=(
                        grown.num_likelihood_evaluations
                        + bracket_sample.num_likelihood_evaluations
                    ),
                )

            class ShrinkCarry(NamedTuple):
                search_step_size: FloatArray
                inside_point: FloatArray
                found_inside: BoolArray
                step_count: IntArray
                num_likelihood_evaluations: IntArray

            def shrink_cond(shrink: ShrinkCarry) -> BoolArray:
                return jnp.logical_not(shrink.found_inside)

            def shrink_body(shrink: ShrinkCarry) -> ShrinkCarry:
                next_step_size = (
                    jnp.asarray(0.5, shrink.search_step_size.dtype)
                    * shrink.search_step_size
                )
                clipped_step_size, hit_support_boundary = (
                    _clip_galilean_step_to_unit_cube_jax(
                        point=old_point,
                        direction=carry.current_direction,
                        step_size=next_step_size,
                    )
                )
                candidate = old_point + clipped_step_size * carry.current_direction
                (
                    candidate_inside,
                    _,
                    candidate_evaluations,
                ) = jax.lax.cond(
                    hit_support_boundary,
                    lambda _: (
                        jnp.asarray(False, mp_policy.bool_dtype),
                        jnp.asarray(-jnp.inf, dtype=constraint.dtype),
                        jnp.zeros((), mp_policy.count_dtype),
                    ),
                    lambda _: strict_inside_and_log_likelihood(candidate),
                    operand=None,
                )
                return ShrinkCarry(
                    search_step_size=next_step_size,
                    inside_point=jnp.where(
                        candidate_inside,
                        candidate,
                        shrink.inside_point,
                    ),
                    found_inside=shrink.found_inside | candidate_inside,
                    step_count=(
                        shrink.step_count
                        + jnp.ones((), mp_policy.count_dtype)
                    ),
                    num_likelihood_evaluations=(
                        shrink.num_likelihood_evaluations
                        + candidate_evaluations
                    ),
                )

            def shrink_from_outside(_: None) -> BoundaryResult:
                at_support_boundary = (
                        _unit_cube_forward_step_limit_jax(
                            old_point,
                            carry.current_direction,
                        )
                        <= jnp.asarray(1e-12, old_point.dtype)
                )

                def reflect_at_current_support(_: None) -> BoundaryResult:
                    return BoundaryResult(
                        next_point=old_point,
                        next_step_size=carry.step_size,
                        outside_point=old_point,
                        num_likelihood_evaluations=(
                            carry.num_likelihood_evaluations
                            + proposal_evaluations
                        ),
                    )

                def shrink_to_inside(_: None) -> BoundaryResult:
                    shrunk = jax.lax.while_loop(
                        shrink_cond,
                        shrink_body,
                        ShrinkCarry(
                            search_step_size=carry.step_size,
                            inside_point=old_point,
                            found_inside=jnp.asarray(False, mp_policy.bool_dtype),
                            step_count=jnp.zeros((), mp_policy.count_dtype),
                            num_likelihood_evaluations=(
                                carry.num_likelihood_evaluations
                                + proposal_evaluations
                            ),
                        ),
                    )
                    bracket_sample = (
                        _sample_galilean_reflection_point_from_bracket_jax(
                            key=boundary_key,
                            inside_point=shrunk.inside_point,
                            outside_point=proposal,
                            strict_inside_fn=strict_inside_and_log_likelihood,
                        )
                    )
                    return BoundaryResult(
                        next_point=bracket_sample.point,
                        next_step_size=shrunk.search_step_size,
                        outside_point=bracket_sample.outside_point,
                        num_likelihood_evaluations=(
                            shrunk.num_likelihood_evaluations
                            + bracket_sample.num_likelihood_evaluations
                        ),
                    )

                return jax.lax.cond(
                    at_support_boundary,
                    reflect_at_current_support,
                    shrink_to_inside,
                    operand=None,
                )

            return jax.lax.cond(
                proposal_inside,
                grow_from_inside,
                shrink_from_outside,
                operand=None,
            )

        def side_body(carry: SideCarry, _) -> tuple[SideCarry, tuple[FloatArray, FloatArray, FloatArray]]:
            alignment = jnp.vdot(carry.current_direction, initial_direction)
            active = (
                    jnp.logical_not(carry.done)
                    & (alignment >= jnp.asarray(0.0, alignment.dtype))
            )

            def active_body(_: None) -> tuple[SideCarry, tuple[FloatArray, FloatArray, FloatArray]]:
                old_point = carry.current_point
                boundary_key, next_key = random.split(carry.key)
                boundary = build_boundary(carry, boundary_key)
                segment = boundary.next_point - old_point
                segment_length = jnp.linalg.norm(segment)
                valid_segment_length = jnp.where(
                    segment_length > jnp.asarray(0.0, segment_length.dtype),
                    segment_length,
                    jnp.asarray(0.0, segment_length.dtype),
                )

                support_normal = _unit_cube_support_normal_jax(
                    boundary.outside_point,
                    carry.current_direction,
                )
                eps = jnp.asarray(1e-12, support_normal.dtype)
                skip_gradient = (
                        (jnp.linalg.norm(support_normal) > eps)
                        | jnp.isneginf(constraint)
                )

                def zero_gradient(_):
                    return jnp.zeros_like(boundary.next_point)

                def likelihood_gradient(_):
                    gradient_tree = grad_log_likelihood_fn(
                        unravel_fn(boundary.next_point)
                    )
                    gradient, _ = _ravel_point(gradient_tree)
                    return gradient

                gradient = jax.lax.cond(
                    skip_gradient,
                    zero_gradient,
                    likelihood_gradient,
                    operand=None,
                )
                next_direction = _reflect_galilean_direction_jax(
                    carry.current_direction,
                    gradient,
                    support_normal,
                )
                next_alignment = jnp.vdot(next_direction, initial_direction)
                next_done = next_alignment < jnp.asarray(
                    0.0,
                    next_alignment.dtype,
                )
                next_carry = SideCarry(
                    current_point=boundary.next_point,
                    current_direction=next_direction,
                    step_size=boundary.next_step_size,
                    key=next_key,
                    done=next_done,
                    num_likelihood_evaluations=(
                        boundary.num_likelihood_evaluations
                    ),
                )
                return next_carry, (
                    old_point,
                    boundary.next_point,
                    valid_segment_length,
                )

            def inactive_body(_: None) -> tuple[SideCarry, tuple[FloatArray, FloatArray, FloatArray]]:
                inactive_length = jnp.asarray(0.0, flat_u0.dtype)
                next_carry = SideCarry(
                    current_point=carry.current_point,
                    current_direction=carry.current_direction,
                    step_size=carry.step_size,
                    key=carry.key,
                    done=jnp.asarray(True, mp_policy.bool_dtype),
                    num_likelihood_evaluations=(
                        carry.num_likelihood_evaluations
                    ),
                )
                return next_carry, (
                    carry.current_point,
                    carry.current_point,
                    inactive_length,
                )

            return jax.lax.cond(
                active,
                active_body,
                inactive_body,
                operand=None,
            )

        final_carry, side_segments = jax.lax.scan(
            side_body,
            SideCarry(
                current_point=flat_u0,
                current_direction=initial_direction,
                step_size=step_size0,
                key=side_key,
                done=jnp.asarray(False, mp_policy.bool_dtype),
                num_likelihood_evaluations=jnp.zeros((), mp_policy.count_dtype),
            ),
            xs=None,
            length=max_reflections,
        )
        segment_starts, segment_ends, segment_lengths = side_segments
        return _GalileanFixedSide(
            segment_starts=segment_starts,
            segment_ends=segment_ends,
            segment_lengths=segment_lengths,
            step_size=final_carry.step_size,
            num_likelihood_evaluations=(
                final_carry.num_likelihood_evaluations
            ),
        )

    positive_key, negative_key, sample_key = random.split(key, 3)
    positive_side = build_side(flat_direction, positive_key)
    negative_side = build_side(-flat_direction, negative_key)
    segment_starts = jnp.concatenate(
        [positive_side.segment_starts, negative_side.segment_starts],
        axis=0,
    )
    segment_ends = jnp.concatenate(
        [positive_side.segment_ends, negative_side.segment_ends],
        axis=0,
    )
    segment_lengths = jnp.concatenate(
        [positive_side.segment_lengths, negative_side.segment_lengths],
        axis=0,
    )
    total_length = jnp.sum(segment_lengths)
    path_key, alpha_key = random.split(sample_key)
    path_position = random.uniform(
        path_key,
        dtype=total_length.dtype,
        minval=jnp.asarray(0.0, total_length.dtype),
        maxval=total_length,
    )
    cumulative_lengths = jnp.cumsum(segment_lengths)
    segment_index = jnp.searchsorted(
        cumulative_lengths,
        path_position,
        side="right",
    )
    segment_index = jnp.minimum(segment_index, segment_lengths.shape[0] - 1)
    alpha = random.uniform(alpha_key, dtype=flat_u0.dtype)
    sampled_segment = (
        segment_ends[segment_index] - segment_starts[segment_index]
    )
    selected_point = (
        segment_starts[segment_index] + alpha * sampled_segment
    )
    selected_point = jnp.where(
        total_length > 0.0,
        selected_point,
        flat_u0,
    )
    next_step_size = jnp.where(
        segment_index < max_reflections,
        positive_side.step_size,
        negative_side.step_size,
    )
    next_step_size = jnp.where(
        total_length > 0.0,
        next_step_size,
        step_size0,
    )
    selected_inside, selected_log_likelihood, selected_evaluations = (
        strict_inside_and_log_likelihood(selected_point)
    )
    final_log_likelihood, final_evaluations = jax.lax.cond(
        selected_inside & jnp.isneginf(constraint),
        lambda _: (
            log_likelihood_flat(selected_point),
            jnp.ones((), mp_policy.count_dtype),
        ),
        lambda _: (
            selected_log_likelihood,
            jnp.zeros((), mp_policy.count_dtype),
        ),
        operand=None,
    )
    selected_point = jnp.where(
        selected_inside,
        selected_point,
        flat_u0,
    )
    log_likelihood = jnp.where(
        selected_inside,
        final_log_likelihood,
        jnp.asarray(log_L0, dtype=constraint.dtype),
    )
    point_tree = unravel_fn(selected_point)
    num_likelihood_evaluations = (
        positive_side.num_likelihood_evaluations
        + negative_side.num_likelihood_evaluations
        + selected_evaluations
        + final_evaluations
    )
    return (
        TreeField(point_tree),
        log_likelihood,
        num_likelihood_evaluations,
        next_step_size,
    )


def _sample_galilean_markov_transition_jax(
        *,
        key: PRNGKey,
        U0: TreeField[UType],
        log_L0: FloatArray,
        direction: TreeField[UType],
        log_L_constraint: FloatArray,
        log_likelihood_fn: Callable[[UType], FloatArray],
        grad_log_likelihood_fn: Callable[[UType], UType],
        initial_step_size: FloatArray,
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> tuple[TreeField[UType], FloatArray, IntArray]:
    """Sample one Galilean transition and discard the adapted step size."""
    point, log_likelihood, num_likelihood_evaluations, _ = (
        _sample_galilean_markov_transition_jax_with_step(
            key=key,
            U0=U0,
            log_L0=log_L0,
            direction=direction,
            log_L_constraint=log_L_constraint,
            log_likelihood_fn=log_likelihood_fn,
            grad_log_likelihood_fn=grad_log_likelihood_fn,
            initial_step_size=initial_step_size,
            max_reflections=max_reflections,
            max_step_halvings=max_step_halvings,
            max_step_doublings=max_step_doublings,
        )
    )
    return point, log_likelihood, num_likelihood_evaluations


@partial(
    jax.jit,
    static_argnames=[
        "log_likelihood_fn",
        "max_reflections",
        "max_step_halvings",
        "max_step_doublings",
    ],
)
def _sample_galilean_markov_transition_jit(
        key: PRNGKey,
        U0: TreeField[UType],
        log_L0: FloatArray,
        direction: TreeField[UType],
        log_L_constraint: FloatArray,
        initial_step_size: FloatArray,
        *,
        log_likelihood_fn: Callable[[UType], FloatArray],
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> tuple[TreeField[UType], FloatArray, IntArray]:
    """Cached JIT wrapper for repeated traced Galilean transitions."""
    return _sample_galilean_markov_transition_jax(
        key=key,
        U0=U0,
        log_L0=log_L0,
        direction=direction,
        log_L_constraint=log_L_constraint,
        log_likelihood_fn=log_likelihood_fn,
        grad_log_likelihood_fn=jax.grad(log_likelihood_fn),
        initial_step_size=initial_step_size,
        max_reflections=max_reflections,
        max_step_halvings=max_step_halvings,
        max_step_doublings=max_step_doublings,
    )


def _sample_galilean_chain_impl(
        *,
        key: PRNGKey,
        log_L_constraint: FloatArray,
        seed_point: SeedPoint,
        log_likelihood_fn: Callable[[UType], FloatArray],
        grad_log_likelihood_fn: Callable[[UType], UType],
        direction_kernel,
        direction_template: TreeField[UType],
        initial_step_size: FloatArray,
        num_slices: int,
        num_phantom: int,
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
    """Run a full Galilean Markov chain with JAX control flow."""

    class ChainCarry(NamedTuple):
        point: TreeField[UType]
        log_likelihood: FloatArray
        step_size: FloatArray
        num_likelihood_evaluations: IntArray

    def chain_body(carry: ChainCarry, keys) -> tuple[ChainCarry, FloatArray]:
        proposal_key, direction_key = keys
        direction = _sample_direction_from_kernel(
            key=direction_key,
            direction_kernel=direction_kernel,
            current_point=direction_template,
        )
        (
            next_point,
            next_log_likelihood,
            delta_evaluations,
            _next_step_size,
        ) = (
            _sample_galilean_markov_transition_jax_with_step(
                key=proposal_key,
                U0=carry.point,
                log_L0=carry.log_likelihood,
                direction=direction,
                log_L_constraint=log_L_constraint,
                log_likelihood_fn=log_likelihood_fn,
                grad_log_likelihood_fn=grad_log_likelihood_fn,
                initial_step_size=initial_step_size,
                max_reflections=max_reflections,
                max_step_halvings=max_step_halvings,
                max_step_doublings=max_step_doublings,
            )
        )
        next_carry = ChainCarry(
            point=next_point,
            log_likelihood=next_log_likelihood,
            step_size=initial_step_size,
            num_likelihood_evaluations=(
                carry.num_likelihood_evaluations + delta_evaluations
            ),
        )
        return next_carry, next_log_likelihood

    proposal_keys = random.split(key, num_slices)
    direction_keys = random.split(random.fold_in(key, 1), num_slices)
    final_carry, chain_log_likelihoods = jax.lax.scan(
        chain_body,
        ChainCarry(
            point=TreeField(seed_point.U0),
            log_likelihood=jnp.asarray(
                seed_point.log_L0,
                dtype=log_L_constraint.dtype,
            ),
            step_size=initial_step_size,
            num_likelihood_evaluations=jnp.asarray(
                0,
                mp_policy.count_dtype,
            ),
        ),
        (proposal_keys, direction_keys),
    )

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
        phantom_log_likelihoods = chain_log_likelihoods[
                                  phantom_start:-1
                                  ]
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
        final_carry.point.tree,
        final_carry.log_likelihood,
        final_carry.num_likelihood_evaluations,
        phantom_samples,
    )


@partial(
    jax.jit,
    static_argnames=[
        "log_likelihood_fn",
        "direction_kernel",
        "num_slices",
        "num_phantom",
        "max_reflections",
        "max_step_halvings",
        "max_step_doublings",
    ],
)
def _sample_galilean_chain_jit(
        key: PRNGKey,
        log_L_constraint: FloatArray,
        seed_point: SeedPoint,
        direction_template: TreeField[UType],
        initial_step_size: FloatArray,
        *,
        log_likelihood_fn: Callable[[UType], FloatArray],
        direction_kernel,
        num_slices: int,
        num_phantom: int,
        max_reflections: int,
        max_step_halvings: int,
        max_step_doublings: int,
) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
    """Cached JIT wrapper for a full Galilean chain."""
    return _sample_galilean_chain_impl(
        key=key,
        log_L_constraint=log_L_constraint,
        seed_point=seed_point,
        log_likelihood_fn=log_likelihood_fn,
        grad_log_likelihood_fn=jax.grad(log_likelihood_fn),
        direction_kernel=direction_kernel,
        direction_template=direction_template,
        initial_step_size=initial_step_size,
        num_slices=num_slices,
        num_phantom=num_phantom,
        max_reflections=max_reflections,
        max_step_halvings=max_step_halvings,
        max_step_doublings=max_step_doublings,
    )


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


def _new_proposal_python(
        *,
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
    Python-loop equivalent of ``_new_proposal`` for dispatched likelihoods.

    The likelihood callable is deliberately invoked on concrete proposal
    values. This keeps runtime-dispatched ``U -> log_L`` probes outside JAX
    tracing while retaining the same straight-line slice/shrinkage semantics.
    """
    if gradient_guided:
        raise NotImplementedError("Gradient guided slice sampler not implemented.")

    max_shrinkage_steps = _resolve_max_shrinkage_steps(max_shrinkage_steps)
    run_key, t_key, step_key, _ = random.split(key, 4)
    num_likelihood_evaluations = jnp.asarray(0, mp_policy.count_dtype)

    left_bound, right_bound = _slice_bounds(
        point_U0=U0,
        direction=direction,
    )
    slice_width = jnp.asarray(slice_width, left_bound.dtype)
    left = left_bound
    right = right_bound
    pick_point = partial(
        _pick_point_in_interval,
        point_U0=U0,
        direction=direction,
    )

    if not no_step_out:
        eps = jnp.asarray(1e-12, left_bound.dtype)
        use_full_slice = bool(np.asarray(jnp.isinf(slice_width)))
        effective_slice_width = jnp.maximum(slice_width, eps)
        place_key, step_key = random.split(step_key)
        uniform_origin = random.uniform(place_key, dtype=left_bound.dtype)
        initial_left = -uniform_origin * effective_slice_width
        initial_right = initial_left + effective_slice_width
        if use_full_slice:
            left = left_bound
            right = right_bound
        else:
            left = jnp.maximum(left_bound, initial_left)
            right = jnp.minimum(right_bound, initial_right)

        def _point_at_t(t: FloatArray) -> TreeField[UType]:
            return U0 + direction * t

        def _outside_slice(t: FloatArray) -> bool:
            nonlocal num_likelihood_evaluations
            log_likelihood = log_likelihood_fn(_point_at_t(t).tree)
            num_likelihood_evaluations += jnp.ones(
                (),
                mp_policy.count_dtype,
            )
            return bool(np.asarray(log_likelihood <= log_L_constraint))

        left_outside_slice = _outside_slice(left)
        right_outside_slice = _outside_slice(right)
        while (
                not use_full_slice
                and (
                    (bool(np.asarray(left > left_bound)))
                    or (bool(np.asarray(right < right_bound)))
                )
                and not (left_outside_slice and right_outside_slice)
        ):
            step_key, choose_key = random.split(step_key, 2)
            can_expand_left = bool(np.asarray(left > left_bound))
            can_expand_right = bool(np.asarray(right < right_bound))
            if can_expand_left and can_expand_right:
                choose_left = bool(
                    np.asarray(
                        random.uniform(choose_key, dtype=left_bound.dtype)
                        < 0.5
                    )
                )
            else:
                choose_left = can_expand_left
            current_width = jnp.maximum(right - left, eps)
            if choose_left:
                candidate_left = jnp.maximum(left_bound, left - current_width)
                left = candidate_left
                left_outside_slice = _outside_slice(candidate_left)
            else:
                candidate_right = jnp.minimum(
                    right_bound,
                    right + current_width,
                )
                right = candidate_right
                right_outside_slice = _outside_slice(candidate_right)

    point_U, t = pick_point(
        key=t_key,
        left=left,
        right=right,
    )
    log_L = jnp.asarray(
        log_likelihood_fn(point_U.tree),
        dtype=log_L_constraint.dtype,
    )
    num_likelihood_evaluations += jnp.ones((), mp_policy.count_dtype)

    if bool(np.asarray(log_L > log_L_constraint)):
        proposal = point_U
        proposal_log_L = log_L
        left_after = left
        right_after = right
    else:
        current_left = left
        current_right = right
        rejected_t = t
        proposal = U0
        proposal_satisfied = False
        left_after = left
        right_after = right
        for _ in range(max_shrinkage_steps):
            current_left, current_right = _shrink_interval(
                t=rejected_t,
                left=current_left,
                right=current_right,
            )
            run_key, proposal_key = random.split(run_key, 2)
            candidate_U, candidate_t = pick_point(
                key=proposal_key,
                left=current_left,
                right=current_right,
            )
            candidate_log_L = jnp.asarray(
                log_likelihood_fn(candidate_U.tree),
                dtype=log_L_constraint.dtype,
            )
            num_likelihood_evaluations += jnp.ones(
                (),
                mp_policy.count_dtype,
            )
            left_after = current_left
            right_after = current_right
            if bool(np.asarray(candidate_log_L > log_L_constraint)):
                proposal = candidate_U
                proposal_log_L = candidate_log_L
                proposal_satisfied = True
                break
            rejected_t = candidate_t
        if not proposal_satisfied:
            if log_L0 is None:
                proposal_log_L = jnp.asarray(
                    log_likelihood_fn(U0.tree),
                    dtype=log_L_constraint.dtype,
                )
                num_likelihood_evaluations += jnp.ones(
                    (),
                    mp_policy.count_dtype,
                )
            else:
                proposal_log_L = jnp.asarray(
                    log_L0,
                    dtype=log_L_constraint.dtype,
                )

    next_slice_width = 2 * (right_after - left_after)
    return (
        proposal,
        proposal_log_L,
        num_likelihood_evaluations,
        direction,
        next_slice_width,
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


def _materialize_split_keys(key: PRNGKey, count: int) -> tuple[PRNGKey, ...]:
    """Split and host-materialize keys for forced Python loops."""
    resolved = operator.index(count)
    if resolved < 0:
        raise ValueError(f"count must be non-negative, got {count}.")
    if resolved == 0:
        return ()
    return tuple(jax.device_get(random.split(key, resolved)))


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
        dispatch_log_likelihood_fn = _context_any(
            adaptation_context,
            (
                "proposal_log_likelihood_fn",
                "dispatch_log_likelihood_fn",
                "log_likelihood_fn",
            ),
        )
        log_likelihood_fn = (
            _StaticLogLikelihoodFn(self.model, args, params)
            if dispatch_log_likelihood_fn is None
            else dispatch_log_likelihood_fn
        )
        force_python_loop = bool(
            _context_get(adaptation_context, "force_python_loop", False)
            or dispatch_log_likelihood_fn is not None
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
            force_jax_galilean = (
                    _context_get(
                        adaptation_context,
                        "force_jax_galilean",
                        False,
                    )
                    is True
            )
            use_jax_galilean = (
                    not force_python_loop
                    and (
                        force_jax_galilean
                        or _has_traced_leaf(
                            (
                                key,
                                log_L_constraint,
                                seed_point.U0,
                                seed_point.log_L0,
                            )
                        )
                    )
            )
            if use_jax_galilean:
                step_size_dtype = jax.tree.leaves(seed_point.U0)[0].dtype
                initial_step_size = jnp.asarray(
                    self.galilean_initial_step_size,
                    dtype=step_size_dtype,
                )
                num_phantom = self.num_phantom()
                if _as_mode_name(direction_kernel) in ISOTROPIC_DIRECTION_KERNELS:
                    return _sample_galilean_chain_jit(
                        key,
                        log_L_constraint,
                        seed_point,
                        direction_template,
                        initial_step_size,
                        log_likelihood_fn=log_likelihood_fn,
                        direction_kernel=direction_kernel,
                        num_slices=num_slices,
                        num_phantom=num_phantom,
                        max_reflections=self.max_galilean_reflections,
                        max_step_halvings=(
                            self.max_galilean_step_halvings
                        ),
                        max_step_doublings=(
                            self.max_galilean_step_doublings
                        ),
                    )
                return _sample_galilean_chain_impl(
                    key=key,
                    log_L_constraint=log_L_constraint,
                    seed_point=seed_point,
                    log_likelihood_fn=log_likelihood_fn,
                    grad_log_likelihood_fn=grad_log_likelihood_fn,
                    direction_kernel=direction_kernel,
                    direction_template=direction_template,
                    initial_step_size=initial_step_size,
                    num_slices=num_slices,
                    num_phantom=num_phantom,
                    max_reflections=self.max_galilean_reflections,
                    max_step_halvings=self.max_galilean_step_halvings,
                    max_step_doublings=self.max_galilean_step_doublings,
                )

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

        def apply_proposal(carry: Carry, proposal_key: PRNGKey):
            if force_python_loop:
                return _new_proposal_python(
                    key=proposal_key,
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
            return _new_proposal(
                key=proposal_key,
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

        def update_carry(
                carry: Carry,
                proposal_result,
                direction: TreeField[UType],
        ) -> Carry:
            U_sample, log_L, num_likelihood_evaluations, _, slice_width = (
                proposal_result
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

        def propose_op(carry: Carry, x: XType) -> Carry:
            proposal_result = apply_proposal(carry, x.key)
            direction = _sample_direction_from_kernel(
                key=x.direction_key,
                direction_kernel=direction_kernel,
                current_point=direction_template,
            )
            return update_carry(carry, proposal_result, direction)

        def propose_op_without_next_direction(carry: Carry, x: XType) -> Carry:
            proposal_result = apply_proposal(carry, x.key)
            return update_carry(carry, proposal_result, carry.direction)

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

        if force_python_loop:
            init_proposal_result = _new_proposal_python(
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
        else:
            init_proposal_result = _new_proposal(
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
        U_sample, log_L, num_likelihood_evaluations, _, slice_width = (
            init_proposal_result
        )
        if num_slices > 1 or not force_python_loop:
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

        loop_count = num_slices - 1
        proposal_key, direction_scan_key = random.split(sample_key, 2)
        if force_python_loop:
            proposal_keys = _materialize_split_keys(proposal_key, loop_count)
            consumed_direction_count = max(loop_count - 1, 0)
            direction_keys = _materialize_split_keys(
                direction_scan_key,
                consumed_direction_count,
            )
            carry = init_carry
            samples = []
            for i, loop_proposal_key in enumerate(proposal_keys):
                proposal_result = apply_proposal(carry, loop_proposal_key)
                if i == loop_count - 1:
                    direction = carry.direction
                else:
                    loop_direction_key = direction_keys[i]
                    direction = _sample_direction_from_kernel(
                        key=loop_direction_key,
                        direction_kernel=direction_kernel,
                        current_point=direction_template,
                    )
                carry = update_carry(
                    carry=carry,
                    proposal_result=proposal_result,
                    direction=direction,
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
            xs = XType(
                key=random.split(proposal_key, loop_count),
                direction_key=random.split(direction_scan_key, loop_count),
                alpha=jnp.linspace(0.5, 1., loop_count)
            )
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
