from __future__ import annotations

import dataclasses
import inspect
import multiprocessing
from typing import NamedTuple

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

import jaxns.constrained_sampler as constrained_sampler
from jaxns.constrained_sampler import (
    UniDimSliceSampler,
    _new_proposal,
    _sample_direction,
    _shrink_interval,
    _slice_bounds,
)
from jaxns.constrained_sampler_distributed import DistributedUniDimSliceSampler
from jaxns.pytree import TreeField
from jaxns.samples import SeedPoint
from tests.distributed_support import QuadraticEvaluator, make_toy_model


ISOTROPIC_MODE_NAMES = {"isotropic", "isotropic_gaussian"}
STRAIGHT_LINE_MODE_NAMES = {"straight_line", "straight_line_perfect"}


class UnsupportedModeCase(NamedTuple):
    name: str
    kwargs: dict[str, object]


class EllipsoidalAdaptationContext(NamedTuple):
    samples_U: jnp.ndarray
    log_likelihoods: jnp.ndarray
    component_means: jnp.ndarray
    component_radii: jnp.ndarray
    component_rotations: jnp.ndarray
    component_integrated_volumes: jnp.ndarray
    valid_mask: jnp.ndarray


class SampleHistoryOnlyEllipsoidalAdaptationContext(NamedTuple):
    samples_U: jnp.ndarray
    log_likelihoods: jnp.ndarray
    valid_mask: jnp.ndarray
    max_num_components: int


@dataclasses.dataclass(slots=True)
class MutableEllipsoidalAdaptationContext:
    samples_U: jnp.ndarray
    log_likelihoods: jnp.ndarray
    component_means: jnp.ndarray
    component_radii: jnp.ndarray
    component_rotations: jnp.ndarray
    component_integrated_volumes: jnp.ndarray
    valid_mask: jnp.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenTestDirectionKernel:
    component_probabilities: tuple[float, ...]
    direction: jnp.ndarray


class ShrinkCase(NamedTuple):
    name: str
    t: float
    expected_left: float
    expected_right: float


def _mode_name(value: object) -> str:
    if isinstance(value, str):
        return value
    for attr_name in ("mode", "name", "value"):
        if hasattr(value, attr_name):
            attr_value = getattr(value, attr_name)
            if isinstance(attr_value, str):
                return attr_value
            if hasattr(attr_value, "value"):
                return str(attr_value.value)
            return str(attr_value)
    return type(value).__name__


def _assert_immutable_direction_config(value: object) -> None:
    if dataclasses.is_dataclass(value):
        assert value.__dataclass_params__.frozen
        return
    if isinstance(value, str):
        return
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return
    pytest.fail(
        "direction_kernel must be stored as an immutable, snapshotted "
        "configuration, not a mutable object"
    )


def _required_sampler_symbol(name: str):
    symbol = getattr(constrained_sampler, name, None)
    if symbol is None:
        pytest.fail(
            f"jaxns.constrained_sampler.{name} is required by Ticket 0008 "
            "direction-kernel and trajectory tests."
        )
    return symbol


def _ellipsoidal_context(
        volumes: tuple[float, float],
) -> EllipsoidalAdaptationContext:
    return EllipsoidalAdaptationContext(
        samples_U=jnp.asarray(
            [
                [0.15, 0.20],
                [0.20, 0.25],
                [0.80, 0.75],
                [0.85, 0.80],
            ]
        ),
        log_likelihoods=jnp.asarray([0.1, 0.2, 0.3, 0.4]),
        component_means=jnp.asarray(
            [
                [0.20, 0.25],
                [0.80, 0.75],
            ]
        ),
        component_radii=jnp.asarray(
            [
                [0.08, 0.20],
                [0.25, 0.10],
            ]
        ),
        component_rotations=jnp.stack(
            [
                jnp.eye(2),
                jnp.asarray([[0.0, -1.0], [1.0, 0.0]]),
            ]
        ),
        component_integrated_volumes=jnp.asarray(volumes),
        valid_mask=jnp.asarray([True, True, True, True]),
    )


def _mutable_ellipsoidal_context(
        volumes: tuple[float, float],
) -> MutableEllipsoidalAdaptationContext:
    context = _ellipsoidal_context(volumes)
    return MutableEllipsoidalAdaptationContext(
        samples_U=context.samples_U,
        log_likelihoods=context.log_likelihoods,
        component_means=context.component_means,
        component_radii=context.component_radii,
        component_rotations=context.component_rotations,
        component_integrated_volumes=context.component_integrated_volumes,
        valid_mask=context.valid_mask,
    )


def _component_probabilities(kernel) -> np.ndarray:
    for attr_name in (
            "component_probabilities",
            "component_weights",
            "mixture_weights",
            "weights",
    ):
        if hasattr(kernel, attr_name):
            return np.asarray(getattr(kernel, attr_name), dtype=float)
    if isinstance(kernel, dict):
        for key in (
                "component_probabilities",
                "component_weights",
                "mixture_weights",
                "weights",
        ):
            if key in kernel:
                return np.asarray(kernel[key], dtype=float)
    pytest.fail(
        "ellipsoidal direction kernel must expose normalized component "
        "selection probabilities."
    )


def _direction_array(sample) -> np.ndarray:
    for attr_name in ("direction", "n_hat", "value"):
        if hasattr(sample, attr_name):
            return np.asarray(getattr(sample, attr_name), dtype=float)
    if isinstance(sample, dict):
        for key in ("direction", "n_hat", "value"):
            if key in sample:
                return np.asarray(sample[key], dtype=float)
    if hasattr(sample, "tree"):
        return np.asarray(sample.tree, dtype=float)
    return np.asarray(sample, dtype=float)


def _point_array(value) -> np.ndarray:
    if hasattr(value, "tree"):
        value = value.tree
    return np.asarray(value, dtype=float)


def _greedy_result_point(result) -> np.ndarray:
    for attr_name in ("proposal", "point", "U", "U_sample"):
        if hasattr(result, attr_name):
            return _point_array(getattr(result, attr_name))
    if isinstance(result, dict):
        for key in ("proposal", "point", "U", "U_sample"):
            if key in result:
                return _point_array(result[key])
    if isinstance(result, tuple) and result:
        return _point_array(result[0])
    pytest.fail("greedy shrinkage result must expose the returned point.")


def _greedy_result_log_likelihood(result) -> float:
    for attr_name in ("log_likelihood", "log_L", "log_L_sample"):
        if hasattr(result, attr_name):
            return float(getattr(result, attr_name))
    if isinstance(result, dict):
        for key in ("log_likelihood", "log_L", "log_L_sample"):
            if key in result:
                return float(result[key])
    if isinstance(result, tuple) and len(result) > 1:
        return float(result[1])
    pytest.fail("greedy shrinkage result must expose the returned likelihood.")


def _greedy_result_used_current_point_fallback(result) -> bool:
    for attr_name in (
            "used_current_point_fallback",
            "current_point_fallback",
            "fallback_to_current_point",
    ):
        if hasattr(result, attr_name):
            return bool(getattr(result, attr_name))
    if isinstance(result, dict):
        for key in (
                "used_current_point_fallback",
                "current_point_fallback",
                "fallback_to_current_point",
        ):
            if key in result:
                return bool(result[key])
    pytest.fail(
        "greedy shrinkage result must say whether the current-point fallback "
        "was used."
    )


def _greedy_result_num_steps(result) -> int:
    for attr_name in ("num_shrinkage_steps", "shrinkage_steps", "num_steps"):
        if hasattr(result, attr_name):
            return int(getattr(result, attr_name))
    if isinstance(result, dict):
        for key in ("num_shrinkage_steps", "shrinkage_steps", "num_steps"):
            if key in result:
                return int(result[key])
    pytest.fail(
        "greedy shrinkage result must expose the termination step count."
    )


def _greedy_trace_value(result, names: tuple[str, ...]) -> np.ndarray:
    containers = [result]
    for container_name in ("diagnostics", "trace", "shrinkage_trace"):
        if hasattr(result, container_name):
            containers.append(getattr(result, container_name))
        if isinstance(result, dict) and container_name in result:
            containers.append(result[container_name])

    for container in containers:
        for name in names:
            if hasattr(container, name):
                return np.asarray(getattr(container, name), dtype=float)
            if isinstance(container, dict) and name in container:
                return np.asarray(container[name], dtype=float)

    pytest.fail(
        "greedy shrinkage result must expose contraction trace values for "
        f"one of {names}."
    )


def _greedy_result_accepted_t(result) -> float:
    return float(
        _greedy_trace_value(
            result,
            (
                "accepted_t",
                "accepted_offset",
                "final_t",
                "final_offset",
            ),
        )
    )


def _always_inside_log_likelihood(u):
    del u
    return jnp.asarray(1.0)


def _narrow_slice_log_likelihood(u):
    u = jnp.asarray(u)
    return -jnp.abs(u[0] - 0.5)


def _only_current_point_is_inside_log_likelihood(u):
    u = jnp.asarray(u)
    return jnp.where(jnp.isclose(u[0], 0.5), 1.0, -1.0)


def _strictly_only_current_point_is_inside_log_likelihood(u):
    u = jnp.asarray(u)
    return jnp.where(u[0] == jnp.asarray(0.5, u.dtype), 1.0, -1.0)


def _inner_strict_window_log_likelihood(u):
    u = jnp.asarray(u)
    return jnp.where(jnp.abs(u[0] - 0.5) < 0.04, 1.0, -1.0)


def _two_mode_log_likelihood(u):
    u = jnp.asarray(u)
    x = u[0]
    in_left_mode = jnp.abs(x - 0.2) < 0.04
    in_right_mode = jnp.abs(x - 0.8) < 0.04
    return jnp.where(jnp.logical_or(in_left_mode, in_right_mode), 1.0, -1.0)


def _new_proposal_strict_current_fallback_worker(queue) -> None:
    try:
        proposal, log_likelihood, num_evals, _, _ = _new_proposal(
            key=random.PRNGKey(101),
            U0=TreeField(jnp.asarray([0.5])),
            direction=TreeField(jnp.asarray([1.0])),
            slice_width=jnp.asarray(jnp.inf),
            no_step_out=True,
            gradient_guided=False,
            log_L_constraint=jnp.asarray(0.0),
            log_likelihood_fn=_strictly_only_current_point_is_inside_log_likelihood,
        )
        queue.put(
            (
                "ok",
                np.asarray(proposal.tree, dtype=float),
                float(log_likelihood),
                int(num_evals),
            )
        )
    except BaseException as error:
        queue.put(("error", type(error).__name__, str(error)))


def test_isotropic_directions_are_unit_vectors_with_symmetric_signs():
    point_template = TreeField(jnp.zeros(3, dtype=jnp.float32))
    keys = random.split(random.PRNGKey(12), 2048)

    directions = np.asarray(
        [
            _sample_direction(key, point_template).tree
            for key in keys
        ]
    )

    assert directions.shape == (2048, 3)
    assert np.all(np.isfinite(directions))
    np.testing.assert_allclose(
        np.linalg.norm(directions, axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.all(np.abs(np.mean(directions, axis=0)) < 0.05)
    positive_fraction = np.mean(directions > 0.0, axis=0)
    assert np.all(positive_fraction > 0.47)
    assert np.all(positive_fraction < 0.53)


def test_isotropic_1d_direction_is_symmetric_across_prng_keys():
    point_template = TreeField(jnp.zeros(1, dtype=jnp.float32))
    keys = random.split(random.PRNGKey(1208), 2048)

    directions = np.asarray(
        [
            _sample_direction(key, point_template).tree[0]
            for key in keys
        ],
        dtype=float,
    )

    assert directions.shape == (2048,)
    assert set(np.unique(directions)) == {-1.0, 1.0}
    positive_fraction = float(np.mean(directions > 0.0))
    assert 0.47 < positive_fraction < 0.53


def test_isotropic_direction_is_independent_of_current_chain_point():
    key = random.PRNGKey(3)
    first_point = TreeField(jnp.asarray([0.10, 0.25, 0.90]))
    second_point = TreeField(jnp.asarray([0.95, 0.50, 0.05]))

    first_direction = _sample_direction(key, first_point)
    second_direction = _sample_direction(key, second_point)

    np.testing.assert_allclose(first_direction.tree, second_direction.tree)


def test_straight_line_bracket_is_maximal_unit_cube_intersection():
    point = TreeField(jnp.asarray([0.20, 0.55, 0.80]))
    raw_direction = jnp.asarray([0.75, -0.50, 0.25])
    direction = TreeField(raw_direction / jnp.linalg.norm(raw_direction))

    left, right = _slice_bounds(point, direction)
    left_endpoint = np.asarray((point + direction * left).tree)
    right_endpoint = np.asarray((point + direction * right).tree)

    assert float(left) < 0.0
    assert float(right) > 0.0
    assert np.all(left_endpoint >= -1e-7)
    assert np.all(left_endpoint <= 1.0 + 1e-7)
    assert np.all(right_endpoint >= -1e-7)
    assert np.all(right_endpoint <= 1.0 + 1e-7)
    assert np.any(np.isclose(left_endpoint, 0.0, atol=1e-7))
    assert np.any(np.isclose(right_endpoint, 1.0, atol=1e-7))

    epsilon = jnp.asarray(1e-4, left.dtype)
    before_left = np.asarray((point + direction * (left - epsilon)).tree)
    after_right = np.asarray((point + direction * (right + epsilon)).tree)
    assert np.any((before_left < 0.0) | (before_left > 1.0))
    assert np.any((after_right < 0.0) | (after_right > 1.0))


def test_straight_line_proposal_uses_no_endpoint_likelihood_evaluations():
    point = TreeField(jnp.asarray([0.25, 0.50, 0.75]))
    raw_direction = jnp.asarray([0.40, 0.60, -0.30])
    direction = TreeField(raw_direction / jnp.linalg.norm(raw_direction))

    proposal, log_likelihood, num_evals, _, _ = _new_proposal(
        key=random.PRNGKey(8),
        U0=point,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=True,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(0.0),
        log_likelihood_fn=_always_inside_log_likelihood,
    )

    assert np.all(np.asarray(proposal.tree) >= 0.0)
    assert np.all(np.asarray(proposal.tree) <= 1.0)
    assert float(log_likelihood) > 0.0
    assert int(num_evals) == 1


def test_straight_line_sampler_explicitly_surfaces_mode_tunneling_risk():
    point = TreeField(jnp.asarray([0.20]))
    direction = TreeField(jnp.asarray([1.0]))

    left, right = _slice_bounds(point, direction)
    barrier_point = (point + direction * jnp.asarray(0.30)).tree
    proposal, log_likelihood, num_evals, _, _ = _new_proposal(
        key=random.PRNGKey(1),
        U0=point,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=True,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(0.0),
        log_likelihood_fn=_two_mode_log_likelihood,
    )

    assert float(left) == pytest.approx(-0.20)
    assert float(right) == pytest.approx(0.80)
    assert float(_two_mode_log_likelihood(barrier_point)) <= 0.0
    assert 0.76 < float(proposal.tree[0]) < 0.84
    assert float(log_likelihood) > 0.0
    assert int(num_evals) == 1


def test_sampler_accepts_explicit_baseline_direction_and_trajectory_modes():
    sampler = UniDimSliceSampler(
        model=make_toy_model(),
        num_slices=2,
        direction_kernel="isotropic",
        trajectory="straight_line",
    )

    assert _mode_name(sampler.direction_kernel) in ISOTROPIC_MODE_NAMES
    assert _mode_name(sampler.trajectory) in STRAIGHT_LINE_MODE_NAMES


def test_direction_kernel_configuration_is_snapshotted_and_immutable():
    sampler = UniDimSliceSampler(
        model=make_toy_model(),
        num_slices=2,
        direction_kernel="isotropic",
        trajectory="straight_line",
    )

    direction_config = sampler.direction_kernel

    assert _mode_name(direction_config) in ISOTROPIC_MODE_NAMES
    _assert_immutable_direction_config(direction_config)


def test_ellipsoidal_component_selection_uses_integrated_volume_weights():
    build_kernel = _required_sampler_symbol(
        "_build_ellipsoidal_direction_kernel"
    )
    select_component = _required_sampler_symbol("_sample_direction_component")
    context = _ellipsoidal_context(volumes=(1.0, 3.0))
    kernel = build_kernel(adaptation_context=context)
    probabilities = _component_probabilities(kernel)

    np.testing.assert_allclose(
        probabilities,
        np.asarray([0.25, 0.75]),
        rtol=1e-6,
        atol=1e-6,
    )

    keys = random.split(random.PRNGKey(28), 3000)
    selected = np.asarray(
        [
            int(select_component(key=key, direction_kernel=kernel))
            for key in keys
        ]
    )

    assert selected.shape == (3000,)
    second_component_fraction = float(np.mean(selected == 1))
    assert 0.72 < second_component_fraction < 0.78


def test_ellipsoidal_kernel_build_requires_sampler_adaptation_context():
    build_kernel = _required_sampler_symbol(
        "_build_ellipsoidal_direction_kernel"
    )
    context = _ellipsoidal_context(volumes=(2.0, 1.0))
    kernel = build_kernel(adaptation_context=context)
    probabilities = _component_probabilities(kernel)

    np.testing.assert_allclose(
        probabilities,
        np.asarray([2.0 / 3.0, 1.0 / 3.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_immutable_direction_config(kernel)

    sampler = UniDimSliceSampler(
        model=make_toy_model(),
        num_slices=2,
        direction_kernel="ellipsoidal",
        trajectory="straight_line",
    )
    assert _mode_name(sampler.direction_kernel) in {
        "ellipsoidal",
        "ellipsoidal_gaussian",
    }
    get_sample_signature = inspect.signature(sampler.get_sample)
    assert "adaptation_context" in get_sample_signature.parameters


def test_ellipsoidal_kernel_can_freeze_from_sample_history_only_context():
    build_kernel = _required_sampler_symbol(
        "_build_ellipsoidal_direction_kernel"
    )
    sample_direction = _required_sampler_symbol(
        "_sample_direction_from_kernel"
    )
    context = SampleHistoryOnlyEllipsoidalAdaptationContext(
        samples_U=jnp.asarray(
            [
                [0.12, 0.18],
                [0.16, 0.21],
                [0.20, 0.24],
                [0.76, 0.78],
                [0.81, 0.82],
                [0.86, 0.85],
            ],
            dtype=jnp.float32,
        ),
        log_likelihoods=jnp.asarray(
            [0.10, 0.15, 0.20, 0.30, 0.35, 0.40],
            dtype=jnp.float32,
        ),
        valid_mask=jnp.asarray([True, True, True, True, True, True]),
        max_num_components=2,
    )
    for precomputed_attr in (
            "component_means",
            "component_radii",
            "component_rotations",
            "component_integrated_volumes",
    ):
        assert not hasattr(context, precomputed_attr)

    kernel = build_kernel(adaptation_context=context)
    probabilities = _component_probabilities(kernel)

    assert probabilities.ndim == 1
    assert probabilities.size >= 1
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=1e-6)
    _assert_immutable_direction_config(kernel)

    direction = _direction_array(
        sample_direction(
            key=random.PRNGKey(94),
            direction_kernel=kernel,
            current_point=TreeField(jnp.asarray([0.25, 0.75])),
        )
    )
    assert direction.shape == (2,)
    assert np.all(np.isfinite(direction))
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-5)


def test_ellipsoidal_direction_kernel_is_frozen_for_entire_chain(monkeypatch):
    _required_sampler_symbol("_build_ellipsoidal_direction_kernel")
    _required_sampler_symbol("_sample_direction_from_kernel")
    context = _mutable_ellipsoidal_context(volumes=(9.0, 1.0))
    build_probabilities: list[tuple[float, ...]] = []
    sampled_probabilities: list[tuple[float, ...]] = []
    sampled_points: list[np.ndarray] = []

    def fake_build_kernel(*, adaptation_context):
        volumes = np.asarray(
            adaptation_context.component_integrated_volumes,
            dtype=float,
        )
        probabilities = tuple(
            float(value) for value in volumes / volumes.sum()
        )
        build_probabilities.append(probabilities)

        adaptation_context.component_integrated_volumes = jnp.asarray(
            [1.0, 9.0]
        )
        return FrozenTestDirectionKernel(
            component_probabilities=probabilities,
            direction=jnp.asarray([1.0]),
        )

    def fake_sample_direction_from_kernel(
            *,
            key,
            direction_kernel,
            current_point,
    ):
        del key
        sampled_probabilities.append(direction_kernel.component_probabilities)
        sampled_points.append(_point_array(current_point))
        return TreeField(direction_kernel.direction)

    monkeypatch.setattr(
        constrained_sampler,
        "_build_ellipsoidal_direction_kernel",
        fake_build_kernel,
        raising=False,
    )
    monkeypatch.setattr(
        constrained_sampler,
        "_sample_direction_from_kernel",
        fake_sample_direction_from_kernel,
        raising=False,
    )

    sampler = UniDimSliceSampler(
        model=make_toy_model(),
        num_slices=4,
        direction_kernel="ellipsoidal",
        trajectory="straight_line",
    )
    sampler.get_sample(
        key=random.PRNGKey(91),
        log_L_constraint=jnp.asarray(-0.20),
        seed_point=SeedPoint(
            U0=jnp.asarray([0.5]),
            log_L0=jnp.asarray(0.0),
        ),
        adaptation_context=context,
    )

    np.testing.assert_allclose(np.asarray(build_probabilities), [[0.9, 0.1]])
    np.testing.assert_allclose(
        np.asarray(context.component_integrated_volumes),
        np.asarray([1.0, 9.0]),
    )
    assert len(sampled_probabilities) >= 2
    np.testing.assert_allclose(
        np.asarray(sampled_probabilities),
        np.tile(np.asarray([0.9, 0.1]), (len(sampled_probabilities), 1)),
    )
    assert all(point.shape == (1,) for point in sampled_points)


def test_configured_ellipsoidal_direction_is_independent_of_current_point():
    build_kernel = _required_sampler_symbol(
        "_build_ellipsoidal_direction_kernel"
    )
    sample_direction = _required_sampler_symbol(
        "_sample_direction_from_kernel"
    )
    kernel = build_kernel(adaptation_context=_ellipsoidal_context((1.0, 1.0)))

    first_direction = _direction_array(
        sample_direction(
            key=random.PRNGKey(12),
            direction_kernel=kernel,
            current_point=TreeField(jnp.asarray([0.10, 0.90])),
        )
    )
    second_direction = _direction_array(
        sample_direction(
            key=random.PRNGKey(12),
            direction_kernel=kernel,
            current_point=TreeField(jnp.asarray([0.90, 0.10])),
        )
    )

    np.testing.assert_allclose(first_direction, second_direction)


def test_ellipsoidal_kernel_is_frozen_against_mid_chain_adaptation_updates():
    build_kernel = _required_sampler_symbol(
        "_build_ellipsoidal_direction_kernel"
    )
    select_component = _required_sampler_symbol("_sample_direction_component")
    first_kernel = build_kernel(
        adaptation_context=_ellipsoidal_context(volumes=(9.0, 1.0))
    )
    second_kernel = build_kernel(
        adaptation_context=_ellipsoidal_context(volumes=(1.0, 9.0))
    )

    np.testing.assert_allclose(
        _component_probabilities(first_kernel),
        np.asarray([0.9, 0.1]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _component_probabilities(second_kernel),
        np.asarray([0.1, 0.9]),
        rtol=1e-6,
        atol=1e-6,
    )

    keys = random.split(random.PRNGKey(41), 2000)
    first_selected = np.asarray(
        [
            int(select_component(key=key, direction_kernel=first_kernel))
            for key in keys
        ]
    )
    second_selected = np.asarray(
        [
            int(select_component(key=key, direction_kernel=second_kernel))
            for key in keys
        ]
    )

    assert float(np.mean(first_selected == 0)) > 0.86
    assert float(np.mean(second_selected == 1)) > 0.86


@pytest.mark.parametrize(
    "case",
    (
        ShrinkCase(
            name="rejected_left_proposal_moves_left_boundary",
            t=-0.25,
            expected_left=-0.25,
            expected_right=1.0,
        ),
        ShrinkCase(
            name="rejected_right_proposal_moves_right_boundary",
            t=0.40,
            expected_left=-1.0,
            expected_right=0.40,
        ),
    ),
    ids=lambda case: case.name,
)
def test_greedy_shrinkage_uses_rejected_point_as_nearest_boundary(case):
    left, right = _shrink_interval(
        t=jnp.asarray(case.t),
        left=jnp.asarray(-1.0),
        right=jnp.asarray(1.0),
    )

    np.testing.assert_allclose(float(left), case.expected_left)
    np.testing.assert_allclose(float(right), case.expected_right)
    assert float(left) <= 0.0 <= float(right)


def test_greedy_shrinkage_documented_step_sequence_keeps_current_bracket():
    rejected_offsets = (0.75, -0.50, 0.25)
    expected_bounds = (
        (-1.0, 0.75),
        (-0.50, 0.75),
        (-0.50, 0.25),
    )
    left = jnp.asarray(-1.0)
    right = jnp.asarray(1.0)

    for rejected_t, expected_bound in zip(rejected_offsets, expected_bounds):
        previous_width = float(right - left)
        left, right = _shrink_interval(
            t=jnp.asarray(rejected_t),
            left=left,
            right=right,
        )

        np.testing.assert_allclose(
            np.asarray([float(left), float(right)]),
            np.asarray(expected_bound),
        )
        assert float(left) <= 0.0 <= float(right)
        assert float(right - left) < previous_width

    accepted_t = jnp.asarray(0.10)
    accepted_point = jnp.asarray([0.5]) + jnp.asarray([1.0]) * accepted_t
    assert float(left) < float(accepted_t) < float(right)
    assert float(_narrow_slice_log_likelihood(accepted_point)) > -0.2


def test_greedy_shrinkage_keeps_current_point_as_fallback():
    left = jnp.asarray(-1.0)
    right = jnp.asarray(1.0)

    for rejected_t in (-0.75, 0.50, -0.10, 0.02):
        previous_width = float(right - left)
        left, right = _shrink_interval(
            t=jnp.asarray(rejected_t),
            left=left,
            right=right,
        )

        assert float(left) <= 0.0 <= float(right)
        assert float(right - left) < previous_width


def test_greedy_shrinkage_exhaustion_returns_current_point_fallback():
    shrink_to_strict = _required_sampler_symbol(
        "_greedy_shrink_to_strict_contour"
    )
    point = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    result = shrink_to_strict(
        key=random.PRNGKey(17),
        U0=point,
        direction=direction,
        left=jnp.asarray(-0.5),
        right=jnp.asarray(0.5),
        log_L0=jnp.asarray(1.0),
        log_L_constraint=jnp.asarray(0.0),
        log_likelihood_fn=_only_current_point_is_inside_log_likelihood,
        max_steps=4,
    )

    np.testing.assert_allclose(_greedy_result_point(result), np.asarray([0.5]))
    np.testing.assert_allclose(_greedy_result_log_likelihood(result), 1.0)
    assert _greedy_result_used_current_point_fallback(result) is True
    assert _greedy_result_num_steps(result) == 4


def test_greedy_strict_path_exponentially_contracts_until_acceptance():
    shrink_to_strict = _required_sampler_symbol(
        "_greedy_shrink_to_strict_contour"
    )
    point = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))
    proposal_t_sequence = jnp.asarray([0.40, -0.20, 0.10, -0.05, 0.025])

    result = shrink_to_strict(
        key=random.PRNGKey(23),
        U0=point,
        direction=direction,
        left=jnp.asarray(-0.5),
        right=jnp.asarray(0.5),
        log_L0=jnp.asarray(1.0),
        log_L_constraint=jnp.asarray(0.0),
        log_likelihood_fn=_inner_strict_window_log_likelihood,
        max_steps=proposal_t_sequence.shape[0],
        proposal_t_sequence=proposal_t_sequence,
    )

    rejected_t = np.ravel(
        _greedy_trace_value(
            result,
            (
                "rejected_t",
                "rejected_ts",
                "rejected_offsets",
                "rejected_proposal_ts",
            ),
        )
    )
    left_bounds = np.ravel(
        _greedy_trace_value(
            result,
            ("left_bounds", "left_trace", "bracket_left_trace"),
        )
    )
    right_bounds = np.ravel(
        _greedy_trace_value(
            result,
            ("right_bounds", "right_trace", "bracket_right_trace"),
        )
    )
    if left_bounds.size == rejected_t.size + 1:
        left_bounds = left_bounds[1:]
        right_bounds = right_bounds[1:]

    expected_rejected_t = np.asarray([0.40, -0.20, 0.10, -0.05])
    expected_bounds = np.asarray(
        [
            [-0.50, 0.40],
            [-0.20, 0.40],
            [-0.20, 0.10],
            [-0.05, 0.10],
        ]
    )

    np.testing.assert_allclose(rejected_t, expected_rejected_t)
    np.testing.assert_allclose(
        np.abs(rejected_t[1:]) / np.abs(rejected_t[:-1]),
        np.full(3, 0.5),
    )
    np.testing.assert_allclose(
        np.column_stack([left_bounds, right_bounds]),
        expected_bounds,
    )
    assert np.all(np.diff(right_bounds - left_bounds) < 0.0)
    assert _greedy_result_used_current_point_fallback(result) is False
    assert _greedy_result_accepted_t(result) == pytest.approx(0.025)
    np.testing.assert_allclose(
        _greedy_result_point(result),
        np.asarray([0.525]),
    )
    assert _greedy_result_log_likelihood(result) > 0.0


def test_greedy_shrinkage_retries_until_sample_satisfies_strict_contour():
    point = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    proposal, log_likelihood, num_evals, _, _ = _new_proposal(
        key=random.PRNGKey(0),
        U0=point,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=True,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-0.1),
        log_likelihood_fn=_narrow_slice_log_likelihood,
    )

    assert int(num_evals) > 1
    assert float(log_likelihood) > -0.1
    assert abs(float(proposal.tree[0]) - 0.5) < 0.1


def test_new_proposal_returns_current_point_when_strict_slice_has_no_width():
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_new_proposal_strict_current_fallback_worker,
        args=(queue,),
    )
    process.start()
    process.join(timeout=20.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        pytest.fail(
            "_new_proposal did not terminate for a strict current-point seed; "
            "production shrinkage needs a bounded current-point fallback."
        )

    assert process.exitcode == 0
    assert not queue.empty()
    status, *payload = queue.get()
    assert status == "ok", payload
    proposal, log_likelihood, num_evals = payload
    np.testing.assert_allclose(proposal, np.asarray([0.5]))
    np.testing.assert_allclose(log_likelihood, 1.0)
    assert 1 <= num_evals <= 64


@pytest.mark.parametrize(
    "case",
    (
        UnsupportedModeCase(
            name="gradient_guided_trajectory",
            kwargs={"trajectory": "gradient_guided"},
        ),
        UnsupportedModeCase(
            name="legacy_gradient_guided_flag",
            kwargs={"gradient_guided": True},
        ),
    ),
    ids=lambda case: case.name,
)
def test_unsupported_trajectory_modes_fail_clearly(case):
    kwargs = {
        "model": make_toy_model(),
        "num_slices": 2,
    }
    kwargs.update(case.kwargs)

    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"(?i)(unsupported|not implemented|galilean|gradient|0012)",
    ):
        UniDimSliceSampler(**kwargs)


def test_legacy_gradient_guided_flag_requires_explicit_galilean_trajectory():
    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"(?i)(gradient_guided|legacy|ambiguous|trajectory|galilean)",
    ):
        UniDimSliceSampler(
            model=make_toy_model(),
            num_slices=2,
            gradient_guided=True,
        )


def test_worker_backed_legacy_gradient_guided_flag_fails_at_construction():
    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"(?i)(gradient_guided|gradient|legacy|unsupported|galilean)",
    ):
        DistributedUniDimSliceSampler(
            model=make_toy_model(),
            evaluator=QuadraticEvaluator(),
            num_slices=2,
            gradient_guided=True,
        )


@pytest.mark.parametrize(
    "trajectory",
    ("gradient_guided",),
)
def test_worker_backed_unsupported_gradient_trajectory_fails_at_construction(
        trajectory,
):
    signature = inspect.signature(DistributedUniDimSliceSampler)
    if "trajectory" not in signature.parameters:
        pytest.fail(
            "DistributedUniDimSliceSampler must expose explicit trajectory "
            "mode arguments so unsupported worker-backed modes fail at "
            "construction."
        )

    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"(?i)(unsupported|not implemented|galilean|gradient|0012)",
    ):
        DistributedUniDimSliceSampler(
            model=make_toy_model(),
            evaluator=QuadraticEvaluator(),
            num_slices=2,
            trajectory=trajectory,
        )
