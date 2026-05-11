from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

import jaxns.constrained_sampler as constrained_sampler
import jaxns.constrained_sampler_distributed as distributed_constrained_sampler
import jaxns.core as core_module
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.constrained_sampler_distributed import DistributedUniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import SeedPoint
from jaxns.termination_condition import TerminationCondition
from tests.distributed_support import QuadraticEvaluator


class GalileanContourCase(NamedTuple):
    seed: jax.Array
    direction: jax.Array
    log_L_constraint: jax.Array
    initial_step_size: jax.Array
    log_likelihood: Callable[[jax.Array], jax.Array]
    grad_log_likelihood: Callable[[jax.Array], jax.Array]


class FakeGalileanTrajectory(NamedTuple):
    points: jax.Array
    segment_lengths: jax.Array


class FakeGalileanSample(NamedTuple):
    point: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class UnitIntervalQuadraticModel(PureDataclassPytree):
    centre: float = 0.5

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return random.uniform(key, shape=(1,), minval=0.0, maxval=1.0)

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        u = jnp.asarray(U)
        return -jnp.square(u[0] - self.centre)

    def log_prior(self, U, args=(), params=None):
        del args, params
        u = jnp.asarray(U)
        inside = jnp.logical_and(u[0] >= 0.0, u[0] <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)


UnitIntervalQuadraticModel.register_pytree()


class UnitIntervalQuadraticEvaluator:
    def __init__(self, centre: float = 0.5):
        self.centre = float(centre)

    def evaluate(self, u):
        u = jnp.ravel(jnp.asarray(u))[0]
        return -jnp.square(u - self.centre)

    def grad_log_likelihood(self, u):
        u = jnp.ravel(jnp.asarray(u))[0]
        return jnp.asarray([-2.0 * (u - self.centre)])

    def gradient(self, u):
        return self.grad_log_likelihood(u)


@dataclasses.dataclass(frozen=True, slots=True)
class MarkedGalileanModel(PureDataclassPytree):
    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return random.uniform(key, shape=(1,), minval=0.0, maxval=1.0)

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        u = jnp.ravel(jnp.asarray(U))[0]
        return jnp.select(
            condlist=[
                jnp.isclose(u, 0.10),
                jnp.isclose(u, 0.90),
                jnp.isclose(u, 0.30),
                jnp.isclose(u, 0.70),
                jnp.isclose(u, 0.60),
                jnp.isclose(u, 0.40),
                jnp.isclose(u, 0.50),
            ],
            choicelist=[
                101.0,
                109.0,
                3.0,
                7.0,
                6.0,
                4.0,
                5.0,
            ],
            default=-100.0,
        )

    def log_prior(self, U, args=(), params=None):
        del args, params
        u = jnp.asarray(U)
        inside = jnp.logical_and(u[0] >= 0.0, u[0] <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)


MarkedGalileanModel.register_pytree()


def _one_dimensional_contour_case() -> GalileanContourCase:
    def log_likelihood(U):
        u = jnp.asarray(U)
        return -jnp.square(u[0] - 0.5)

    def grad_log_likelihood(U):
        u = jnp.asarray(U)
        return jnp.asarray([-2.0 * (u[0] - 0.5)])

    return GalileanContourCase(
        seed=jnp.asarray([0.5]),
        direction=jnp.asarray([1.0]),
        log_L_constraint=jnp.asarray(-0.04),
        initial_step_size=jnp.asarray(0.05),
        log_likelihood=log_likelihood,
        grad_log_likelihood=grad_log_likelihood,
    )


def _two_dimensional_circle_contour_case() -> GalileanContourCase:
    centre = jnp.asarray([0.5, 0.5])
    direction = jnp.asarray([3.0, 4.0])
    direction = direction / jnp.linalg.norm(direction)

    def log_likelihood(U):
        u = jnp.asarray(U)
        return -jnp.sum(jnp.square(u - centre))

    def grad_log_likelihood(U):
        u = jnp.asarray(U)
        return -2.0 * (u - centre)

    return GalileanContourCase(
        seed=centre,
        direction=direction,
        log_L_constraint=jnp.asarray(-0.0625),
        initial_step_size=jnp.asarray(0.04),
        log_likelihood=log_likelihood,
        grad_log_likelihood=grad_log_likelihood,
    )


def _constant_inside_contour_case() -> GalileanContourCase:
    def log_likelihood(U):
        del U
        return jnp.asarray(1.0)

    def grad_log_likelihood(U):
        return jnp.zeros_like(jnp.asarray(U))

    return GalileanContourCase(
        seed=jnp.asarray([0.5]),
        direction=jnp.asarray([1.0]),
        log_L_constraint=jnp.asarray(0.0),
        initial_step_size=jnp.asarray(0.05),
        log_likelihood=log_likelihood,
        grad_log_likelihood=grad_log_likelihood,
    )


def _required_sampler_symbol(name: str):
    symbol = getattr(constrained_sampler, name, None)
    if symbol is None:
        pytest.fail(
            f"jaxns.constrained_sampler.{name} is required by Ticket 0012 "
            "Galilean sampler tests."
        )
    return symbol


def _as_point_array(value) -> np.ndarray:
    if hasattr(value, "tree"):
        value = value.tree
    return np.asarray(value, dtype=float)


def _trajectory_points(trajectory) -> np.ndarray:
    points = None
    for attr_name in (
            "points",
            "U_points",
            "trajectory_points",
            "vertices",
    ):
        if hasattr(trajectory, attr_name):
            points = getattr(trajectory, attr_name)
            break

    if points is None and isinstance(trajectory, dict):
        for key in ("points", "U_points", "trajectory_points", "vertices"):
            if key in trajectory:
                points = trajectory[key]
                break

    if points is None and isinstance(trajectory, tuple) and trajectory:
        points = trajectory[0]

    if points is None:
        pytest.fail("Galilean trajectory must expose ordered points.")

    points_array = _as_point_array(points)
    if points_array.ndim == 1:
        points_array = points_array[:, None]
    return points_array


def _segment_lengths(trajectory) -> np.ndarray:
    for attr_name in (
            "segment_lengths",
            "lengths",
            "trajectory_segment_lengths",
    ):
        if hasattr(trajectory, attr_name):
            return _as_point_array(getattr(trajectory, attr_name))
    if isinstance(trajectory, dict):
        for key in (
                "segment_lengths",
                "lengths",
                "trajectory_segment_lengths",
        ):
            if key in trajectory:
                return _as_point_array(trajectory[key])

    points = _trajectory_points(trajectory)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def _terminal_direction(trajectory) -> np.ndarray:
    for attr_name in ("terminal_direction", "final_direction"):
        if hasattr(trajectory, attr_name):
            return _as_point_array(getattr(trajectory, attr_name))

    if isinstance(trajectory, dict):
        for key in ("terminal_direction", "final_direction"):
            if key in trajectory:
                return _as_point_array(trajectory[key])

    directions = None
    for attr_name in ("directions", "trajectory_directions"):
        if hasattr(trajectory, attr_name):
            directions = getattr(trajectory, attr_name)
            break
    if isinstance(trajectory, dict):
        for key in ("directions", "trajectory_directions"):
            if key in trajectory:
                directions = trajectory[key]
                break
    if (
            directions is None
            and isinstance(trajectory, tuple)
            and len(trajectory) > 1
    ):
        directions = trajectory[1]
    if directions is None:
        pytest.fail(
            "Galilean side trajectory must expose its terminal direction."
        )

    direction_array = _as_point_array(directions)
    return direction_array[-1]


def _sampled_point(sample) -> np.ndarray:
    for attr_name in ("point", "point_U", "U", "U_sample"):
        if hasattr(sample, attr_name):
            return _as_point_array(getattr(sample, attr_name))
    if isinstance(sample, dict):
        for key in ("point", "point_U", "U", "U_sample"):
            if key in sample:
                return _as_point_array(sample[key])
    if isinstance(sample, tuple):
        return _as_point_array(sample[0])
    return _as_point_array(sample)


def _phantom_log_l_and_mask(phantom_samples) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(phantom_samples.log_L, dtype=float),
        np.asarray(phantom_samples.valid_mask, dtype=bool),
    )


def _build_galilean_side(case: GalileanContourCase, **overrides):
    build_side = _required_sampler_symbol("_build_galilean_side")
    kwargs = dict(
        U0=case.seed,
        direction=case.direction,
        log_L_constraint=case.log_L_constraint,
        initial_step_size=case.initial_step_size,
        log_likelihood_fn=case.log_likelihood,
        grad_log_likelihood_fn=case.grad_log_likelihood,
        max_reflections=8,
        max_step_halvings=16,
        max_step_doublings=16,
    )
    kwargs.update(overrides)
    return build_side(**kwargs)


def _build_galilean_trajectory(case: GalileanContourCase, **overrides):
    build_trajectory = _required_sampler_symbol("_build_galilean_trajectory")
    kwargs = dict(
        U0=case.seed,
        direction=case.direction,
        log_L_constraint=case.log_L_constraint,
        initial_step_size=case.initial_step_size,
        log_likelihood_fn=case.log_likelihood,
        grad_log_likelihood_fn=case.grad_log_likelihood,
        max_reflections=8,
        max_step_halvings=16,
        max_step_doublings=16,
    )
    kwargs.update(overrides)
    return build_trajectory(**kwargs)


def test_galilean_reflection_is_unit_norm_and_reversible_on_planar_wall():
    reflect_direction = _required_sampler_symbol("_reflect_galilean_direction")
    incoming = jnp.asarray([3.0, -4.0])
    incoming = incoming / jnp.linalg.norm(incoming)
    unit_normal = jnp.asarray([1.0, 0.0])

    reflected = reflect_direction(incoming, unit_normal)
    round_tripped = reflect_direction(reflected, unit_normal)

    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(reflected)),
        1.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(reflected),
        np.asarray([-incoming[0], incoming[1]]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(round_tripped),
        np.asarray(incoming),
        rtol=1e-6,
        atol=1e-6,
    )


def test_galilean_side_terminates_when_direction_u_turns():
    case = _one_dimensional_contour_case()
    side = _build_galilean_side(case)

    terminal_direction = _terminal_direction(side)
    alignment = float(np.dot(terminal_direction, np.asarray(case.direction)))

    assert alignment < 0.0
    assert _trajectory_points(side).shape[0] >= 2


def test_galilean_trajectory_extends_from_seed_and_is_strict():
    case = _one_dimensional_contour_case()
    trajectory = _build_galilean_trajectory(case)
    points = _trajectory_points(trajectory)
    offsets = points[:, 0] - float(case.seed[0])
    log_likelihoods = np.asarray(
        [float(case.log_likelihood(jnp.asarray(point))) for point in points],
        dtype=float,
    )

    assert np.any(offsets < -0.05)
    assert np.any(offsets > 0.05)
    assert np.count_nonzero(np.isclose(offsets, 0.0, atol=1e-8)) == 1
    assert np.all(log_likelihoods > float(case.log_L_constraint))


def test_galilean_trajectory_respects_unit_cube_support_for_broad_contours():
    sample_trajectory = _required_sampler_symbol("_sample_galilean_trajectory")

    def log_likelihood(U):
        u = jnp.asarray(U)
        return -jnp.square(u[0] - 0.5)

    def grad_log_likelihood(U):
        u = jnp.asarray(U)
        return jnp.asarray([-2.0 * (u[0] - 0.5)])

    trajectory = _build_galilean_trajectory(
        GalileanContourCase(
            seed=jnp.asarray([0.5]),
            direction=jnp.asarray([1.0]),
            log_L_constraint=jnp.asarray(-1.0),
            initial_step_size=jnp.asarray(0.2),
            log_likelihood=log_likelihood,
            grad_log_likelihood=grad_log_likelihood,
        )
    )
    points = _trajectory_points(trajectory)
    segment_lengths = _segment_lengths(trajectory)

    assert np.all(points >= -1e-8)
    assert np.all(points <= 1.0 + 1e-8)

    keys = random.split(random.PRNGKey(202), 512)
    samples = np.asarray(
        [
            _sampled_point(
                sample_trajectory(
                    key=key,
                    points=jnp.asarray(points),
                    segment_lengths=jnp.asarray(segment_lengths),
                )
            )
            for key in keys
        ],
        dtype=float,
    )
    assert np.all(samples >= -1e-8)
    assert np.all(samples <= 1.0 + 1e-8)


def test_built_galilean_trajectory_is_reversible_under_direction_flip():
    case = _one_dimensional_contour_case()
    forward = _build_galilean_trajectory(case)
    reverse = _build_galilean_trajectory(
        case._replace(direction=-case.direction)
    )

    forward_points = _trajectory_points(forward)
    reverse_points = _trajectory_points(reverse)
    forward_lengths = _segment_lengths(forward)
    reverse_lengths = _segment_lengths(reverse)

    assert forward_points.shape == reverse_points.shape
    np.testing.assert_allclose(
        forward_points,
        reverse_points[::-1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        forward_lengths,
        reverse_lengths[::-1],
        rtol=1e-6,
        atol=1e-6,
    )


def test_built_galilean_trajectory_in_2d_has_direction_flip_symmetry():
    case = _two_dimensional_circle_contour_case()
    centre = np.asarray(case.seed, dtype=float)
    initial_direction = np.asarray(case.direction, dtype=float)
    forward = _build_galilean_trajectory(case)
    reverse = _build_galilean_trajectory(
        case._replace(direction=-case.direction)
    )

    forward_points = _trajectory_points(forward)
    reverse_points = _trajectory_points(reverse)
    forward_lengths = _segment_lengths(forward)
    reverse_lengths = _segment_lengths(reverse)
    offsets = forward_points - centre
    cross_track_offsets = (
        offsets[:, 0] * initial_direction[1]
        - offsets[:, 1] * initial_direction[0]
    )
    log_likelihoods = np.asarray(
        [
            float(case.log_likelihood(jnp.asarray(point)))
            for point in forward_points
        ]
    )

    assert forward_points.shape == reverse_points.shape
    assert forward_points.shape[1] == 2
    np.testing.assert_allclose(
        forward_points,
        reverse_points[::-1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        forward_points,
        2.0 * centre - forward_points[::-1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        forward_lengths,
        reverse_lengths[::-1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(cross_track_offsets, 0.0, atol=1e-6)
    assert np.all(log_likelihoods > float(case.log_L_constraint))


def test_galilean_trajectory_sampling_weights_segments_by_length():
    sample_trajectory = _required_sampler_symbol("_sample_galilean_trajectory")
    points = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 3.0],
        ]
    )
    segment_lengths = jnp.asarray([1.0, 3.0])
    keys = random.split(random.PRNGKey(12), 4000)

    samples = np.asarray(
        [
            _sampled_point(
                sample_trajectory(
                    key=key,
                    points=points,
                    segment_lengths=segment_lengths,
                )
            )
            for key in keys
        ],
        dtype=float,
    )

    on_short_segment = np.isclose(samples[:, 1], 0.0, atol=1e-7)
    short_segment_fraction = float(np.mean(on_short_segment))

    assert 0.22 < short_segment_fraction < 0.28


def test_galilean_trajectory_sampling_is_uniform_within_selected_segment():
    sample_trajectory = _required_sampler_symbol("_sample_galilean_trajectory")
    points = jnp.asarray(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
        ]
    )
    segment_lengths = jnp.asarray([4.0, 1.0])
    keys = random.split(random.PRNGKey(43), 9000)

    samples = np.asarray(
        [
            _sampled_point(
                sample_trajectory(
                    key=key,
                    points=points,
                    segment_lengths=segment_lengths,
                )
            )
            for key in keys
        ],
        dtype=float,
    )

    on_long_segment = np.isclose(samples[:, 1], 0.0, atol=1e-7)
    long_segment_fraction = float(np.mean(on_long_segment))
    long_x = samples[on_long_segment, 0]
    quartile_counts, _ = np.histogram(long_x, bins=np.asarray([0, 1, 2, 3, 4]))
    quartile_fractions = quartile_counts / np.sum(quartile_counts)

    assert 0.77 < long_segment_fraction < 0.83
    assert np.all(quartile_fractions > 0.22)
    assert np.all(quartile_fractions < 0.28)
    np.testing.assert_allclose(np.mean(long_x), 2.0, atol=0.08)


def test_galilean_sampler_keeps_only_markov_states_as_phantoms():
    model = UnitIntervalQuadraticModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    U_sample, log_likelihood, _, phantom_samples = sampler.get_sample(
        key=random.PRNGKey(3),
        log_L_constraint=jnp.asarray(-0.04),
        seed_point=SeedPoint(
            U0=jnp.asarray([0.5]),
            log_L0=jnp.asarray(0.0),
        ),
    )

    assert sampler.num_phantom() == 2
    assert phantom_samples.U_samples is None
    assert phantom_samples.log_L.shape == (2,)
    assert phantom_samples.valid_mask.shape == (2,)
    assert np.all(np.asarray(phantom_samples.valid_mask))
    assert float(log_likelihood) > -0.04
    assert float(model.log_likelihood(U_sample)) > -0.04
    assert np.all(np.asarray(phantom_samples.log_L) > -0.04)


def test_galilean_sampler_runs_through_public_nested_sampler_run():
    model = UnitIntervalQuadraticModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    ns = NestedSampler(
        model=model,
        sampler=sampler,
        target_num_live_points=2,
        max_samples=4,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=4),
        store_phantom_samples=True,
    )

    state = ns.run(random.PRNGKey(33))

    num_samples = int(state.num_samples)
    assert num_samples == 4
    assert np.all(
        np.asarray(state.samples.U_samples[:num_samples], dtype=float)
        >= -1e-8
    )
    assert np.all(
        np.asarray(state.samples.U_samples[:num_samples], dtype=float)
        <= 1.0 + 1e-8
    )
    assert np.all(
        np.asarray(state.samples.log_likelihoods[:num_samples], dtype=float)
        > -np.inf
    )


def test_galilean_run_until_goal_uses_pure_core_epoch_without_python_parent_selection(
        monkeypatch,
):
    model = UnitIntervalQuadraticModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    ns = NestedSampler(
        model=model,
        sampler=sampler,
        target_num_live_points=2,
        max_samples=4,
        shell_size=1,
        store_phantom_samples=True,
    )

    def fail_python_parent_selection(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "direct pure-core Galilean must not call select_parent_work"
        )

    epoch_steps = []
    original_depth_epoch = core_module._pure_core_depth_epoch_jax

    def recording_depth_epoch(*args, **kwargs):
        result = original_depth_epoch(*args, **kwargs)
        epoch_steps.append(int(np.asarray(result.history.num_steps)))
        return result

    monkeypatch.setattr(
        core_module,
        "select_parent_work",
        fail_python_parent_selection,
    )
    monkeypatch.setattr(
        core_module,
        "_pure_core_depth_epoch_jax",
        recording_depth_epoch,
    )

    state = ns.run_until_goal(
        goal_cond=lambda state: int(state.num_samples) >= 4,
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
        key=random.PRNGKey(33),
        max_goal_iterations=3,
    )

    num_samples = int(state.num_samples)
    assert num_samples == 4
    assert epoch_steps
    assert sum(epoch_steps) >= 1
    active_u = np.asarray(state.samples.U_samples[:num_samples], dtype=float)
    assert np.all(active_u >= -1e-8)
    assert np.all(active_u <= 1.0 + 1e-8)
    active_log_l = np.asarray(
        state.samples.log_likelihoods[:num_samples],
        dtype=float,
    )
    assert np.all(np.isfinite(active_log_l))
    phantom = state.samples.phantom_samples
    assert phantom.log_L.shape[1] == sampler.num_phantom()
    valid_phantom = np.asarray(phantom.valid_mask[:num_samples], dtype=bool)
    phantom_log_l = np.asarray(phantom.log_L[:num_samples], dtype=float)
    assert np.all(np.isfinite(phantom_log_l[valid_phantom]))


def test_galilean_run_until_goal_does_not_use_eager_python_transition(
        monkeypatch,
):
    model = UnitIntervalQuadraticModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    ns = NestedSampler(
        model=model,
        sampler=sampler,
        target_num_live_points=2,
        max_samples=4,
        shell_size=1,
        store_phantom_samples=True,
    )

    def fail_eager_transition(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "direct pure-core Galilean must use the traced-safe streaming "
            "transition, including root initialization"
        )

    monkeypatch.setattr(
        constrained_sampler,
        "_sample_galilean_markov_transition",
        fail_eager_transition,
    )

    state = ns.run_until_goal(
        goal_cond=lambda state: int(state.num_samples) >= 4,
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
        key=random.PRNGKey(41),
        max_goal_iterations=3,
    )

    assert int(state.num_samples) == 4


def test_galilean_standalone_sampler_uses_eager_transition(monkeypatch):
    model = UnitIntervalQuadraticModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    def fail_jax_transition(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "standalone Galilean sampler calls should keep the eager "
            "trajectory path unless explicitly forced or traced"
        )

    monkeypatch.setattr(
        constrained_sampler,
        "_sample_galilean_markov_transition_jax",
        fail_jax_transition,
    )

    U_sample, log_likelihood, _, phantom_samples = sampler.get_sample(
        key=random.PRNGKey(52),
        log_L_constraint=jnp.asarray(-0.04),
        seed_point=SeedPoint(
            U0=jnp.asarray([0.5]),
            log_L0=jnp.asarray(0.0),
        ),
    )

    assert float(log_likelihood) > -0.04
    assert float(model.log_likelihood(U_sample)) > -0.04
    assert np.all(np.asarray(phantom_samples.log_L, dtype=float) > -0.04)


def test_galilean_distributed_sampler_uses_eager_transition(monkeypatch):
    model = UnitIntervalQuadraticModel()
    sampler = DistributedUniDimSliceSampler(
        model=model,
        evaluator=UnitIntervalQuadraticEvaluator(centre=0.5),
        num_slices=2,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    eager_calls = []
    original_eager_transition = (
        distributed_constrained_sampler._sample_galilean_markov_transition
    )

    def recording_eager_transition(*args, **kwargs):
        eager_calls.append(True)
        return original_eager_transition(*args, **kwargs)

    def fail_jax_transition(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "distributed Galilean sampler calls should keep the eager "
            "worker-backed trajectory path"
        )

    monkeypatch.setattr(
        distributed_constrained_sampler,
        "_sample_galilean_markov_transition",
        recording_eager_transition,
    )
    monkeypatch.setattr(
        constrained_sampler,
        "_sample_galilean_markov_transition_jit",
        fail_jax_transition,
    )

    U_sample, log_likelihood, _, phantom_samples = sampler.get_sample(
        key=random.PRNGKey(53),
        log_L_constraint=jnp.asarray(-0.04),
        seed_point=SeedPoint(
            U0=jnp.asarray([0.5]),
            log_L0=jnp.asarray(0.0),
        ),
    )

    assert eager_calls
    assert float(log_likelihood) > -0.04
    assert float(model.log_likelihood(U_sample)) > -0.04
    assert np.all(np.asarray(phantom_samples.log_L, dtype=float) > -0.04)


def test_jax_galilean_does_not_evaluate_likelihood_outside_unit_cube():
    transition = _required_sampler_symbol(
        "_sample_galilean_markov_transition_jax"
    )
    outside_likelihood_calls = []

    def record_call(point):
        point = np.asarray(point, dtype=float)
        outside_likelihood_calls.append(
            bool(np.any((point < -1e-8) | (point > 1.0 + 1e-8)))
        )

    def log_likelihood(U):
        u = jnp.asarray(U)
        jax.debug.callback(record_call, u)
        return -jnp.square(u[0] - 0.5)

    def grad_log_likelihood(U):
        u = jnp.asarray(U)
        return jnp.asarray([-2.0 * (u[0] - 0.5)])

    point, log_likelihood_value, _, = transition(
        key=random.PRNGKey(71),
        U0=jnp.asarray([0.5]),
        log_L0=jnp.asarray(0.0),
        direction=jnp.asarray([1.0]),
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=log_likelihood,
        grad_log_likelihood_fn=grad_log_likelihood,
        initial_step_size=jnp.asarray(0.2),
        max_step_halvings=16,
        max_step_doublings=16,
    )

    assert not any(outside_likelihood_calls)
    assert np.all(np.asarray(point.tree, dtype=float) >= -1e-8)
    assert np.all(np.asarray(point.tree, dtype=float) <= 1.0 + 1e-8)
    assert float(log_likelihood_value) > -1.0


def test_jax_galilean_reflects_from_unit_cube_support_normal():
    transition = _required_sampler_symbol(
        "_sample_galilean_markov_transition_jax"
    )

    def log_likelihood(U):
        u = jnp.asarray(U)
        return -jnp.square(u[1] - 0.5)

    def grad_log_likelihood(U):
        u = jnp.asarray(U)
        return jnp.asarray([0.0, -2.0 * (u[1] - 0.5)])

    point, log_likelihood_value, _ = transition(
        key=random.PRNGKey(93),
        U0=jnp.asarray([0.98, 0.5]),
        log_L0=jnp.asarray(0.0),
        direction=jnp.asarray([1.0, 0.0]),
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=log_likelihood,
        grad_log_likelihood_fn=grad_log_likelihood,
        initial_step_size=jnp.asarray(0.05),
        max_step_halvings=16,
        max_step_doublings=16,
    )

    point_array = np.asarray(point.tree, dtype=float)
    assert np.all(point_array >= -1e-8)
    assert np.all(point_array <= 1.0 + 1e-8)
    assert float(log_likelihood_value) > -1.0


def test_galilean_internal_points_are_excluded_from_phantom_clusters(
        monkeypatch,
):
    model = MarkedGalileanModel()
    completed_chain_states = iter(
        [
            jnp.asarray([0.30]),
            jnp.asarray([0.70]),
            jnp.asarray([0.60]),
            jnp.asarray([0.40]),
        ]
    )

    def fake_build_trajectory(**kwargs):
        del kwargs
        return FakeGalileanTrajectory(
            points=jnp.asarray([[0.10], [0.50], [0.90]]),
            segment_lengths=jnp.asarray([0.40, 0.40]),
        )

    def fake_sample_trajectory(**kwargs):
        del kwargs
        return FakeGalileanSample(point=next(completed_chain_states))

    monkeypatch.setattr(
        constrained_sampler,
        "_build_galilean_trajectory",
        fake_build_trajectory,
        raising=False,
    )
    monkeypatch.setattr(
        constrained_sampler,
        "_sample_galilean_trajectory",
        fake_sample_trajectory,
        raising=False,
    )
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    U_sample, log_likelihood, _, phantom_samples = sampler.get_sample(
        key=random.PRNGKey(13),
        log_L_constraint=jnp.asarray(0.0),
        seed_point=SeedPoint(
            U0=jnp.asarray([0.50]),
            log_L0=jnp.asarray(5.0),
        ),
    )

    phantom_log_l = np.asarray(phantom_samples.log_L, dtype=float)

    assert float(log_likelihood) == 4.0
    np.testing.assert_allclose(np.asarray(U_sample), np.asarray([0.40]))
    np.testing.assert_allclose(phantom_log_l, np.asarray([7.0, 6.0]))
    assert not np.any(np.isin(phantom_log_l, np.asarray([101.0, 109.0])))


def test_galilean_reflection_rejects_degenerate_gradient_normal():
    reflect_direction = _required_sampler_symbol("_reflect_galilean_direction")

    with pytest.raises(
        (ValueError, FloatingPointError),
        match=r"(?i)(gradient|normal|degenerate|zero)",
    ):
        reflect_direction(
            jnp.asarray([1.0, 0.0]),
            jnp.asarray([0.0, 0.0]),
        )


def test_galilean_side_fails_clearly_when_max_reflections_exhausted():
    case = _one_dimensional_contour_case()

    with pytest.raises(
        (ValueError, RuntimeError, FloatingPointError),
        match=r"(?i)(max_reflections|reflection|limit)",
    ):
        _build_galilean_side(case, max_reflections=0)


def test_galilean_side_fails_clearly_when_boundary_search_is_degenerate():
    case = _constant_inside_contour_case()

    with pytest.raises(
        (ValueError, RuntimeError, FloatingPointError),
        match=r"(?i)(boundary|doubling|gradient|degenerate|limit)",
    ):
        _build_galilean_side(
            case,
            max_reflections=2,
            max_step_halvings=2,
            max_step_doublings=2,
        )


@pytest.mark.parametrize(
    "sampler_factory",
    [
        lambda: UniDimSliceSampler(
            model=UnitIntervalQuadraticModel(),
            num_slices=2,
            trajectory="gradient_guided",
        ),
        lambda: DistributedUniDimSliceSampler(
            model=UnitIntervalQuadraticModel(),
            evaluator=QuadraticEvaluator(centre=0.5),
            num_slices=2,
            trajectory="gradient_guided",
        ),
    ],
    ids=["local", "distributed"],
)
def test_unsupported_gradient_informed_modes_fail_clearly(sampler_factory):
    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"(?i)(gradient|mode|unsupported|galilean|trajectory)",
    ):
        sampler_factory()


@pytest.mark.parametrize(
    "sampler_factory",
    [
        lambda: UniDimSliceSampler(
            model=UnitIntervalQuadraticModel(),
            num_slices=2,
            gradient_guided=True,
        ),
        lambda: DistributedUniDimSliceSampler(
            model=UnitIntervalQuadraticModel(),
            evaluator=QuadraticEvaluator(centre=0.5),
            num_slices=2,
            gradient_guided=True,
        ),
    ],
    ids=["local", "distributed"],
)
def test_legacy_gradient_guided_flag_requires_explicit_galilean_trajectory(
        sampler_factory,
):
    with pytest.raises(
        (ValueError, NotImplementedError),
        match=r"(?i)(gradient_guided|legacy|ambiguous|trajectory|galilean)",
    ):
        sampler_factory()


def test_galilean_local_and_worker_backed_paths_match():
    model = UnitIntervalQuadraticModel()
    seed_point = SeedPoint(
        U0=jnp.asarray([0.5]),
        log_L0=jnp.asarray(0.0),
    )
    sampler_kwargs = dict(
        model=model,
        num_slices=4,
        no_step_out=True,
        trajectory="galilean",
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    local_sampler = UniDimSliceSampler(**sampler_kwargs)
    worker_sampler = DistributedUniDimSliceSampler(
        evaluator=UnitIntervalQuadraticEvaluator(centre=0.5),
        **sampler_kwargs,
    )
    local_result = local_sampler.get_sample(
        key=random.PRNGKey(101),
        log_L_constraint=jnp.asarray(-0.04),
        seed_point=seed_point,
    )
    worker_result = worker_sampler.get_sample(
        key=random.PRNGKey(101),
        log_L_constraint=jnp.asarray(-0.04),
        seed_point=seed_point,
    )
    local_u, local_log_l, local_num_evals, local_phantoms = local_result
    worker_u, worker_log_l, worker_num_evals, worker_phantoms = worker_result

    np.testing.assert_allclose(np.asarray(worker_u), np.asarray(local_u))
    np.testing.assert_allclose(
        np.asarray(worker_log_l),
        np.asarray(local_log_l),
    )
    assert int(worker_num_evals) == int(local_num_evals)

    local_phantom_log_l, local_phantom_mask = _phantom_log_l_and_mask(
        local_phantoms
    )
    worker_phantom_log_l, worker_phantom_mask = _phantom_log_l_and_mask(
        worker_phantoms
    )
    np.testing.assert_allclose(worker_phantom_log_l, local_phantom_log_l)
    np.testing.assert_array_equal(worker_phantom_mask, local_phantom_mask)
