from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxns.allocation as allocation
from jaxns.constrained_sampler import AbstractSampler
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import PhantomSamples
from jaxns.samples import Samples
from jaxns.samples import SeedPoint
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition


@dataclasses.dataclass(frozen=True, slots=True)
class ToyModel(PureDataclassPytree):
    centre: float = 0.25

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, minval=0.0, maxval=1.0)

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
        return -jnp.square(jnp.asarray(U) - self.centre)

    def log_prior(self, U, args=(), params=None):
        del args, params
        inside = jnp.logical_and(U >= 0.0, U <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)


ToyModel.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class PoisonModel(PureDataclassPytree):
    """Model that proves invalid public options fail before any sampling work."""

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del key, args, params
        raise AssertionError("validation should happen before sampling")

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
        del U, args, params, allow_nan
        raise AssertionError("validation should happen before likelihood work")

    def log_prior(self, U, args=(), params=None):
        del args, params
        return jnp.zeros_like(U)


PoisonModel.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicContourSampler(PureDataclassPytree, AbstractSampler):
    num_phantom_samples: int = 0

    def num_phantom(self) -> int:
        return self.num_phantom_samples

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            args=(),
            params=None,
    ):
        del key, args, params
        finite_constraint = jnp.where(
            jnp.isneginf(log_L_constraint),
            jnp.asarray(0.0, dtype=jnp.asarray(log_L_constraint).dtype),
            log_L_constraint,
        )
        log_L = finite_constraint + jnp.asarray(0.25, dtype=finite_constraint.dtype)
        U_sample = jnp.asarray(jnp.clip(0.1 + 0.1 * log_L, 0.0, 1.0))
        U_sample = jnp.asarray(U_sample, dtype=jnp.asarray(seed_point.U0).dtype)
        phantom_offsets = jnp.arange(
            self.num_phantom_samples,
            dtype=log_L.dtype,
        )
        phantom_log_L = log_L + 0.01 * (phantom_offsets + 1.0)
        phantom_samples = PhantomSamples(
            U_samples=jnp.full((self.num_phantom_samples,), U_sample),
            valid_mask=jnp.ones((self.num_phantom_samples,), dtype=bool),
            log_L=phantom_log_L,
        )
        return (
            U_sample,
            log_L,
            jnp.asarray(1, dtype=jnp.int32),
            phantom_samples,
        )


DeterministicContourSampler.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class ArgsParamsAssertingModel(PureDataclassPytree):
    expected_args: tuple
    expected_params: dict[str, float]

    def _assert_context(self, args, params) -> None:
        assert args == self.expected_args
        assert params == self.expected_params

    def U_ndims(self, args=(), params=None) -> int:
        self._assert_context(args, params)
        return 1

    def sample_U(self, key, args=(), params=None):
        del key
        self._assert_context(args, params)
        return jnp.asarray(0.25)

    def transform_to_X(self, U, args=(), params=None):
        self._assert_context(args, params)
        return U

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del allow_nan
        self._assert_context(args, params)
        return jnp.asarray(U) + params["offset"]

    def log_prior(self, U, args=(), params=None):
        self._assert_context(args, params)
        return jnp.zeros_like(U)


ArgsParamsAssertingModel.register_pytree()


@dataclasses.dataclass(frozen=True, slots=True)
class ArgsParamsAssertingSampler(PureDataclassPytree, AbstractSampler):
    expected_args: tuple
    expected_params: dict[str, float]

    def num_phantom(self) -> int:
        return 0

    def get_sample(
            self,
            key,
            log_L_constraint,
            seed_point: SeedPoint,
            args=(),
            params=None,
    ):
        del key
        assert args == self.expected_args
        assert params == self.expected_params
        assert float(log_L_constraint) == -np.inf
        phantom_samples = PhantomSamples(
            U_samples=jnp.zeros((0,), dtype=jnp.asarray(seed_point.U0).dtype),
            valid_mask=jnp.zeros((0,), dtype=bool),
            log_L=jnp.zeros((0,), dtype=jnp.asarray(seed_point.log_L0).dtype),
        )
        return (
            seed_point.U0,
            seed_point.log_L0 + jnp.asarray(1.0),
            jnp.asarray(1, dtype=jnp.int32),
            phantom_samples,
        )


ArgsParamsAssertingSampler.register_pytree()


PUBLIC_ALLOCATION_TARGETS = (
    "uniform",
    "evidence_improving",
    "posterior_improving",
)


def make_toy_model() -> ToyModel:
    return ToyModel()


class GoalObservation(NamedTuple):
    state_type: type
    root_out_degree: int
    state_num_samples: int
    total_num_samples: int


class SelectionPlan(NamedTuple):
    target_K: jax.Array
    current_K: jax.Array
    unit_peak_utility: jax.Array
    log_L_blocks: jax.Array
    valid: jax.Array
    volume_path: allocation.VolumePath


def _make_nested_sampler() -> NestedSampler:
    model = make_toy_model()
    return NestedSampler(
        model=model,
        sampler=UniDimSliceSampler(
            model=model,
            num_slices=3,
            no_step_out=True,
            collect_phantom_samples=True,
            phantom_burn_in=1,
        ),
        target_num_live_points=2,
        max_samples=8,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=8),
        batch_size=1,
    )


def _make_poison_nested_sampler() -> NestedSampler:
    return NestedSampler(
        model=PoisonModel(),
        sampler=DeterministicContourSampler(),
        target_num_live_points=2,
        max_samples=4,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=4),
        batch_size=1,
    )


def _make_deterministic_nested_sampler(
        *,
        max_samples: int = 8,
        shell_size: int = 1,
        num_phantom: int = 0,
) -> NestedSampler:
    return NestedSampler(
        model=make_toy_model(),
        sampler=DeterministicContourSampler(num_phantom_samples=num_phantom),
        target_num_live_points=2,
        max_samples=max_samples,
        shell_size=shell_size,
        termination_condition=TerminationCondition(max_samples=max_samples),
        batch_size=1,
    )


def _make_samples(
        *,
        log_likelihoods: tuple[float, ...],
        out_degree: tuple[int, ...],
        log_L_constraints: tuple[float, ...] | None = None,
        max_samples: int = 8,
        num_phantom: int = 0,
) -> Samples:
    if log_L_constraints is None:
        log_L_constraints = tuple([-np.inf] * len(log_likelihoods))
    pad_count = max_samples - len(log_likelihoods)
    assert pad_count >= 0

    phantom_log_L = np.full((max_samples, num_phantom), -np.inf, dtype=float)
    phantom_valid = np.zeros((max_samples, num_phantom), dtype=bool)
    phantom_U = np.zeros((max_samples, num_phantom), dtype=float)
    return Samples(
        log_L_constraints=jnp.asarray(
            tuple(log_L_constraints) + tuple([np.inf] * pad_count),
        ),
        log_likelihoods=jnp.asarray(
            tuple(log_likelihoods) + tuple([np.inf] * pad_count),
        ),
        U_samples=jnp.asarray(
            tuple(0.1 + 0.1 * value for value in log_likelihoods)
            + tuple([0.0] * pad_count),
        ),
        out_degree=jnp.asarray(
            tuple(out_degree) + tuple([0] * pad_count),
            dtype=jnp.int32,
        ),
        num_likelihood_evaluations=jnp.asarray(
            tuple([1] * len(log_likelihoods)) + tuple([0] * pad_count),
            dtype=jnp.int32,
        ),
        phantom_samples=PhantomSamples(
            U_samples=jnp.asarray(phantom_U),
            valid_mask=jnp.asarray(phantom_valid),
            log_L=jnp.asarray(phantom_log_L),
        ),
    )


def _make_state(
        *,
        root_out_degree: int,
        log_likelihoods: tuple[float, ...],
        out_degree: tuple[int, ...],
        log_L_constraints: tuple[float, ...] | None = None,
        max_samples: int = 8,
        num_phantom: int = 0,
) -> State:
    samples = _make_samples(
        log_likelihoods=log_likelihoods,
        out_degree=out_degree,
        log_L_constraints=log_L_constraints,
        max_samples=max_samples,
        num_phantom=num_phantom,
    )
    log_L_supremum = jnp.max(samples.log_likelihoods[:len(log_likelihoods)])
    supremum_idx = jnp.argmax(samples.log_likelihoods[:len(log_likelihoods)])
    return State(
        root_out_degree=jnp.asarray(root_out_degree, dtype=jnp.int32),
        samples=samples,
        num_samples=jnp.asarray(len(log_likelihoods), dtype=jnp.int32),
        log_L_supremum=log_L_supremum,
        U_supremum=samples.U_samples[supremum_idx],
        model=make_toy_model(),
        args=(),
        params=None,
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
    )


def _immediate_goal(
        observations: list[GoalObservation],
) -> Callable[[State], bool]:
    def goal_cond(state: State) -> bool:
        result = state.to_result()
        observations.append(
            GoalObservation(
                state_type=type(state),
                root_out_degree=int(state.root_out_degree),
                state_num_samples=int(state.num_samples),
                total_num_samples=int(result.total_num_samples),
            )
        )
        return True

    return goal_cond


def _goal_after_observations(
        observations: list[GoalObservation],
        *,
        required_observations: int,
) -> Callable[[State], bool]:
    def goal_cond(state: State) -> bool:
        result = state.to_result()
        observations.append(
            GoalObservation(
                state_type=type(state),
                root_out_degree=int(state.root_out_degree),
                state_num_samples=int(state.num_samples),
                total_num_samples=int(result.total_num_samples),
            )
        )
        return len(observations) >= required_observations

    return goal_cond


def _goal_after_state_counts(
        observed_num_samples: list[int],
        *,
        required_observations: int,
) -> Callable[[State], bool]:
    def goal_cond(state: State) -> bool:
        observed_num_samples.append(int(state.num_samples))
        return len(observed_num_samples) >= required_observations

    return goal_cond


def _accepted_work_snapshot(
        state: State,
        *,
        initial_num_samples: int,
) -> tuple[tuple[float, ...], tuple[int, ...], int]:
    num_samples = int(state.num_samples)
    constraints = tuple(
        float(value)
        for value in np.asarray(
            state.samples.log_L_constraints[initial_num_samples:num_samples]
        )
    )
    out_degree = tuple(
        int(value)
        for value in np.asarray(state.samples.out_degree[:num_samples])
    )
    return constraints, out_degree, int(state.root_out_degree)


def _selection_plan(
        *,
        target_K: tuple[int, ...],
        current_K: tuple[int, ...],
        log_L_blocks: tuple[float, ...],
        valid: tuple[bool, ...] | None = None,
        X: tuple[float, ...] | None = None,
) -> SelectionPlan:
    if valid is None:
        valid = tuple([True] * len(target_K))
    if X is None:
        X = tuple(float(0.5 ** (idx + 1)) for idx in range(len(target_K)))
    X_prev = (1.0,) + X[:-1]
    return SelectionPlan(
        target_K=jnp.asarray(target_K, dtype=jnp.int32),
        current_K=jnp.asarray(current_K, dtype=jnp.int32),
        unit_peak_utility=jnp.ones((len(target_K),), dtype=float),
        log_L_blocks=jnp.asarray(log_L_blocks, dtype=float),
        valid=jnp.asarray(valid, dtype=bool),
        volume_path=allocation.VolumePath(
            X_prev=jnp.asarray(X_prev, dtype=float),
            X=jnp.asarray(X, dtype=float),
            shell_mass=jnp.asarray(
                tuple(prev - current for prev, current in zip(X_prev, X)),
                dtype=float,
            ),
        ),
    )


def test_nested_sampler_exposes_v3_run_until_goal_public_api():
    run_signature = inspect.signature(NestedSampler.run_until_goal)
    resume_signature = inspect.signature(NestedSampler.resume_until_goal)

    assert "goal_cond" in run_signature.parameters
    assert "depth_cond" in run_signature.parameters
    assert "allocation_target" in run_signature.parameters
    assert run_signature.parameters["allocation_target"].default == "uniform"
    assert any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in run_signature.parameters.values()
    )

    assert "state" in resume_signature.parameters
    assert "goal_cond" in resume_signature.parameters
    assert "depth_cond" in resume_signature.parameters
    assert "allocation_target" in resume_signature.parameters
    assert (
        resume_signature.parameters["allocation_target"].default
        == "uniform"
    )
    assert any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in resume_signature.parameters.values()
    )


@pytest.mark.parametrize("allocation_target", PUBLIC_ALLOCATION_TARGETS)
def test_run_until_goal_accepts_public_modes_and_passes_state_to_goal_cond(
        allocation_target,
):
    ns = _make_nested_sampler()
    observations: list[GoalObservation] = []

    state = ns.run_until_goal(
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target=allocation_target,
    )

    assert isinstance(state, State)
    assert observations == [
        GoalObservation(
            state_type=State,
            root_out_degree=2,
            state_num_samples=2,
            total_num_samples=2,
        )
    ]
    assert int(state.root_out_degree) == 2
    assert int(state.num_samples) == 2


def test_run_until_goal_initializes_root_constraints_and_phantom_clusters():
    ns = _make_nested_sampler()
    observations: list[GoalObservation] = []

    state = ns.run_until_goal(
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
    )
    result = state.to_result()
    num_samples = int(state.num_samples)

    assert num_samples == int(state.root_out_degree) == 2
    np.testing.assert_array_equal(
        np.asarray(state.samples.log_L_constraints[:num_samples]),
        np.asarray([-np.inf, -np.inf]),
    )
    assert state.samples.phantom_samples.log_L.shape[0] >= num_samples
    assert state.samples.phantom_samples.log_L.shape[1] == ns.sampler.num_phantom()
    assert result.log_L_phantom.shape[0] >= num_samples
    assert result.log_L_phantom.shape[1] == ns.sampler.num_phantom()
    np.testing.assert_array_equal(
        np.asarray(result.valid_phantom[:num_samples]),
        np.ones((num_samples,), dtype=bool),
    )


def test_v3_root_initialization_forwards_args_and_params_to_model_and_sampler():
    expected_args = ("context", 7)
    expected_params = {"offset": 0.5}
    ns = NestedSampler(
        model=ArgsParamsAssertingModel(
            expected_args=expected_args,
            expected_params=expected_params,
        ),
        sampler=ArgsParamsAssertingSampler(
            expected_args=expected_args,
            expected_params=expected_params,
        ),
        target_num_live_points=1,
        max_samples=2,
        shell_size=1,
        args=expected_args,
        params=expected_params,
        termination_condition=TerminationCondition(max_samples=2),
        batch_size=1,
    )
    observations: list[GoalObservation] = []

    state = ns.run_until_goal(
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=2),
        allocation_target="uniform",
        key=jax.random.PRNGKey(41),
    )

    assert int(state.num_samples) == 1
    np.testing.assert_allclose(
        np.asarray(state.samples.log_likelihoods[:1]),
        np.asarray([1.75]),
    )


@pytest.mark.parametrize("allocation_target", PUBLIC_ALLOCATION_TARGETS)
def test_resume_until_goal_accepts_public_modes_and_passes_existing_state(
        allocation_target,
):
    ns = _make_nested_sampler()
    initial_observations: list[GoalObservation] = []
    initial_state = ns.run_until_goal(
        goal_cond=_immediate_goal(initial_observations),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
    )

    resume_observations: list[GoalObservation] = []
    resumed_state = ns.resume_until_goal(
        state=initial_state,
        goal_cond=_immediate_goal(resume_observations),
        depth_cond=TerminationCondition(max_samples=6),
        allocation_target=allocation_target,
    )

    assert isinstance(resumed_state, State)
    assert resume_observations == [
        GoalObservation(
            state_type=State,
            root_out_degree=int(initial_state.root_out_degree),
            state_num_samples=int(initial_state.num_samples),
            total_num_samples=int(initial_state.num_samples),
        )
    ]
    assert int(resumed_state.num_samples) == int(initial_state.num_samples)


def test_run_until_goal_evaluates_goal_before_first_depth_run():
    ns = _make_nested_sampler()
    observations: list[GoalObservation] = []

    state = ns.run_until_goal(
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
        delta_K=2,
    )

    assert observations == [
        GoalObservation(
            state_type=State,
            root_out_degree=2,
            state_num_samples=2,
            total_num_samples=2,
        )
    ]
    assert int(state.num_samples) == 2


def test_run_until_goal_checks_goal_only_at_outer_loop_boundaries():
    ns = _make_nested_sampler()
    observations: list[GoalObservation] = []

    state = ns.run_until_goal(
        goal_cond=_goal_after_observations(
            observations,
            required_observations=2,
        ),
        depth_cond=TerminationCondition(max_samples=3),
        allocation_target="uniform",
        delta_K=1,
    )

    assert observations == [
        GoalObservation(
            state_type=State,
            root_out_degree=2,
            state_num_samples=2,
            total_num_samples=2,
        ),
        GoalObservation(
            state_type=State,
            root_out_degree=2,
            state_num_samples=3,
            total_num_samples=3,
        ),
    ]
    assert int(state.num_samples) == 3


def test_inner_loop_continues_until_depth_condition_before_goal_recheck():
    initial_state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(1, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, 0.0),
        max_samples=8,
    )
    ns = _make_deterministic_nested_sampler(max_samples=8)
    observations: list[int] = []

    state = ns.resume_until_goal(
        state=initial_state,
        goal_cond=_goal_after_state_counts(
            observations,
            required_observations=2,
        ),
        depth_cond=TerminationCondition(max_samples=5),
        allocation_target="uniform",
        delta_K=4,
        key=jax.random.PRNGKey(53),
    )

    assert observations == [3, 5]
    assert int(state.num_samples) == 5


def test_run_until_goal_rejects_invalid_delta_k_before_initial_work():
    ns = _make_poison_nested_sampler()
    observations: list[GoalObservation] = []

    with pytest.raises(ValueError, match="delta_K"):
        ns.run_until_goal(
            goal_cond=_immediate_goal(observations),
            depth_cond=TerminationCondition(max_samples=4),
            allocation_target="uniform",
            delta_K=0,
        )

    assert observations == []


def test_run_until_goal_validates_allocation_target_before_initial_work():
    ns = _make_poison_nested_sampler()
    observations: list[GoalObservation] = []

    with pytest.raises(ValueError, match="allocation_target|unsupported_target"):
        ns.run_until_goal(
            goal_cond=_immediate_goal(observations),
            depth_cond=TerminationCondition(max_samples=4),
            allocation_target="unsupported_target",
        )

    assert observations == []


def test_uniform_delta_k_is_accepted_by_depth_limited_public_resume():
    initial_state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(1, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, 0.0),
        max_samples=8,
    )
    observed_low: list[int] = []
    observed_high: list[int] = []

    low_delta_state = _make_deterministic_nested_sampler(
        max_samples=8,
    ).resume_until_goal(
        state=initial_state,
        goal_cond=_goal_after_state_counts(
            observed_low,
            required_observations=3,
        ),
        depth_cond=TerminationCondition(max_samples=7),
        allocation_target="uniform",
        delta_K=1,
        key=jax.random.PRNGKey(11),
    )
    high_delta_state = _make_deterministic_nested_sampler(
        max_samples=8,
    ).resume_until_goal(
        state=initial_state,
        goal_cond=_goal_after_state_counts(
            observed_high,
            required_observations=3,
        ),
        depth_cond=TerminationCondition(max_samples=7),
        allocation_target="uniform",
        delta_K=3,
        key=jax.random.PRNGKey(11),
    )

    assert observed_low[0] == 3
    assert observed_high[0] == 3
    assert len(observed_low) >= 2
    assert len(observed_high) >= 2
    assert int(low_delta_state.num_samples) == 7
    assert int(high_delta_state.num_samples) == 7


def test_public_allocation_modes_choose_different_parent_work_sets():
    initial_state = _make_state(
        root_out_degree=3,
        log_likelihoods=(
            float(np.log(1.0)),
            float(np.log(3.0)),
            float(np.log(4.0)),
            float(np.log(20.0)),
        ),
        out_degree=(1, 1, 1, 0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf, float(np.log(4.0))),
        max_samples=10,
    )
    work_by_mode = {}

    for allocation_target in PUBLIC_ALLOCATION_TARGETS:
        observations: list[int] = []
        state = _make_deterministic_nested_sampler(
            max_samples=10,
            shell_size=2,
        ).resume_until_goal(
            state=initial_state,
            goal_cond=_goal_after_state_counts(
                observations,
                required_observations=2,
            ),
            depth_cond=TerminationCondition(max_samples=6),
            allocation_target=allocation_target,
            delta_K=3,
            key=jax.random.PRNGKey(23),
        )
        work_by_mode[allocation_target] = _accepted_work_snapshot(
            state,
            initial_num_samples=4,
        )

    assert len(set(work_by_mode.values())) == len(PUBLIC_ALLOCATION_TARGETS)


def test_select_parent_work_targets_exact_under_allocated_strict_parent():
    state = _make_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(1, 1, 0),
        log_L_constraints=(-np.inf, 0.0, 1.0),
        max_samples=3,
    )
    plan = _selection_plan(
        target_K=(1, 2, 1),
        current_K=(1, 1, 1),
        log_L_blocks=(0.0, 1.0, 2.0),
    )

    parent_work = allocation.select_parent_work(
        key=jax.random.PRNGKey(101),
        state=state,
        plan=plan,
        num_parents=1,
    )

    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_idxs),
        np.asarray([0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_log_L_constraints),
        np.asarray([0.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.target_block_idxs),
        np.asarray([1]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_block_idxs),
        np.asarray([0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.fallback_to_root),
        np.asarray([False]),
    )


def test_select_parent_work_uses_volume_weighted_strict_parent_blocks():
    state = _make_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0, 2.0, 3.0),
        out_degree=(1, 1, 1, 0),
        log_L_constraints=(-np.inf, 0.0, 1.0, 2.0),
        max_samples=4,
    )
    plan = _selection_plan(
        target_K=(1, 1, 1, 2),
        current_K=(1, 1, 1, 1),
        log_L_blocks=(0.0, 1.0, 2.0, 3.0),
        X=(1.0, 0.5, 0.2, 0.1),
    )

    parent_work = allocation.select_parent_work(
        key=jax.random.PRNGKey(2),
        state=state,
        plan=plan,
        num_parents=1,
    )

    # For target block 3, the documented strict-parent weights are
    # proportional to X_3 / X_g = [0.1, 0.2, 0.5] for blocks 0, 1, and 2.
    # With PRNGKey(2), this selects block 1, not the first or nearest parent.
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_idxs),
        np.asarray([1]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_log_L_constraints),
        np.asarray([1.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.target_block_idxs),
        np.asarray([3]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_block_idxs),
        np.asarray([1]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.fallback_to_root),
        np.asarray([False]),
    )


def test_select_parent_work_schedules_multiple_items_for_same_block_deficit():
    state = _make_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0),
        out_degree=(1, 0),
        log_L_constraints=(-np.inf, 0.0),
        max_samples=2,
    )
    plan = _selection_plan(
        target_K=(1, 4),
        current_K=(1, 1),
        log_L_blocks=(0.0, 1.0),
    )

    parent_work = allocation.select_parent_work(
        key=jax.random.PRNGKey(109),
        state=state,
        plan=plan,
        num_parents=3,
    )

    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_idxs),
        np.asarray([0, 0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_log_L_constraints),
        np.asarray([0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.target_block_idxs),
        np.asarray([1, 1, 1]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_block_idxs),
        np.asarray([0, 0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.fallback_to_root),
        np.asarray([False, False, False]),
    )


def test_select_parent_work_schedules_zero_utility_base_target_deficits():
    state = _make_state(
        root_out_degree=3,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(0, 0, 0),
        max_samples=3,
    )
    plan = _selection_plan(
        target_K=(3, 3, 3),
        current_K=(3, 2, 1),
        log_L_blocks=(0.0, 1.0, 2.0),
        X=(1.0, 0.5, 0.25),
    )
    plan = plan._replace(
        unit_peak_utility=jnp.asarray([0.5, 1.0, 0.0]),
    )

    parent_work = allocation.select_parent_work(
        key=jax.random.PRNGKey(55),
        state=state,
        plan=plan,
        num_parents=3,
    )

    assert 2 in set(np.asarray(parent_work.target_block_idxs).tolist())


def test_select_parent_work_falls_back_to_root_when_no_strict_parent_exists():
    state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 0.0),
        out_degree=(0, 0),
        log_L_constraints=(-np.inf, -np.inf),
        max_samples=2,
    )
    plan = _selection_plan(
        target_K=(3,),
        current_K=(2,),
        log_L_blocks=(0.0,),
    )

    parent_work = allocation.select_parent_work(
        key=jax.random.PRNGKey(103),
        state=state,
        plan=plan,
        num_parents=1,
    )

    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_idxs),
        np.asarray([-1]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_log_L_constraints),
        np.asarray([-np.inf]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.target_block_idxs),
        np.asarray([0]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.parent_block_idxs),
        np.asarray([-1]),
    )
    np.testing.assert_array_equal(
        np.asarray(parent_work.fallback_to_root),
        np.asarray([True]),
    )


def test_accept_parent_work_preserves_in_flight_parent_indices():
    state = _make_state(
        root_out_degree=1,
        log_likelihoods=(0.0, 1.0),
        out_degree=(1, 0),
        log_L_constraints=(-np.inf, 0.0),
        max_samples=3,
    )
    stale_plan = _selection_plan(
        target_K=(1, 2),
        current_K=(1, 1),
        log_L_blocks=(0.0, 1.0),
    )
    fresh_plan = _selection_plan(
        target_K=(2, 1),
        current_K=(1, 1),
        log_L_blocks=(0.0, 1.0),
    )

    in_flight_work = allocation.select_parent_work(
        key=jax.random.PRNGKey(107),
        state=state,
        plan=stale_plan,
        num_parents=1,
    )
    recomputed_work = allocation.select_parent_work(
        key=jax.random.PRNGKey(107),
        state=state,
        plan=fresh_plan,
        num_parents=1,
    )
    new_samples = _make_samples(
        log_likelihoods=(1.5,),
        out_degree=(0,),
        log_L_constraints=(0.0,),
        max_samples=1,
    )

    accepted = allocation.accept_parent_work(
        state=state,
        parent_work=in_flight_work,
        new_samples=new_samples,
    )

    np.testing.assert_array_equal(
        np.asarray(in_flight_work.parent_idxs),
        np.asarray([0]),
    )
    np.testing.assert_array_equal(
        np.asarray(recomputed_work.parent_idxs),
        np.asarray([-1]),
    )
    assert int(accepted.root_out_degree) == 1
    assert int(accepted.num_samples) == 3
    np.testing.assert_array_equal(
        np.asarray(accepted.samples.out_degree[:3]),
        np.asarray([2, 0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(accepted.samples.log_L_constraints[:3]),
        np.asarray([-np.inf, 0.0, 0.0]),
    )


def test_selected_parents_are_strict_valid_contours_or_sentinel_fallback():
    initial_state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 1.0, 2.0),
        out_degree=(1, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, 0.0),
        max_samples=6,
    )
    observations: list[int] = []

    state = _make_deterministic_nested_sampler(max_samples=6).resume_until_goal(
        state=initial_state,
        goal_cond=_goal_after_state_counts(
            observations,
            required_observations=2,
        ),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
        delta_K=1,
        key=jax.random.PRNGKey(29),
    )

    initial_contours = set(np.asarray(initial_state.samples.log_likelihoods[:3]))
    for sample_idx in range(3, int(state.num_samples)):
        constraint = float(state.samples.log_L_constraints[sample_idx])
        child_log_L = float(state.samples.log_likelihoods[sample_idx])
        assert child_log_L > constraint
        assert np.isneginf(constraint) or constraint in initial_contours


def test_no_seed_plateau_fallback_increments_root_out_degree():
    initial_state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 0.0),
        out_degree=(0, 0),
        log_L_constraints=(-np.inf, -np.inf),
        max_samples=5,
    )
    observations: list[int] = []

    state = _make_deterministic_nested_sampler(max_samples=5).resume_until_goal(
        state=initial_state,
        goal_cond=_goal_after_state_counts(
            observations,
            required_observations=2,
        ),
        depth_cond=TerminationCondition(max_samples=3),
        allocation_target="uniform",
        delta_K=1,
        key=jax.random.PRNGKey(31),
    )

    assert int(state.root_out_degree) == 3
    assert int(state.num_samples) == 3
    assert float(state.samples.log_L_constraints[2]) == -np.inf
    np.testing.assert_array_equal(
        np.asarray(state.samples.out_degree[:2]),
        np.asarray([0, 0]),
    )


def test_first_v3_allocation_after_root_initialization_uses_iteration_zero():
    initial_state = _make_state(
        root_out_degree=2,
        log_likelihoods=(0.0, 0.0),
        out_degree=(0, 0),
        log_L_constraints=(-np.inf, -np.inf),
        max_samples=4,
    )
    observations: list[int] = []

    state = _make_deterministic_nested_sampler(max_samples=4).resume_until_goal(
        state=initial_state,
        goal_cond=_goal_after_state_counts(
            observations,
            required_observations=2,
        ),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
        delta_K=1,
        key=jax.random.PRNGKey(37),
    )

    assert observations == [2, 2]
    assert int(state.num_samples) == 2


def test_unsupported_legacy_depth_conditions_fail_explicitly_for_v3_run():
    ns = _make_deterministic_nested_sampler(max_samples=3)

    with pytest.raises(
            (NotImplementedError, ValueError),
            match="depth|evidence_uncert|v3",
    ):
        ns.run_until_goal(
            goal_cond=lambda state: False,
            depth_cond=TerminationCondition(evidence_uncert=0.5, max_samples=3),
            allocation_target="uniform",
            key=jax.random.PRNGKey(39),
        )


def test_acceptance_updates_only_selected_parent_out_degrees_and_constraints():
    samples = _make_samples(
        log_likelihoods=(0.0, 1.0, 2.0, 3.0),
        out_degree=(1, 1, 1, 0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf, 2.0),
        max_samples=6,
    )
    selected_parent_idxs = jnp.asarray([2, 0])
    new_samples = _make_samples(
        log_likelihoods=(2.25, 0.25),
        out_degree=(0, 0),
        log_L_constraints=(2.0, 0.0),
        max_samples=2,
    )

    accepted = samples.append_samples(
        insert_idx=jnp.asarray(4),
        parent_idxs=selected_parent_idxs,
        samples=new_samples,
        delta_parent_out_degree=jnp.asarray([1, 1], dtype=samples.out_degree.dtype),
    )

    np.testing.assert_array_equal(
        np.asarray(accepted.out_degree[:6]),
        np.asarray([2, 1, 2, 0, 0, 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(accepted.log_L_constraints[4:6]),
        np.asarray(samples.log_likelihoods[np.asarray(selected_parent_idxs)]),
    )


def test_unknown_allocation_target_fails_explicitly_for_run_and_resume():
    ns = _make_nested_sampler()
    observations: list[GoalObservation] = []

    allocation_target_error = "allocation_target|unsupported_target"

    with pytest.raises(ValueError, match=allocation_target_error):
        ns.run_until_goal(
            goal_cond=_immediate_goal(observations),
            depth_cond=TerminationCondition(max_samples=4),
            allocation_target="unsupported_target",
        )

    initial_state = ns.run_until_goal(
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="uniform",
    )

    with pytest.raises(ValueError, match=allocation_target_error):
        ns.resume_until_goal(
            state=initial_state,
            goal_cond=_immediate_goal(observations),
            depth_cond=TerminationCondition(max_samples=6),
            allocation_target="unsupported_target",
        )


def test_invalid_posterior_utility_option_fails_for_run_and_resume():
    ns = _make_deterministic_nested_sampler(max_samples=4)
    observations: list[GoalObservation] = []

    with pytest.raises(ValueError, match="posterior_utility"):
        ns.run_until_goal(
            goal_cond=_immediate_goal(observations),
            depth_cond=TerminationCondition(max_samples=3),
            allocation_target="posterior_improving",
            posterior_utility="not-a-utility",
            key=jax.random.PRNGKey(43),
        )

    initial_state = ns.run_until_goal(
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=3),
        allocation_target="uniform",
        key=jax.random.PRNGKey(45),
    )

    with pytest.raises(ValueError, match="posterior_utility"):
        ns.resume_until_goal(
            state=initial_state,
            goal_cond=_immediate_goal(observations),
            depth_cond=TerminationCondition(max_samples=4),
            allocation_target="posterior_improving",
            posterior_utility="not-a-utility",
            key=jax.random.PRNGKey(47),
        )


def test_conservative_posterior_utility_option_is_public_for_run_and_resume():
    ns = _make_deterministic_nested_sampler(max_samples=4)
    observations: list[GoalObservation] = []

    initial_state = ns.run_until_goal(
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=3),
        allocation_target="posterior_improving",
        posterior_utility="conservative",
        key=jax.random.PRNGKey(49),
    )

    resumed_state = ns.resume_until_goal(
        state=initial_state,
        goal_cond=_immediate_goal(observations),
        depth_cond=TerminationCondition(max_samples=4),
        allocation_target="posterior_improving",
        posterior_utility="conservative",
        key=jax.random.PRNGKey(51),
    )

    assert isinstance(initial_state, State)
    assert isinstance(resumed_state, State)


def test_legacy_run_and_resume_remain_fixed_live_point_compatibility_path():
    ns = _make_nested_sampler()

    state = ns.run(key=jax.random.PRNGKey(57))
    resumed = ns.resume(state, key=jax.random.PRNGKey(59))

    assert isinstance(state, State)
    assert isinstance(resumed, State)
    assert int(state.num_samples) <= int(ns.max_samples)
    assert int(resumed.num_samples) >= int(state.num_samples)
    assert "legacy fixed-live-point" in NestedSampler.run.__doc__
    assert "legacy fixed-live-point" in NestedSampler.resume.__doc__
