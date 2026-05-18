import numpy as np
import pytest
from jax import numpy as jnp, random

import jaxns.constrained_sampler as constrained_sampler
from jaxns.constrained_sampler import _new_proposal
from jaxns.constrained_sampler import _new_proposal_python
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.em_gmm import DirectionAdaptationContext
from jaxns.pytree import TreeField
from jaxns.samples import SeedPoint


class _SliceToyModel:
    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan: bool = True,
    ):
        del args, params, allow_nan
        x = jnp.ravel(jnp.asarray(U))[0]
        return -jnp.square(x - jnp.asarray(0.37, dtype=x.dtype))


def _log_likelihood_1d(U):
    x = U[0]
    return -(x - 0.5) ** 2


def _needle_likelihood_1d(U):
    x = U[0]
    return jnp.where(jnp.equal(x, 0.5), 0.0, -1.0)


def _near_float32_contour_likelihood_1d(U):
    x = U[0]
    endpoint_log_l = jnp.asarray(-0.99999999, dtype=jnp.float64)
    accepted_log_l = jnp.asarray(0.0, dtype=jnp.float64)
    return jnp.where((x < 0.45) | (x > 0.55), endpoint_log_l, accepted_log_l)


def _left_interval_likelihood_1d(U):
    x = U[0]
    return jnp.where(x < jnp.asarray(0.25, dtype=x.dtype), 0.0, -1.0)


def _frozen_direction_context() -> DirectionAdaptationContext:
    return DirectionAdaptationContext(
        component_means=jnp.asarray([[0.5]]),
        component_radii=jnp.asarray([[1.0]]),
        component_rotations=jnp.asarray([[[1.0]]]),
        component_probabilities=jnp.asarray([1.0]),
        component_integrated_volumes=jnp.asarray([1.0]),
        kernel_version=17,
        allocation_target="uniform",
    )


def _run_slice_sampler_loop(*, force_python_loop: bool):
    model = _SliceToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=5,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
        direction_kernel="ellipsoidal",
        max_shrinkage_steps=16,
    )
    seed_point = SeedPoint(
        U0=jnp.asarray([0.25]),
        log_L0=model.log_likelihood(jnp.asarray([0.25])),
    )
    return sampler.get_sample(
        key=random.PRNGKey(42),
        log_L_constraint=jnp.asarray(-0.25),
        seed_point=seed_point,
        adaptation_context={
            "force_python_loop": force_python_loop,
            "direction_adaptation_context": _frozen_direction_context(),
        },
    )


def _run_step_out_slice_sampler_loop(
        *,
        force_python_loop: bool,
        num_slices: int,
):
    model = _SliceToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        no_step_out=False,
        collect_phantom_samples=True,
        phantom_burn_in=0,
        direction_kernel="ellipsoidal",
        max_shrinkage_steps=16,
    )
    seed_point = SeedPoint(
        U0=jnp.asarray([0.25]),
        log_L0=model.log_likelihood(jnp.asarray([0.25])),
    )
    return sampler.get_sample(
        key=random.PRNGKey(42),
        log_L_constraint=jnp.asarray(-0.25),
        seed_point=seed_point,
        adaptation_context={
            "force_python_loop": force_python_loop,
            "direction_adaptation_context": _frozen_direction_context(),
        },
    )


class _CountingProposalLikelihood:
    def __init__(self, model: _SliceToyModel):
        self.model = model
        self.calls = []

    def __call__(self, U):
        self.calls.append(np.asarray(U).copy())
        return self.model.log_likelihood(U)


class _RecordingLikelihood:
    def __init__(self, log_likelihood_fn):
        self.log_likelihood_fn = log_likelihood_fn
        self.calls = []

    def __call__(self, U):
        self.calls.append(np.asarray(U).copy())
        return self.log_likelihood_fn(U)


def _run_dispatched_forced_slice_sampler():
    model = _SliceToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
        direction_kernel="ellipsoidal",
        max_shrinkage_steps=16,
    )
    seed_point = SeedPoint(
        U0=jnp.asarray([0.25]),
        log_L0=model.log_likelihood(jnp.asarray([0.25])),
    )
    proposal_log_likelihood = _CountingProposalLikelihood(model)
    result = sampler.get_sample(
        key=random.PRNGKey(42),
        log_L_constraint=jnp.asarray(-0.25),
        seed_point=seed_point,
        adaptation_context={
            "proposal_log_likelihood_fn": proposal_log_likelihood,
            "force_python_loop": True,
            "direction_adaptation_context": _frozen_direction_context(),
        },
    )
    return model, proposal_log_likelihood, result


def _run_dispatched_forced_step_out_sampler():
    model = _SliceToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        no_step_out=False,
        collect_phantom_samples=False,
        direction_kernel="ellipsoidal",
        max_shrinkage_steps=16,
    )
    seed_point = SeedPoint(
        U0=jnp.asarray([0.25]),
        log_L0=model.log_likelihood(jnp.asarray([0.25])),
    )
    proposal_log_likelihood = _CountingProposalLikelihood(model)
    result = sampler.get_sample(
        key=random.PRNGKey(42),
        log_L_constraint=jnp.asarray(-0.25),
        seed_point=seed_point,
        adaptation_context={
            "proposal_log_likelihood_fn": proposal_log_likelihood,
            "force_python_loop": True,
            "direction_adaptation_context": _frozen_direction_context(),
        },
    )
    return proposal_log_likelihood, result


def _patch_geometry_helper_spies(monkeypatch):
    calls = {
        "_slice_bounds": 0,
        "_pick_point_in_interval": 0,
        "_shrink_interval": 0,
    }

    def spy(name, original):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for name in calls:
        monkeypatch.setattr(
            constrained_sampler,
            name,
            spy(name, getattr(constrained_sampler, name)),
        )

    return calls


def test_unidim_slice_sampler_forced_dispatch_preserves_semantics():
    model, proposal_log_likelihood, result = _run_dispatched_forced_slice_sampler()

    U_sample, log_L, num_evals, phantom_samples = result
    log_L_constraint = jnp.asarray(-0.25)

    assert int(num_evals) == 5
    assert len(proposal_log_likelihood.calls) == int(num_evals)
    assert float(log_L) > float(log_L_constraint)
    np.testing.assert_allclose(
        np.asarray(model.log_likelihood(U_sample)),
        np.asarray(log_L),
    )
    np.testing.assert_allclose(
        np.asarray(U_sample),
        np.asarray([0.5694930886407388]),
    )
    np.testing.assert_allclose(
        np.asarray(log_L),
        np.asarray(-0.03979749241542166),
    )
    assert phantom_samples.U_samples is None
    assert phantom_samples.valid_mask.shape == (2,)
    assert phantom_samples.log_L.shape == (2,)
    np.testing.assert_array_equal(
        np.asarray(phantom_samples.valid_mask),
        np.asarray([True, True]),
    )
    assert np.all(np.asarray(phantom_samples.log_L) > float(log_L_constraint))
    np.testing.assert_allclose(
        np.asarray(phantom_samples.log_L),
        np.asarray([-0.22724820340151832, -0.07628074487208507]),
    )


def test_new_proposal_python_no_step_out_mixed_dtype_matches_jax():
    U0 = TreeField(jnp.asarray([0.25], dtype=jnp.float32))
    direction = TreeField(jnp.asarray([1.0], dtype=jnp.float64))
    kwargs = dict(
        key=random.PRNGKey(11),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf, dtype=jnp.float32),
        no_step_out=True,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0, dtype=jnp.float64),
        log_likelihood_fn=_log_likelihood_1d,
        log_L0=jnp.asarray(-0.0625, dtype=jnp.float64),
        max_shrinkage_steps=4,
    )

    jax_point, jax_log_l, jax_evals, _, jax_width = _new_proposal(**kwargs)
    python_point, python_log_l, python_evals, python_direction, python_width = (
        _new_proposal_python(**kwargs)
    )

    np.testing.assert_array_equal(
        np.asarray(python_point.tree),
        np.asarray(jax_point.tree),
    )
    np.testing.assert_array_equal(np.asarray(python_log_l), np.asarray(jax_log_l))
    assert int(python_evals) == int(jax_evals)
    np.testing.assert_array_equal(np.asarray(python_width), np.asarray(jax_width))
    np.testing.assert_array_equal(
        np.asarray(python_direction.tree),
        np.asarray(direction.tree),
    )


def test_new_proposal_python_step_out_endpoint_mixed_dtype_matches_jax():
    U0 = TreeField(jnp.asarray([0.5], dtype=jnp.float32))
    direction = TreeField(jnp.asarray([1.0], dtype=jnp.float32))
    kwargs = dict(
        key=random.PRNGKey(123),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(0.1, dtype=jnp.float32),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0, dtype=jnp.float32),
        log_likelihood_fn=_near_float32_contour_likelihood_1d,
        log_L0=jnp.asarray(0.0, dtype=jnp.float64),
        max_shrinkage_steps=4,
    )

    jax_point, jax_log_l, jax_evals, _, jax_width = _new_proposal(**kwargs)
    python_point, python_log_l, python_evals, python_direction, python_width = (
        _new_proposal_python(**kwargs)
    )

    np.testing.assert_array_equal(
        np.asarray(python_point.tree),
        np.asarray(jax_point.tree),
    )
    np.testing.assert_array_equal(np.asarray(python_log_l), np.asarray(jax_log_l))
    assert int(python_evals) == int(jax_evals)
    assert int(python_evals) == 12
    np.testing.assert_array_equal(np.asarray(python_width), np.asarray(jax_width))
    np.testing.assert_array_equal(
        np.asarray(python_direction.tree),
        np.asarray(direction.tree),
    )


def test_new_proposal_python_rejected_budget_matches_jax():
    U0 = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    for max_shrinkage_steps in (0, 2):
        kwargs = dict(
            key=random.PRNGKey(7),
            U0=U0,
            direction=direction,
            slice_width=jnp.asarray(jnp.inf),
            no_step_out=True,
            gradient_guided=False,
            log_L_constraint=jnp.asarray(-0.5),
            log_likelihood_fn=_needle_likelihood_1d,
            log_L0=jnp.asarray(0.0),
            max_shrinkage_steps=max_shrinkage_steps,
        )

        jax_point, jax_log_l, jax_evals, _, jax_width = _new_proposal(**kwargs)
        python_point, python_log_l, python_evals, _, python_width = (
            _new_proposal_python(**kwargs)
        )

        np.testing.assert_array_equal(
            np.asarray(python_point.tree),
            np.asarray(jax_point.tree),
        )
        np.testing.assert_array_equal(
            np.asarray(python_log_l),
            np.asarray(jax_log_l),
        )
        assert int(python_evals) == int(jax_evals)
        assert int(python_evals) == max_shrinkage_steps + 1
        np.testing.assert_array_equal(
            np.asarray(python_width),
            np.asarray(jax_width),
        )


def test_new_proposal_python_defers_log_l0_none_fallback_until_exhausted():
    U0 = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))
    kwargs = dict(
        key=random.PRNGKey(1),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=True,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-0.5),
        max_shrinkage_steps=4,
    )

    jax_point, jax_log_l, jax_evals, _, jax_width = _new_proposal(
        **kwargs,
        log_likelihood_fn=_left_interval_likelihood_1d,
        log_L0=None,
    )
    recording_likelihood = _RecordingLikelihood(_left_interval_likelihood_1d)
    python_point, python_log_l, python_evals, _, python_width = (
        _new_proposal_python(
            **kwargs,
            log_likelihood_fn=recording_likelihood,
            log_L0=None,
        )
    )

    np.testing.assert_array_equal(
        np.asarray(python_point.tree),
        np.asarray(jax_point.tree),
    )
    np.testing.assert_array_equal(
        np.asarray(python_log_l),
        np.asarray(jax_log_l),
    )
    assert int(jax_evals) == 2
    assert int(python_evals) == int(jax_evals)
    assert len(recording_likelihood.calls) == int(jax_evals)
    assert recording_likelihood.calls[0][0] > 0.25
    assert recording_likelihood.calls[1][0] < 0.25
    assert not any(
        np.array_equal(call, np.asarray(U0.tree))
        for call in recording_likelihood.calls
    )
    np.testing.assert_array_equal(
        np.asarray(python_width),
        np.asarray(jax_width),
    )


def test_unidim_slice_sampler_forced_dispatch_step_out_counts_width_evals():
    proposal_log_likelihood, result = _run_dispatched_forced_step_out_sampler()

    U_sample, log_L, num_evals, phantom_samples = result

    assert int(num_evals) == len(proposal_log_likelihood.calls)
    assert int(num_evals) >= 4
    assert float(log_L) > -0.25
    assert 0.0 <= float(np.asarray(U_sample)[0]) <= 1.0
    assert phantom_samples.U_samples is None
    assert phantom_samples.valid_mask.shape == (0,)
    width_evals = np.sort(
        np.ravel(
            np.asarray(
                [
                    proposal_log_likelihood.calls[1],
                    proposal_log_likelihood.calls[2],
                ]
            )
        )
    )
    assert width_evals[0] == 0.0
    assert 0.0 < width_evals[1] < 1.0


def test_unidim_slice_sampler_pure_jax_path_still_uses_jax_geometry_helpers(
        monkeypatch,
):
    helper_calls = _patch_geometry_helper_spies(monkeypatch)

    _run_slice_sampler_loop(force_python_loop=False)

    assert helper_calls["_slice_bounds"] > 0
    assert helper_calls["_pick_point_in_interval"] > 0
    assert helper_calls["_shrink_interval"] > 0


@pytest.mark.parametrize("num_slices", (1, 2, 5))
def test_unidim_slice_sampler_forced_python_loop_key_schedule(
        monkeypatch,
        num_slices,
):
    proposal_keys = []
    direction_keys = []
    materialized_key_counts = []
    original_new_proposal_python = constrained_sampler._new_proposal_python
    original_sample_direction = constrained_sampler._sample_direction_from_kernel
    original_materialize_split_keys = constrained_sampler._materialize_split_keys

    def recording_new_proposal_python(*args, **kwargs):
        proposal_keys.append(np.asarray(kwargs["key"]))
        return original_new_proposal_python(*args, **kwargs)

    def recording_sample_direction_from_kernel(*args, **kwargs):
        direction_keys.append(np.asarray(kwargs["key"]))
        return original_sample_direction(*args, **kwargs)

    def recording_materialize_split_keys(*args, **kwargs):
        count = kwargs["count"] if "count" in kwargs else args[1]
        materialized_key_counts.append(count)
        return original_materialize_split_keys(*args, **kwargs)

    monkeypatch.setattr(
        constrained_sampler,
        "_new_proposal_python",
        recording_new_proposal_python,
    )

    monkeypatch.setattr(
        constrained_sampler,
        "_sample_direction_from_kernel",
        recording_sample_direction_from_kernel,
    )

    monkeypatch.setattr(
        constrained_sampler,
        "_materialize_split_keys",
        recording_materialize_split_keys,
    )

    model = _SliceToyModel()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=0,
        direction_kernel="ellipsoidal",
        trajectory="straight_line",
        max_shrinkage_steps=16,
    )
    seed_point = SeedPoint(
        U0=jnp.asarray([0.25]),
        log_L0=model.log_likelihood(jnp.asarray([0.25])),
    )

    sampler.get_sample(
        key=random.PRNGKey(42),
        log_L_constraint=jnp.asarray(-0.25),
        seed_point=seed_point,
        adaptation_context={
            "force_python_loop": True,
            "direction_adaptation_context": _frozen_direction_context(),
        },
    )

    loop_count = num_slices - 1
    direction_key, sample_key = random.split(random.PRNGKey(42), 2)
    sample_key, init_sample_key, init_direction_key = random.split(
        sample_key,
        3,
    )
    proposal_key, direction_scan_key = random.split(sample_key, 2)
    expected_proposal_keys = random.split(proposal_key, loop_count)
    expected_direction_keys = random.split(direction_scan_key, loop_count)

    assert materialized_key_counts == [loop_count, max(loop_count - 1, 0)]
    assert len(proposal_keys) == num_slices
    np.testing.assert_array_equal(proposal_keys[0], np.asarray(init_sample_key))
    assert len(proposal_keys[1:]) == loop_count
    for actual_key, expected_key in zip(
            proposal_keys[1:],
            expected_proposal_keys,
    ):
        np.testing.assert_array_equal(actual_key, np.asarray(expected_key))

    assert len(direction_keys) == num_slices
    np.testing.assert_array_equal(direction_keys[0], np.asarray(direction_key))
    if num_slices > 1:
        np.testing.assert_array_equal(
            direction_keys[1],
            np.asarray(init_direction_key),
        )
    for actual_key, expected_key in zip(
            direction_keys[2:],
            expected_direction_keys[:-1],
    ):
        np.testing.assert_array_equal(actual_key, np.asarray(expected_key))


@pytest.mark.parametrize("num_slices", (2, 5))
def test_unidim_slice_sampler_step_out_fused_and_python_loop_outputs_match(
        num_slices,
):
    python_result = _run_step_out_slice_sampler_loop(
        force_python_loop=True,
        num_slices=num_slices,
    )
    fused_result = _run_step_out_slice_sampler_loop(
        force_python_loop=False,
        num_slices=num_slices,
    )

    python_u, python_log_l, python_evals, python_phantoms = python_result
    fused_u, fused_log_l, fused_evals, fused_phantoms = fused_result

    np.testing.assert_array_equal(np.asarray(fused_u), np.asarray(python_u))
    np.testing.assert_array_equal(
        np.asarray(fused_log_l),
        np.asarray(python_log_l),
    )
    assert int(fused_evals) == int(python_evals)
    assert fused_phantoms.U_samples is None
    assert python_phantoms.U_samples is None
    np.testing.assert_array_equal(
        np.asarray(fused_phantoms.valid_mask),
        np.asarray(python_phantoms.valid_mask),
    )
    np.testing.assert_array_equal(
        np.asarray(fused_phantoms.log_L),
        np.asarray(python_phantoms.log_L),
    )


def test_unidim_slice_sampler_auto_forces_dispatched_likelihood(monkeypatch):
    proposal_calls = []
    cumulative_calls = []
    original_new_proposal_python = constrained_sampler._new_proposal_python

    def recording_new_proposal_python(*args, **kwargs):
        proposal_calls.append((args, kwargs))
        return original_new_proposal_python(*args, **kwargs)

    def recording_cumulative_op_static(*args, **kwargs):
        cumulative_calls.append((args, kwargs))
        raise AssertionError("cumulative_op_static should not be used")

    monkeypatch.setattr(
        constrained_sampler,
        "_new_proposal_python",
        recording_new_proposal_python,
    )
    monkeypatch.setattr(
        constrained_sampler,
        "cumulative_op_static",
        recording_cumulative_op_static,
    )

    model = _SliceToyModel()
    proposal_log_likelihood = _CountingProposalLikelihood(model)
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        collect_phantom_samples=False,
        direction_kernel="ellipsoidal",
        trajectory="straight_line",
        max_shrinkage_steps=16,
    )
    seed_point = SeedPoint(
        U0=jnp.asarray([0.25]),
        log_L0=model.log_likelihood(jnp.asarray([0.25])),
    )

    _, _, num_evals, _ = sampler.get_sample(
        key=random.PRNGKey(42),
        log_L_constraint=jnp.asarray(-0.25),
        seed_point=seed_point,
        adaptation_context={
            "proposal_log_likelihood_fn": proposal_log_likelihood,
            "direction_adaptation_context": _frozen_direction_context(),
        },
    )

    assert len(proposal_calls) == 3
    assert cumulative_calls == []
    assert len(proposal_log_likelihood.calls) == int(num_evals)


def test_unidim_slice_sampler_fused_and_python_loop_outputs_match():
    python_result = _run_slice_sampler_loop(force_python_loop=True)
    fused_result = _run_slice_sampler_loop(force_python_loop=False)

    python_u, python_log_l, python_evals, python_phantoms = python_result
    fused_u, fused_log_l, fused_evals, fused_phantoms = fused_result

    np.testing.assert_array_equal(np.asarray(fused_u), np.asarray(python_u))
    np.testing.assert_array_equal(
        np.asarray(fused_log_l),
        np.asarray(python_log_l),
    )
    assert int(fused_evals) == int(python_evals)
    assert fused_phantoms.U_samples is None
    assert python_phantoms.U_samples is None
    np.testing.assert_array_equal(
        np.asarray(fused_phantoms.valid_mask),
        np.asarray(python_phantoms.valid_mask),
    )
    np.testing.assert_array_equal(
        np.asarray(fused_phantoms.log_L),
        np.asarray(python_phantoms.log_L),
    )


def test_unidim_slice_sampler_fused_mode_uses_scan_helper(monkeypatch):
    calls = []
    original = constrained_sampler.cumulative_op_static

    def recording_cumulative_op_static(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        constrained_sampler,
        "cumulative_op_static",
        recording_cumulative_op_static,
    )

    _run_slice_sampler_loop(force_python_loop=False)

    assert len(calls) == 1


def test_new_proposal_nonperfect_first_uses_full_slice_width():
    U0 = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    point_U, log_L, num_evals, _, next_slice_width = _new_proposal(
        key=random.PRNGKey(0),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    np.testing.assert_allclose(next_slice_width, 2.0)
    assert 0.0 <= float(point_U.tree[0]) <= 1.0
    assert float(log_L) > -1.0
    assert int(num_evals) >= 1


def test_new_proposal_nonperfect_finite_width_clips_and_steps_out():
    U0 = TreeField(jnp.asarray([0.95]))
    direction = TreeField(jnp.asarray([1.0]))

    _, _, num_evals, _, next_slice_width = _new_proposal(
        key=random.PRNGKey(1),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(0.05),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    assert int(num_evals) > 1
    assert float(next_slice_width) > 0.0
    np.testing.assert_allclose(next_slice_width, 2.0)


def test_new_proposal_nonperfect_reuses_previous_width():
    U0 = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    point_U, _, _, direction_1, slice_width_1 = _new_proposal(
        key=random.PRNGKey(2),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    _, _, num_evals_2, _, slice_width_2 = _new_proposal(
        key=random.PRNGKey(3),
        U0=point_U,
        direction=direction_1,
        slice_width=slice_width_1,
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    assert isinstance(direction_1, TreeField)
    assert not isinstance(direction_1.tree, TreeField)
    np.testing.assert_allclose(direction_1.tree, jnp.asarray([1.0]))
    assert jnp.isfinite(slice_width_1)
    np.testing.assert_allclose(slice_width_1, 2.0)
    assert int(num_evals_2) > 1
    assert float(slice_width_2) > 0.0
