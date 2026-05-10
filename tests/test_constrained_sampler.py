import numpy as np
from jax import numpy as jnp, random

import jaxns.constrained_sampler as constrained_sampler
from jaxns.constrained_sampler import _new_proposal
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
