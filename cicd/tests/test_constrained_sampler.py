import dataclasses

import jax
import numpy as np
from jax import numpy as jnp
from jax import random

from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    EllipsoidalDirection,
    UniDimSliceSampler,
    _new_proposal,
    _sample_complete_chains,
    _sample_ellipsoidal_direction,
    sample_request,
)
from jaxns.multi_ellipsoid_utils import (
    component_probabilities,
    component_probabilities_reference,
    empty_sampler_data,
)
from jaxns.pytree import PureDataclassPytree, TreeField
from jaxns.samples import SeedPoint


@dataclasses.dataclass(slots=True, frozen=True)
class QuadraticModel(PureDataclassPytree):
    """Small traceable model for fixed-stream sampler comparisons."""

    centre: jax.Array  # [D]

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan=True,
    ):
        del args, params, allow_nan
        return -jnp.sum(jnp.square(U - self.centre))


QuadraticModel.register_pytree()


def _request(width: int) -> ConstrainedSampleRequest:
    seeds = jnp.linspace(0.35, 0.65, width * 2).reshape((width, 2))
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    log_likelihoods = jax.vmap(model.log_likelihood)(seeds)
    return ConstrainedSampleRequest(
        keys=random.split(random.PRNGKey(244), width),
        valid=jnp.ones((width,), dtype=jnp.bool_),
        log_L_constraints=jnp.full((width,), -0.25),
        seed_points=SeedPoint(
            U0=seeds,
            log_L0=log_likelihoods,
        ),
        sampler_data=None,
    )


def test_slice_continuations_preserve_complete_chain_outputs():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=5,
        collect_phantom_samples=True,
        phantom_burn_in=2,
    )
    request = _request(width=6)
    reference = jax.jit(
        lambda value: _sample_complete_chains(sampler, value)
    )(request)
    continued = jax.jit(
        lambda value: sample_request(sampler, value)
    )(request)

    # The fixed logical IDs, random streams, phantom prefix, and counters must
    # survive removal of the barrier between individual slice transitions.
    for expected, actual in zip(
        jax.tree.leaves(reference),
        jax.tree.leaves(continued),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_slice_continuations_handle_one_scalar_transition():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=1,
        collect_phantom_samples=False,
    )
    request = _request(width=1)
    result = jax.jit(
        lambda value: sample_request(sampler, value)
    )(request)

    assert result.log_likelihoods.shape == (1,)
    assert result.phantom_samples.log_L.shape == (1, 0)
    assert int(result.num_likelihood_evaluations[0]) >= 1


def test_nonperfect_batch_keeps_complete_chain_reference():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=2,
        no_step_out=False,
    )
    request = _request(width=2)
    reference = _sample_complete_chains(sampler, request)
    observed = sample_request(sampler, request)

    for expected, actual in zip(
        jax.tree.leaves(reference),
        jax.tree.leaves(observed),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _log_likelihood_1d(U):
    x = U[0]
    return -(x - 0.5) ** 2


def test_new_proposal_nonperfect_first_uses_full_slice_width():
    U0 = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    point_U, log_L, num_evals, _, next_slice_width, _ = _new_proposal(
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

    _, _, num_evals, _, next_slice_width, _ = _new_proposal(
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

    point_U, _, _, direction_1, slice_width_1, _ = _new_proposal(
        key=random.PRNGKey(2),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    _, _, num_evals_2, _, slice_width_2, _ = _new_proposal(
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


def test_component_selection_is_strict_and_volume_weighted():
    data = empty_sampler_data(num_components=3, dimension=2)
    data = dataclasses.replace(
        data,
        log_volumes=jnp.log(jnp.asarray([1.0, 3.0, 8.0])),
        log_L_max=jnp.asarray([0.0, 2.0, 5.0]),
        valid=jnp.asarray([True, True, False]),
    )
    probabilities = component_probabilities(data, jnp.asarray(0.0))
    reference = component_probabilities_reference(
        np.log(np.asarray([1.0, 3.0, 8.0])),
        np.asarray([0.0, 2.0, 5.0]),
        np.asarray([True, True, False]),
        0.0,
    )

    # The first peak equals the strict contour and is therefore ineligible;
    # invalid high-volume geometry must not leak into selection either.
    np.testing.assert_allclose(probabilities, reference)
    np.testing.assert_allclose(probabilities, np.asarray([0.0, 1.0, 0.0]))


def test_ellipsoidal_direction_jit_vmap_and_degenerate_fallback():
    data = empty_sampler_data(num_components=2, dimension=2)
    data = dataclasses.replace(
        data,
        radii=jnp.asarray([[8.0, 1.0], [1.0, 8.0]]),
        rotations=jnp.repeat(jnp.eye(2)[None, :, :], 2, axis=0),
        log_volumes=jnp.zeros((2,)),
        log_L_max=jnp.asarray([0.0, 2.0]),
        valid=jnp.asarray([True, True]),
    )
    keys = random.split(random.PRNGKey(13), 2048)
    draw = jax.jit(jax.vmap(
        lambda key: _sample_ellipsoidal_direction(
            key,
            TreeField(jnp.asarray([0.5, 0.5])),
            jnp.asarray(0.0),
            data,
            0.0,
        ).tree
    ))
    directions = draw(keys)
    np.testing.assert_allclose(
        jnp.linalg.norm(directions, axis=1),
        1.0,
        rtol=1e-6,
    )
    assert jnp.mean(jnp.square(directions[:, 1])) > 0.75

    invalid = dataclasses.replace(
        data,
        radii=jnp.full_like(data.radii, jnp.nan),
        valid=jnp.zeros_like(data.valid),
    )
    fallback = _sample_ellipsoidal_direction(
        random.PRNGKey(14),
        TreeField(jnp.asarray([0.5, 0.5])),
        jnp.asarray(10.0),
        invalid,
        0.0,
    ).tree
    assert jnp.all(jnp.isfinite(fallback))
    np.testing.assert_allclose(jnp.linalg.norm(fallback), 1.0, rtol=1e-6)


def test_ellipsoidal_configuration_validation():
    with np.testing.assert_raises_regex(ValueError, "num_components"):
        EllipsoidalDirection(num_components=0)

    # A concrete model is unnecessary for checking the static configuration.
    with np.testing.assert_raises_regex(ValueError, "prob_isotropic"):
        UniDimSliceSampler(
            model=None,
            num_slices=2,
            direction=EllipsoidalDirection(prob_isotropic=1.1),
        )
