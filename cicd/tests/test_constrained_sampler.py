import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from jaxns.constrained_sampler import (
    EllipsoidalDirection,
    UniDimSliceSampler,
    sample_request,
)
from jaxns.pytree import PureDataclassPytree, TreeField
from jaxns.samples import SeedPoint
from jaxns.sampling.batching import (
    sample_complete_chains as _sample_complete_chains,
)
from jaxns.sampling.ellipsoid import (
    component_probabilities,
    component_probabilities_reference,
    empty_sampler_data,
)
from jaxns.sampling.protocol import ConstrainedSampleRequest
from jaxns.sampling.slice import (
    _new_proposal,
    _sample_ellipsoidal_direction,
)


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


@dataclasses.dataclass(slots=True, frozen=True)
class CircularModel(PureDataclassPytree):
    """Traceable likelihood using minimum-image circular displacement."""

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
        displacement = jnp.mod(U - self.centre + 0.5, 1.0) - 0.5
        return -jnp.sum(jnp.square(displacement))


CircularModel.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class CylinderModel(PureDataclassPytree):
    """Traceable likelihood with one circular and one hard coordinate."""

    centre: jax.Array  # [2]

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan=True,
    ):
        del args, params, allow_nan
        circular = jnp.mod(U[0] - self.centre[0] + 0.5, 1.0) - 0.5
        hard = U[1] - self.centre[1]
        displacement = jnp.asarray([circular, hard])
        return -jnp.sum(jnp.square(displacement / 0.1))


CylinderModel.register_pytree()


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


def _periodic_request(width: int) -> ConstrainedSampleRequest:
    seeds = jnp.linspace(0.01, 0.99, width * 2).reshape((width, 2))
    model = CircularModel(centre=jnp.asarray([0.99, 0.01]))
    log_likelihoods = jax.vmap(model.log_likelihood)(seeds)
    return ConstrainedSampleRequest(
        keys=random.split(random.PRNGKey(275), width),
        valid=jnp.ones((width,), dtype=jnp.bool_),
        log_L_constraints=jnp.full((width,), -0.5),
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
        num_slices=32,
        collect_phantom_samples=True,
        max_phantom_samples=2,
    )
    request = _request(width=8)
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


def test_phantom_capacity_retains_start_prefix_and_excludes_classic():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    seed = SeedPoint(
        U0=jnp.asarray([0.35, 0.65]),
        log_L0=model.log_likelihood(jnp.asarray([0.35, 0.65])),
    )
    complete_prefix_sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        collect_phantom_samples=True,
    )
    bounded_sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        collect_phantom_samples=True,
        max_phantom_samples=2,
    )
    key = random.PRNGKey(284)

    complete = complete_prefix_sampler.get_sample(
        key,
        jnp.asarray(-0.25),
        seed,
    )
    bounded = bounded_sampler.get_sample(
        key,
        jnp.asarray(-0.25),
        seed,
    )

    # Four generated transitions contain exactly three eligible phantoms; the
    # fourth output is reserved for the classic replacement. Changing only the
    # retained width cannot change that replacement or its evaluation count.
    assert complete[3].log_L.shape == (3,)
    assert bounded[3].log_L.shape == (2,)
    np.testing.assert_array_equal(bounded[3].log_L, complete[3].log_L[:2])
    np.testing.assert_array_equal(
        bounded[3].U_samples,
        complete[3].U_samples[:2],
    )
    np.testing.assert_array_equal(bounded[0], complete[0])
    np.testing.assert_array_equal(bounded[1], complete[1])
    np.testing.assert_array_equal(bounded[2], complete[2])
    assert not np.array_equal(
        np.asarray(complete[0]),
        complete[3].U_samples[-1],
    )


def test_phantom_capacity_validation_and_burn_in_deprecation():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))

    for non_python_integer in (np.int64(2), jnp.asarray(2)):
        with pytest.raises(TypeError, match="Python integer"):
            UniDimSliceSampler(
                model=model,
                num_slices=4,
                collect_phantom_samples=True,
                max_phantom_samples=non_python_integer,
            )
    with pytest.raises(ValueError, match="positive"):
        UniDimSliceSampler(
            model=model,
            num_slices=4,
            collect_phantom_samples=True,
            max_phantom_samples=0,
        )
    with pytest.raises(ValueError, match="num_slices - 1"):
        UniDimSliceSampler(
            model=model,
            num_slices=4,
            collect_phantom_samples=True,
            max_phantom_samples=4,
        )
    with pytest.raises(ValueError, match="collect_phantom_samples"):
        UniDimSliceSampler(
            model=model,
            num_slices=4,
            max_phantom_samples=1,
        )
    with pytest.warns(DeprecationWarning, match="phantom_burn_in"):
        legacy = UniDimSliceSampler(
            model=model,
            num_slices=4,
            collect_phantom_samples=True,
            phantom_burn_in=1,
        )
    assert legacy.num_phantom() == 2


def test_periodic_slice_continuations_preserve_complete_chain_outputs():
    """The pool scheduler preserves each random-chart scalar transition."""
    model = CircularModel(centre=jnp.asarray([0.99, 0.01]))
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=32,
        collect_phantom_samples=True,
        max_phantom_samples=2,
    )._with_periodic((True, True))
    request = _periodic_request(width=8)
    reference = jax.jit(
        lambda value: _sample_complete_chains(sampler, value)
    )(request)
    continued = jax.jit(
        lambda value: sample_request(sampler, value)
    )(request)

    for expected, actual in zip(
        jax.tree.leaves(reference),
        jax.tree.leaves(continued),
        strict=True,
    ):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        if np.issubdtype(expected.dtype, np.inexact):
            # Separate complete-chain and continuation lowering can reassociate
            # the final circular subtraction by ordinary machine roundoff.
            tolerance = 32 * np.finfo(expected.dtype).eps
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
            )
        else:
            np.testing.assert_array_equal(actual, expected)
    canonical_values = (
        *jax.tree.leaves(continued.U_samples),
        *jax.tree.leaves(continued.phantom_samples.U_samples),
    )
    for leaf in canonical_values:
        assert np.all(np.asarray(leaf) >= 0.0)
        assert np.all(np.asarray(leaf) < 1.0)


def test_periodic_scalar_and_vmapped_chains_share_one_transition_law():
    """Batch width changes execution only, not a logical chain's result."""
    width = 4
    model = CircularModel(centre=jnp.asarray([0.99, 0.01]))
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=8,
    )._with_periodic((True, True))
    request = _periodic_request(width)

    vmapped = sample_request(sampler, request)
    scalar_results = []
    for lane in range(width):
        scalar_request = jax.tree.map(
            lambda value, index=lane: value[index:index + 1],
            request,
        )
        scalar_results.append(sample_request(sampler, scalar_request))
    scalar = jax.tree.map(
        lambda *values: jnp.concatenate(values, axis=0),
        *scalar_results,
    )

    for expected, actual in zip(
        jax.tree.leaves(vmapped),
        jax.tree.leaves(scalar),
        strict=True,
    ):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        if np.issubdtype(expected.dtype, np.inexact):
            tolerance = 32 * np.finfo(expected.dtype).eps
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
            )
        else:
            np.testing.assert_array_equal(actual, expected)


def test_random_charts_preserve_uniform_circular_measure():
    """A flat circular slice remains uniform after random chart mixtures."""
    U0 = TreeField(jnp.asarray([0.99]))
    direction = TreeField(jnp.asarray([1.0]))

    def draw(key):
        point, _, _, _, _, _ = _new_proposal(
            key=key,
            U0=U0,
            direction=direction,
            slice_width=jnp.asarray(jnp.inf),
            no_step_out=True,
            gradient_guided=False,
            log_L_constraint=jnp.asarray(-1.0),
            log_likelihood_fn=lambda value: jnp.asarray(0.0),
            periodic=(True,),
        )
        return point.tree[0]

    samples = jax.jit(jax.vmap(draw))(
        random.split(random.PRNGKey(276), 8192)
    )
    histogram, _ = np.histogram(
        np.asarray(samples),
        bins=8,
        range=(0.0, 1.0),
    )

    np.testing.assert_allclose(np.mean(samples), 0.5, atol=0.02)
    np.testing.assert_allclose(
        histogram,
        np.full((8,), 1024),
        rtol=0.15,
    )
    assert np.all(np.asarray(samples) >= 0.0)
    assert np.all(np.asarray(samples) < 1.0)


def test_periodic_coordinates_reject_euclidean_ellipsoids():
    """A seam-splitting Euclidean GMM cannot silently choose directions."""
    sampler = UniDimSliceSampler(
        model=CircularModel(centre=jnp.asarray([0.5])),
        num_slices=5,
        direction=EllipsoidalDirection(num_components=2),
    )._with_periodic((True,))

    with pytest.raises(ValueError, match="EllipsoidalDirection"):
        sampler.validate_core(1)


def test_mixed_cylinder_crosses_only_the_periodic_seam():
    """Endpoint adjacency is periodic while the companion cube face is hard."""
    width = 256
    centre = jnp.asarray([0.99, 0.5])
    model = CylinderModel(centre=centre)
    seeds = jnp.broadcast_to(centre, (width, 2))
    request = ConstrainedSampleRequest(
        keys=random.split(random.PRNGKey(278), width),
        valid=jnp.ones((width,), dtype=jnp.bool_),
        log_L_constraints=jnp.full((width,), -2.0),
        seed_points=SeedPoint(
            U0=seeds,
            log_L0=jnp.zeros((width,)),
        ),
        sampler_data=None,
    )
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=16,
    )._with_periodic((True, False))

    samples = np.asarray(sample_request(sampler, request).U_samples)

    assert np.any(samples[:, 0] < 0.1)
    assert np.any(samples[:, 0] > 0.9)
    assert np.all(samples[:, 1] > 0.35)
    assert np.all(samples[:, 1] < 0.65)
    assert np.all(samples >= 0.0)
    assert np.all(samples < 1.0)


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


def test_narrow_batch_keeps_complete_chain_reference():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    sampler = UniDimSliceSampler(model=model, num_slices=40)
    request = _request(width=4)
    reference = _sample_complete_chains(sampler, request)
    observed = sample_request(sampler, request)

    for expected, actual in zip(
        jax.tree.leaves(reference),
        jax.tree.leaves(observed),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_continuation_outer_jit_captures_registered_function_args():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    sampler = UniDimSliceSampler(model=model, num_slices=32)
    request = _request(width=8)
    registered_args = (lambda value: value,)

    observed = jax.jit(
        lambda value: sample_request(
            sampler,
            value,
            args=registered_args,
        )
    )(request)

    assert observed.log_likelihoods.shape == (8,)
    assert np.all(np.asarray(observed.log_likelihoods) > -0.25)


def test_slice_continuations_preserve_gmm_direction_law():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    direction = EllipsoidalDirection(
        num_components=1,
        min_effective_samples=3,
        population_size=3,
        prob_isotropic=0.01,
    )
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=32,
        collect_phantom_samples=True,
        max_phantom_samples=2,
        direction=direction,
    )
    data = empty_sampler_data(num_components=1, dimension=2)
    data = dataclasses.replace(
        data,
        radii=jnp.asarray([[2.0, 0.5]]),
        rotations=jnp.eye(2)[None],
        log_volumes=jnp.zeros((1,)),
        log_L_max=jnp.ones((1,)),
        valid=jnp.ones((1,), dtype=jnp.bool_),
    )
    request = dataclasses.replace(_request(width=8), sampler_data=data)
    reference = _sample_complete_chains(sampler, request)
    continued = sample_request(sampler, request)

    for expected, actual in zip(
        jax.tree.leaves(reference),
        jax.tree.leaves(continued),
        strict=True,
    ):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        if np.issubdtype(expected.dtype, np.inexact):
            # A transition-ordered scan preserves the reference reduction
            # order up to ordinary machine roundoff. Keep this deliberately
            # much tighter than a scientific tolerance: a larger discrepancy
            # can change a difficult chain's subsequent slice path.
            tolerance = 32 * np.finfo(expected.dtype).eps
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=tolerance,
                atol=tolerance,
            )
        else:
            np.testing.assert_array_equal(actual, expected)


def test_slice_continuations_do_not_execute_scheduler_padding():
    model = QuadraticModel(centre=jnp.asarray([0.45, 0.55]))
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=32,
        collect_phantom_samples=True,
        max_phantom_samples=1,
    )
    request = _request(width=8)
    padded_request = dataclasses.replace(
        request,
        valid=jnp.asarray([True] + [False] * 7),
    )
    scalar_request = ConstrainedSampleRequest(
        keys=request.keys[:1],
        valid=request.valid[:1],
        log_L_constraints=request.log_L_constraints[:1],
        seed_points=SeedPoint(
            U0=jax.tree.map(
                lambda values: values[:1],
                request.seed_points.U0,
            ),
            log_L0=request.seed_points.log_L0[:1],
        ),
        sampler_data=None,
    )

    reference = sample_request(sampler, scalar_request)
    continued = sample_request(sampler, padded_request)

    # An invalid tail lane is transport/storage padding, not a logical chain.
    # It remains a filler device lane only while the valid chain is active.
    assert int(continued.num_likelihood_evaluations[1]) == 0
    assert not bool(continued.phantom_samples.valid_mask[1, 0])
    np.testing.assert_allclose(
        np.asarray(continued.log_likelihoods[0]),
        np.asarray(reference.log_likelihoods[0]),
        rtol=1e-5,
        atol=1e-6,
    )
    for expected, actual in zip(
        jax.tree.leaves(reference.U_samples),
        jax.tree.leaves(continued.U_samples),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual[0]),
            np.asarray(expected[0]),
            rtol=1e-5,
            atol=1e-6,
        )


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
