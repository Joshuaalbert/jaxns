"""Regressions for slice chords anchored on exact unit-cube boundaries."""

import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.test_constrained_sampler import QuadraticModel, _request
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.pytree import TreeField
from jaxns.samples import SeedPoint
from jaxns.sampling.batching import sample_complete_chains
from jaxns.sampling.slice import _new_proposal, _slice_bounds


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("point,direction,expected", [
    pytest.param([0.25, 0.5], [1.0, 0.1], [-0.25, 0.75], id="interior"),
    pytest.param([0.0, 0.5], [1.0, 0.1], [0.0, 1.0], id="lower-inward"),
    pytest.param([0.0, 0.5], [-1.0, 0.1], [-1.0, 0.0], id="lower-outward"),
    pytest.param([1.0, 0.5], [-1.0, 0.1], [0.0, 1.0], id="upper-inward"),
    pytest.param([1.0, 0.5], [1.0, 0.1], [-1.0, 0.0], id="upper-outward"),
    pytest.param([0.0, 0.5], [0.0, 1.0], [-0.5, 0.5], id="lower-tangent"),
    pytest.param([1.0, 0.5], [-0.0, 1.0], [-0.5, 0.5], id="upper-tangent"),
    pytest.param([0.5, 0.25], [0.0, -1.0], [-0.75, 0.25], id="interior-axis"),
    pytest.param([0.0, 1.0], [1.0, -2.0], [0.0, 0.5], id="corner-inward"),
    pytest.param([0.0, 1.0], [-1.0, 2.0], [-0.5, 0.0], id="corner-outward"),
    pytest.param([0.0, 1.0], [0.0, -1.0], [0.0, 1.0], id="corner-tangent"),
    pytest.param([0.0, 0.0], [1.0, -1.0], [0.0, 0.0], id="corner-blocked"),
])
def test_slice_bounds_intersect_cube_faces(point, direction, expected, structured, dtype):
    anchor = jnp.asarray(point, dtype=dtype)
    heading = jnp.asarray(direction, dtype=dtype)
    if structured:
        anchor = {"scalar": anchor[0], "nested": (anchor[1:],)}
        heading = {"scalar": heading[0], "nested": (heading[1:],)}
    bounds = _slice_bounds(TreeField(anchor), TreeField(heading))

    np.testing.assert_array_equal(np.asarray(bounds), expected)
    assert all(bound.dtype == dtype for bound in bounds)


@pytest.mark.parametrize("face", [0.0, 1.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_face_slice_proposals_explore_valid_chord(face, sign):
    anchor = TreeField(jnp.asarray([face, 0.5]))
    direction = TreeField(jnp.asarray([sign, 0.1]))

    def draw(key):
        point, log_likelihood, calls, _, _ = _new_proposal(
            key, anchor, direction, jnp.asarray(0.0),
            lambda value: jnp.asarray(1.0),
        )
        return point.tree, log_likelihood, calls

    samples, log_likelihoods, calls = jax.jit(jax.vmap(draw))(
        jax.random.split(jax.random.PRNGKey(313), 512),
    )
    samples = np.asarray(samples)
    assert np.all((samples >= 0.0) & (samples <= 1.0))
    # A constant likelihood admits the full chord. The face coordinate must
    # explore [0, 1], rather than accepting an unchanged boundary seed.
    np.testing.assert_allclose(samples[:, 0].mean(), 0.5, atol=0.05)
    np.testing.assert_allclose(samples[:, 0].std(), 1 / np.sqrt(12), atol=0.025)
    np.testing.assert_allclose(
        samples[:, 1], 0.5 + 0.1 * (samples[:, 0] - face) / sign,
        atol=1e-7,
    )
    np.testing.assert_array_equal(log_likelihoods, np.ones(512))
    np.testing.assert_array_equal(calls, np.ones(512))


@pytest.mark.parametrize("covariance", [False, True])
def test_boundary_seeds_escape_in_scalar_and_continued_chains(covariance):
    model = QuadraticModel(centre=jnp.asarray([0.5, 0.5]))
    seeds = jnp.asarray([
        [0.0, 0.2], [0.0, 0.8], [1.0, 0.2], [1.0, 0.8],
        [0.2, 0.0], [0.8, 0.0], [0.2, 1.0], [0.8, 1.0],
    ])
    request = dataclasses.replace(
        _request(width=8),
        log_L_constraints=jnp.full(8, -1.0),
        seed_points=SeedPoint(
            U0=seeds, log_L0=jax.vmap(model.log_likelihood)(seeds),
        ),
    )
    # Eight lanes and 32 transitions exercise the continued execution path;
    # the complete-chain reference exercises the scalar transition routine.
    sampler = UniDimSliceSampler(num_slices=32, num_phantom_samples=3)
    if covariance:
        sampler = sampler._with_direction_cholesky(
            jnp.asarray([[0.1, 0.0], [0.05, 0.3]]),
        )
    scalar = jax.jit(
        lambda value: sample_complete_chains(sampler, value, model=model),
    )(request)
    continued = jax.jit(
        lambda value: sampler.get_samples(value, model=model),
    )(request)
    for expected, actual in zip(
        jax.tree.leaves(scalar), jax.tree.leaves(continued), strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    for result in (scalar, continued):
        for points in (result.U_samples, result.phantom_samples.U_samples):
            assert np.all((np.asarray(points) > 0.0) & (np.asarray(points) < 1.0))
        np.testing.assert_array_equal(result.num_likelihood_evaluations, np.full(8, 32))
