"""Reversal and independent stationary-reference checks for the experiment."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.issue_312.reflection_sampler import (
    ReflectionSampler,
    cube_drift,
    reflection_trajectory,
)


def test_cube_drift_matches_multiple_face_crossings():
    point = jnp.array([0.1, 0.9, 0.5])
    velocity = jnp.array([-2.3, 4.7, -0.2])
    position, direction = cube_drift(point, velocity, 1.0)
    np.testing.assert_allclose(position, [0.2, 0.4, 0.3], atol=1e-15)
    np.testing.assert_allclose(direction, [2.3, -4.7, -0.2])
    reverse, reverse_velocity = cube_drift(position, -direction, 1.0)
    np.testing.assert_allclose(reverse, point, atol=1e-15)
    np.testing.assert_allclose(reverse_velocity, -velocity)


@pytest.mark.parametrize("boundary_only", [False, True])
def test_midpoint_reflection_reversal_and_jacobian(boundary_only):
    def likelihood(point):
        return -((point[0] - 0.4 - 0.2 * jnp.sin(5 * point[1])) / 0.03) ** 2

    def mapping(phase):
        point, velocity = reflection_trajectory(
            phase[:2], phase[2:], jnp.asarray(0.12), 11, jnp.asarray(-1.0),
            jax.value_and_grad(likelihood), boundary_only,
        )
        return jnp.concatenate([point, -velocity])

    # The test includes cube-face crossings and curved, position-dependent
    # reflection planes. No point lies on a discontinuous branch boundary.
    phase = jnp.array([0.35, 0.98, -0.8, 1.2])
    mapped = mapping(phase)
    np.testing.assert_allclose(mapping(mapped), phase, rtol=1e-10, atol=1e-10)
    jacobian = jax.jacfwd(mapping)(phase)
    np.testing.assert_allclose(abs(np.linalg.det(jacobian)), 1, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(mapped[2:]), np.linalg.norm(phase[2:]))


def test_reflection_rejections_and_evaluation_accounting():
    def likelihood(point):
        return -((point[0] - 0.5 - 0.2 * jnp.sin(6 * point[1])) / 0.001) ** 2

    point = jnp.array([0.5 + 0.2 * jnp.sin(1.2), 0.2])
    sampler = ReflectionSampler(num_cycles=20, slices_per_cycle=0, num_bounces=5, step_size=0.1)
    final, (points, values) = jax.jit(sampler.run_chain, static_argnums=4)(
        jax.random.PRNGKey(12), point, likelihood(point), jnp.asarray(-1.0), likelihood,
    )
    changes = np.any(np.diff(np.concatenate([point[None], points]), axis=0) != 0, axis=1)
    assert 0 < final.accepted < 20
    assert final.accepted == np.sum(changes)
    assert final.evaluations == 120
    assert final.gradients == 100
    assert np.all(np.asarray(values) > -1.0)
    assert np.all((np.asarray(points) >= 0) & (np.asarray(points) <= 1))


def test_reflection_time_reversal_on_stiff_ridge():
    def likelihood(point):
        residual = point[0] - 0.5 - 0.3 * jnp.sin(2 * jnp.pi * point[1])
        return -0.5 * (
            (residual / 1e-5) ** 2 + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2)
        )

    rng = np.random.default_rng(33123)
    points = rng.uniform(size=(256, 18))
    points[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * points[:, 1]) + rng.normal(size=256) * 1e-5
    points[:, 2:16] = 0.5 + rng.normal(size=(256, 14)) * 0.05
    velocities = rng.normal(size=points.shape)

    @jax.jit
    @jax.vmap
    def advance(point, velocity):
        return reflection_trajectory(
            point, velocity, jnp.asarray(0.003), 10, jnp.asarray(-15.0),
            jax.value_and_grad(likelihood),
        )

    forward, direction = advance(jnp.asarray(points), jnp.asarray(velocities))
    reverse, reverse_direction = advance(forward, -direction)
    np.testing.assert_allclose(reverse, points, rtol=0, atol=2e-9)
    np.testing.assert_allclose(reverse_direction, -velocities, rtol=0, atol=2e-8)


@pytest.mark.parametrize("boundary_only", [False, True])
def test_reflection_preserves_uniform_curved_tube(boundary_only):
    # This curved two-dimensional tube has constant width and exact uniform
    # x1. Reflection-only proposals isolate the new kernel from slice moves.
    rng = np.random.default_rng(3312)
    points = rng.uniform(size=(4096, 2))
    points[:, 0] = 0.5 + 0.2 * np.sin(2 * np.pi * points[:, 1]) + 0.04 * (points[:, 0] - 0.5)

    def likelihood(point):
        return -((point[0] - 0.5 - 0.2 * jnp.sin(2 * jnp.pi * point[1])) / 0.02) ** 2

    sampler = ReflectionSampler(
        num_cycles=9, slices_per_cycle=0, num_bounces=10, step_size=0.02,
        boundary_only=boundary_only,
    )

    @jax.jit
    @jax.vmap
    def chain(key, point):
        return sampler.run_chain(key, point, likelihood(point), jnp.asarray(-1.0), likelihood)[0]

    final = chain(jax.random.split(jax.random.PRNGKey(442), len(points)), jnp.asarray(points))
    final_points = np.asarray(final.point)
    residual = (
        final_points[:, 0] - 0.5 - 0.2 * np.sin(2 * np.pi * final_points[:, 1])
    ) / 0.04 + 0.5
    from scipy.stats import kstest

    assert kstest(final_points[:, 1], "uniform").statistic < 5 / np.sqrt(len(points))
    assert kstest(residual, "uniform").statistic < 5 / np.sqrt(len(points))
    assert np.all(np.asarray(final.log_likelihood) > -1.0)
    assert np.sqrt(np.mean((final_points[:, 1] - points[:, 1]) ** 2)) > 0.1
