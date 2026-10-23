"""Experimental midpoint reflections, with exact cube drifts and endpoint tests.

The drift/reflect/drift map is volume preserving and reversible after a
velocity flip. It need not find an exact likelihood-boundary intersection.
Hug reflections use the gradient at every midpoint; the Galilean variant
reflects only at midpoints outside the likelihood constraint. Both decisions
depend on position alone, never on the sign of the incoming velocity.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
from jax import numpy as jnp

from jaxns.constrained_sampler import AbstractSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, TreeField, pytree_ravel
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.slice import _new_proposal


class ReflectionState(NamedTuple):
    point: jax.Array
    log_likelihood: jax.Array
    evaluations: jax.Array
    gradients: jax.Array
    accepted: jax.Array


def cube_drift(
    point: jax.Array, velocity: jax.Array, time: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact straight flight with specular reflection at every cube face.

    Folding an unfolded trajectory handles any number of face crossings.
    This is a phase-space map: folding position without flipping velocity
    would break time reversal of the surrounding reflection integrator.
    """
    unfolded = jnp.mod(point + time * velocity, 2.0)
    folded = 1.0 - jnp.abs(1.0 - unfolded)
    return folded, velocity * jnp.where(unfolded < 1.0, 1.0, -1.0)


def reflection_trajectory(
    point: jax.Array,
    velocity: jax.Array,
    step_size: jax.Array,
    num_bounces: int,
    contour: jax.Array,
    value_and_grad: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    boundary_only: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """A fixed-length, palindromic, norm-preserving phase-space proposal.

    Each midpoint reflection is an involution at fixed position with unit
    absolute Jacobian. The cube drifts have the same volume and reversal
    properties. Reversing endpoint velocity retraces the whole trajectory.
    Do not stop early or shrink the step when encountering an invalid point:
    the complete path and the endpoint rejection are part of this proposal.
    """
    def bounce(unused_index, carry):
        location, direction = cube_drift(*carry, 0.5 * step_size)
        value, gradient = value_and_grad(location)
        norm = jnp.linalg.norm(gradient)
        normal = gradient / jnp.where(norm > 0, norm, 1.0)
        reflect = norm > 0
        if boundary_only:
            reflect = reflect & (value <= contour)
        reflected = direction - 2 * jnp.dot(direction, normal) * normal
        direction = jnp.where(reflect, reflected, direction)
        return cube_drift(location, direction, 0.5 * step_size)

    return jax.lax.fori_loop(0, num_bounces, bounce, (point, velocity))


@dataclass(frozen=True, slots=True)
class ReflectionSampler(AbstractSampler, PureDataclassPytree):
    """Alternate ordinary slice moves and Metropolised reflection paths.

    Isotropic Gaussian velocities have a position-independent density that
    is unchanged by the reflections. For the uniform constrained prior the
    MH test is therefore just the strict true endpoint likelihood constraint.
    Straight slice moves supply motion across likelihood contours; Hug alone
    is designed to stay near its initial level set.
    """

    num_cycles: int = 9
    slices_per_cycle: int = 10
    num_bounces: int = 10
    step_size: float = 0.01
    boundary_only: bool = False
    num_phantom_samples: int = 0

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [
            "num_cycles", "slices_per_cycle", "num_bounces", "step_size",
            "boundary_only", "num_phantom_samples",
        ])

    def __post_init__(self):
        if self.num_cycles < 1 or self.slices_per_cycle < 0 or self.num_bounces < 0:
            raise ValueError("Positive cycles and nonnegative slice/bounce counts required.")
        if self.slices_per_cycle + self.num_bounces == 0:
            raise ValueError("A cycle must contain at least one proposal.")
        if not 0 < self.step_size < float("inf"):
            raise ValueError("Step size must be finite and positive.")
        if not 0 <= self.num_phantom_samples < self.num_cycles:
            raise ValueError("Only complete cycle endpoints can be retained as phantoms.")

    def num_phantom(self) -> int:
        return self.num_phantom_samples

    def max_num_phantom(self) -> int:
        return self.num_cycles - 1

    def _with_phantom_count(self, num_phantom_samples: int):
        return replace(self, num_phantom_samples=num_phantom_samples)

    def run_chain(
        self, key: jax.Array, point: jax.Array, log_likelihood: jax.Array,
        contour: jax.Array, log_likelihood_fn: Callable[[jax.Array], jax.Array],
    ) -> tuple[ReflectionState, tuple[jax.Array, jax.Array]]:
        """Expose gradient counts and acceptance without changing the core API."""
        if point.dtype != jnp.float64:
            raise ValueError("Use float64 cube coordinates for this experiment.")
        zero = jnp.asarray(0, mp_policy.count_dtype)
        initial = ReflectionState(point, log_likelihood, zero, zero, zero)
        value_and_grad = jax.value_and_grad(log_likelihood_fn)

        def cycle(current, cycle_key):
            slice_key, velocity_key, time_key = jax.random.split(cycle_key, 3)

            def slice_step(carry, step_key):
                direction_key, proposal_key = jax.random.split(step_key)
                direction = jax.random.normal(direction_key, point.shape, dtype=point.dtype)
                direction /= jnp.linalg.norm(direction)
                proposed, value, evaluations, _, _ = _new_proposal(
                    key=proposal_key, U0=TreeField(carry.point),
                    direction=TreeField(direction), log_L_constraint=contour,
                    log_likelihood_fn=log_likelihood_fn,
                )
                return carry._replace(
                    point=proposed.tree, log_likelihood=value,
                    evaluations=carry.evaluations + evaluations,
                ), None

            current, _ = jax.lax.scan(
                slice_step, current, jax.random.split(slice_key, self.slices_per_cycle),
            )
            if self.num_bounces:
                velocity = jax.random.normal(velocity_key, point.shape, dtype=point.dtype)
                # Jitter is independent of position and fixed for the whole
                # proposal, so the reversed path uses the same time steps.
                step_size = self.step_size * jax.random.uniform(
                    time_key, dtype=point.dtype, minval=0.8, maxval=1.2,
                )
                proposed, _ = reflection_trajectory(
                    current.point, velocity, step_size, self.num_bounces,
                    contour, value_and_grad, self.boundary_only,
                )
                value = log_likelihood_fn(proposed)
                accepted = jnp.isfinite(value) & (value > contour)
                current = ReflectionState(
                    jnp.where(accepted, proposed, current.point),
                    jnp.where(accepted, value, current.log_likelihood),
                    current.evaluations + self.num_bounces + 1,
                    current.gradients + self.num_bounces,
                    current.accepted + accepted.astype(mp_policy.count_dtype),
                )
            # Rejected trajectories remain self-transitions. Conditioning on
            # eventual acceptance would change the stationary distribution.
            return current, (current.point, current.log_likelihood)

        return jax.lax.scan(cycle, initial, jax.random.split(key, self.num_cycles))

    def get_sample(
        self, key, log_L_constraint, seed_point: SeedPoint,
        *, model: Model, args=(), params=None,
    ):
        point, unravel = pytree_ravel(seed_point.U0)

        def likelihood(value):
            return model.log_likelihood(unravel(value), args=args, params=params)

        final, (points, values) = self.run_chain(
            key, point, seed_point.log_L0, log_L_constraint, likelihood,
        )
        phantom = PhantomSamples(
            U_samples=jax.vmap(unravel)(points[:self.num_phantom_samples]),
            log_L=values[:self.num_phantom_samples],
            valid_mask=jnp.asarray(self.num_phantom_samples > 0),
        )
        return unravel(final.point), final.log_likelihood, final.evaluations, phantom


ReflectionSampler.register_pytree()
