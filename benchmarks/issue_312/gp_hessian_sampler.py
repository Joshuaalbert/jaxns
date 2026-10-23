"""Frozen GP predictive-mean Hessian directions with an endpoint MH ratio."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
from jax import numpy as jnp

from benchmarks.issue_312.gp_geometry import GP, negative_predictive_hessian
from jaxns.constrained_sampler import AbstractSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, TreeField, pytree_ravel
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.slice import _new_proposal


class Geometry(NamedTuple):
    rotation: jax.Array  # [D, D], eigenvectors in columns
    precision: jax.Array  # [D]


class ChainState(NamedTuple):
    point: jax.Array
    log_likelihood: jax.Array
    geometry: Geometry
    evaluations: jax.Array
    accepted: jax.Array


def local_geometry(gp: GP, point: jax.Array, scale: jax.Array) -> Geometry:
    eigenvalues, rotation = jnp.linalg.eigh(negative_predictive_hessian(gp, point))
    # Keep the oracle experiment's prior-variance cap and contour scaling.
    # Only the source of the Hessian changes in this comparison.
    return Geometry(rotation, jnp.maximum(eigenvalues / scale, 12.0))


def log_direction_density(direction: jax.Array, geometry: Geometry) -> jax.Array:
    projected = geometry.rotation.T @ direction
    quadratic = jnp.sum(geometry.precision * projected ** 2)
    return (
        0.5 * jnp.sum(jnp.log(geometry.precision))
        - 0.5 * len(direction) * jnp.log(quadratic)
    )


@dataclass(frozen=True, slots=True)
class GPHessianSliceSampler(AbstractSampler, PureDataclassPytree):
    """Use a learned mean's local Hessian, frozen for the complete chain.

    log_likelihood_peak and quadratic_dimension retain the oracle experiment's
    scale so this first GP comparison isolates curvature learning. They are
    not estimated by the GP, and this is not a general production sampler.
    """

    gp: GP
    log_likelihood_peak: float
    quadratic_dimension: int = 15
    num_slices: int = 90
    num_phantom_samples: int = 0
    correct_mh: bool = True

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [
            "log_likelihood_peak", "quadratic_dimension", "num_slices",
            "num_phantom_samples", "correct_mh",
        ])

    def __post_init__(self):
        if self.num_slices < 1 or self.quadratic_dimension < 1:
            raise ValueError("Slices and quadratic dimension must be positive.")
        if not 0 <= self.num_phantom_samples < self.num_slices:
            raise ValueError("Phantoms must exclude the final transition.")

    def num_phantom(self) -> int:
        return self.num_phantom_samples

    def max_num_phantom(self) -> int:
        return self.num_slices - 1

    def _with_phantom_count(self, num_phantom_samples: int):
        return replace(self, num_phantom_samples=num_phantom_samples)

    def validate_core(self, dimension: int) -> None:
        if dimension != self.gp.input_mean.shape[0]:
            raise ValueError("GP and model dimensions differ.")

    def run_chain(
        self, key: jax.Array, point: jax.Array, log_likelihood: jax.Array,
        contour: jax.Array, log_likelihood_fn: Callable[[jax.Array], jax.Array],
    ) -> tuple[ChainState, tuple[jax.Array, jax.Array]]:
        if point.dtype != jnp.float64:
            raise ValueError("Use float64 cube coordinates for this experiment.")
        scale = 2 * jnp.maximum(self.log_likelihood_peak - contour, 1e-12) / (
            self.quadratic_dimension + 2
        )
        zero = jnp.asarray(0, mp_policy.count_dtype)
        initial = ChainState(
            point, log_likelihood, local_geometry(self.gp, point, scale), zero, zero,
        )

        def transition(current, step_key):
            direction_key, proposal_key, accept_key = jax.random.split(step_key, 3)
            normal = jax.random.normal(direction_key, point.shape, dtype=point.dtype)
            direction = current.geometry.rotation @ (normal / jnp.sqrt(
                current.geometry.precision,
            ))
            direction /= jnp.linalg.norm(direction)
            candidate, candidate_log_l, evaluations, _, _ = _new_proposal(
                key=proposal_key, U0=TreeField(current.point),
                direction=TreeField(direction), log_L_constraint=contour,
                log_likelihood_fn=log_likelihood_fn,
            )
            proposed_geometry = local_geometry(self.gp, candidate.tree, scale)
            log_ratio = (
                log_direction_density(direction, proposed_geometry)
                - log_direction_density(direction, current.geometry)
            )
            accepted = jnp.log(jax.random.uniform(
                accept_key, dtype=point.dtype,
            )) < jnp.minimum(0.0, log_ratio)
            if not self.correct_mh:
                accepted = jnp.asarray(True)
            # The same frozen GP defines BOTH endpoint laws. Rejections stay
            # in the chain; no refit or retry-until-accept is permitted here.
            selected = jax.tree.map(
                lambda new, old: jnp.where(accepted, new, old),
                (candidate.tree, candidate_log_l, proposed_geometry),
                (current.point, current.log_likelihood, current.geometry),
            )
            updated = ChainState(
                *selected, current.evaluations + evaluations,
                current.accepted + accepted.astype(mp_policy.count_dtype),
            )
            return updated, (updated.point, updated.log_likelihood)

        return jax.lax.scan(transition, initial, jax.random.split(key, self.num_slices))

    def get_sample(
        self, key, log_L_constraint, seed_point: SeedPoint,
        *, model: Model, args=(), params=None,
    ):
        point, unravel = pytree_ravel(seed_point.U0)

        def likelihood(value):
            return model.log_likelihood(unravel(value), args=args, params=params)

        final, (points, log_likelihoods) = self.run_chain(
            key, point, seed_point.log_L0, log_L_constraint, likelihood,
        )
        phantom = PhantomSamples(
            U_samples=jax.vmap(unravel)(points[:self.num_phantom_samples]),
            log_L=log_likelihoods[:self.num_phantom_samples],
            valid_mask=jnp.asarray(self.num_phantom_samples > 0),
        )
        return unravel(final.point), final.log_likelihood, final.evaluations, phantom


GPHessianSliceSampler.register_pytree()
