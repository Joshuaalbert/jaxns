"""Shared MH-corrected slice chain for frozen predictive Hessian experiments."""

from collections.abc import Callable
from typing import NamedTuple

import jax
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import TreeField
from jaxns.sampling.slice import _new_proposal


class Geometry(NamedTuple):
    rotation: jax.Array  # [D, D], eigenvectors in columns
    precision: jax.Array  # [D]


class ChainState(NamedTuple):
    point: jax.Array  # [D]
    log_likelihood: jax.Array  # []
    geometry: Geometry
    evaluations: jax.Array  # []
    accepted: jax.Array  # []


def log_direction_density(direction: jax.Array, geometry: Geometry) -> jax.Array:
    """Angular central Gaussian density, omitting the sphere-area constant."""
    projected = geometry.rotation.T @ direction
    quadratic = jnp.sum(geometry.precision * projected ** 2)
    return (
        0.5 * jnp.sum(jnp.log(geometry.precision))
        - 0.5 * len(direction) * jnp.log(quadratic)
    )


def sample_hessian_chain(
    key: jax.Array,
    point: jax.Array,
    log_likelihood: jax.Array,
    contour: jax.Array,
    log_likelihood_fn: Callable[[jax.Array], jax.Array],
    geometry_at: Callable[[jax.Array], Geometry],
    num_slices: int,
    correct_mh: bool = True,
) -> tuple[ChainState, tuple[jax.Array, jax.Array]]:
    """Use one immutable geometry function for both endpoints of every step.

    This is the existing GP chain factored out so the NN comparison changes
    only the learned geometry. The true constraint, PRNG order and retained
    self-transitions are shared. It is not a production direction interface.
    """
    if point.dtype != jnp.float64:
        raise ValueError("Use float64 cube coordinates for this experiment.")
    zero = jnp.asarray(0, mp_policy.count_dtype)
    initial = ChainState(point, log_likelihood, geometry_at(point), zero, zero)

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
        proposed_geometry = geometry_at(candidate.tree)
        log_ratio = (
            log_direction_density(direction, proposed_geometry)
            - log_direction_density(direction, current.geometry)
        )
        accepted = jnp.log(jax.random.uniform(
            accept_key, dtype=point.dtype,
        )) < jnp.minimum(0.0, log_ratio)
        if not correct_mh:
            accepted = jnp.asarray(True)
        # The same frozen predictor defines BOTH endpoint laws. Rejections
        # stay in the chain; no refit or retry-until-accept is permitted here.
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

    return jax.lax.scan(transition, initial, jax.random.split(key, num_slices))
