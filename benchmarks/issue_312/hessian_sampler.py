"""Toy-only local Hessian directions with a per-transition MH correction.

This is an experimental AbstractSampler implementation, not a production
direction API. Its geometry is an immutable analytic function of position.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import log, pi
from typing import NamedTuple

import jax
from jax import numpy as jnp

from jaxns.constrained_sampler import AbstractSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, TreeField, pytree_ravel
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.slice import _new_proposal


class Geometry(NamedTuple):
    """Spectral factor of the only non-diagonal block of the precision."""

    rotation: jax.Array  # [2, 2], eigenvectors in columns
    precision: jax.Array  # [18], in the block eigenbasis then original axes


class ChainState(NamedTuple):
    point: jax.Array
    log_likelihood: jax.Array
    geometry: Geometry
    evaluations: jax.Array
    accepted: jax.Array


def negative_hessian_block(
    point: jax.Array, sigma_stiff: float, curvature: float,
) -> jax.Array:
    """Exact (x0, x1) block of minus the toy log-likelihood Hessian."""
    phase = 2 * jnp.pi * point[1]
    residual = point[0] - (0.5 + curvature * jnp.sin(phase))
    slope = 2 * jnp.pi * curvature * jnp.cos(phase)
    second = -(2 * jnp.pi) ** 2 * curvature * jnp.sin(phase)
    # The residual-times-curvature term makes the exact Hessian indefinite on
    # part of the tube. Dropping it would instead be a Gauss-Newton experiment.
    return jnp.array([
        [1.0, -slope],
        [-slope, slope ** 2 - residual * second],
    ], dtype=point.dtype) / sigma_stiff ** 2


def local_geometry(
    point: jax.Array,
    sigma_stiff: float,
    curvature: float,
    precision_floor: float,
    contour_variance_scale: jax.Array = 1.0,
) -> Geometry:
    """Floor eigenvalues of the exact negative Hessian before inversion.

    The default floor of 12 is the inverse variance of a unit uniform prior.
    Fourteen Gaussian axes have precision 400; the two unused axes have zero
    Hessian eigenvalues and therefore receive the floor. Only the 2x2 coupled
    block needs decomposition; this is the full 18D spectral regularisation.
    """
    eigenvalues, rotation = jnp.linalg.eigh(
        negative_hessian_block(point, sigma_stiff, curvature),
    )
    precision = jnp.concatenate([
        eigenvalues,
        jnp.full((14,), 1 / 0.05 ** 2, point.dtype),
        jnp.zeros((2,), point.dtype),
    ])
    return Geometry(rotation, jnp.maximum(
        precision / contour_variance_scale, precision_floor,
    ))


def draw_direction(key: jax.Array, geometry: Geometry) -> jax.Array:
    """Normalize a Gaussian with covariance equal to inverse precision."""
    normal = jax.random.normal(key, (18,), dtype=geometry.precision.dtype)
    scaled = normal / jnp.sqrt(geometry.precision)
    direction = jnp.concatenate([
        geometry.rotation @ scaled[:2], scaled[2:],
    ])
    return direction / jnp.linalg.norm(direction)


def log_direction_density(direction: jax.Array, geometry: Geometry) -> jax.Array:
    """Angular central Gaussian density, omitting the sphere-area constant."""
    projected = jnp.concatenate([
        geometry.rotation.T @ direction[:2], direction[2:],
    ])
    # Evaluate in the eigenbasis to avoid cancellation in an ill-conditioned
    # dense quadratic form. The Gaussian's radial coordinate was discarded,
    # so its ordinary Gaussian density is not the correct MH proposal factor.
    quadratic = jnp.sum(geometry.precision * projected ** 2)
    return 0.5 * jnp.sum(jnp.log(geometry.precision)) - 9 * jnp.log(quadratic)


@dataclass(frozen=True, slots=True)
class ToyHessianSliceSampler(AbstractSampler, PureDataclassPytree):
    """Straight slice moves guided by the toy's regularised exact Hessian."""

    num_slices: int = 90
    sigma_stiff: float = 1e-5
    curvature: float = 0.3
    precision_floor: float = 12.0
    scale_to_contour: bool = True
    num_phantom_samples: int = 0
    correct_mh: bool = True  # False is only a validation negative control.

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [
            "num_slices", "sigma_stiff", "curvature", "precision_floor",
            "num_phantom_samples", "correct_mh",
            "scale_to_contour",
        ])

    def __post_init__(self):
        if self.num_slices < 1 or self.sigma_stiff <= 0:
            raise ValueError("Slices and stiff width must be positive.")
        if not 0 < self.precision_floor < float("inf"):
            raise ValueError("Precision floor must be finite and positive.")
        if not 0 <= self.num_phantom_samples < self.num_slices:
            raise ValueError("Phantoms must exclude the final transition.")

    def num_phantom(self) -> int:
        return self.num_phantom_samples

    def max_num_phantom(self) -> int:
        return self.num_slices - 1

    def _with_phantom_count(self, num_phantom_samples: int):
        return replace(self, num_phantom_samples=num_phantom_samples)

    def validate_core(self, dimension: int) -> None:
        if dimension != 18:
            raise ValueError("The analytic Hessian is specific to the 18D toy.")

    def run_chain(
        self,
        key: jax.Array,
        point: jax.Array,
        log_likelihood: jax.Array,
        contour: jax.Array,
        log_likelihood_fn: Callable[[jax.Array], jax.Array],
    ) -> tuple[ChainState, tuple[jax.Array, jax.Array]]:
        """Return diagnostics separately from the unchanged sampler contract."""
        if point.dtype != jnp.float64:
            raise ValueError("Use float64 cube coordinates for this experiment.")

        contour_variance_scale = 1.0
        if self.scale_to_contour:
            peak = -log(self.sigma_stiff * (2 * pi) ** 0.5) - 14 * log(0.05 * (2 * pi) ** 0.5)
            # A uniform 15D quadratic contour has covariance 2*gap/(15+2)
            # times inverse Hessian. The fixed contour sets this scale for
            # BOTH endpoints. The precision floor caps broad axes at prior
            # variance, avoiding posterior-sized directions during startup.
            contour_variance_scale = 2 * jnp.maximum(peak - contour, 1e-12) / 17

        def geometry_at(value):
            return local_geometry(
                value, self.sigma_stiff, self.curvature, self.precision_floor,
                contour_variance_scale,
            )

        zero = jnp.asarray(0, mp_policy.count_dtype)
        initial = ChainState(point, log_likelihood, geometry_at(point), zero, zero)

        def transition(current, step_key):
            direction_key, proposal_key, accept_key = jax.random.split(step_key, 3)
            direction = draw_direction(direction_key, current.geometry)
            candidate, candidate_log_l, evaluations, _, _ = _new_proposal(
                key=proposal_key,
                U0=TreeField(current.point),
                direction=TreeField(direction),
                log_L_constraint=contour,
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
            if not self.correct_mh:
                accepted = jnp.asarray(True)
            # Rejections are self-transitions and count toward num_slices.
            # Retrying until acceptance would sample a biased jump chain.
            selected_point, selected_log_l, selected_geometry = jax.tree.map(
                lambda new, old: jnp.where(accepted, new, old),
                (candidate.tree, candidate_log_l, proposed_geometry),
                (current.point, current.log_likelihood, current.geometry),
            )
            updated = ChainState(
                selected_point, selected_log_l, selected_geometry,
                current.evaluations + evaluations,
                current.accepted + accepted.astype(mp_policy.count_dtype),
            )
            return updated, (updated.point, updated.log_likelihood)

        return jax.lax.scan(
            transition, initial, jax.random.split(key, self.num_slices),
        )

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


ToyHessianSliceSampler.register_pytree()
