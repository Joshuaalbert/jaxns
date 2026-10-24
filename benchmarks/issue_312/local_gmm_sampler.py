"""Position-dependent GMM component directions with a mixture MH correction.

This benchmark uses Gaussian component responsibilities at the current point.
It does not change the released contour-volume component selector or fit API.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp

from jaxns.constrained_sampler import AbstractSampler
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, TreeField, pytree_ravel
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.gmm import GaussianMixture
from jaxns.sampling.slice import _new_proposal


class GMMGeometry(NamedTuple):
    centres: jax.Array  # [K, D]
    cholesky: jax.Array  # [K, D, D]
    inverse_cholesky: jax.Array  # [K, D, D]
    log_determinants: jax.Array  # [K]
    log_masses: jax.Array  # [K]


def factor_mixture(mixture: GaussianMixture) -> GMMGeometry:
    """Validate and factor a host-boundary fit once, not at each slice step."""
    centres = np.asarray(mixture.centres, dtype=np.float64)
    covariances = np.asarray(mixture.covariances, dtype=np.float64)
    masses = np.where(np.asarray(mixture.valid), np.asarray(mixture.log_masses), -np.inf)
    if centres.ndim != 2 or covariances.shape != (len(centres), centres.shape[1], centres.shape[1]):
        raise ValueError("Expected K centres and K square covariance matrices.")
    if (masses.shape != (len(centres),) or not np.any(np.isfinite(masses))
            or np.any(np.isnan(masses) | np.isposinf(masses))):
        raise ValueError("The GMM needs at least one finite component mass.")
    if not np.all(np.isfinite(centres)) or not np.all(np.isfinite(covariances)):
        raise ValueError("All GMM factors must be finite, including masked components.")
    if not np.allclose(covariances, covariances.swapaxes(-1, -2), rtol=1e-12, atol=1e-15):
        raise ValueError("GMM covariance matrices must be symmetric.")
    cholesky = np.linalg.cholesky(covariances)
    inverse = np.linalg.inv(cholesky)
    log_det = 2 * np.sum(np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1)
    return GMMGeometry(*map(jnp.asarray, (centres, cholesky, inverse, log_det, masses)))


def log_responsibilities(geometry: GMMGeometry, point: jax.Array) -> jax.Array:
    delta = point - geometry.centres
    whitened = jnp.einsum("kij,kj->ki", geometry.inverse_cholesky, delta)
    logits = geometry.log_masses - 0.5 * (
        geometry.log_determinants + jnp.sum(whitened ** 2, axis=1)
    )
    return logits - logsumexp(logits)


def component_log_angular_density(geometry: GMMGeometry, direction: jax.Array) -> jax.Array:
    """Normalized-Gaussian densities, omitting their common sphere-area factor."""
    whitened = jnp.einsum("kij,j->ki", geometry.inverse_cholesky, direction)
    return -0.5 * (
        geometry.log_determinants + len(direction) * jnp.log(jnp.sum(whitened ** 2, axis=1))
    )


def log_direction_density(
    selection: jax.Array, angular_density: jax.Array, iso_prob: float,
) -> jax.Array:
    """Marginalize every component and the isotropic branch at each endpoint."""
    # A uniform sphere has density one in the same omitted-area convention.
    return jnp.logaddexp(
        jnp.log(iso_prob), jnp.log1p(-iso_prob) + logsumexp(selection + angular_density),
    )


class GMMChainState(NamedTuple):
    point: jax.Array
    log_likelihood: jax.Array
    selection: jax.Array
    evaluations: jax.Array
    accepted: jax.Array
    component_counts: jax.Array  # [K + 1], final entry is isotropic


@dataclass(frozen=True, slots=True)
class LocalGMMSliceSampler(AbstractSampler, PureDataclassPytree):
    """Use frozen component covariances with location-dependent selection.

    Each component supplies a zero-mean displacement law, not a proposed
    position at its fitted centre. The true constrained slice is followed by
    a correction using the full marginal angular density. Setting
    local_selection=False supplies the paired, state-independent control.
    """

    geometry: GMMGeometry
    num_slices: int = 90
    iso_prob: float = 0.01
    local_selection: bool = True
    correct_mh: bool = True
    num_phantom_samples: int = 0

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [
            "num_slices", "iso_prob", "local_selection", "correct_mh", "num_phantom_samples",
        ])

    def __post_init__(self):
        if self.num_slices < 1 or not 0 <= self.iso_prob <= 1:
            raise ValueError("Use a positive slice count and isotropic probability in [0, 1].")
        if not 0 <= self.num_phantom_samples < self.num_slices:
            raise ValueError("Phantoms must exclude the final transition.")

    def num_phantom(self) -> int:
        return self.num_phantom_samples

    def max_num_phantom(self) -> int:
        return self.num_slices - 1

    def _with_phantom_count(self, num_phantom_samples: int):
        return replace(self, num_phantom_samples=num_phantom_samples)

    def validate_core(self, dimension: int) -> None:
        if dimension != self.geometry.centres.shape[1]:
            raise ValueError("GMM and model dimensions differ.")

    def run_chain(
        self, key: jax.Array, point: jax.Array, log_likelihood: jax.Array,
        contour: jax.Array, log_likelihood_fn: Callable[[jax.Array], jax.Array],
    ) -> tuple[GMMChainState, tuple[jax.Array, jax.Array]]:
        if point.dtype != jnp.float64:
            raise ValueError("Use float64 cube coordinates for this experiment.")
        geometry = self.geometry

        def selection_at(value):
            if self.local_selection:
                return log_responsibilities(geometry, value)
            return geometry.log_masses - logsumexp(geometry.log_masses)

        zero = jnp.asarray(0, mp_policy.count_dtype)
        components = len(geometry.centres)
        initial = GMMChainState(
            point, log_likelihood, selection_at(point), zero, zero,
            jnp.zeros((components + 1,), dtype=mp_policy.count_dtype),
        )

        def transition(current, step_key):
            component_key, normal_key, iso_key, proposal_key, accept_key = jax.random.split(
                step_key, 5,
            )
            component = jax.random.categorical(component_key, current.selection)
            normal = jax.random.normal(normal_key, point.shape, dtype=point.dtype)
            isotropic = jax.random.bernoulli(iso_key, self.iso_prob)
            direction = jnp.where(isotropic, normal, geometry.cholesky[component] @ normal)
            direction /= jnp.linalg.norm(direction)
            candidate, candidate_log_l, evaluations, _, _ = _new_proposal(
                key=proposal_key, U0=TreeField(current.point), direction=TreeField(direction),
                log_L_constraint=contour, log_likelihood_fn=log_likelihood_fn,
            )
            proposed_selection = selection_at(candidate.tree)
            angular_density = component_log_angular_density(geometry, direction)
            log_ratio = (
                log_direction_density(proposed_selection, angular_density, self.iso_prob)
                - log_direction_density(current.selection, angular_density, self.iso_prob)
            )
            accepted = jnp.log(jax.random.uniform(accept_key, dtype=point.dtype)) < jnp.minimum(
                0.0, log_ratio,
            )
            if not self.correct_mh:
                accepted = jnp.asarray(True)
            # Retain the old state on rejection, including its responsibilities.
            # Neither a selected-component-only ratio nor retry-until-accept
            # implements this marginal-mixture Metropolis transition.
            selected = jax.tree.map(
                lambda new, old: jnp.where(accepted, new, old),
                (candidate.tree, candidate_log_l, proposed_selection),
                (current.point, current.log_likelihood, current.selection),
            )
            updated = GMMChainState(
                *selected, current.evaluations + evaluations,
                current.accepted + accepted.astype(mp_policy.count_dtype),
                current.component_counts.at[jnp.where(isotropic, components, component)].add(1),
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


LocalGMMSliceSampler.register_pytree()
