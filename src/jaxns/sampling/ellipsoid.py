import dataclasses
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax import vmap
from jax._src.scipy.special import gammaln
from jax.scipy.special import logsumexp

from jaxns.mixed_precision import mp_policy
from jaxns.optional import import_matplotlib
from jaxns.pytree import PureDataclassPytree
from jaxns.sampling.gmm import (
    GaussianMixture,
    e_step,
    fit_gmm,
    initialise_gmm,
)
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType


class EllipsoidParams(NamedTuple):
    mu: FloatArray  # [K, D] Ellipsoids centres
    radii: FloatArray  # [K, D] Ellsipoids radii
    rotation: FloatArray  # [K, D, D] Ellipsoids rotation matrices


@dataclasses.dataclass(slots=True, frozen=True)
class SamplerData(PureDataclassPytree):
    """Persistent fitted-likelihood geometry used only by the sampler."""

    mixture: GaussianMixture
    centres: FloatArray  # [K, D]
    radii: FloatArray  # [K, D]
    rotations: FloatArray  # [K, D, D]
    log_volumes: FloatArray  # [K] covariance ellipsoid volumes at radius one
    log_L_at_mean: FloatArray  # [K] fitted component likelihood at its mean
    valid: BoolArray  # [K]
    enabled: BoolArray  # [] whether fitted directions may be selected
    iso_prob: FloatArray  # [] independent isotropic safety probability
    num_samples: IntArray  # [] samples used by the last successful update
    num_updates: IntArray  # [] successful updates
    num_directions: IntArray  # [] directions used by completed chains
    num_isotropic: IntArray  # [] used isotropic directions


SamplerData.register_pytree()


def empty_sampler_data(
        num_components: int,
        dimension: int,
) -> SamplerData:
    """Create fixed-shape invalid geometry for an isotropic startup."""
    centres = jnp.zeros(
        (num_components, dimension),
        mp_policy.measure_dtype,
    )
    covariances = jnp.repeat(
        jnp.eye(dimension, dtype=mp_policy.measure_dtype)[None, :, :],
        num_components,
        axis=0,
    )
    valid = jnp.zeros((num_components,), mp_policy.bool_dtype)
    mixture = GaussianMixture(
        centres=centres,
        covariances=covariances,
        log_masses=jnp.full(
            (num_components,),
            -jnp.inf,
            mp_policy.measure_dtype,
        ),
        valid=valid,
    )
    return SamplerData(
        mixture=mixture,
        centres=centres,
        radii=jnp.zeros_like(centres),
        rotations=jnp.repeat(
            jnp.eye(dimension, dtype=mp_policy.measure_dtype)[None, :, :],
            num_components,
            axis=0,
        ),
        log_volumes=jnp.full(
            (num_components,),
            -jnp.inf,
            mp_policy.measure_dtype,
        ),
        log_L_at_mean=jnp.full(
            (num_components,),
            -jnp.inf,
            mp_policy.measure_dtype,
        ),
        valid=valid,
        enabled=jnp.asarray(False, mp_policy.bool_dtype),
        iso_prob=jnp.asarray(1.0, mp_policy.measure_dtype),
        num_samples=jnp.asarray(0, mp_policy.count_dtype),
        num_updates=jnp.asarray(0, mp_policy.count_dtype),
        num_directions=jnp.asarray(0, mp_policy.count_dtype),
        num_isotropic=jnp.asarray(0, mp_policy.count_dtype),
    )


def log_ellipsoid_volume(radii):
    D = radii.shape[0]
    return mp_policy.cast_to_measure(
        jnp.log(2.) - jnp.log(D) + 0.5 * D * jnp.log(jnp.pi) - gammaln(0.5 * D) + jnp.sum(jnp.log(radii)))


def bounding_ellipsoid(points: UType, mask: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Use empirical mean and covariance as approximation to bounding ellipse.

    Args:
        points: [N, D] points to fit ellipsoids to
        mask: [N] mask of which points to consider

    Returns:
        mu, cov
    """
    mu = jnp.average(points, weights=mask, axis=0)
    dx = points - mu
    cov = jnp.average(dx[:, :, None] * dx[:, None, :], weights=mask, axis=0)
    return mp_policy.cast_to_measure(mu), mp_policy.cast_to_measure(cov)


def covariance_to_rotational(cov: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    (x - mu)^T inv(cov) (x - mu) = (x - mu)^T J @ J.T (x - mu)

    where J.T is composed of un-rotation and un-scaling:

    J.T = diag(1/radii) @ rotation.T <==> J = rotation @ diag(1/radii)

    Now since, cov = U @ diag(s) @ V.H we have

    J @ J.T = inv(U @ diag(s) @ V.H) = V @ diag(1/s) @ U.H

    ==> J.T = diag(1/sqrt(s)) @ U.H
    ==> radii = sqrt(s), rotation = U

    Args:
        cov:

    Returns:
        radii, rotation
    """
    u, s, _ = jnp.linalg.svd(cov)
    radii_min = jnp.finfo(s.dtype).eps
    radii = jnp.maximum(jnp.sqrt(s), radii_min)
    rotation = u
    return radii, rotation


def ellipsoid_params(points: UType, mask: FloatArray) -> EllipsoidParams:
    """
    If the ellipsoid is defined by

    (x - mu)^T C (x - mu) = 1

    where C = L @ L.T and L = diag(1/radii) @ rotation.T

    then this returns the mu, radius and rotation matrices of the ellipsoid.

    Args:
        points: [N, D] points to fit ellipsoids to
        mask: [N] mask of which points to consider

    Returns:
        mu [D], radii [D] rotation [D,D]
    """
    # get ellipsoid mean and covariance
    mu, Sigma = bounding_ellipsoid(points=points, mask=mask)
    radii, rotation = covariance_to_rotational(Sigma)

    # Compute scale factor for radii to enclose all points.
    # for all i (points[i] - mu) @ inv(Sigma) / scale**2 @ (points[i] - mu) <= 1
    # for all i (points[i] - mu) @ (L @ L.T) @ (points[i] - mu) <= scale**2
    rho = vmap(lambda x: maha_ellipsoid(x=x, mu=mu, radii=radii, rotation=rotation))(points)

    rho_max = jnp.max(jnp.where(mask, rho, jnp.zeros((), rho.dtype)))
    radii *= jnp.sqrt(rho_max)

    return EllipsoidParams(mu=mu, radii=radii, rotation=rotation)


def _component_ellipsoid(
        points: FloatArray,
        log_L: FloatArray,
        centre: FloatArray,
        covariance: FloatArray,
        weights: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
    """Fit one Gaussian likelihood ellipsoid without a sample hull."""
    dimension = covariance.shape[0]
    total = jnp.sum(weights)
    square_total = jnp.sum(jnp.square(weights))
    effective = jnp.square(total) / jnp.maximum(
        square_total,
        jnp.finfo(weights.dtype).eps,
    )
    eigenvalues = jnp.linalg.eigvalsh(covariance)
    largest_eigenvalue = jnp.max(eigenvalues)
    well_conditioned = (
        (largest_eigenvalue > 0.0)
        & (
            jnp.min(eigenvalues)
            > jnp.finfo(covariance.dtype).eps
            * dimension
            * largest_eigenvalue
        )
    )
    radii, rotation = covariance_to_rotational(covariance)
    log_volume = log_ellipsoid_volume(radii)
    distance = vmap(
        lambda point: maha_ellipsoid(
            point,
            centre,
            radii,
            rotation,
        )
    )(points)
    # For log L(U) = h - rho(U)^2 / 2, every stored observation estimates
    # the component height h at its fitted mean. Responsibility-weighted
    # regression uses the likelihood values themselves, so a narrow density
    # component cannot invent an unsupported high likelihood through its
    # normalizing determinant.
    height_observations = log_L + 0.5 * distance  # [N]
    height_mask = (  # [N]
        (weights > 0.0) & jnp.isfinite(height_observations)
    )
    height_weights = jnp.where(height_mask, weights, 0.0)  # [N]
    height_observations = jnp.where(
        height_mask,
        height_observations,
        0.0,
    )  # [N]
    height_total = jnp.sum(height_weights)  # []
    safe_height_total = jnp.maximum(
        height_total,
        jnp.finfo(weights.dtype).eps,
    )  # []
    log_L_at_mean = jnp.sum(
        height_weights * height_observations
    ) / safe_height_total  # []
    finite = (  # []
        jnp.all(jnp.isfinite(radii))
        & jnp.all(radii > 0.0)
        & jnp.all(jnp.isfinite(rotation))
        & jnp.isfinite(log_volume)
        & jnp.isfinite(log_L_at_mean)
    )
    valid = (  # []
        finite
        & well_conditioned
        & (effective >= dimension + 1)
        & (height_total > 0.0)
    )
    return radii, rotation, log_volume, log_L_at_mean, valid


def update_sampler_data(
        key: PRNGKey,
        data: SamplerData,
        points: FloatArray,
        log_L: FloatArray,
        log_sample_weights: FloatArray,
        mask: BoolArray,
        num_samples: IntArray,
        *,
        n_iters: int,
        iso_prob: float,
        regularisation: float,
) -> SamplerData:
    """Fit a likelihood GMM from the complete current classic population.

    A failed candidate cannot poison a running sampler. Valid old component
    geometry survives an empty or singular candidate. The fitted mixture is a
    normalized posterior-density proxy in homogeneous U-space. Stored
    likelihood values then calibrate every component on the likelihood scale.
    """
    # Warm refinement reuses the fitted likelihood surrogate. The first fit
    # uses deterministic posterior-mass quantiles, so this explicit state
    # operation consumes no nested-sampling randomness.
    initial = jax.lax.cond(
        jnp.any(data.mixture.valid),
        lambda unused: data.mixture,
        lambda unused: initialise_gmm(
            key,
            points,
            data.centres.shape[0],
            mask=mask,
            log_sample_weights=log_sample_weights,
            regularisation=regularisation,
        ),
        operand=None,
    )
    mixture, _, normalized_weights = fit_gmm(
        key,
        points,
        data.centres.shape[0],
        mask=mask,
        log_sample_weights=log_sample_weights,
        initial=initial,
        n_iters=n_iters,
        regularisation=regularisation,
    )

    log_responsibilities = e_step(
        points,
        mixture.centres,
        mixture.covariances,
        mixture.log_masses,
        mask,
    )  # [K, N]
    component_weights = (
        jnp.exp(log_responsibilities) * normalized_weights[None, :]
    )  # [K, N]
    components = jnp.arange(data.centres.shape[0], dtype=mp_policy.index_dtype)
    candidate = vmap(
        lambda component: _component_ellipsoid(
            points,
            log_L,
            mixture.centres[component],
            mixture.covariances[component],
            component_weights[component],
        )
    )(components)
    radii, rotations, log_volumes, log_L_at_mean, valid = candidate
    centres = jnp.where(valid[:, None], mixture.centres, data.centres)
    radii = jnp.where(valid[:, None], radii, data.radii)
    rotations = jnp.where(valid[:, None, None], rotations, data.rotations)
    log_volumes = jnp.where(valid, log_volumes, data.log_volumes)
    log_L_at_mean = jnp.where(
        valid,
        log_L_at_mean,
        data.log_L_at_mean,
    )
    combined_valid = valid | data.valid
    any_candidate = jnp.any(valid)
    candidate_data = SamplerData(
        mixture=mixture,
        centres=centres,
        radii=radii,
        rotations=rotations,
        log_volumes=log_volumes,
        log_L_at_mean=log_L_at_mean,
        valid=combined_valid,
        enabled=jnp.any(combined_valid),
        iso_prob=jnp.asarray(iso_prob, mp_policy.measure_dtype),
        num_samples=num_samples.astype(mp_policy.count_dtype),
        num_updates=data.num_updates + jnp.asarray(1, mp_policy.count_dtype),
        num_directions=data.num_directions,
        num_isotropic=data.num_isotropic,
    )
    return jax.lax.cond(
        any_candidate,
        lambda unused: candidate_data,
        lambda unused: data,
        operand=None,
    )


def component_probabilities(
        data: SamplerData,
        log_L_constraint: FloatArray,
) -> FloatArray:
    """Return fitted super-level ellipsoid probabilities with shape ``[K]``."""
    log_volumes, eligible = _trimmed_log_volumes(data, log_L_constraint)
    logits = jnp.where(eligible, log_volumes, -jnp.inf)
    normalizer = logsumexp(logits)
    return jnp.where(eligible, jnp.exp(logits - normalizer), 0.0)


def _trimmed_log_volumes(
        data: SamplerData,
        log_L_constraint: FloatArray,
) -> tuple[FloatArray, BoolArray]:
    """Trim fitted Gaussian ellipsoids at one likelihood high-water mark."""
    height = data.log_L_at_mean - log_L_constraint
    eligible = data.enabled & data.valid & (height > 0.0)
    dimension = jnp.asarray(data.radii.shape[1], data.log_volumes.dtype)
    radius_squared = 2.0 * jnp.maximum(height, 0.0)
    log_volumes = (
        data.log_volumes
        + 0.5 * dimension * jnp.log(jnp.maximum(radius_squared, 1e-300))
    )
    return log_volumes, eligible


def component_probabilities_reference(
        log_volumes: np.ndarray,
        log_L_at_mean: np.ndarray,
        valid: np.ndarray,
        enabled: bool,
        log_L_constraint: float,
        dimension: int,
) -> np.ndarray:
    """NumPy reference for fitted super-level ellipsoid selection."""
    height = np.asarray(log_L_at_mean) - log_L_constraint
    eligible = enabled & np.asarray(valid) & (height > 0.0)
    if not np.any(eligible):
        return np.zeros_like(log_volumes, dtype=float)
    trimmed = np.asarray(log_volumes) + 0.5 * dimension * np.log(
        2.0 * np.maximum(height, np.finfo(float).tiny)
    )
    shifted = np.where(
        eligible,
        trimmed - np.max(trimmed[eligible]),
        -np.inf,
    )
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def ellipsoid_to_circle(point: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> FloatArray:
    """
    Apply a linear map that would turn an ellipsoid into a sphere.
    Args:
        point: [D] point to transform
        mu: [D] center of ellipse
        radii: [D] radii of ellipse
        rotation: [D,D] rotation matrix of ellipse

    Returns:
        a transformed point of shape [D]
    """
    return jnp.diag(jnp.reciprocal(radii)) @ rotation.T @ (point - mu)


def circle_to_ellipsoid(point: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> FloatArray:
    """
    Apple a linear map that would turn a sphere into an ellipsoid

    Args:
        point: [D] point to transform
        mu: [D] center of ellipse
        radii: [D] radii of ellipse
        rotation: [D,D] rotation matrix of ellipse

    Returns:
        a transformed point of shape [D]
    """
    return mu + (rotation @ jnp.diag(radii) @ point)


def maha_ellipsoid(x: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> FloatArray:
    """
    Compute the Mahalanobis distance.

    Args:
        x: point [D]
        mu: center of ellipse [D]
        radii: radii of ellipse [D]
        rotation: rotation matrix [D, D]

    Returns:
        The Mahalanobis distance of `x` to `mu`.
    """
    u_circ = ellipsoid_to_circle(x, mu, radii, rotation)
    return u_circ @ u_circ


def point_in_ellipsoid(x: FloatArray, mu: FloatArray, radii: FloatArray, rotation: FloatArray) -> BoolArray:
    """
    Determine if a given point is inside a closed ellipse.

    Args:
        x: point [D]
        mu: center of ellipse [D]
        radii: radii of ellipse [D]
        rotation: rotation matrix [D, D]

    Returns:
        True iff x is inside the closed ellipse
    """
    # maha can be slightly bigger than 1, e.g.  maha=1.0000000000000004
    # jax.debug.print("maha={maha}",maha=maha_ellipsoid(x, mu, radii, rotation))
    return jnp.less_equal(maha_ellipsoid(x, mu, radii, rotation), jnp.asarray(1. + 1e-10, x.dtype))


def plot_ellipses(params: EllipsoidParams, show: bool = True):
    """
    Plots ellipses.

    Args:
        params: ellipsoid parameters to plot
        show: whether to show figure
    """
    plt = import_matplotlib()
    theta = jnp.linspace(0., 2 * jnp.pi, 100)
    circle = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    for mu, radii, rotation in zip(params.mu, params.radii, params.rotation):
        ellipse = vmap(
            circle_to_ellipsoid,
            in_axes=(0, None, None, None),
        )(circle, mu, radii, rotation)
        plt.plot(ellipse[:, 0], ellipse[:, 1], c=np.random.uniform(size=3))
    if show:
        plt.show()
