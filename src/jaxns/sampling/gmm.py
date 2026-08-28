"""Small full-covariance Gaussian-mixture fit used by direction geometry."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random, vmap
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.types import BoolArray, FloatArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class GaussianMixture(PureDataclassPytree):
    """Persistent parameters of a fixed-width Gaussian mixture."""

    centres: FloatArray  # [K, D]
    covariances: FloatArray  # [K, D, D]
    log_masses: FloatArray  # [K]
    valid: BoolArray  # [K]


GaussianMixture.register_pytree()


def _sample_weights(
        data: FloatArray,
        mask: BoolArray | None,
        log_sample_weights: FloatArray | None,
) -> tuple[BoolArray, FloatArray]:
    """Return a finite data mask and normalized non-negative row weights."""
    if mask is None:
        mask = jnp.ones((data.shape[0],), mp_policy.bool_dtype)
    mask = mask & jnp.all(jnp.isfinite(data), axis=1)
    if log_sample_weights is None:
        weights = mask.astype(mp_policy.measure_dtype)
    else:
        usable = mask & jnp.isfinite(log_sample_weights)
        largest = jnp.max(jnp.where(usable, log_sample_weights, -jnp.inf))
        weights = jnp.where(
            usable,
            jnp.exp(log_sample_weights - largest),
            jnp.asarray(0.0, mp_policy.measure_dtype),
        )
    total = jnp.sum(weights)
    # A caller may present only zero-weight rows while a run is still young.
    # Falling back to the finite mask keeps the fit defined; the external
    # effective-information gate still prevents installing that geometry.
    fallback = mask.astype(mp_policy.measure_dtype)
    weights = jnp.where(total > 0.0, weights, fallback)
    weights = weights / jnp.maximum(jnp.sum(weights), 1.0)
    return mask, weights


def _regularisation(covariance: FloatArray, regularisation: float) -> FloatArray:
    """Scale a dimensionless ridge to the population's average variance."""
    dimension = covariance.shape[-1]
    scale = jnp.trace(covariance) / jnp.asarray(
        dimension,
        covariance.dtype,
    )
    floor = jnp.asarray(jnp.finfo(covariance.dtype).eps, covariance.dtype)
    return jnp.asarray(regularisation, covariance.dtype) * jnp.maximum(
        scale,
        floor,
    )


def initialise_gmm(
        key: PRNGKey,
        data: FloatArray,
        n_components: int,
        mask: BoolArray | None = None,
        log_sample_weights: FloatArray | None = None,
        regularisation: float = 1e-6,
) -> GaussianMixture:
    """Initialise a mixture from weighted observations.

    The initial covariance is shared and diagonal. Component centres are
    sampled without replacement from the fitted population so empty slots do
    not all begin at the same location.
    """
    if n_components < 1:
        raise ValueError("n_components must be positive.")
    if data.ndim != 2:
        raise ValueError(f"data must have shape [N, D], got {data.shape}.")
    if n_components > data.shape[0]:
        raise ValueError("n_components cannot exceed the data capacity.")

    finite, weights = _sample_weights(data, mask, log_sample_weights)
    # Zero weights do not protect reductions from ``0 * NaN``. Replace masked
    # storage explicitly so padding can never poison an otherwise valid fit.
    safe_data = jnp.where(finite[:, None], data, 0.0)
    if mask is None and log_sample_weights is None:
        # Preserve the established unweighted initialization and therefore
        # deterministic component labels for callers of the historical API.
        choices = random.choice(
            key,
            data.shape[0],
            shape=(n_components,),
            replace=False,
        )
    else:
        # Weighted direction fits use deterministic mass quantiles. Their
        # initial centres therefore depend on valid scientific rows, not on
        # padded storage capacity or the random implementation for a larger
        # categorical array. This is required for transparent growth to be
        # exactly resumable across recompilations.
        cumulative = jnp.cumsum(weights)
        targets = (
            jnp.arange(n_components, dtype=weights.dtype) + 0.5
        ) / jnp.asarray(n_components, weights.dtype)
        choices = jnp.searchsorted(
            cumulative,
            targets,
            side="left",
        )
        choices = jnp.minimum(
            choices,
            jnp.asarray(data.shape[0] - 1, choices.dtype),
        )
    centres = safe_data[choices]
    mean = jnp.sum(weights[:, None] * safe_data, axis=0)
    centered = safe_data - mean
    variance = jnp.sum(weights[:, None] * jnp.square(centered), axis=0)
    covariance = jnp.diag(variance)
    ridge = _regularisation(covariance, regularisation)
    covariance = covariance + ridge * jnp.eye(
        data.shape[1],
        dtype=data.dtype,
    )
    covariances = jnp.repeat(covariance[None, :, :], n_components, axis=0)
    log_masses = jnp.full(
        (n_components,),
        -jnp.log(jnp.asarray(n_components, mp_policy.measure_dtype)),
        mp_policy.measure_dtype,
    )
    valid = jnp.ones((n_components,), mp_policy.bool_dtype)
    return GaussianMixture(
        centres=mp_policy.cast_to_measure(centres),
        covariances=mp_policy.cast_to_measure(covariances),
        log_masses=log_masses,
        valid=valid,
    )


def initialize_params(key, data, n_components: int):
    """Compatibility wrapper returning the historical parameter tuple."""
    mixture = initialise_gmm(key, data, n_components)
    return mixture.centres, mixture.covariances, mixture.log_masses


def e_step(
        data: FloatArray,
        means: FloatArray,
        covariances: FloatArray,
        log_weights: FloatArray,
        mask: BoolArray | None,
) -> FloatArray:
    """Compute component responsibilities for each observation.

    Args:
        data: Observations with shape ``[N, D]``.
        means: Component centres with shape ``[K, D]``.
        covariances: Component covariances with shape ``[K, D, D]``.
        log_weights: Normalized component log masses with shape ``[K]``.
        mask: Valid observation mask with shape ``[N]``.

    Returns:
        Log responsibilities with shape ``[K, N]``. Masked columns are
        exactly negative infinity rather than ``NaN``.
    """
    if mask is None:
        mask = jnp.ones((data.shape[0],), mp_policy.bool_dtype)
    safe_data = jnp.where(mask[:, None], data, 0.0)
    log_density = vmap(
        lambda centre, covariance: multivariate_normal.logpdf(
            safe_data,
            centre,
            covariance,
        )
    )(means, covariances)
    weighted = log_density + log_weights[:, None]
    normalizer = logsumexp(weighted, axis=0)
    responsibilities = weighted - normalizer[None, :]
    return jnp.where(mask[None, :], responsibilities, -jnp.inf)


def _maximisation(
        data: FloatArray,
        log_responsibilities: FloatArray,
        sample_weights: FloatArray,
        previous: GaussianMixture,
        regularisation: float,
) -> GaussianMixture:
    """Perform one weighted M step while retaining collapsed components."""
    safe_data = jnp.where(sample_weights[:, None] > 0.0, data, 0.0)
    responsibilities = jnp.exp(log_responsibilities)
    weighted = responsibilities * sample_weights[None, :]
    component_mass = jnp.sum(weighted, axis=1)
    safe_mass = jnp.maximum(
        component_mass,
        jnp.asarray(jnp.finfo(data.dtype).eps, data.dtype),
    )
    centres = (
        jnp.einsum("kn,nd->kd", weighted, safe_data)
        / safe_mass[:, None]
    )
    centered = safe_data[None, :, :] - centres[:, None, :]
    covariances = jnp.einsum(
        "kn,knd,kne->kde",
        weighted,
        centered,
        centered,
    ) / safe_mass[:, None, None]
    ridges = vmap(lambda covariance: _regularisation(
        covariance,
        regularisation,
    ))(covariances)
    identity = jnp.eye(data.shape[1], dtype=data.dtype)
    covariances = covariances + ridges[:, None, None] * identity[None, :, :]

    finite = (
        (component_mass > jnp.finfo(component_mass.dtype).eps)
        & jnp.all(jnp.isfinite(centres), axis=1)
        & jnp.all(jnp.isfinite(covariances), axis=(1, 2))
    )
    # Warm refinement must never erase a previously healthy component merely
    # because the current weighted population assigns it negligible mass.
    centres = jnp.where(
        finite[:, None],
        centres,
        previous.centres,
    )
    covariances = jnp.where(
        finite[:, None, None],
        covariances,
        previous.covariances,
    )
    retained_mass = jnp.exp(previous.log_masses)
    masses = jnp.where(finite, component_mass, retained_mass)
    valid = finite | previous.valid
    masses = jnp.where(valid, masses, 0.0)
    masses = masses / jnp.maximum(jnp.sum(masses), 1.0)
    log_masses = jnp.where(valid, jnp.log(masses), -jnp.inf)
    return GaussianMixture(
        centres=mp_policy.cast_to_measure(centres),
        covariances=mp_policy.cast_to_measure(covariances),
        log_masses=mp_policy.cast_to_measure(log_masses),
        valid=valid,
    )


def m_step(data, log_responsibilities):
    """Compatibility wrapper for an unweighted maximisation step."""
    n_components = log_responsibilities.shape[0]
    previous = GaussianMixture(
        centres=jnp.zeros((n_components, data.shape[1]), data.dtype),
        covariances=jnp.repeat(
            jnp.eye(data.shape[1], dtype=data.dtype)[None, :, :],
            n_components,
            axis=0,
        ),
        log_masses=jnp.full(
            (n_components,),
            -jnp.log(jnp.asarray(n_components, data.dtype)),
            data.dtype,
        ),
        valid=jnp.zeros((n_components,), mp_policy.bool_dtype),
    )
    weights = jnp.full(
        (data.shape[0],),
        1.0 / data.shape[0],
        data.dtype,
    )
    mixture = _maximisation(
        data,
        log_responsibilities,
        weights,
        previous,
        regularisation=1e-4,
    )
    return mixture.centres, mixture.covariances, mixture.log_masses


def fit_gmm(
        key: PRNGKey,
        data: FloatArray,
        n_components: int,
        mask: BoolArray | None = None,
        log_sample_weights: FloatArray | None = None,
        initial: GaussianMixture | None = None,
        n_iters: int = 10,
        regularisation: float = 1e-6,
) -> tuple[GaussianMixture, FloatArray, FloatArray]:
    """Run a bounded, warm-startable weighted EM fit.

    Returns:
        The fitted mixture, hard component assignments ``[N]``, and normalized
        observation weights ``[N]``.
    """
    if n_iters < 1:
        raise ValueError("n_iters must be positive.")
    input_mask = mask
    input_log_sample_weights = log_sample_weights
    finite, weights = _sample_weights(data, mask, log_sample_weights)
    mixture = initial
    if mixture is None:
        mixture = initialise_gmm(
            key,
            data,
            n_components,
            mask=input_mask,
            log_sample_weights=input_log_sample_weights,
            regularisation=regularisation,
        )

    def iteration(_, current):
        responsibilities = e_step(
            data,
            current.centres,
            current.covariances,
            current.log_masses,
            finite,
        )
        return _maximisation(
            data,
            responsibilities,
            weights,
            current,
            regularisation,
        )

    mixture = lax.fori_loop(0, n_iters, iteration, mixture)
    responsibilities = e_step(
        data,
        mixture.centres,
        mixture.covariances,
        mixture.log_masses,
        finite,
    )
    cluster_id = jnp.argmax(responsibilities, axis=0)
    return mixture, cluster_id, weights


def em_gmm(
        key,
        data,
        n_components,
        mask: jax.Array | None = None,
        n_iters=10,
        tol=1e-6,
):
    """Fit a GMM while retaining the historical public return structure."""
    del tol
    mixture, cluster_id, _ = fit_gmm(
        key,
        data,
        n_components,
        mask=mask,
        n_iters=n_iters,
        regularisation=1e-4,
    )
    params = (
        mixture.centres,
        mixture.covariances,
        mixture.log_masses,
    )
    return cluster_id, params, jnp.asarray(n_iters, mp_policy.count_dtype)


def em_gmm_reference(
        data: np.ndarray,
        initial: GaussianMixture,
        sample_weights: np.ndarray,
        mask: np.ndarray,
        n_iters: int,
        regularisation: float = 1e-6,
) -> GaussianMixture:
    """Pure NumPy reference for deterministic warm-started EM updates."""
    data = np.asarray(data)
    safe_data = np.where(mask[:, None], data, 0.0)
    weights = np.where(mask, sample_weights, 0.0)
    weights = weights / np.sum(weights)
    centres = np.asarray(initial.centres).copy()
    covariances = np.asarray(initial.covariances).copy()
    log_masses = np.asarray(initial.log_masses).copy()
    valid = np.asarray(initial.valid).copy()
    dimension = data.shape[1]

    for _ in range(n_iters):
        log_density = []
        for centre, covariance in zip(centres, covariances):
            sign, log_det = np.linalg.slogdet(covariance)
            delta = safe_data - centre
            quadratic = np.einsum(
                "nd,de,ne->n",
                delta,
                np.linalg.inv(covariance),
                delta,
            )
            log_density.append(
                -0.5 * (dimension * np.log(2.0 * np.pi) + log_det + quadratic)
                if sign > 0
                else np.full((data.shape[0],), -np.inf)
            )
        weighted_log_density = np.stack(log_density) + log_masses[:, None]
        largest = np.max(weighted_log_density, axis=0)
        responsibilities = np.exp(weighted_log_density - largest[None, :])
        responsibilities /= np.sum(responsibilities, axis=0, keepdims=True)
        responsibilities[:, ~mask] = 0.0
        joint = responsibilities * weights[None, :]
        component_mass = np.sum(joint, axis=1)
        finite = component_mass > np.finfo(data.dtype).eps
        for component in range(centres.shape[0]):
            if not finite[component]:
                continue
            centres[component] = (
                joint[component] @ safe_data / component_mass[component]
            )
            delta = safe_data - centres[component]
            covariance = np.einsum(
                "n,nd,ne->de",
                joint[component],
                delta,
                delta,
            ) / component_mass[component]
            scale = max(
                np.trace(covariance) / dimension,
                np.finfo(data.dtype).eps,
            )
            covariances[component] = (
                covariance
                + regularisation * scale * np.eye(dimension, dtype=data.dtype)
            )
        masses = np.where(finite, component_mass, np.exp(log_masses))
        valid = finite | valid
        masses = np.where(valid, masses, 0.0)
        masses /= np.sum(masses)
        log_masses = np.where(valid, np.log(masses), -np.inf)

    return GaussianMixture(
        centres=jnp.asarray(centres),
        covariances=jnp.asarray(covariances),
        log_masses=jnp.asarray(log_masses),
        valid=jnp.asarray(valid),
    )
