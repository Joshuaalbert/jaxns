"""Experimental contour-covariance geometry for local slice sampling."""

import dataclasses

import jax
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree, pytree_ravel
from jaxns.samples import Samples
from jaxns.types import FloatArray, IntArray


@dataclasses.dataclass(frozen=True, slots=True)
class CovarianceDirections(PureDataclassPytree):
    """Frozen geometry and its last refresh position in expected log volume."""

    cholesky: FloatArray  # [D, D]
    log_X: FloatArray  # []
    log_L_constraint: FloatArray  # []
    num_points: IntArray  # [] classics used by the last fit
    num_updates: IntArray  # []
    relative_regularisation: FloatArray  # []
    refresh_efolds: FloatArray  # []


CovarianceDirections.register_pytree()


def empty_covariance_directions(
        dimension: int,
        relative_regularisation: float,
        refresh_efolds: float,
) -> CovarianceDirections:
    """Start with isotropic geometry until the first constrained batch."""
    return CovarianceDirections(
        cholesky=jnp.eye(dimension, dtype=mp_policy.measure_dtype),
        log_X=jnp.asarray(0.0, mp_policy.measure_dtype),
        log_L_constraint=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        num_points=jnp.asarray(0, mp_policy.count_dtype),
        num_updates=jnp.asarray(0, mp_policy.count_dtype),
        relative_regularisation=jnp.asarray(
            relative_regularisation, mp_policy.measure_dtype,
        ),
        refresh_efolds=jnp.asarray(refresh_efolds, mp_policy.measure_dtype),
    )


def refresh_covariance_directions(
        data: CovarianceDirections,
        samples: Samples,
        num_samples: IntArray,
        log_L_constraint: FloatArray,
        log_X: FloatArray,
) -> CovarianceDirections:
    """Refit above the dispatch contour after an expected-volume e-fold.

    The fit uses every stored classic above the contour, with equal weights.
    This is direction geometry only: seed eligibility and race weights retain
    their existing definitions. No likelihood is evaluated here.
    """
    due = (
        (data.num_updates == 0)
        | (jnp.abs(log_X - data.log_X) >= data.refresh_efolds)
    )

    def fit(unused):
        points = jax.vmap(lambda point: pytree_ravel(point)[0])(
            samples.U_samples,
        ).astype(data.cholesky.dtype)  # [N, D], covariance in measure precision
        selected = (
            (jnp.arange(points.shape[0]) < num_samples)
            & (samples.log_likelihoods > log_L_constraint)
        )  # [N]
        count = jnp.sum(selected, dtype=mp_policy.count_dtype)
        selected_points = jnp.where(selected[:, None], points, 0.0)
        mean = jnp.sum(selected_points, axis=0) / jnp.maximum(count, 1)
        centred = jnp.where(selected[:, None], points - mean, 0.0)
        # A centred matrix product avoids both cancellation in E[xx']-E[x]E[x]'
        # and an [N,D,D] outer-product intermediate.
        covariance = centred.T @ centred / jnp.maximum(count - 1, 1)
        dimension = points.shape[1]
        variance = jnp.trace(covariance) / dimension
        # The isotropic core scales with this contour's mean variance.
        # A completely collapsed population uses the unit-cube prior variance;
        # multiplying zero covariance by a relative ridge cannot restore rank.
        scale = jnp.where(variance > 0.0, variance, 1.0 / 12.0)
        covariance += (
            data.relative_regularisation * scale
            * jnp.eye(dimension, dtype=covariance.dtype)
        )
        cholesky = jnp.linalg.cholesky(covariance)
        usable = (count >= 2) & jnp.all(jnp.isfinite(cholesky))
        return dataclasses.replace(
            data,
            cholesky=jnp.where(usable, cholesky, data.cholesky),
            log_X=jnp.where(usable, log_X, data.log_X),
            log_L_constraint=jnp.where(
                usable, log_L_constraint, data.log_L_constraint,
            ),
            num_points=jnp.where(usable, count, data.num_points),
            num_updates=data.num_updates + usable.astype(mp_policy.count_dtype),
        )

    return jax.lax.cond(due, fit, lambda unused: data, operand=None)
