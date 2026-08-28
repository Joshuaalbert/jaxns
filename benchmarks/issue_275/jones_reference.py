"""Independent quadrature reference for the periodic Jones-scalar problem.

The constant phase is integrated analytically through a modified Bessel
function. Measurement uncertainty is integrated in log coordinates before
tensor Gauss-Legendre quadrature is applied to DTEC and clock. This is
deliberately independent of JAXNS shrinkage and sampling.
"""

from __future__ import annotations

import argparse

import jax
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import i0e, i1e, logsumexp

TEC_CONVERSION = -8.4479745  # rad MHz / mTECU
CLOCK_CONVERSION = 2.0e-3 * np.pi  # rad / MHz / ns
NUM_CHANNELS = 20
FREQUENCIES_MHZ = np.linspace(700.0, 702.6, NUM_CHANNELS)
TRUE_DTEC_MTECU = 90.0
TRUE_CLOCK_NS = 0.5
TRUE_CONSTANT_RAD = np.pi - 0.03
TRUE_UNCERTAINTY = 0.1
UNCERTAINTY_PRIOR_SCALE = 0.25

jax.config.update("jax_enable_x64", True)


def observed_gains() -> np.ndarray:
    """Return the fixed noisy real/imaginary scalar gains used by the test."""
    phase = (
        TRUE_DTEC_MTECU * TEC_CONVERSION / FREQUENCIES_MHZ
        + TRUE_CLOCK_NS * CLOCK_CONVERSION * FREQUENCIES_MHZ
        + TRUE_CONSTANT_RAD
    )
    mean = np.concatenate([np.cos(phase), np.sin(phase)])
    noise = np.asarray(
        jax.random.normal(
            jax.random.PRNGKey(275),
            shape=(2 * NUM_CHANNELS,),
            dtype=jax.numpy.float64,
        )
    )
    return mean + TRUE_UNCERTAINTY * noise


def _normalised_nodes(order: int, low: float, high: float):
    """Map Legendre nodes to a uniform expectation on ``[low, high]``."""
    nodes, weights = leggauss(order)
    values = low + 0.5 * (nodes + 1.0) * (high - low)
    return values, 0.5 * weights


def _constant_and_uncertainty_summary(
        resultant: np.ndarray,
        data_energy: float,
        log_uncertainty_nodes: np.ndarray,
        log_uncertainty_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate constant/noise evidence and first constant-phase moment."""
    num_channels = NUM_CHANNELS
    log_sigma = log_uncertainty_nodes[None, :]
    inverse_variance = np.exp(-2.0 * log_sigma)
    magnitude = np.abs(resultant)
    bessel_argument = magnitude[:, None] * inverse_variance

    # Combining exp(x) from I0(x) with the residual exponential avoids
    # overflow precisely where the phase model fits the complex gains well.
    log_integrand = (
        -num_channels * np.log(2.0 * np.pi)
        + (1.0 - 2.0 * num_channels) * log_sigma
        - 0.5 * (
            data_energy - 2.0 * magnitude[:, None]
        ) * inverse_variance
        + np.log(i0e(bessel_argument))
        + 0.5 * np.log(2.0 / np.pi)
        - np.log(UNCERTAINTY_PRIOR_SCALE)
        - 0.5
        * np.exp(2.0 * log_sigma)
        / np.square(UNCERTAINTY_PRIOR_SCALE)
    )
    log_terms = log_integrand + log_uncertainty_weights[None, :]
    log_evidence = logsumexp(log_terms, axis=1)
    # [Q, S] normalized uncertainty quadrature weights conditional on the
    # current DTEC/clock point after analytically integrating constant phase.
    uncertainty_weights = np.exp(log_terms - log_evidence[:, None])
    bessel_ratio = i1e(bessel_argument) / i0e(bessel_argument)
    direction = np.divide(
        resultant,
        magnitude,
        out=np.zeros_like(resultant),
        where=magnitude > 0.0,
    )
    constant_moment = direction * np.sum(
        uncertainty_weights * bessel_ratio,
        axis=1,
    )
    return log_evidence, constant_moment


def reference_summary(
        order: int,
        uncertainty_order: int,
) -> tuple[float, complex]:
    """Calculate evidence and constant moment by deterministic quadrature."""
    gains = observed_gains()
    complex_gains = gains[:NUM_CHANNELS] + 1j * gains[NUM_CHANNELS:]
    data_energy = float(np.sum(np.square(gains)) + NUM_CHANNELS)

    dtec, dtec_weights = _normalised_nodes(order, -300.0, 300.0)
    clock, clock_weights = _normalised_nodes(order, -2.0, 2.0)
    base_phase = (
        dtec[:, None, None]
        * TEC_CONVERSION
        / FREQUENCIES_MHZ[None, None, :]
        + clock[None, :, None]
        * CLOCK_CONVERSION
        * FREQUENCIES_MHZ[None, None, :]
    )
    resultant = np.sum(
        complex_gains[None, None, :] * np.exp(-1j * base_phase),
        axis=-1,
    )

    sigma_nodes, sigma_weights = leggauss(uncertainty_order)
    # The posterior is concentrated near sigma=0.1. In log coordinates these
    # bounds also cover the negligible small-sigma and HalfNormal tails.
    log_sigma = -9.0 + 5.0 * (sigma_nodes + 1.0)
    log_sigma_weights = np.log(5.0 * sigma_weights)

    flat_resultant = resultant.reshape(-1)
    log_inner = np.empty(flat_resultant.shape, dtype=np.float64)
    constant_moment = np.empty_like(flat_resultant, dtype=np.complex128)
    chunk_size = 4096
    for start in range(0, flat_resultant.size, chunk_size):
        stop = min(start + chunk_size, flat_resultant.size)
        log_inner[start:stop], constant_moment[start:stop] = (
            _constant_and_uncertainty_summary(
                flat_resultant[start:stop],
                data_energy,
                log_sigma,
                log_sigma_weights,
            )
        )
    log_inner = log_inner.reshape(resultant.shape)
    constant_moment = constant_moment.reshape(resultant.shape)
    log_weights = (
        np.log(dtec_weights)[:, None]
        + np.log(clock_weights)[None, :]
    )
    log_terms = log_inner + log_weights
    log_evidence = float(logsumexp(log_terms))
    weights = np.exp(log_terms - log_evidence)
    return log_evidence, complex(np.sum(weights * constant_moment))


def reference_log_evidence(order: int, uncertainty_order: int) -> float:
    """Calculate log evidence using deterministic nested quadrature."""
    log_evidence, _ = reference_summary(order, uncertainty_order)
    return log_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-order", type=int, default=512)
    parser.add_argument("--uncertainty-order", type=int, default=384)
    args = parser.parse_args()

    order = 32
    while order <= args.max_order:
        log_evidence, constant_moment = reference_summary(
            order=order,
            uncertainty_order=args.uncertainty_order,
        )
        print(
            f"order={order:4d} log_Z={log_evidence:.12f} "
            f"constant_moment={constant_moment.real:+.12f}"
            f"{constant_moment.imag:+.12f}j"
        )
        order *= 2


if __name__ == "__main__":
    main()
