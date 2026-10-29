"""Fixed-shape contour policies and pre-chain choices for the online experiment.

Only frozen arrays enter compiled sampling. Host learning is separate, and the
choice key never replaces the original reflection/slice random stream.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree

COMPRESSION_KNOTS = np.arange(65, dtype=np.float64) * 2.0
INITIAL_LOG_PARAMETERS = np.log(np.array([0.003, 0.03]))
CHOICE_NAMESPACE = 3122901


@dataclass(frozen=True, slots=True)
class OnlinePolicy(PureDataclassPytree):
    """Small dynamic sampler operands; values never enter compilation cache keys."""

    log_parameters: jax.Array  # [65, 2], desired h and nominal duration
    support: jax.Array  # [2], learned compression limits, NaN when untrained
    version: jax.Array  # []

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [])


OnlinePolicy.register_pytree()


class ChosenParameters(NamedTuple):
    num_bounces: jax.Array
    step_size: jax.Array
    nominal_duration: jax.Array
    requested_step: jax.Array
    requested_duration: jax.Array
    choice: jax.Array
    bound_clipped: jax.Array
    duration_capped: jax.Array
    extrapolated: jax.Array
    extrapolation_clamped: jax.Array
    compression_clipped: jax.Array


def initial_policy() -> OnlinePolicy:
    return OnlinePolicy(
        jnp.asarray(np.tile(INITIAL_LOG_PARAMETERS, (65, 1))),
        jnp.full(2, jnp.nan, jnp.float64),
        jnp.asarray(0, mp_policy.count_dtype),
    )


def choose_parameters(
    policy: OnlinePolicy, key: jax.Array, compression: jax.Array,
    *, explore: bool = True,
) -> ChosenParameters:
    """Choose before seeing position, then freeze B/h for the entire chain."""
    if policy.log_parameters.shape != (65, 2) or policy.support.shape != (2,):
        raise ValueError('Expected policy [65, 2] and learned support [2].')
    clipped_c = jnp.clip(compression, 0., 128.)
    logs = jax.vmap(
        lambda values: jnp.interp(clipped_c, jnp.asarray(COMPRESSION_KNOTS), values),
        in_axes=1,
    )(policy.log_parameters)
    outside = (compression < policy.support[0]) | (compression > policy.support[1])
    distance = jnp.maximum(policy.support[0] - compression, compression - policy.support[1])
    return choose_log_parameters(logs, key, explore=explore, extrapolated=outside,
                                 extrapolation_clamped=outside & (distance > 2.),
                                 compression_clipped=compression != clipped_c)


def choose_log_parameters(
    logs: jax.Array, key: jax.Array, *, explore: bool = True,
    extrapolated: jax.Array = False, extrapolation_clamped: jax.Array = False,
    compression_clipped: jax.Array = False,
) -> ChosenParameters:
    """Shared rounding/randomization after an explicitly owned policy lookup."""
    # Exact initial constants preserve the control path despite exp(log(x)) rounding.
    base = jnp.where(logs == jnp.asarray(INITIAL_LOG_PARAMETERS),
                     jnp.asarray([.003, .03]), jnp.exp(logs))
    choice = jnp.asarray(0, mp_policy.count_dtype)
    if explore:
        draw = jax.random.uniform(jax.random.fold_in(key, CHOICE_NAMESPACE), dtype=jnp.float64)
        choice = jnp.sum(draw >= jnp.asarray([.9, .925, .95, .975])).astype(
            mp_policy.count_dtype,
        )
    factors = jnp.asarray([[1., 1.], [1 / 1.5, 1.], [1.5, 1.],
                           [1., 1 / 1.5], [1., 1.5]])
    requested = base * factors[choice]  # [2], before bounds and integer resolution
    bounded = jnp.clip(requested, jnp.asarray([.001, .003]), jnp.asarray([.03, .1]))
    duration = jnp.minimum(bounded[1], 256 * bounded[0])
    ratio = duration / bounded[0]
    nearest = jnp.rint(ratio)
    tolerance = 8 * jnp.finfo(jnp.float64).eps * jnp.maximum(1., jnp.abs(ratio))
    ratio = jnp.where(jnp.abs(ratio - nearest) <= tolerance, nearest, ratio)
    bounces = jnp.clip(jnp.ceil(ratio), 1, 256).astype(mp_policy.count_dtype)
    step = duration / bounces
    # Keep exactly the fixed control even when division lowering uses a reciprocal.
    step = jnp.where((duration == .03) & (bounces == 10), .003, step)
    return ChosenParameters(
        bounces, step, duration, requested[0], requested[1], choice,
        jnp.any(requested != bounded), bounded[1] > 256 * bounded[0],
        extrapolated, extrapolation_clamped, compression_clipped,
    )


def predict_table(anchors: np.ndarray, learned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy interpolation with bounded, two-e-fold extrapolation.

    Anchors live at stratum centres (1, 3, ..., 127). This function is also the
    independent reference for the data passed to the compiled interpolator.
    """
    if anchors.shape != (64, 2) or learned.shape != (64,):
        raise ValueError('Expected 64 two-parameter anchors and their validity mask.')
    selected = np.flatnonzero(learned)
    if not len(selected):
        return np.tile(INITIAL_LOG_PARAMETERS, (65, 1)), np.full(2, np.nan)
    x = 2. * selected + 1.
    y = anchors[selected]
    table = np.column_stack([
        np.interp(COMPRESSION_KNOTS, x, y[:, dimension]) for dimension in range(2)
    ])
    if len(selected) > 1:
        lower_slope = np.clip((y[1] - y[0]) / (x[1] - x[0]), -np.log(1.25), np.log(1.25))
        upper_slope = np.clip((y[-1] - y[-2]) / (x[-1] - x[-2]), -np.log(1.25), np.log(1.25))
        below, above = COMPRESSION_KNOTS < x[0], COMPRESSION_KNOTS > x[-1]
        table[below] = y[0] + np.clip(COMPRESSION_KNOTS[below] - x[0], -2., 0.)[:, None] * (
            lower_slope
        )
        table[above] = y[-1] + np.clip(COMPRESSION_KNOTS[above] - x[-1], 0., 2.)[:, None] * (
            upper_slope
        )
    return table, np.array([x[0], x[-1]])
