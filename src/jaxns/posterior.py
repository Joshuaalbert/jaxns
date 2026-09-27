"""Equally weighted posterior observations and posterior integration."""

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import TypeVar

import jax
import jax.numpy as jnp

from jaxns.cumulative_ops import batch_reduce
from jaxns.log_semiring import LogSpace
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.types import FloatArray, UType, XType

MF = TypeVar("MF")


@dataclasses.dataclass(frozen=True, slots=True)
class PosteriorSamples(PureDataclassPytree):
    """An equally weighted empirical posterior, without run diagnostics."""

    U_samples: UType  # [N, ...] unit-hypercube pytree leaves
    X_samples: XType  # [N, ...] transformed parameter pytree leaves
    log_L: FloatArray  # [N]

    def integrate_fn_over_posterior(
            self,
            fn: Callable[[XType], MF],
            *,
            semi_positive: bool = False,
            batch_size: int | None = None,
    ) -> MF:
        """Average a function of X over the equally weighted observations.

        Args:
            fn: Function applied to one transformed parameter sample. Its
                output may be a pytree of arrays.
            semi_positive: Whether every output value is non-negative.
            batch_size: Optional bound on samples evaluated simultaneously.

        Returns:
            The empirical expectation with the function's pytree structure.
        """
        return _integrate_equal_samples(
            self, fn, semi_positive=semi_positive, batch_size=batch_size,
        )


PosteriorSamples.register_pytree()


@partial(
    jax.jit, inline=True,
    static_argnames=["fn", "semi_positive", "batch_size"],
)
def _integrate_equal_samples(
        samples: PosteriorSamples,
        fn: Callable[[XType], MF],
        *,
        semi_positive: bool,
        batch_size: int | None,
) -> MF:
    num_samples = samples.log_L.shape[0]
    log_weights = jnp.full(
        (num_samples,), -jnp.log(num_samples), mp_policy.measure_dtype,
    )
    return _integrate_posterior(
        samples.X_samples, log_weights, fn,
        semi_positive=semi_positive, batch_size=batch_size,
    )


@partial(
    jax.jit, inline=True,
    static_argnames=["fn", "semi_positive", "batch_size"],
)
def _integrate_posterior(
        X_samples: XType,
        log_weights: FloatArray,
        fn: Callable[[XType], MF],
        *,
        semi_positive: bool,
        batch_size: int | None,
) -> MF:
    """Apply the same signed, stable integration rule to either measure."""
    def kernel(x):
        weight, X = x
        values = fn(X)

        def increment(value):
            if semi_positive:
                # The function returns ordinary values, not their logarithms.
                f = LogSpace(jnp.log(value))
            else:
                f = LogSpace.from_signed_value(value)
            return (weight * f).value

        return jax.tree.map(increment, values)

    return batch_reduce(
        kernel,
        xs=(LogSpace(log_weights), X_samples),
        reduce_fn=jnp.sum,
        batch_size=batch_size,
        vectorised_kernel=False,
    )
