"""Restore fixed nominal trajectory time using the previously frozen resolution."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import numpy as jnp

from benchmarks.issue_312.reflection_sampler import ReflectionSampler
from benchmarks.issue_312.tuning_schedule import FrozenContourReflectionSampler
from jaxns.mixed_precision import mp_policy


class ReflectionParameters(NamedTuple):
    num_bounces: jax.Array
    step_size: jax.Array


@dataclass(frozen=True, slots=True)
class DurationContourReflectionSampler(FrozenContourReflectionSampler):
    """Toy-only fixed-duration paths with count determined by the parent contour.

    The inherited schedule is desired resolution, not a new acceptance target.
    Count and nominal step are fixed before the chain and all trajectory jitter.
    No trajectory outcome can change either parameter.
    """

    def parameters_for_contour(self, contour: jax.Array) -> ReflectionParameters:
        desired = self.step_for_contour(contour)
        count = jnp.ceil(0.03 / desired).astype(mp_policy.count_dtype)
        early = self.peak - contour > self.gaps[-1]
        # Explicit early control avoids any ceil sensitivity to .03/.003 rounding.
        count = jnp.where(early, 10, count)
        step = jnp.where(early, 0.003, 0.03 / count)
        return ReflectionParameters(count, step)

    def run_chain(self, key, point, log_likelihood, contour, log_likelihood_fn):
        count, step = self.parameters_for_contour(contour)
        return ReflectionSampler.run_chain(
            self, key, point, log_likelihood, contour, log_likelihood_fn,
            contour_step_size=step, contour_num_bounces=count,
        )


DurationContourReflectionSampler.register_pytree()
