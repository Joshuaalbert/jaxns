"""Frozen parent-contour schedule around the unchanged reflection chain law."""

from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp

from benchmarks.issue_312.reflection_sampler import ReflectionSampler


@dataclass(frozen=True, slots=True)
class FrozenContourReflectionSampler(ReflectionSampler):
    """Toy-only lookup; no seed dependence and no online adaptation.

    The small static schedule is frozen once per process. Inheriting the
    benchmark sampler keeps get_sample, phantom slicing and random splitting
    identical; only the base step supplied to its pure chain function changes.
    """

    step_size: float = 0.003
    gaps: tuple[float, ...] = (1.0, 49.0)
    steps: tuple[float, ...] = (0.003, 0.003)
    peak: float = 0.0

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [
            "num_cycles", "slices_per_cycle", "num_bounces", "step_size",
            "boundary_only", "num_phantom_samples", "gaps", "steps", "peak",
        ])

    def __post_init__(self):
        ReflectionSampler.__post_init__(self)
        if (len(self.gaps) < 2 or len(self.gaps) != len(self.steps)
                or not np.all(np.diff(self.gaps) > 0)
                or not np.all(np.isfinite(self.gaps)) or min(self.gaps) <= 0
                or not np.all(np.isfinite(self.steps)) or min(self.steps) <= 0
                or not np.isfinite(self.peak)):
            raise ValueError("Schedule needs increasing positive gaps and finite positive steps.")

    def step_for_contour(self, contour: jax.Array) -> jax.Array:
        """Interpolate log h in log gap; early contours use the common fixed h."""
        # This exact constant specialisation is also the RNG-equivalence control.
        if min(self.steps) == max(self.steps) == self.step_size:
            return jnp.asarray(self.step_size, jnp.float64)
        gap = self.peak - contour
        log_gap = jnp.log(jnp.maximum(gap, self.gaps[0]))
        step = jnp.exp(jnp.interp(
            log_gap, jnp.log(jnp.asarray(self.gaps)), jnp.log(jnp.asarray(self.steps)),
        ))
        return jnp.where(gap > self.gaps[-1], self.step_size, step)

    def run_chain(self, key, point, log_likelihood, contour, log_likelihood_fn):
        return ReflectionSampler.run_chain(
            self, key, point, log_likelihood, contour, log_likelihood_fn,
            contour_step_size=self.step_for_contour(contour),
        )


FrozenContourReflectionSampler.register_pytree()
