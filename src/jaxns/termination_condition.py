"""Cheap expectation-based conditions for the compiled depth loop."""

import dataclasses
from functools import partial

import jax
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.types import BoolArray, FloatArray, IntArray


@dataclasses.dataclass(slots=True, frozen=True)
class TerminationCondition(PureDataclassPytree):
    """Conditions that may bound a depth epoch or a default run.

    All statistical fields are evaluated from the expected classic block
    shrinkage path. Monte Carlo shrinkage, including optional phantom
    conditioning, remains a result-level operation and is never placed in the
    compiled depth-loop condition.
    """

    ess: FloatArray | None = None  # []
    evidence_uncert: FloatArray | None = None  # []
    dlogZ: FloatArray | None = None  # []
    max_samples: IntArray | None = None  # []
    max_num_likelihood_evaluations: IntArray | None = None  # []
    log_L_target: FloatArray | None = None  # []
    log_L_contour_target: FloatArray | None = None  # []
    efficiency_threshold: FloatArray | None = None  # []
    rtol: FloatArray | None = None  # []
    atol: FloatArray | None = None  # []
    cummax_XL_frac: FloatArray | None = None  # []


TerminationCondition.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class TerminationRegister(PureDataclassPytree):
    """Online expectation summary consumed by depth and goal conditions."""

    num_samples_used: IntArray  # []
    num_likelihood_evaluations: IntArray  # []
    log_Z_mean: FloatArray  # []
    log_Z_uncert: FloatArray  # []
    remaining_evidence_fraction: FloatArray  # []
    posterior_tail_fraction: FloatArray  # []
    ess: FloatArray  # []
    log_L_max: FloatArray  # []
    log_L_contour_max: FloatArray  # []
    efficiency_shrinkage: FloatArray  # []
    plateau: BoolArray  # []
    no_seed_points: BoolArray  # []
    relative_spread: FloatArray  # []
    absolute_spread: FloatArray  # []

    def is_done(
            self,
            term_cond: TerminationCondition,
    ) -> tuple[BoolArray, IntArray]:
        """Return a scalar done flag and the compatible bit-mask reason."""
        return _is_done(self, term_cond)


TerminationRegister.register_pytree()


@partial(jax.jit, inline=True)
def _is_done(
        self: TerminationRegister,
        term_cond: TerminationCondition,
) -> tuple[BoolArray, IntArray]:
    done_values = (
        jnp.asarray(False, mp_policy.bool_dtype),
        jnp.asarray(0, mp_policy.count_dtype),
    )

    def add(current, bit_done, bit):
        next_done = jnp.logical_or(current[0], bit_done)
        next_reason = current[1] + jnp.where(
            bit_done,
            jnp.asarray(2 ** bit, mp_policy.count_dtype),
            jnp.asarray(0, mp_policy.count_dtype),
        )
        return next_done, next_reason

    if term_cond.max_samples is not None:
        done_values = add(
            done_values,
            self.num_samples_used >= term_cond.max_samples,
            0,
        )
    if term_cond.evidence_uncert is not None:
        done_values = add(
            done_values,
            self.log_Z_uncert <= term_cond.evidence_uncert,
            1,
        )
    if term_cond.dlogZ is not None:
        done_values = add(
            done_values,
            self.remaining_evidence_fraction < term_cond.dlogZ,
            2,
        )
    if term_cond.ess is not None:
        done_values = add(done_values, self.ess >= term_cond.ess, 3)
    if term_cond.max_num_likelihood_evaluations is not None:
        done_values = add(
            done_values,
            self.num_likelihood_evaluations
            >= term_cond.max_num_likelihood_evaluations,
            4,
        )
    if term_cond.log_L_target is not None:
        done_values = add(
            done_values,
            self.log_L_max >= term_cond.log_L_target,
            5,
        )
    if term_cond.log_L_contour_target is not None:
        done_values = add(
            done_values,
            self.log_L_contour_max >= term_cond.log_L_contour_target,
            6,
        )
    if term_cond.efficiency_threshold is not None:
        done_values = add(
            done_values,
            self.efficiency_shrinkage < term_cond.efficiency_threshold,
            7,
        )
    done_values = add(done_values, self.plateau, 8)
    if term_cond.rtol is not None:
        done_values = add(
            done_values,
            self.relative_spread < term_cond.rtol,
            9,
        )
    if term_cond.atol is not None:
        done_values = add(
            done_values,
            self.absolute_spread < term_cond.atol,
            10,
        )
    done_values = add(done_values, self.no_seed_points, 11)
    if term_cond.cummax_XL_frac is not None:
        done_values = add(
            done_values,
            self.posterior_tail_fraction < term_cond.cummax_XL_frac,
            12,
        )
    return done_values
