from __future__ import annotations

import dataclasses
import warnings
from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp
from jax import random

from jaxns.cumulative_ops import cumulative_op_static
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, TreeField
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.batching import (
    evaluate_request,
)
from jaxns.sampling.batching import (
    sample_complete_chains as _sample_complete_chains,
)
from jaxns.sampling.continuation import (
    _continue_slice_chains,
)
from jaxns.sampling.ellipsoid import SamplerData, empty_sampler_data
from jaxns.sampling.protocol import (
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    LikelihoodEvaluation,
    LikelihoodRequest,
)
from jaxns.sampling.seeding import get_seed_point
from jaxns.sampling.slice import (
    _draw_ellipsoidal_direction,
    _new_proposal,
    _sample_direction,
)
from jaxns.types import BoolArray, FloatArray, IntArray, UType

__all__ = [
    "AbstractSampler",
    "ConstrainedSampleBatch",
    "ConstrainedSampleRequest",
    "EllipsoidalDirection",
    "LikelihoodEvaluation",
    "LikelihoodRequest",
    "UniDimSliceSampler",
    "evaluate_request",
    "get_seed_point",
    "sample_request",
]

# Continuation bookkeeping dominates cheap, short or narrow batches. The
# issue-244 matrix shows material physical-work savings at 32 slice transitions
# and eight lanes. At eight lanes this costs 7.5% on a cheap GMM benchmark but
# removes 39.8% of physical calls; narrower requests keep the complete-chain
# reference because their bookkeeping cost outweighed the smaller saving.
MIN_CONTINUATION_SLICES = 32
MIN_CONTINUATION_CHAINS = 8



def sample_request(
        sampler: AbstractSampler,
        request: ConstrainedSampleRequest,
        *,
        args=(),
        params=None,
) -> ConstrainedSampleBatch:
    """Execute one local or worker-side constrained-sampling batch.

    Samplers own their batch execution because only the sampler knows whether
    its data-dependent work can be continued between likelihood evaluations.
    The base implementation retains complete-chain ``vmap`` as the reference
    and fallback for samplers without an explicit batching strategy.
    """

    return sampler.get_samples(
        request,
        args=args,
        params=params,
    )



class AbstractSampler(ABC):
    """
    Performs sampling from the prior within a likelihood constraint, to produce i.i.d. samples for nested sampling.
    The sampler is assumed to be stateless and pure.
    """

    @abstractmethod
    def num_phantom(self) -> int:
        """
        Get the number of phantom samples to produce per real sample. Note that the number of phantom samples may be less than this if the sampler fails to produce enough valid phantom samples, but it will never be more than this.

        Returns:
            the number of phantom samples to produce per real sample.
        """
        ...

    @abstractmethod
    def get_sample(self, key, log_L_constraint: FloatArray, seed_point: SeedPoint, args=(), params=None) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            seed_point: a seed point to begin sampling from

        Returns:
            U_sample: an i.i.d. sample within the constraint
            log_L: the log-likelihood of the sample
            num_likelihood_evaluations: number of likelihood evaluations used to produce the sample
            phantom_samples: samples that satisfy the constraint but were not accepted. Can be used for various things, e.g. estimating evidence uncertainty.
        """
        ...

    def get_sample_with_diagnostics(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            args=(),
            params=None,
            sampler_data: SamplerData | None = None,
    ) -> tuple[
        UType,
        FloatArray,
        IntArray,
        PhantomSamples,
        IntArray,
        IntArray,
    ]:
        """Sample through the uniform core interface with zero diagnostics."""
        if sampler_data is not None:
            raise ValueError(
                "This sampler does not accept adaptive direction data."
            )
        U_sample, log_L, num_evals, phantom = self.get_sample(
            key,
            log_L_constraint,
            seed_point,
            args=args,
            params=params,
        )
        zero = jnp.asarray(0, mp_policy.count_dtype)
        return U_sample, log_L, num_evals, phantom, zero, zero

    def get_samples(
            self,
            request: ConstrainedSampleRequest,
            *,
            args=(),
            params=None,
    ) -> ConstrainedSampleBatch:
        """Sample a batch through the complete-chain reference path.

        Subclasses may override this when they can preserve their scalar law
        while scheduling likelihood evaluations more efficiently.
        """
        return _sample_complete_chains(
            self,
            request,
            args=args,
            params=params,
        )

    def uses_adaptive_directions(self) -> bool:
        """Return whether the core must maintain contour direction geometry."""
        return False

    def direction_config(self) -> EllipsoidalDirection | None:
        """Return adaptive direction configuration through an explicit API."""
        return None

    def initial_sampler_data(self, dimension: int) -> SamplerData | None:
        """Construct optional sampler state for a new nested-sampling run."""
        del dimension
        return None

    def _with_periodic(self, periodic: tuple[bool, ...]) -> AbstractSampler:
        """Return a sampler configured for static U-space topology."""
        if periodic:
            raise ValueError(
                f"{type(self).__name__} does not support periodic U-space "
                "coordinates."
            )
        return self

    def validate_core(self, dimension: int) -> None:
        """Validate sampler compatibility with the current race-tree core."""
        del dimension


def _take_phantom_prefix(cumulative_samples, num_phantom: int):
    """Take retained generated transitions from the start of a chain."""
    return jax.tree.map(lambda x: x[:num_phantom], cumulative_samples)


@dataclasses.dataclass(slots=True, frozen=True)
class EllipsoidalDirection:
    """Configuration for opt-in warm-refined ellipsoidal directions.

    Args:
        num_components: Fixed maximum number of Gaussian components.
        min_effective_samples: Kish effective sample gate. ``None`` uses four
            times the full-covariance minimum per component.
        num_iterations: Bounded EM iterations performed at each update.
        population_size: Fixed number of likelihood-ordered classic rows used
            by a fit. This keeps compiled fit shapes independent of sample
            storage capacity.
        prob_isotropic: Independent isotropic fallback probability per slice
            transition.
        regularisation: Dimensionless covariance ridge relative to variance.
    """

    num_components: int = 4
    min_effective_samples: int | None = None
    num_iterations: int = 10
    population_size: int = 1024
    prob_isotropic: float = 1e-2
    regularisation: float = 1e-6

    def __post_init__(self):
        if self.num_components < 1:
            raise ValueError("num_components must be positive.")
        if self.min_effective_samples is not None and self.min_effective_samples < 1:
            raise ValueError("min_effective_samples must be positive.")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive.")
        if self.population_size < self.num_components:
            raise ValueError(
                "population_size must be at least num_components."
            )
        if not 0.0 <= self.prob_isotropic <= 1.0:
            raise ValueError("prob_isotropic must be between zero and one.")
        if self.regularisation <= 0.0:
            raise ValueError("regularisation must be positive.")


@dataclasses.dataclass(slots=True, frozen=True)
class UniDimSliceSampler(AbstractSampler, PureDataclassPytree):
    """
    Slice sampler for a single dimension.

    Args:
        model: AbstractModel
        num_slices: number of slices between acceptance. Note: some other software use units of prior dimension.
        no_step_out: if true then perform exponential shrinkage from maximal bounds, requiring no step-out procedure.
            Otherwise, uses a doubling procedure (exponentially finding bracket).
            Note: Perfect is a misnomer, as perfection also depends on the number of slices between acceptance.
        gradient_guided: if true then do HMC with householder reflections.
        collect_phantom_samples: if true, then collect phantom samples
    """

    model: Model
    num_slices: int
    no_step_out: bool = True
    gradient_guided: bool = False
    collect_phantom_samples: bool = False
    phantom_burn_in: int | None = None
    direction: EllipsoidalDirection | None = None
    # Internal scalar topology derived from JAXCTX metadata by NestedSampler.
    # Keeping this private avoids a second user-supplied flat-index API.
    _periodic: tuple[bool, ...] = ()

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, [
            'num_slices',
            'no_step_out',
            'gradient_guided',
            'collect_phantom_samples',
            'phantom_burn_in',
            'direction',
            '_periodic',
        ])

    def _check(self):
        if self.num_slices < 1:
            raise ValueError(f"num_slices should be >= 1, got {self.num_slices}.")
        if self.gradient_guided:
            warnings.warn("Gradient guided slice sampler is experimental and will likely change.")

    def num_phantom(self) -> int:
        if self.collect_phantom_samples:
            if self.phantom_burn_in is None:
                burn_in = int(self.num_slices * 0.1)
            else:
                burn_in = self.phantom_burn_in
            return self.num_slices - 1 - burn_in
        return 0

    def uses_adaptive_directions(self) -> bool:
        return self.direction is not None

    def direction_config(self) -> EllipsoidalDirection | None:
        return self.direction

    def initial_sampler_data(self, dimension: int) -> SamplerData | None:
        if self.direction is None:
            return None
        return empty_sampler_data(
            self.direction.num_components,
            dimension,
        )

    def _with_periodic(
            self,
            periodic: tuple[bool, ...],
    ) -> UniDimSliceSampler:
        """Install the model-derived static topology on this sampler."""
        return dataclasses.replace(self, _periodic=periodic)

    def validate_core(self, dimension: int) -> None:
        if not self.no_step_out:
            raise ValueError(
                "The current core requires perfect/no-step-out bracketing."
            )
        if self.gradient_guided:
            raise ValueError(
                "Gradient-guided sampling is not implemented in this core."
            )
        if self._periodic and len(self._periodic) != dimension:
            raise ValueError(
                "Periodic U-space topology does not match model dimension."
            )
        if self._periodic and self.direction is not None:
            raise ValueError(
                "EllipsoidalDirection does not yet support periodic U-space "
                "coordinates; use isotropic directions."
            )
        if self.direction is None:
            return
        min_effective_samples = self.direction.min_effective_samples
        if min_effective_samples is None:
            min_effective_samples = (
                4 * self.direction.num_components * (dimension + 1)
            )
        if min_effective_samples > self.direction.population_size:
            raise ValueError(
                "EllipsoidalDirection.population_size must be at least its "
                "resolved min_effective_samples."
            )

    def get_sample(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            args=(),
            params=None,
            sampler_data: SamplerData | None = None,
    ) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
        output = self._get_sample_with_diagnostics(
            key,
            log_L_constraint,
            seed_point,
            args=args,
            params=params,
            sampler_data=sampler_data,
        )
        return output[:4]

    def get_sample_with_diagnostics(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            args=(),
            params=None,
            sampler_data: SamplerData | None = None,
    ) -> tuple[
        UType,
        FloatArray,
        IntArray,
        PhantomSamples,
        IntArray,
        IntArray,
    ]:
        """Sample one chain and expose diagnostics only for adaptive directions."""
        output = self._get_sample_with_diagnostics(
            key,
            log_L_constraint,
            seed_point,
            args=args,
            params=params,
            sampler_data=sampler_data,
        )
        if self.direction is not None:
            return output
        # The existing isotropic core reports no direction diagnostics. Keep
        # those internal counters unobserved so XLA can eliminate their work
        # and the explicit sampler API does not change user-facing results.
        zero = jnp.asarray(0, mp_policy.count_dtype)
        return *output[:4], zero, zero

    def get_samples(
            self,
            request: ConstrainedSampleRequest,
            *,
            args=(),
            params=None,
    ) -> ConstrainedSampleBatch:
        """Continue slice chains between fixed-width likelihood calls."""
        if (
            not self.no_step_out
            or self.gradient_guided
            or self.num_slices < MIN_CONTINUATION_SLICES
            or request.log_L_constraints.shape[0] < MIN_CONTINUATION_CHAINS
        ):
            # Continuations model the release sampler's perfect bracket. Keep
            # the scalar implementation as the explicit reference and as the
            # compatibility owner for other trajectory constructions.
            return _sample_complete_chains(
                self,
                request,
                args=args,
                params=params,
            )
        return _continue_slice_chains(
            self,
            request,
            args=args,
            params=params,
        )

    def _get_sample_with_diagnostics(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            args=(),
            params=None,
            sampler_data: SamplerData | None = None,
    ) -> tuple[
        UType,
        FloatArray,
        IntArray,
        PhantomSamples,
        IntArray,
        IntArray,
    ]:
        """Execute the slice chain and return its complete internal counters."""

        class XType(NamedTuple):
            key: jax.Array
            alpha: jax.Array

        def log_likelihood_fn(U):
            return self.model.log_likelihood(
                U,
                args=args,
                params=params,
                allow_nan=False,
            )

        class Carry(NamedTuple):
            U_sample: TreeField[UType]
            log_L_constraint: FloatArray
            log_L: FloatArray
            num_likelihood_evaluations: IntArray
            direction: TreeField[UType]
            slice_width: FloatArray
            direction_isotropic: BoolArray
            num_directions: IntArray
            num_isotropic: IntArray

        def propose_op(carry: Carry, x: XType) -> Carry:
            (
                U_sample,
                log_L,
                num_likelihood_evaluations,
                direction,
                slice_width,
                direction_isotropic,
            ) = _new_proposal(
                key=x.key,
                U0=carry.U_sample,
                direction=carry.direction,
                slice_width=carry.slice_width,
                no_step_out=self.no_step_out,
                gradient_guided=self.gradient_guided,
                log_L_constraint=carry.log_L_constraint,
                log_likelihood_fn=log_likelihood_fn,
                periodic=self._periodic,
                sampler_data=sampler_data,
                prob_isotropic=(
                    1.0
                    if self.direction is None
                    else self.direction.prob_isotropic
                ),
            )

            carry = Carry(
                U_sample=U_sample,
                log_L_constraint=carry.log_L_constraint,
                log_L=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations + carry.num_likelihood_evaluations,
                direction=direction,
                slice_width=slice_width,
                direction_isotropic=direction_isotropic,
                num_directions=(
                    carry.num_directions
                    + jnp.asarray(1, mp_policy.count_dtype)
                ),
                num_isotropic=(
                    carry.num_isotropic
                    + carry.direction_isotropic.astype(mp_policy.count_dtype)
                ),
            )
            return carry

        direction_key, sample_key = jax.random.split(key, 2)

        if sampler_data is None:
            init_direction = _sample_direction(
                direction_key,
                TreeField(seed_point.U0),
            )
            init_direction_isotropic = jnp.asarray(
                True,
                mp_policy.bool_dtype,
            )
        else:
            init_direction, init_direction_isotropic = (
                _draw_ellipsoidal_direction(
                    direction_key,
                    TreeField(seed_point.U0),
                    log_L_constraint,
                    sampler_data,
                    self.direction.prob_isotropic,
                )
            )
        slice_width_dtype = jax.tree.leaves(seed_point.U0)[0].dtype

        # Initial proposal determines the width used by perfect bracketing.
        sample_key, init_sample_key = random.split(sample_key, 2)

        (
            U_sample,
            log_L,
            num_likelihood_evaluations,
            init_direction,
            slice_width,
            next_direction_isotropic,
        ) = _new_proposal(
            key=init_sample_key,
            U0=TreeField(seed_point.U0),
            direction=init_direction,
            slice_width=jnp.asarray(jnp.inf, slice_width_dtype),
            no_step_out=True,
            gradient_guided=self.gradient_guided,
            log_L_constraint=log_L_constraint,
            log_likelihood_fn=log_likelihood_fn,
            periodic=self._periodic,
            sampler_data=sampler_data,
            prob_isotropic=(
                1.0
                if self.direction is None
                else self.direction.prob_isotropic
            ),
        )

        init_carry = Carry(
            U_sample=U_sample,
            log_L_constraint=log_L_constraint,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
            direction=init_direction,
            slice_width=slice_width,
            direction_isotropic=next_direction_isotropic,
            num_directions=jnp.asarray(1, mp_policy.count_dtype),
            num_isotropic=init_direction_isotropic.astype(
                mp_policy.count_dtype
            ),
        )

        xs = XType(
            key=random.split(sample_key, self.num_slices - 1),
            alpha=jnp.linspace(0.5, 1., self.num_slices - 1)
        )
        final_carry, cumulative_samples = cumulative_op_static(
            op=propose_op,
            init=init_carry,
            xs=xs
        )

        # concat initial sample to cumulative samples
        cumulative_samples = jax.tree.map(
            lambda x, y: jnp.concatenate([x[None], y], axis=0),
            init_carry,
            cumulative_samples
        )

        # The final transition is the classic replacement. Retain generated
        # transitions from the start of the chain, never the end adjacent to
        # the classic sample. The input seed is an existing classic sample and
        # is not duplicated as a phantom observation.
        assert self.num_phantom() <= self.num_slices - 1, "num_phantom() should be in [0, num_slices - 1]"

        phantom_fraction = _take_phantom_prefix(
            cumulative_samples,
            self.num_phantom(),
        )
        phantom_samples = PhantomSamples(
            U_samples=phantom_fraction.U_sample.tree,
            log_L=phantom_fraction.log_L,
            valid_mask=jnp.ones(phantom_fraction.log_L.shape, mp_policy.bool_dtype)
        )

        U_sample = final_carry.U_sample.tree
        log_L_sample = final_carry.log_L
        num_likelihood_evaluations = final_carry.num_likelihood_evaluations

        return (
            U_sample,
            log_L_sample,
            num_likelihood_evaluations,
            phantom_samples,
            final_carry.num_directions,
            final_carry.num_isotropic,
        )


UniDimSliceSampler.register_pytree()
