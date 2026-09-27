from __future__ import annotations

import dataclasses
import operator
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
    sample_complete_chains as _sample_complete_chains,
)
from jaxns.sampling.continuation import (
    _continue_slice_chains,
)
from jaxns.sampling.ellipsoid import SamplerData
from jaxns.sampling.protocol import (
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
)
from jaxns.sampling.slice import (
    _draw_ellipsoidal_direction,
    _new_proposal,
    _sample_direction,
)
from jaxns.types import BoolArray, FloatArray, IntArray, UType

__all__ = [
    "AbstractSampler",
    "UniDimSliceSampler",
]

# Continuation bookkeeping dominates cheap, short or narrow batches. The
# issue-244 matrix shows material physical-work savings at 32 slice transitions
# and eight lanes. At eight lanes this costs 7.5% on a cheap GMM benchmark but
# removes 39.8% of physical calls; narrower requests keep the complete-chain
# reference because their bookkeeping cost outweighed the smaller saving.
MIN_CONTINUATION_SLICES = 32
MIN_CONTINUATION_CHAINS = 8



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
    def get_sample(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            *,
            model: Model,
            args=(),
            params=None,
    ) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            model: Model owned by the calling run.
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
            *,
            model: Model,
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
            model=model,
            args=args,
            params=params,
        )
        zero = jnp.asarray(0, mp_policy.count_dtype)
        return U_sample, log_L, num_evals, phantom, zero, zero

    def get_samples(
            self,
            request: ConstrainedSampleRequest,
            *,
            model: Model,
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
            model=model,
            args=args,
            params=params,
        )

    def _with_periodic(self, periodic: tuple[bool, ...]) -> AbstractSampler:
        """Return a sampler configured for static U-space topology."""
        if periodic:
            raise ValueError(
                f"{type(self).__name__} does not support periodic U-space "
                "coordinates."
            )
        return self

    def _with_phantom_capacity(
            self,
            max_phantom_samples: int | None,
            dimension: int,
    ) -> AbstractSampler:
        """Apply a high-level retained capacity through an explicit API."""
        del dimension
        if (
            max_phantom_samples is not None
            and max_phantom_samples != self.num_phantom()
        ):
            raise ValueError(
                "max_phantom_samples must match the fixed capacity of a "
                "custom sampler."
            )
        return self

    def validate_core(self, dimension: int) -> None:
        """Validate sampler compatibility with the current race-tree core."""
        del dimension


def _take_phantom_prefix(cumulative_samples, num_phantom: int):
    """Take retained generated transitions from the start of a chain."""
    return jax.tree.map(lambda x: x[:num_phantom], cumulative_samples)


@dataclasses.dataclass(slots=True, frozen=True)
class UniDimSliceSampler(AbstractSampler, PureDataclassPytree):
    """Stateless slice transitions using perfect unit-hypercube bracketing.

    Args:
        num_slices: Number of transitions per classic replacement.
        collect_phantom_samples: Whether to retain intermediate chain states.
        max_phantom_samples: Retained start-prefix capacity. ``None`` retains
            every intermediate transition when used directly. Runners resolve
            it to at most one model dimension. The final classic is excluded.
    """

    num_slices: int
    collect_phantom_samples: bool = False
    max_phantom_samples: int | None = None
    # Internal scalar topology derived from JAXCTX metadata by NestedSampler.
    # Keeping this private avoids a second user-supplied flat-index API.
    _periodic: tuple[bool, ...] = ()

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, [
            'num_slices',
            'collect_phantom_samples',
            'max_phantom_samples',
            '_periodic',
        ])

    def __post_init__(self):
        try:
            num_slices = operator.index(self.num_slices)
        except TypeError as error:
            raise TypeError("num_slices must be a Python integer.") from error
        if num_slices is not self.num_slices:
            raise TypeError("num_slices must be a Python integer.")
        if num_slices < 1:
            raise ValueError(f"num_slices should be >= 1, got {self.num_slices}.")
        if (
            not self.collect_phantom_samples
            and self.max_phantom_samples is not None
        ):
            raise ValueError(
                "A phantom capacity requires collect_phantom_samples=True."
            )
        if self.max_phantom_samples is not None:
            try:
                max_phantom_samples = operator.index(
                    self.max_phantom_samples
                )
            except TypeError as error:
                raise TypeError(
                    "max_phantom_samples must be a Python integer or None."
                ) from error
            if max_phantom_samples is not self.max_phantom_samples:
                raise TypeError(
                    "max_phantom_samples must be a Python integer or None."
                )
            if max_phantom_samples < 1:
                raise ValueError("max_phantom_samples must be positive.")
            if max_phantom_samples > self.num_slices - 1:
                raise ValueError(
                    "max_phantom_samples cannot exceed num_slices - 1: "
                    f"got {max_phantom_samples} for "
                    f"num_slices={self.num_slices}."
                )

    def num_phantom(self) -> int:
        if not self.collect_phantom_samples:
            return 0
        if self.max_phantom_samples is not None:
            return operator.index(self.max_phantom_samples)
        # A low-level sampler with no explicit memory bound retains the whole
        # eligible prefix. NestedSampler supplies its dimension-sized default.
        return self.num_slices - 1

    def _with_periodic(
            self,
            periodic: tuple[bool, ...],
    ) -> UniDimSliceSampler:
        """Install the model-derived static topology on this sampler."""
        return dataclasses.replace(self, _periodic=periodic)

    def _with_phantom_capacity(
            self,
            max_phantom_samples: int | None,
            dimension: int,
    ) -> UniDimSliceSampler:
        """Resolve NestedSampler's default or explicit retained capacity."""
        if not self.collect_phantom_samples:
            if max_phantom_samples is not None:
                raise ValueError(
                    "max_phantom_samples requires "
                    "collect_phantom_samples=True."
                )
            return self
        if max_phantom_samples is None:
            if self.max_phantom_samples is not None:
                # A capacity set directly on the low-level sampler is already
                # explicit and takes precedence over the high-level default.
                return self
            max_phantom_samples = min(dimension, self.num_slices - 1)
        elif (
            self.max_phantom_samples is not None
            and max_phantom_samples != self.max_phantom_samples
        ):
            raise ValueError(
                "max_phantom_samples disagrees with the capacity configured "
                "on the custom slice sampler."
            )
        if max_phantom_samples == 0:
            return self
        if max_phantom_samples == self.num_phantom():
            return self
        return dataclasses.replace(
            self,
            max_phantom_samples=max_phantom_samples,
        )

    def validate_core(self, dimension: int) -> None:
        if self._periodic and len(self._periodic) != dimension:
            raise ValueError(
                "Periodic U-space topology does not match model dimension."
            )

    def get_sample(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            *,
            model: Model,
            args=(),
            params=None,
            sampler_data: SamplerData | None = None,
    ) -> tuple[UType, FloatArray, IntArray, PhantomSamples]:
        output = self._get_sample_with_diagnostics(
            key,
            log_L_constraint,
            seed_point,
            model=model,
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
            *,
            model: Model,
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
            model=model,
            args=args,
            params=params,
            sampler_data=sampler_data,
        )
        if sampler_data is not None:
            return output
        # Direction counters describe use of a state-owned fitted law. Exact
        # isotropic startup has no such state, so both scalar and continuation
        # paths expose zeros and preserve the reference result contract.
        zero = jnp.asarray(0, mp_policy.count_dtype)
        return *output[:4], zero, zero

    def get_samples(
            self,
            request: ConstrainedSampleRequest,
            *,
            model: Model,
            args=(),
            params=None,
    ) -> ConstrainedSampleBatch:
        """Continue slice chains between fixed-width likelihood calls."""
        if (
            self.num_slices < MIN_CONTINUATION_SLICES
            or request.log_L_constraints.shape[0] < MIN_CONTINUATION_CHAINS
        ):
            # Keep scalar chains as the correctness reference and avoid
            # continuation bookkeeping for short or narrow requests.
            sampler_data = request.sampler_data
            if sampler_data is None:
                return _sample_complete_chains(
                    self,
                    request,
                    model=model,
                    args=args,
                    params=params,
                )

            def sample_disabled_fit(unused):
                # Keep the fitted state on the caller's request, but execute
                # the exact plain-isotropic key stream inside this branch.
                isotropic_request = dataclasses.replace(
                    request,
                    sampler_data=None,
                )
                sampled = _sample_complete_chains(
                    self,
                    isotropic_request,
                    model=model,
                    args=args,
                    params=params,
                )
                direction_counts = jnp.full_like(  # [S]
                    sampled.num_directions,
                    self.num_slices,
                )
                return dataclasses.replace(
                    sampled,
                    num_directions=direction_counts,
                    num_isotropic=direction_counts,
                )

            # The scalar flag is shared across lanes and transitions, so the
            # disabled runtime does not enter any fitted-direction operation.
            return jax.lax.cond(
                sampler_data.enabled,
                lambda unused: _sample_complete_chains(
                    self,
                    request,
                    model=model,
                    args=args,
                    params=params,
                ),
                sample_disabled_fit,
                operand=None,
            )
        return _continue_slice_chains(
            self,
            request,
            model=model,
            args=args,
            params=params,
        )

    def _get_sample_with_diagnostics(
            self,
            key,
            log_L_constraint: FloatArray,
            seed_point: SeedPoint,
            *,
            model: Model,
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

        def log_likelihood_fn(U):
            return model.log_likelihood(
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
            direction_isotropic: BoolArray
            num_directions: IntArray
            num_isotropic: IntArray

        def propose_op(carry: Carry, x: XType) -> Carry:
            (
                U_sample,
                log_L,
                num_likelihood_evaluations,
                direction,
                direction_isotropic,
            ) = _new_proposal(
                key=x.key,
                U0=carry.U_sample,
                direction=carry.direction,
                log_L_constraint=carry.log_L_constraint,
                log_likelihood_fn=log_likelihood_fn,
                periodic=self._periodic,
                sampler_data=sampler_data,
            )

            carry = Carry(
                U_sample=U_sample,
                log_L_constraint=carry.log_L_constraint,
                log_L=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations + carry.num_likelihood_evaluations,
                direction=direction,
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
                )
            )

        # Preserve the first-transition key split before the remaining scan.
        sample_key, init_sample_key = random.split(sample_key, 2)

        (
            U_sample,
            log_L,
            num_likelihood_evaluations,
            init_direction,
            next_direction_isotropic,
        ) = _new_proposal(
            key=init_sample_key,
            U0=TreeField(seed_point.U0),
            direction=init_direction,
            log_L_constraint=log_L_constraint,
            log_likelihood_fn=log_likelihood_fn,
            periodic=self._periodic,
            sampler_data=sampler_data,
        )

        init_carry = Carry(
            U_sample=U_sample,
            log_L_constraint=log_L_constraint,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
            direction=init_direction,
            direction_isotropic=next_direction_isotropic,
            num_directions=jnp.asarray(1, mp_policy.count_dtype),
            num_isotropic=init_direction_isotropic.astype(
                mp_policy.count_dtype
            ),
        )

        xs = XType(
            key=random.split(sample_key, self.num_slices - 1),
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
