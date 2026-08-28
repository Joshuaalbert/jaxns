"""Immutable request and response schemas shared by sampler runtimes."""

from __future__ import annotations

import dataclasses

from jaxns.pytree import PureDataclassPytree
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.ellipsoid import SamplerData
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType


@dataclasses.dataclass(slots=True, frozen=True)
class LikelihoodRequest(PureDataclassPytree):
    """Unit-hypercube points whose likelihoods must run on a worker."""

    U_samples: UType  # [S, ...] unit-hypercube pytree leaves


LikelihoodRequest.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class LikelihoodEvaluation(PureDataclassPytree):
    """Worker-computed likelihood values aligned with one request."""

    log_likelihoods: FloatArray  # [S]


LikelihoodEvaluation.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class ConstrainedSampleBatch(PureDataclassPytree):
    """Fixed-width output of concurrent constrained-sampler lanes."""

    U_samples: UType  # [S, ...] unit-hypercube pytree leaves
    log_likelihoods: FloatArray  # [S]
    num_likelihood_evaluations: IntArray  # [S]
    phantom_samples: PhantomSamples  # [S, P, ...]
    num_directions: IntArray  # [S]
    num_isotropic: IntArray  # [S]


ConstrainedSampleBatch.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class ConstrainedSampleRequest(PureDataclassPytree):
    """Self-contained worker input for one constrained-sampling batch.

    The model, arguments, parameters, and sampler configuration are registered
    once per worker session. A request therefore carries only data that can
    change between batches. Keeping seed coordinates in the request also means
    an in-flight task never points back into a coordinator device buffer.
    """

    keys: PRNGKey  # [S, 2]
    valid: BoolArray  # [S]
    log_L_constraints: FloatArray  # [S]
    seed_points: SeedPoint  # U0 [S, ...], log_L0 [S]
    sampler_data: SamplerData | None  # [K, D, D] and component data


ConstrainedSampleRequest.register_pytree()
