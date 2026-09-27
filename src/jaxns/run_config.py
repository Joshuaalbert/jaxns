"""Shared construction defaults, independent of local or worker execution."""

import dataclasses
import operator
from typing import Literal

import jax.numpy as jnp
from jaxctx import CtxParams

from jaxns.constrained_sampler import AbstractSampler, UniDimSliceSampler
from jaxns.depth_condition import DepthCondition
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model

SAMPLES_PER_ROOT = 1000
INITIAL_BATCHES = 64
AllocationTarget = Literal[
    "uniform", "evidence_improving", "posterior_improving",
]


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedRunConfig:
    """Construction result consumed once by the owning execution runner."""

    root_allocation_degree: int
    replacement_width: int | None
    max_samples: int | None
    sampler: AbstractSampler
    max_phantom_samples: int
    depth_condition: DepthCondition
    initial_capacity: int
    delta_K: int


def resolve_run_config(
        *,
        execution: Literal["local", "distributed"],
        model: Model,
        args: tuple,
        params: CtxParams | None,
        root_allocation_degree: int | None,
        max_samples: int | None,
        sampler: AbstractSampler | None,
        depth_condition: DepthCondition | None,
        collect_phantom_samples: bool,
        max_phantom_samples: int | None,
        allocation_target: AllocationTarget,
        delta_K: int | None,
        initial_capacity: int | None,
        unlimited_samples: bool,
        replacement_width: int | None = None,
) -> ResolvedRunConfig:
    """Resolve defaults without sampling or depending on worker topology."""
    periodic = model._periodic_coordinates(args, params)
    # The aligned metadata already carries one flag per scalar base-space
    # coordinate, so it also supplies dimension without a second model
    # init trace. Collapse all-false flags before sampler construction to
    # preserve the exact pre-periodic key schedule and compiled hot path.
    U_ndims = len(periodic)
    if not any(periodic):
        periodic = ()
    root_degree = root_allocation_degree
    if root_degree is None:
        # Match v2's robust default number of independent Markov chains.
        # Merely recording phantoms must not change the sampled race tree.
        root_degree = max(1, 30 * U_ndims)
    if root_degree <= 0:
        raise ValueError("root_allocation_degree must be positive.")
    if allocation_target not in (
        "uniform", "evidence_improving", "posterior_improving",
    ):
        raise ValueError("Unknown allocation_target.")
    if execution == "distributed" and replacement_width is not None:
        raise ValueError("Distributed execution has no replacement_width.")
    if execution == "local" and replacement_width is None:
        # A wider vmap increases exposure to the slowest data-dependent
        # rejection loop in the batch. Ten chains per dimension retains
        # useful CPU batching without the long-tail cost observed at the
        # former half-root width on multimodal problems.
        replacement_width = min(root_degree, max(1, 10 * U_ndims))
    if unlimited_samples and max_samples is not None:
        raise ValueError(
            "unlimited_samples=True conflicts with a finite max_samples."
        )
    if not unlimited_samples:
        if max_samples is None:
            max_samples = max(
                root_degree + (replacement_width or 1),
                SAMPLES_PER_ROOT * root_degree,
            )
        max_samples = int(max_samples)
        if max_samples < root_degree:
            raise ValueError("max_samples must hold all root samples.")
    if delta_K is None:
        if execution == "distributed" or allocation_target == "uniform":
            # Uniform iteration k targets d_0 + delta_K * k. Matching the
            # increment to d_0 adds one root population at each completed
            # goal-loop iteration: d_0, 2 d_0, 3 d_0, and so on.
            delta_K = root_degree
        else:
            # Utility allocation defines a direct gap, so one replacement
            # width normally keeps every vmapped lane scientifically busy.
            delta_K = replacement_width
    if delta_K <= 0 or (
        replacement_width is not None and replacement_width <= 0
    ):
        raise ValueError("replacement_width and delta_K must be positive.")

    if sampler is None:
        num_slices = max(1, 5 * U_ndims)
        sampler = UniDimSliceSampler(
            num_slices=num_slices,
            collect_phantom_samples=collect_phantom_samples,
        )
    if max_phantom_samples is not None:
        try:
            max_phantom_samples = operator.index(max_phantom_samples)
        except TypeError as error:
            raise TypeError(
                "max_phantom_samples must be an integer or None."
            ) from error
    sampler = sampler._with_phantom_capacity(
        max_phantom_samples,
        U_ndims,
    )
    sampler = sampler._with_periodic(periodic)
    sampler.validate_core(U_ndims)

    if depth_condition is None:
        depth_condition = DepthCondition(
            # Match the released v2 scientific stopping goal exactly so
            # accuracy/performance comparisons cannot benefit from an
            # earlier termination threshold.
            dlogZ=jnp.log1p(
                jnp.asarray(1e-3, mp_policy.measure_dtype)
            ),
        )
    if initial_capacity is None:
        # Preallocating the full default maximum makes every fixed-shape
        # block scan pay for unused padding. Start with enough room for a
        # useful number of replacement batches, then grow geometrically.
        if execution == "local":
            initial_capacity = (
                root_degree + INITIAL_BATCHES * replacement_width
            )
        else:
            initial_capacity = root_degree + 10 * delta_K
    initial_capacity = int(initial_capacity)
    if initial_capacity < root_degree:
        raise ValueError("initial_capacity must hold all root samples.")
    if max_samples is not None:
        initial_capacity = min(initial_capacity, max_samples)

    return ResolvedRunConfig(
        root_allocation_degree=int(root_degree),
        replacement_width=(
            None if replacement_width is None else int(replacement_width)
        ),
        max_samples=max_samples,
        sampler=sampler,
        max_phantom_samples=int(sampler.num_phantom()),
        depth_condition=depth_condition,
        initial_capacity=initial_capacity,
        delta_K=int(delta_K),
    )
