"""Reference scalar/vmapped execution for constrained-sampling requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.samples import SeedPoint
from jaxns.sampling.protocol import (
    ConstrainedSampleBatch,
    ConstrainedSampleRequest,
    LikelihoodEvaluation,
    LikelihoodRequest,
)

if TYPE_CHECKING:
    from jaxns.constrained_sampler import AbstractSampler


def evaluate_request(
        model: Model,
        request: LikelihoodRequest,
        *,
        args=(),
        params=None,
) -> LikelihoodEvaluation:
    """Evaluate likelihoods without involving constrained-chain state."""

    def evaluate_one(U):
        return model.log_likelihood(
            U,
            args=args,
            params=params,
            allow_nan=False,
        ).astype(mp_policy.measure_dtype)

    batch_size = jax.tree.leaves(request.U_samples)[0].shape[0]
    if batch_size == 1:
        log_likelihoods = evaluate_one(
            jax.tree.map(lambda values: values[0], request.U_samples)
        )[None]
    else:
        log_likelihoods = jax.vmap(evaluate_one)(request.U_samples)
    return LikelihoodEvaluation(log_likelihoods=log_likelihoods)


def sample_complete_chains(
        sampler: AbstractSampler,
        request: ConstrainedSampleRequest,
        *,
        args=(),
        params=None,
) -> ConstrainedSampleBatch:
    """Run the reference complete-chain scalar or ``vmap`` implementation.

    The local depth loop and worker program own the enclosing JIT boundary.
    Keeping this compositional helper undecorated lets those boundaries capture
    registered session objects such as notebook functions in ``args``; a
    nested JIT would instead try to interpret them as dynamic arrays.
    """

    def sample_one(sample_key, constraint, seed_u, seed_log_likelihood):
        seed = SeedPoint(U0=seed_u, log_L0=seed_log_likelihood)
        return sampler.get_sample_with_diagnostics(
            sample_key,
            constraint,
            seed,
            args=args,
            params=params,
            sampler_data=request.sampler_data,
        )

    batch_size = request.log_L_constraints.shape[0]
    if batch_size == 1:
        sampled = sample_one(
            request.keys[0],
            request.log_L_constraints[0],
            jax.tree.map(lambda values: values[0], request.seed_points.U0),
            request.seed_points.log_L0[0],
        )
        sampled = jax.tree.map(lambda value: value[None], sampled)
    else:
        sampled = jax.vmap(sample_one)(
            request.keys,
            request.log_L_constraints,
            request.seed_points.U0,
            request.seed_points.log_L0,
        )
    (
        U_samples,
        log_likelihoods,
        num_evals,
        phantom_samples,
        num_directions,
        num_isotropic,
    ) = sampled
    return ConstrainedSampleBatch(
        U_samples=U_samples,
        log_likelihoods=log_likelihoods,
        num_likelihood_evaluations=num_evals,
        phantom_samples=phantom_samples,
        num_directions=num_directions,
        num_isotropic=num_isotropic,
    )
