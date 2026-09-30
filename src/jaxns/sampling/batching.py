"""Reference scalar/vmapped execution for constrained-sampling requests."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.samples import PhantomSamples, SeedPoint
from jaxns.sampling.prior import sample_prior
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
        model: Model,
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
            model=model,
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


def sample_request(
        sampler: AbstractSampler,
        request: ConstrainedSampleRequest,
        *,
        model: Model,
        args=(),
        params=None,
) -> ConstrainedSampleBatch:
    """Execute one local or worker-side constrained-sampling batch.

    Samplers own their batch execution because only the sampler knows whether
    its data-dependent work can be continued between likelihood evaluations.
    The base implementation retains complete-chain ``vmap`` as the reference
    and fallback for samplers without an explicit batching strategy.
    Sentinel lanes draw directly from the prior. Returned phantom data keeps
    likelihoods and one validity flag per chain, without transient coordinates.
    """

    # The scheduler knows the parent identity before it is discarded. Draw
    # roots from the prior directly, so zero likelihood is accepted without
    # inventing a logarithm of the negative sentinel or adding sample metadata.
    from_root = request.valid & request.from_root
    chain_valid = request.valid & ~request.from_root
    width = request.valid.shape[0]
    empty = ConstrainedSampleBatch(
        U_samples=jax.tree.map(jnp.zeros_like, request.seed_points.U0),
        log_likelihoods=jnp.full_like(request.log_L_constraints, -jnp.inf),
        num_likelihood_evaluations=jnp.zeros((width,), mp_policy.count_dtype),
        phantom_samples=PhantomSamples(
            # Scheduled work persists likelihoods only. Drop coordinates here
            # as well as at commit, including for custom constrained samplers.
            U_samples=None,
            log_L=jnp.full(
                (width, sampler.num_phantom()),
                -jnp.inf,
                mp_policy.measure_dtype,
            ),
            valid_mask=jnp.zeros((width,), mp_policy.bool_dtype),
        ),
        num_directions=jnp.zeros((width,), mp_policy.count_dtype),
        num_isotropic=jnp.zeros((width,), mp_policy.count_dtype),
    )

    def draw_roots(unused):
        U, log_L, num_evals = jax.vmap(
            lambda key, valid: sample_prior(
                key, model, args, params, valid=valid,
            )
        )(request.keys, from_root)
        return dataclasses.replace(
            empty, U_samples=U, log_likelihoods=log_L,
            num_likelihood_evaluations=num_evals,
        )

    def draw_chains(unused):
        # Scalar-chain vmap can execute padded branches too. Give every root
        # or padding lane a real non-root seed/contour, so speculative work
        # cannot get trapped above an empty zero-likelihood support.
        first = jnp.argmax(chain_valid)

        def fill(values):
            mask = chain_valid.reshape((width,) + (1,) * (values.ndim - 1))
            return jnp.where(mask, values, jnp.take(values, first, axis=0))

        chains = dataclasses.replace(
            request,
            valid=chain_valid,
            from_root=jnp.zeros_like(chain_valid),
            log_L_constraints=fill(request.log_L_constraints),
            seed_points=jax.tree.map(fill, request.seed_points),
        )
        sampled = sampler.get_samples(
            chains, model=model, args=args, params=params,
        )
        if sampled.phantom_samples.valid_mask.shape != (width,):
            raise ValueError("Phantom validity must be one scalar per chain.")
        return dataclasses.replace(
            sampled,
            # Custom samplers may count in int32. Scheduled prior and chain
            # branches share the run's count dtype, as stored samples do.
            num_likelihood_evaluations=(
                sampled.num_likelihood_evaluations.astype(
                    mp_policy.count_dtype,
                )
            ),
            phantom_samples=dataclasses.replace(
                sampled.phantom_samples,
                U_samples=None,
                valid_mask=sampled.phantom_samples.valid_mask & chain_valid,
            ),
        )

    roots = jax.lax.cond(
        jnp.any(from_root), draw_roots, lambda _: empty, None,
    )
    chains = jax.lax.cond(
        jnp.any(chain_valid), draw_chains, lambda _: empty, None,
    )

    def select(root_values, chain_values):
        mask = from_root.reshape((width,) + (1,) * (root_values.ndim - 1))
        return jnp.where(mask, root_values, chain_values)

    return jax.tree.map(select, roots, chains)
