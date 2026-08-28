"""Build the immutable initial race state from prior samples."""

from functools import partial

import jax
from jax import numpy as jnp

from jaxns.algorithm.race_tree import initialise_likelihood_order
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.samples import PhantomSamples, Samples
from jaxns.state import State
from jaxns.types import PRNGKey


@partial(
    jax.jit,
    static_argnames=(
        "root_degree",
        "sample_capacity",
        "num_phantom",
    ),
)
def _sample_init_state(
        key: PRNGKey,
        model: Model,
        args,
        params,
        *,
        root_degree: int,
        sample_capacity: int,
        num_phantom: int,
) -> State:
    """Draw the root sentinel children with a single vectorised prior call."""

    def sample_root(root_key):
        def draw(draw_key):
            U = model.sample_U(draw_key, args=args, params=params)
            log_L = model.log_likelihood(
                U,
                args=args,
                params=params,
                allow_nan=False,
            ).astype(mp_policy.measure_dtype)
            return draw_key, U, log_L, jnp.asarray(1, mp_policy.count_dtype)

        draw_key, U, log_L, num_evals = draw(root_key)

        def invalid(carry):
            _, _, likelihood, _ = carry
            return likelihood <= -jnp.inf

        def redraw(carry):
            old_key, _, _, old_evals = carry
            next_key, proposal_key = jax.random.split(old_key)
            _, next_U, next_log_L, _ = draw(proposal_key)
            return next_key, next_U, next_log_L, old_evals + 1

        _, U, log_L, num_evals = jax.lax.while_loop(
            invalid,
            redraw,
            (draw_key, U, log_L, num_evals),
        )
        return U, log_L, num_evals

    U_samples, log_likelihoods, num_evals = jax.vmap(sample_root)(
        jax.random.split(key, root_degree)
    )
    return _build_init_state(
        model,
        args,
        params,
        U_samples,
        log_likelihoods,
        num_evals,
        sample_capacity=sample_capacity,
        num_phantom=num_phantom,
    )


@partial(
    jax.jit,
    inline=True,
    static_argnames=("sample_capacity", "num_phantom"),
)
def _build_init_state(
        model: Model,
        args,
        params,
        U_samples,
        log_likelihoods,
        num_evals,
        *,
        sample_capacity: int,
        num_phantom: int,
) -> State:
    """Build root race state from already evaluated prior-space points."""
    root_degree = log_likelihoods.shape[0]
    phantom_U = None
    root_samples = Samples(
        # -inf is the sentinel contour. It is also sufficient to recognise
        # root children later; no persistent parent identity is required.
        log_L_constraints=jnp.full(
            (root_degree,),
            -jnp.inf,
            mp_policy.measure_dtype,
        ),
        log_likelihoods=log_likelihoods,
        U_samples=U_samples,
        out_degree=jnp.zeros((root_degree,), mp_policy.count_dtype),
        num_likelihood_evaluations=num_evals,
        phantom_samples=PhantomSamples(
            U_samples=phantom_U,
            valid_mask=jnp.zeros(
                (root_degree, num_phantom),
                mp_policy.bool_dtype,
            ),
            log_L=jnp.full(
                (root_degree, num_phantom),
                -jnp.inf,
                mp_policy.measure_dtype,
            ),
        ),
    ).resize(sample_capacity)
    supremum_idx = jnp.argmax(log_likelihoods)
    return State(
        root_out_degree=jnp.asarray(root_degree, mp_policy.count_dtype),
        samples=root_samples,
        num_samples=jnp.asarray(root_degree, mp_policy.count_dtype),
        log_L_supremum=log_likelihoods[supremum_idx],
        U_supremum=jax.tree.map(lambda u: u[supremum_idx], U_samples),
        termination_reason=jnp.asarray(0, mp_policy.count_dtype),
        model=model,
        args=args,
        params=params,
        likelihood_order=initialise_likelihood_order(
            root_samples.log_likelihoods,
            jnp.asarray(root_degree, mp_policy.count_dtype),
        ),
    )

