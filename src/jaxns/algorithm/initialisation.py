"""Build the immutable initial race state from prior samples."""

from functools import partial

import jax
from jax import numpy as jnp

from jaxns.algorithm.race_tree import initialise_likelihood_order
from jaxns.mixed_precision import mp_policy
from jaxns.model import Model
from jaxns.samples import PhantomSamples, Samples
from jaxns.sampling.prior import sample_prior
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

    U_samples, log_likelihoods, num_evals = jax.vmap(
        lambda root_key: sample_prior(root_key, model, args, params)
    )(
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
        # Root draws and real zero contours share the stored -inf boundary.
        # Root identity is needed only while sampling: roots are independent
        # prior draws, have no phantom chain, and increment root_out_degree.
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
                (root_degree,),
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
