"""Small deterministic race-tree fixtures shared by core unit tests."""

import jax
import numpy as np
from jax import numpy as jnp

from cicd.tests.distributed_support import make_toy_model
from jaxns.samples import PhantomSamples, Samples
from jaxns.state import State


def make_state(
        *,
        root_out_degree: int,
        log_likelihoods: tuple[float, ...],
        out_degree: tuple[int, ...],
        log_L_constraints: tuple[float, ...] | None = None,
        max_samples: int = 8,
        num_phantom: int = 0,
) -> State:
    """Construct a padded, valid append-order race state."""
    if log_L_constraints is None:
        log_L_constraints = tuple([-np.inf] * len(log_likelihoods))
    pad_count = max_samples - len(log_likelihoods)
    phantom_log_L = np.full((max_samples, num_phantom), -np.inf)
    phantom_valid = np.zeros((max_samples, num_phantom), dtype=bool)
    samples = Samples(
        log_L_constraints=jnp.asarray(
            log_L_constraints + tuple([np.inf] * pad_count)
        ),
        log_likelihoods=jnp.asarray(
            log_likelihoods + tuple([np.inf] * pad_count)
        ),
        U_samples=jnp.asarray(
            tuple(0.1 + 0.1 * value for value in log_likelihoods)
            + tuple([0.0] * pad_count)
        ),
        out_degree=jnp.asarray(
            out_degree + tuple([0] * pad_count),
            dtype=jnp.int32,
        ),
        num_likelihood_evaluations=jnp.asarray(
            tuple([1] * len(log_likelihoods)) + tuple([0] * pad_count),
            dtype=jnp.int32,
        ),
        phantom_samples=PhantomSamples(
            U_samples=None,
            valid_mask=jnp.asarray(phantom_valid),
            log_L=jnp.asarray(phantom_log_L),
        ),
    )
    supremum_idx = jnp.argmax(samples.log_likelihoods[:len(log_likelihoods)])
    return State(
        root_out_degree=jnp.asarray(root_out_degree, dtype=jnp.int32),
        samples=samples,
        num_samples=jnp.asarray(len(log_likelihoods), dtype=jnp.int32),
        log_L_supremum=samples.log_likelihoods[supremum_idx],
        U_supremum=samples.U_samples[supremum_idx],
        model=make_toy_model(),
        termination_reason=jnp.asarray(0, dtype=jnp.int32),
        random_key=jax.random.PRNGKey(290),
        goal_key=jax.random.PRNGKey(291),
    )
