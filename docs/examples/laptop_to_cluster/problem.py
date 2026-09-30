"""Shared model and scientific settings for the laptop-to-cluster tutorial."""

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jaxns.model import Model
from jaxns.priors import Prior

D = 2


def prior_model():
    x = Prior(
        tfp.distributions.Uniform(
            low=jnp.full((D,), -5.0),
            high=jnp.full((D,), 5.0),
        ),
        name="x",
    ).realise()
    return jnp.sum(tfp.distributions.Normal(0.0, 0.5).log_prob(x))


model = Model(prior_model=prior_model)

# Allocation controls the scientific work. Keep it fixed when changing the
# execution topology, whose defaults need not use the same allocation step.
run_settings = {
    "root_allocation_degree": 30 * D,
    "delta_K": 30 * D,
    "allocation_target": "evidence_improving",
    "collect_phantom_samples": True,
    "initial_capacity": 512,
    "unlimited_samples": True,
    "verbose": True,
}
