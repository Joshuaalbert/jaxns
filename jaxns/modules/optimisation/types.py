from typing import NamedTuple, Dict

from jax import numpy as jnp

from jaxns.internals.types import Reservoir


class GlobalOptimiserState(NamedTuple):
    key: jnp.ndarray
    done: jnp.ndarray
    reservoir: Reservoir  # Arrays to hold samples taken from the reservoir.
    num_steps: jnp.ndarray  # the step index of the algorithm, where one step is a single consumption step.
    num_samples: jnp.ndarray  # how many samples have been drawn
    num_likelihood_evaluations: jnp.ndarray  # how many times the likelihood has been evaluated
    patience_steps: jnp.ndarray  # how many steps since goal incremental improvement
    termination_reason: jnp.ndarray  # this will be an int reflecting the reason for termination


class GlobalOptimiserResults(NamedTuple):
    samples: Dict[str, jnp.ndarray]  # Dict of arrays with leading dimension num_samples
    total_num_samples: jnp.ndarray  # int, the total number of samples collected.
    log_L_samples: jnp.ndarray  # log(L) of each sample
    total_num_likelihood_evaluations: jnp.ndarray  # how many likelihood evaluations were made in total,
    log_efficiency: jnp.ndarray  # log(total_num_samples / total_num_likelihood_evaluations)
    termination_reason: jnp.ndarray  # this will be an int reflecting the reason for termination
    log_L_max: jnp.ndarray  # maximum likelihood value obtained
    sample_L_max: Dict[str, jnp.ndarray]  # sample at the log_L_max point