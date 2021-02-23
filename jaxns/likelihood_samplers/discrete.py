from collections import namedtuple
from jax import numpy as jnp, random
from jax.lax import while_loop

import logging

logger = logging.getLogger(__name__)


DiscreteSamplerState = namedtuple('DiscreteSliceSamplerState', ['num_outcomes'])
DiscreteSamplingResults = namedtuple('MultiEllipsoidSamplingResults',
                                           ['key', 'num_likelihood_evaluations', 'u_new', 'log_L_new'])

def init_discrete_sampler_state(num_outcomes):
    return DiscreteSamplerState(num_outcomes=num_outcomes)

def sample_discrete_subspace(key,
                            log_L_constraint,
                            log_likelihood_from_U,
                            sampler_state: DiscreteSamplerState):
    """
    Samples a discrete subspace using uniform sampling without replacement.
    Since at lease of the points satisfies the likelihood constraint, then there is at least one feasible solution.

    Args:
        subspace: list of discrete priors.

    Returns:
        new_u, new_log_L, num_new_f_evals
    """
    # get number of outcomes
    num_outcomes = sampler_state.num_outcomes

    def sample_without_replacement(state):
        (done, key, available_slots, _, _, num_f_evals) = state
        # at least one should always work, because it's in the live_points already
        prob = available_slots / jnp.sum(available_slots)
        key, choice_key = random.split(key, 2)
        u_test = random.choice(choice_key, sampler_state.num_outcomes,
                               shape=(), p=prob)
        available_slots = (jnp.arange(num_outcomes) == u_test) & available_slots
        test_log_L = log_likelihood_from_U(u_test)
        done = test_log_L > log_L_constraint
        return (done, key, available_slots, u_test, test_log_L, num_f_evals + 1)

    (done, key, _, new_u, new_log_L, num_f_evals) = while_loop(lambda state: ~state[0],
                                                               sample_without_replacement,
                                                               (jnp.asarray(False),
                                                                key,
                                                                jnp.ones(num_outcomes, dtype=jnp.bool_),
                                                                jnp.asarray(0),
                                                                log_L_constraint,
                                                                jnp.asarray(0)))

    return DiscreteSamplingResults(key, num_f_evals, new_u, new_log_L)
