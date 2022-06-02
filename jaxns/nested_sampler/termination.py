from jax import numpy as jnp
from jaxns.internals.types import int_type


def termination_condition(num_samples, log_Z_var, log_Z_remaining_upper, log_Z_upper, ess, num_likelihood_evaluations,
                          num_steps, all_plateau, likelihood_contour,
                          *,
                          termination_live_evidence_frac=None,
                          termination_ess=None,
                          termination_evidence_uncert=None,
                          termination_max_num_steps=None,
                          termination_max_samples=None,
                          termination_max_num_likelihood_evaluations=None,
                          termination_likelihood_contour=None
                          ):
    """

    :param num_samples:
    :param log_Z_var:
    :param log_Z_mean:
    :param prev_log_Z_mean:
    :param ess:
    :param num_likelihood_evaluations:
    :param num_steps:
    :param all_plateau:
    :param termination_live_evidence_frac:
    :param termination_ess:
    :param termination_evidence_uncert:
    :param termination_max_num_steps:
    :param termination_max_samples:
    :param termination_max_num_likelihood_evaluations:
    :return:
    """
    termination_condition = jnp.asarray(0, int_type)
    done = jnp.asarray(False)

    def _set_done_bit(bit, bit_reason, done, termination_condition):
        done = done | bit
        termination_condition += jnp.where(bit,
                                           jnp.asarray(2 ** bit_reason, int_type),
                                           jnp.asarray(0, int_type))
        return done, termination_condition

    if termination_max_samples is not None:
        # used all points
        reached_max_samples = num_samples >= termination_max_samples
        done, termination_condition = _set_done_bit(reached_max_samples, 0,
                                                    done=done, termination_condition=termination_condition)
    if termination_evidence_uncert is not None:
        # dynamic stopping condition, for G=0
        evidence_uncert_low_enough = log_Z_var <= jnp.square(termination_evidence_uncert)
        done, termination_condition = _set_done_bit(evidence_uncert_low_enough, 1,
                                                    done=done, termination_condition=termination_condition)
    if termination_live_evidence_frac is not None:
        # dynamic stopping condition, for stopping static run
        # Z_remaining/(Z_remaining + Z_current) < delta
        small_remaining_evidence = log_Z_remaining_upper - log_Z_upper < jnp.log(termination_live_evidence_frac)
        done, termination_condition = _set_done_bit(small_remaining_evidence, 2,
                                                    done=done, termination_condition=termination_condition)
    if termination_ess is not None:
        # Kish's ESS = [sum weights]^2 / [sum weights^2]
        ess_reached = ess >= termination_ess
        done, termination_condition = _set_done_bit(ess_reached, 3,
                                                    done=done, termination_condition=termination_condition)
    if termination_max_num_steps is not None:
        max_steps_reached = num_steps >= termination_max_num_steps
        done, termination_condition = _set_done_bit(max_steps_reached, 4,
                                                    done=done, termination_condition=termination_condition)
    if termination_max_num_likelihood_evaluations is not None:
        too_max_likelihood_evaluations = num_likelihood_evaluations >= termination_max_num_likelihood_evaluations
        done, termination_condition = _set_done_bit(too_max_likelihood_evaluations, 5,
                                                    done=done, termination_condition=termination_condition)

    if all_plateau is not None:
        done, termination_condition = _set_done_bit(all_plateau, 6,
                                                    done=done, termination_condition=termination_condition)

    if termination_likelihood_contour is not None:
        likeihood_contour_reached = likelihood_contour >= termination_likelihood_contour
        done, termination_condition = _set_done_bit(likeihood_contour_reached, 7,
                                                    done=done, termination_condition=termination_condition)

    return done, termination_condition
