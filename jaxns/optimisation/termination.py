from jax import numpy as jnp
from jaxns.internals.types import int_type

def termination_condition(new_log_L_min, new_log_L_max, patience_steps, num_likelihood_evaluations, num_steps, *,
                          termination_patience=None,
                          termination_frac_likelihood_improvement=None,
                          termination_likelihood_contour=None,
                          termination_max_num_steps=None,
                          termination_max_num_likelihood_evaluations=None
                          ):
    """
    Decide whether to terminate the nested sampler.

    Args:
        termination_stats: TerminationStats
        termination_live_evidence_frac: Stop when mean log evidence changes by less than this amount
        termination_ess: Stop when this many effective samples acquired.
        termination_likelihood_contour: Stop when this likelihood contour reached.
        termination_evidence_uncert: Stop when uncertainty in log evidence is below this.
        termination_max_num_steps: Stop when this many steps taken.
        termination_max_samples: Stop when this many samples taken.

    Returns:
        done: bool, whether to Stop
        termination_condition: int, binary flag recording the reasons for stopping (if more than one).
    """
    termination_condition = jnp.asarray(0, int_type)
    done = jnp.asarray(False)

    def _set_done_bit(bit, bit_reason, done, termination_condition):
        done = done | bit
        termination_condition += jnp.where(bit,
                                           jnp.asarray(2 ** bit_reason, int_type),
                                           jnp.asarray(0, int_type))
        return done, termination_condition

    if termination_frac_likelihood_improvement is not None:
        not_enough_improvement = new_log_L_max - new_log_L_min <= jnp.log1p(termination_frac_likelihood_improvement)
        if termination_patience is not None:
            not_enough_patience = patience_steps >= termination_patience
            done, termination_condition = _set_done_bit(not_enough_patience, 0,
                                                        done=done, termination_condition=termination_condition)
        else:
            done, termination_condition = _set_done_bit(not_enough_improvement, 1,
                                                        done=done, termination_condition=termination_condition)
    if termination_likelihood_contour is not None:
        likelihood_contour_reached = new_log_L_max >= termination_likelihood_contour
        done, termination_condition = _set_done_bit(likelihood_contour_reached, 2,
                                                    done=done, termination_condition=termination_condition)
    if termination_max_num_steps is not None:
        too_max_steps_used = num_steps >= termination_max_num_steps
        done, termination_condition = _set_done_bit(too_max_steps_used, 3,
                                                    done=done, termination_condition=termination_condition)
    if termination_max_num_likelihood_evaluations is not None:
        too_max_likelihood_evaluations = num_likelihood_evaluations >= termination_max_num_likelihood_evaluations
        done, termination_condition = _set_done_bit(too_max_likelihood_evaluations, 4,
                                                    done=done, termination_condition=termination_condition)

    on_plateau = new_log_L_max == new_log_L_min
    done, termination_condition = _set_done_bit(on_plateau, 5,
                                                    done=done, termination_condition=termination_condition)

    return done, termination_condition
