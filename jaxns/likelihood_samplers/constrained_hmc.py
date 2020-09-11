from collections import namedtuple

from jax import value_and_grad, random, numpy as jnp
from jax.lax import cond, while_loop, dynamic_update_slice

CHMCSamplerState = namedtuple('CHMCSamplerState',
                              ['step_size'])


def init_chmc_sampler_state(num_live_points):
    return CHMCSamplerState(step_size=0.33 * jnp.ones(num_live_points))


def constrained_hmc(key, log_L_constraint, live_points_U,
                    last_live_point, loglikelihood_from_constrained,
                    prior_transform, sampler_state, max_steps, i_replace, log_X_mean):
    """
    Samples from the prior restricted to the likelihood constraint.
    This undoes the shrinkage at each step to approximate a bound on the contours.
    First it does a scaling on each dimension.

    Args:
        key:
        log_L_constraint:
        live_points_U:
        spawn_point_U:
        loglikelihood_from_constrained:

    Returns:

    """

    N, D = live_points_U.shape

    def constraint(U):
        return loglikelihood_from_constrained(**prior_transform(U)) - log_L_constraint

    val_and_grad_from_U = value_and_grad(constraint)

    step_size = jnp.exp(jnp.log(0.33) + log_X_mean/D)

    def transition(key, u_init, step_size):
        p_key, sample_key = random.split(key, 2)
        p_init = random.normal(p_key, shape=u_init.shape)

        def body(state):
            (i, u, p, num_f, distance, num_bounce, results_array) = state
            # half step forward
            u = u + step_size * p
            # check violation
            C, C_grad = val_and_grad_from_U(u)
            n = C_grad / jnp.linalg.norm(C_grad)
            # bounce off contours and boundaries of
            p_bounce = jnp.where(((u > 1.) | (u < 0.)) | jnp.isnan(n),
                                 -p,
                                 p - 2. * (p @ n) * n)
            # update momentum
            p = jnp.where(C >= 0., p, p_bounce)
            # update counters
            num_bounce = jnp.where(C >= 0., num_bounce, num_bounce + 1)
            distance = distance + step_size
            num_f = num_f + 1
            results_array = dynamic_update_slice(results_array, u[None, :], [i, 0])
            return (i + 1, u, p, num_f, distance, num_bounce, results_array)

        results_array = jnp.zeros((max_steps, D))
        (num_steps, u_test, p_test, num_f, distance, num_bounce, results_array) = while_loop(
            lambda state: (state[0] < max_steps) & (state[5] < 1),
            body,
            (0, u_init, p_init, 0, 0., 0, results_array))
        i = random.randint(sample_key, shape=(), minval=0, maxval=num_steps)
        u_test = results_array[i, :]
        step_size = jnp.where(num_bounce > 0, distance / num_bounce, distance)
        return u_test, num_f, step_size

    def _cond(state):
        (key, u_test, x_test, log_L_test, num_f, step_size) = state
        return log_L_test <= log_L_constraint

    def _body(state):
        (key, u_test, x_test, log_L_test, num_f, step_size) = state
        key, transition_key, select_key = random.split(key, 3)
        i = random.randint(select_key, (), 0, live_points_U.shape[0])
        u_init = live_points_U[i, :]
        # step_size = sampler_state.step_size[i]
        u_test, new_num_f, step_size = transition(transition_key, u_init, step_size)
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        num_f = num_f + new_num_f + 1
        return (key, u_test, x_test, log_L_test, num_f, step_size)

    (key, u_new, x_new, log_L_new, num_f, step_size) = while_loop(_cond,
                                                                  _body,
                                                                  (key, live_points_U[0, :], last_live_point,
                                                                   log_L_constraint, 0, step_size))
    # sampler_state = sampler_state._replace(step_size=
    #                                        dynamic_update_slice(sampler_state.step_size, step_size[None], [i_replace]))
    CHMCResults = namedtuple('CHMCResults',
                             ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new', 'sampler_state'])
    return CHMCResults(key, num_f, u_new, x_new, log_L_new, sampler_state)
