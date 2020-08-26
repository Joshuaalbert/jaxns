from collections import namedtuple

from jax import value_and_grad, random, numpy as jnp
from jax.lax import cond, while_loop

CHMCSamplerState = namedtuple('CHMCSamplerState',
                                   ['mvee_u'])

def init_chmc_sampler_state(num_live_points, whiten=True):
    return CHMCSamplerState(mvee_u=None)

def constrained_hmc(key, log_L_constraint, live_points_U,
                    last_live_point, loglikelihood_from_constrained,
                    prior_transform, T, sampler_state):
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

    def constraint(U):
        return loglikelihood_from_constrained(**prior_transform(U)) - log_L_constraint

    val_and_grad_from_U = value_and_grad(constraint)

    def transition(key, u_init, step_size, T):
        p_init = random.normal(key, shape=u_init.shape)

        def body(state):
            (i, u, p, num_f) = state
            u = u + step_size * p
            C = constraint(u)
            num_f = num_f + 1

            def bounce(args):
                p, u, num_f = args
                _, C_grad = val_and_grad_from_U(u)
                n = C_grad / jnp.linalg.norm(C_grad)
                return p - 2. * (p @ n) * n, num_f + 1

            def no_bounce(args):
                p, num_f = args
                return p, num_f

            (p, num_f) = cond(C >= 0.,
                              (p, num_f),
                              no_bounce,
                              (p, u, num_f),
                              bounce)

            return (i + 1, u, p, num_f)

        (_, u_test, p_test, num_f) = while_loop(lambda state: state[0] < T,
                                                body,
                                                (0, u_init, p_init, 0))
        u_test = u_test + step_size * p_test
        return u_test, num_f

    def _cond(state):
        (key, u_test, x_test, log_L_test, num_f) = state
        return log_L_test <= log_L_constraint

    def _body(state):
        (key, u_test, x_test, log_L_test, num_f) = state
        key, transition_key, select_key = random.split(key, 3)
        u_init = live_points_U[random.randint(select_key, (), 0, live_points_U.shape[0]), :]
        # dist = jnp.abs(u_init - points)
        # dist = jnp.where(dist == 0., jnp.inf, dist)
        step_size = jnp.std(live_points_U, axis=0)/T
        u_test, _num_f = transition(transition_key, u_init, step_size, T)

        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(**x_test)
        num_f = num_f + _num_f + 1
        return (key, u_test, x_test, log_L_test, num_f)

    (key, u_new, x_new, log_L_new, num_f) = while_loop(_cond,
                                                       _body,
                                                       (key, live_points_U[0, :], last_live_point, log_L_constraint, 0))

    CHMCResults = namedtuple('CHMCResults',
                             ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new', 'sampler_state'])
    return CHMCResults(key, num_f, u_new, x_new, log_L_new, CHMCSamplerState(mvee_u=None))