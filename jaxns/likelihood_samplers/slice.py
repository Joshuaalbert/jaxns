from collections import namedtuple

from jaxns.utils import random_ortho_matrix
from jax import vmap, numpy as jnp, random, value_and_grad
from jax.lax import while_loop, scan


def slice_sampling_poly(key, log_L_constraint, live_points, cluster_id,
                        spawn_point, spawn_point_cluster_id, num_repeats,
                        loglikelihood_from_U, sampler_state):
    """
    Given a spawn point inside the feasible regions, perform a series of
    1D slice samplines.

    Args:
        key: PRNGKey
        spawn_point: [M] point to spawn from.
        live_points: [N, M] all live points_U.
        cluster_id: [N] int, the id of the live points_U assigning to a cluster.
        spawn_point_cluster_id: int, cluster if of the spawn point.
        num_repeats: Will cycle through `num_repeats*ndims` random slices.
        loglikelihood_from_U: likelihood_from_U callable

    Returns:

    """

    batched_loglikelihood_from_U = vmap(loglikelihood_from_U)

    def _step_out(x, p, t_L, t_R):
        StepOutState = namedtuple('StepOutState', [
            'bracket', 'num_likelihood_evaluations'])
        w = t_L + t_R
        bracket = jnp.stack([-t_L, t_R])  # 2
        delta = 0.5 * jnp.stack([-w, w])

        # import pylab as plt
        #
        # for i in jnp.unique(cluster_id):
        #     plt.scatter(points[cluster_id == i, 0], points[cluster_id == i, 1])
        # plt.plot([spawn_point_U[0], spawn_point_U[0] + t_R * p[0]], [spawn_point_U[1], spawn_point_U[1] + t_R * p[1]])
        # plt.plot([spawn_point_U[0], spawn_point_U[0] - t_L * p[0]], [spawn_point_U[1], spawn_point_U[1] - t_L * p[1]])
        # plt.show()

        def body(state):
            state = state._replace(bracket=state.bracket + delta,
                                   num_likelihood_evaluations=state.num_likelihood_evaluations + 2)
            return state

        def cond(state):
            # 2, M
            check_point = x + p * state.bracket[:, None]
            check_log_likelihood = batched_loglikelihood_from_U(check_point)
            return ~jnp.all(check_log_likelihood <= log_L_constraint)

        state = StepOutState(bracket=bracket, num_likelihood_evaluations=2)
        state = while_loop(cond,
                           body, state)
        # print(state)
        return state

    def _uniformly_sample_1d_slice(key, x, p, L, R):
        Uniform1dSampleState = namedtuple('Uniform1dSampleState', ['key', 'done', 'L', 'R', 'x', 'log_L',
                                                                   'num_likelihood_evaluations'])

        def body(state):
            key, sample_key = random.split(state.key, 2)
            t = state.L + (state.R - state.L) * random.uniform(sample_key)
            x_test = x + t * p
            log_L_test = loglikelihood_from_U(x_test)
            done = log_L_test > log_L_constraint
            L = jnp.where(t < 0., t, state.L)
            R = jnp.where(t > 0., t, state.R)
            state = state._replace(key=key,
                                   done=done,
                                   L=L,
                                   R=R,
                                   log_L=log_L_test,
                                   x=x_test,
                                   num_likelihood_evaluations=state.num_likelihood_evaluations + 1
                                   )
            return state

        state = Uniform1dSampleState(key=key,
                                     done=False,
                                     L=L,
                                     R=R,
                                     x=x,
                                     log_L=log_L_constraint,
                                     num_likelihood_evaluations=0)

        state = while_loop(lambda state: ~state.done,
                           body,
                           state)
        return state

    OuterSliceSampleState = namedtuple('OuterSliceSampleState', [
        'key', 'x', 'log_L', 'num_likelihood_evaluations'])
    OuterSliceSampleResults = namedtuple('OuterSliceSampleResults', [
        'phantom_x', 'phantom_log_L'])
    InnerSliceSampleState = namedtuple('InnerSliceSampleState', [
        'key', 'i', 'x', 'log_L', 't_R', 't_L', 'num_likelihood_evaluations'
    ])
    outer_state = OuterSliceSampleState(key=key, x=spawn_point, log_L=log_L_constraint,
                                        num_likelihood_evaluations=0)

    def outer_body(outer_state, X):
        key, R_key = random.split(outer_state.key, 2)
        # M,M
        R = random_ortho_matrix(R_key, spawn_point.shape[0])
        # warp so that we explore sampling space

        # print(cholesky, R)
        # Rprime = triangular_solve(cholesky, R, lower=True, transpose_a=True)
        # print(Rprime)

        # initial L, R for each direction
        # t_R[i] = max_(k) (points[k,j] - spawn_point_U[j]) @ R[j,i]
        # t_L[i] = max_(k) (points[k,j] - spawn_point_U[j]) @ -R[j,i]

        # N, M
        dx = live_points - spawn_point
        # [N, M]
        t = dx @ R
        t = jnp.where(cluster_id[:, None] == spawn_point_cluster_id, t, 0.)
        # [M]
        t_R = jnp.maximum(jnp.max(t, axis=0), 0.)
        t_L = jnp.maximum(jnp.max(-t, axis=0), 0.)

        inner_state = InnerSliceSampleState(
            key=key, i=0, x=outer_state.x, log_L=log_L_constraint, t_L=t_L, t_R=t_R,
            num_likelihood_evaluations=0)

        def inner_body(inner_state):
            """
            Perform series of 1D slice samplings through random O(num_parent, R) basis,
            deformed from the sample space to Unit cube space.
            """
            p = R[:, inner_state.i]
            t_L = inner_state.t_L[inner_state.i]
            t_R = inner_state.t_R[inner_state.i]
            stepout_res = _step_out(inner_state.x, p, t_L, t_R)
            uniform_1d_sample_res = _uniformly_sample_1d_slice(inner_state.key,
                                                               inner_state.x,
                                                               p,
                                                               stepout_res.bracket[0],
                                                               stepout_res.bracket[1])
            inner_state = inner_state._replace(key=uniform_1d_sample_res.key,
                                               i=inner_state.i + 1,
                                               x=uniform_1d_sample_res.x,
                                               log_L=uniform_1d_sample_res.log_L,
                                               num_likelihood_evaluations=inner_state.num_likelihood_evaluations
                                                                          + stepout_res.num_likelihood_evaluations
                                                                          + uniform_1d_sample_res.num_likelihood_evaluations)
            return inner_state

        inner_state = while_loop(lambda state: state.i < R.shape[1],
                                 inner_body,
                                 inner_state)
        outer_state = outer_state._replace(key=inner_state.key,
                                           x=inner_state.x,
                                           log_L=inner_state.log_L,
                                           num_likelihood_evaluations=outer_state.num_likelihood_evaluations
                                                                      + inner_state.num_likelihood_evaluations)
        outer_result = OuterSliceSampleResults(phantom_x=inner_state.x,
                                               phantom_log_L=inner_state.log_L)
        return outer_state, outer_result

    outer_state, outer_result = scan(outer_body, outer_state,
                                     (jnp.arange(num_repeats),))
    # remove last one which is the same as our next sampled point.
    outer_result = outer_result._replace(phantom_x=outer_result.phantom_x[:-1, :],
                                         phantom_log_L=outer_result.phantom_log_L[:-1])
    SliceSampleResult = namedtuple('SliceSampleResult', ['key', 'x', 'log_L', 'phantom_x', 'phantom_log_L',
                                                         'num_likelihood_evaluations'])

    return SliceSampleResult(key=outer_state.key, x=outer_state.x,
                             log_L=outer_state.log_L,
                             phantom_x=outer_result.phantom_x,
                             phantom_log_L=outer_result.phantom_log_L,
                             num_likelihood_evaluations=outer_state.num_likelihood_evaluations)


SliceSamplerState = namedtuple('SliceSamplerState',
                                   ['L'])

def init_slice_sampler_state(num_live_points, whiten=True):
    return SliceSamplerState(L=None)

def slice_sampling(key, log_L_constraint, live_points_U,
                   dead_point,
                   num_slices,
                   loglikelihood_from_constrained,
                   prior_transform, sampler_state):

    def constraint(U):
        return loglikelihood_from_constrained(**prior_transform(U))

    C, C_grad = vmap(value_and_grad(constraint))(live_points_U)

    num_f = live_points_U.shape[0]

    # N, M
    normals = C_grad / jnp.linalg.norm(C_grad, axis=-1, keepdims=True)

    def _slice_body(state):
        (key, i, u_init, x_init, _, num_f) = state

        key, p_key, beta_key = random.split(key, 3)

        p = random.normal(p_key, shape=u_init.shape)
        # M
        p = p / jnp.linalg.norm(p)
        # N, M
        dx = u_init - live_points_U
        pn = normals @ p
        # N
        t = vmap(lambda n, dx, pn: (n @ dx) / pn)(normals, dx, pn)
        t_L = jnp.min(t)
        t_R = jnp.max(t)
        print(t_L, t_R)


        # # N
        # t_sorted = jnp.sort(t)
        # print(t_sorted)
        # s_R = jnp.where((t_sorted > 0.) & (pn < 0), -1., 1.)
        # t_R = t_sorted[get_interval(s_R)[1]]
        # _t_sorted = jnp.flip(t_sorted)
        # s_L = jnp.where((_t_sorted < 0.) & (pn > 0), -1., 1.)
        # t_L = _t_sorted[get_interval(s_L)[1]]
        #
        # t_shrink = random.beta(beta_key, points.shape_dict[0], 1) ** jnp.reciprocal(points.shape_dict[1])
        # t_R = t_R / t_shrink
        # t_L = t_L / t_shrink
        # print(t_L, t_R)

        def _sample_cond(state):
            (key, u_test, x_test, log_L_test, _) = state
            return log_L_test <= log_L_constraint

        def _sample_body(state):
            # (t_test, _, _, log_L_test) = state
            (key, _, _, _, num_f) = state
            key, t_key = random.split(key, 2)
            t_sample = random.uniform(t_key, (), minval=t_L, maxval=t_R)
            u_test = u_init + t_sample * p
            x_test = prior_transform(u_test)
            log_L_test = loglikelihood_from_constrained(**x_test)
            return (key, u_test, x_test, log_L_test, num_f+1)

        (key, u_next, x_next, log_L_next, _num_f) = while_loop(_sample_cond,
                                                       _sample_body,
                                                       (key, u_init, x_init, log_L_constraint, 0))
        return (key, i+1, u_next, x_next, log_L_next, num_f+_num_f)

    key, select_key = random.split(key, 2)
    u_init = live_points_U[random.randint(select_key, (), 0, live_points_U.shape[0]), :]

    (key, _, u_new, x_new, log_L_new, num_likelihood_evaluations) = while_loop(lambda state: state[1] < num_slices,
                                                   _slice_body,
                                                   (key, 0, u_init, dead_point, log_L_constraint, num_f))

    SliceSamplingResults = namedtuple('SliceSamplingResults',
                                      ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new', 'sampler_state'])
    return SliceSamplingResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, SliceSamplerState(L=None))