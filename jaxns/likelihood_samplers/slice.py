from collections import namedtuple
import numpy as np
from jax import vmap, numpy as jnp, random
from jax.scipy.special import logsumexp
from jax.lax import while_loop, scan, cond, dynamic_update_slice
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering, maha_ellipsoid, circle_to_ellipsoid, ellipsoid_to_circle, log_ellipsoid_volume

SliceSamplerState = namedtuple('SliceSamplerState',
                               ['cluster_id', 'mu', 'radii', 'rotation', 'num_k', 'num_fev_ma'])


def init_slice_sampler_state(key, live_points_U, depth, log_X, num_slices):
    cluster_id, (mu, radii, rotation) = ellipsoid_clustering(key, live_points_U, depth, log_X)
    num_k = jnp.bincount(cluster_id, minlength=0, length=mu.shape[0])
    return SliceSamplerState(cluster_id=cluster_id, mu=mu, radii=radii, rotation=rotation,
                                      num_k=num_k, num_fev_ma=jnp.asarray(num_slices*live_points_U.shape[1] + 2.))

def recalculate_sampler_state(key, live_points_U, sampler_state, log_X):
    num_clusters = sampler_state.mu.shape[0]  # 2^(depth-1)
    depth = np.log2(num_clusters) + 1
    depth = depth.astype(np.int_)
    return init_slice_sampler_state(key, live_points_U, depth, log_X, 1)._replace(num_fev_ma=sampler_state.num_fev_ma)

def evolve_sampler_state(sampler_state, live_points_U):
    """
    Evolve all ellipsoids by shrinking each ellipsoid by n/(n+1)
    V(S) = sum_k V(S_k)
         = (sum_k n_k/n) V(S)
         = (sum_k n_k/n) sum_k V(E_k)
    V(S) -> V(S) n/(n+1)
    => sum_k V(E_k) -> (sum_k V(E_k)) n/(n+1)
    Args:
        sampler_state:
        num_live_points:
        log_X:
        k_from:

    Returns:

    """
    N, D = live_points_U.shape
    new_radii = jnp.exp(jnp.log(sampler_state.radii)
                        + (jnp.log(N) - jnp.log(N + 1.)) / D)
    sampler_state = sampler_state._replace(radii=new_radii)
    return sampler_state

def slice_sampling(key,
                   log_L_constraint,
                   live_points_U,
                   num_slices,
                   loglikelihood_from_constrained,
                   prior_transform,
                   i_min,
                    log_X,
                   sampler_state: SliceSamplerState):
    N, D = live_points_U.shape

    # subtract the i_min
    k_from = sampler_state.cluster_id[i_min]
    num_k = dynamic_update_slice(sampler_state.num_k, sampler_state.num_k[k_from, None] - 1, k_from[None])
    sampler_state = sampler_state._replace(num_k=num_k)
    ###
    # evolve ellipsoids or potentially recalculate ellipsoids
    # sampler_state = evolve_sampler_state(sampler_state, live_points_U)

    def log_likelihood_from_U(U):
        return loglikelihood_from_constrained(**prior_transform(U))

    def slice_sample_1d(key, x, n, w):
        """
        Perform a 1D slice sampling along x + t*n where n is a unit vector.
        The typical scale is given by w.
        We constrain t so that the sampling is within the unit cube.

            x + t*n = 1
            t1 = (1-x)/n
            t1_right = where(t1>0, min(t1), inf)
            t1_left = where(t1<0, max(t1), -inf)
            x + t*n = 0
            t0 = max(-x/n)
            t0_right = where(t0>0, min(t0), inf)
            t0_left = where(t0<0, max(t0), -inf)
            right_bound = jnp.minimum(t0_right,t1_right)
            left_bound = jnp.maximum(t0_left,t1_left)

        Args:
            key: PRNG
            x: point to sample from
            n: unit vector
            w: typical scale along the 1D slice.

        Returns:

        """

        t1 = (1. - x) / n
        t1_right = jnp.min(jnp.where(t1 >= 0., t1, jnp.inf))
        t1_left = jnp.max(jnp.where(t1 <= 0., t1, -jnp.inf))
        t0 = -x / n
        t0_right = jnp.min(jnp.where(t0 >= 0., t0, jnp.inf))
        t0_left = jnp.max(jnp.where(t0 <= 0., t0, -jnp.inf))
        right_bound = jnp.minimum(t0_right, t1_right)
        left_bound = jnp.maximum(t0_left, t1_left)
        # find bracket, within unit-cube
        key, brack_key = random.split(key, 2)
        w = jnp.minimum(w, right_bound - left_bound)

        left = jnp.maximum(- random.uniform(brack_key, minval=0., maxval=w), left_bound)
        right = jnp.minimum(left + w, right_bound)
        f_left = log_likelihood_from_U(x + left * n)
        f_right = log_likelihood_from_U(x + right * n)
        num_f_eval = 2
        do_step_out = (f_left > log_L_constraint) | (f_right > log_L_constraint)

        # print('do step out',do_step_out,'f_left', f_left, 'f_right', f_right, 'w', w, 'left_bound', left_bound, 'right_bound', right_bound, 'left', left, 'right', right)
        # if jnp.isnan(w):
        #     plot(left,right)

        def step_out_body(state):
            (done, num_f_eval, left, f_left, right, f_right) = state

            def step_left(args):
                # print('step left')
                left, f_left = args
                left = jnp.maximum(left - 0.5 * w, left_bound)
                return 1, left, log_likelihood_from_U(x + left * n)

            def step_right(args):
                # print("step right")
                right, f_right = args
                right = jnp.minimum(right + 0.5 * w, right_bound)
                return 1, right, log_likelihood_from_U(x + right * n)

            (n_left, new_left, f_left) = cond((f_left > log_L_constraint) & (jnp.abs(left - left_bound) < 1e-15),
                                          step_left,
                                          lambda _: (0, left, f_left),
                                          (left, f_left))
            (n_right, new_right, f_right) = cond((f_right > log_L_constraint) & (jnp.abs(right - right_bound) < 1e-15),
                                             step_right,
                                             lambda _: (0, right, f_right),
                                             (right, f_right))
            done = ((f_left <= log_L_constraint) & (f_right <= log_L_constraint)) \
                   | ((jnp.abs(new_left - left_bound) < 1e-15) & (jnp.abs(new_right - right_bound) < 1e-15)) \
                    | ((left == new_left) & (right == new_right))
            # print('step out', 'f_left', f_left, 'f_right', f_right, 'left_bound', left_bound,
            #       'right_bound', right_bound, 'left', left, 'right', right, 'w', w)
            # plot(left,right)
            return (done, num_f_eval + n_left + n_right, new_left, f_left, new_right, f_right)

        (_, num_f_eval, left, f_left, right, f_right) = while_loop(lambda state: ~state[0],
                                                                   step_out_body,
                                                                   (~do_step_out, num_f_eval, left, f_left, right,
                                                                    f_right))

        # shrinkage step
        def shrink_body(state):
            (_, num_f_eval, key, left, right, _, _) = state
            key, t_key = random.split(key, 2)
            t = random.uniform(t_key, minval=left, maxval=right)
            x_t = x + t * n
            f_t = log_likelihood_from_U(x_t)
            done = f_t > log_L_constraint
            left = jnp.where(t < 0., t, left)
            right = jnp.where(t > 0., t, right)
            # print('shrink','t', t, 'left',left,'right', right, 'f', f_t)
            # plot(left, right)
            return (done, num_f_eval + 1, key, left, right, x_t, f_t)

        (done, num_f_eval, key, left, right, x_t, f_t) = while_loop(lambda state: ~state[0],
                                                                    shrink_body,
                                                                    (
                                                                    jnp.asarray(False), num_f_eval, key, left, right, x,
                                                                    log_L_constraint))

        return key, x_t, f_t, num_f_eval

    # use the members of cluster to circularise the search.
    # let R map ellipsoidal to circular
    # let C map circular to ellipsoidal
    # if (x-f).C^T.C.(x-f) <= 1 then
    # select unit vector n distributed on ellipse:
    # n = C.n_circ / |C.n_circ|
    # Selecting x + n*t we ask how large can t be approximately and still remain inside the bounding ellipsoid.
    # Solve (x + n*t-f).R^T.R.(x + n*t-f) = 1
    # Solve (x-f + n*t).R^T.R.(x-f + n*t) = 1
    # Solve (x-f).R^T.R.(x-f) + 2*t*n.R^T.R.(x-f) + t**2 * n.R^T.R.n = 1

    def which_cluster(x):
        dist = vmap(lambda mu, radii, rotation: maha_ellipsoid(x, mu, radii, rotation))(sampler_state.mu,
                                                                                 sampler_state.radii,
                                                                                 sampler_state.rotation)
        return jnp.argmin(jnp.where(jnp.isnan(dist), jnp.inf, dist))

    def slice_body(state, X):
        (key, num_f_eval0, x, _) = state

        k = which_cluster(x)
        mu_k = sampler_state.mu[k,:]
        radii_k = sampler_state.radii[k,:]
        rotation_k = sampler_state.rotation[k,:,:]

        key, n_key = random.split(key, 2)

        n = random.normal(n_key, shape=x.shape)
        n /= jnp.linalg.norm(n)
        origin = jnp.zeros(n.shape)

        # M
        # get w by circularising or getting
        Ln = ellipsoid_to_circle(n, origin, radii_k, rotation_k)
        Ldx = ellipsoid_to_circle(x-mu_k, origin, radii_k, rotation_k)

        a = Ln @ Ln
        b = 2. * Ln @ Ldx
        c = Ldx @ Ldx - 1.
        w = jnp.sqrt(b ** 2 - 4. * a * c) / a
        w = jnp.where(jnp.isnan(w), jnp.max(radii_k), w)

        (key, x_t, f_t, num_f_eval) = slice_sample_1d(key, x, n, w)

        return (key, num_f_eval0 + num_f_eval, x_t, f_t), ()

    key, select_key = random.split(key, 2)
    i = random.randint(select_key, shape=(), minval=0,maxval=N)
    u_init = live_points_U[i, :]

    (key, num_likelihood_evaluations, u_new, log_L_new), _ = scan(slice_body,
                                                               (key, jnp.asarray(0), u_init, log_L_constraint),
                                                               (jnp.arange(num_slices),),
                                                               unroll=1)
    x_new = prior_transform(u_new)
    ellipsoid_select = which_cluster(u_new)

    cluster_id = dynamic_update_slice(sampler_state.cluster_id,
                                      ellipsoid_select[None],
                                      i_min[None])

    num_k = dynamic_update_slice(sampler_state.num_k,
                                 sampler_state.num_k[ellipsoid_select, None] + 1,
                                 ellipsoid_select[None])
    sampler_state = sampler_state._replace(cluster_id=cluster_id, num_k=num_k)

    log_volumes = vmap(lambda radii: log_ellipsoid_volume(radii))(sampler_state.radii)
    log_F = logsumexp(log_volumes) - log_X

    # V(E_k) > 2 V(S_k)
    # |S_k| < D+1
    # jnp.any(log_volumes > jnp.log(sampler_state.num_k) - jnp.log(N) + log_X + jnp.log(2.))
    # jnp.any(sampler_state.num_k == D)
    too_many_func_evals = D*2#step out and a shrink per dimension
    do_recalculate = (num_likelihood_evaluations > sampler_state.num_fev_ma + too_many_func_evals) \
                     | (log_F < 0.)

    tau = 1. / N
    sampler_state = sampler_state._replace(
        num_fev_ma=sampler_state.num_fev_ma * (1. - tau) + tau * num_likelihood_evaluations)

    # print('do_recalculate', do_recalculate, 'num fev', num_likelihood_evaluations, '/', sampler_state.num_fev_ma,
    #       'V(E)/V(S)', jnp.exp(log_F), 'V(E_k)', jnp.exp(log_volumes), '2V(S_k)',
    #       jnp.exp(jnp.log(sampler_state.num_k) - jnp.log(N) + log_X + jnp.log(2.)),
    #       'num_k', sampler_state.num_k)

    key, recalc_key = random.split(key, 2)

    sampler_state = cond(do_recalculate,
                         lambda args: recalculate_sampler_state(*args),
                         lambda _: sampler_state,
                         (recalc_key, live_points_U, sampler_state, log_X))

    # if do_recalculate:
    #     plot(-1,1)

    SliceSamplingResults = namedtuple('SliceSamplingResults',
                                      ['key', 'num_likelihood_evaluations', 'u_new', 'x_new', 'log_L_new',
                                       'sampler_state'])
    return SliceSamplingResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new, sampler_state)
