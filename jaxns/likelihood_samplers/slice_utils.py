from jax import random, numpy as jnp, vmap
from jax.lax import while_loop, cond

from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_to_circle, maha_ellipsoid


def slice_sample_1d(key, x, n, w, log_L_constraint, log_likelihood_from_U, do_stepout=False, midpoint_shrink=False,
                    midpoint_shrink_thresh=100):
    """
    Perform a 1D slice sampling along x + t*n where n is a unit vector.
    The typical scale is given by w.
    We constrain t so that the sampling is within the unit cube.

        x + t*num_options = 1
        t1 = (1-x)/num_options
        t1_right = where(t1>0, min(t1), inf)
        t1_left = where(t1<0, max(t1), -inf)
        x + t*num_options = 0
        t0 = max(-x/num_options)
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
    # w = jnp.minimum(w, right_bound - left_bound)

    left = jnp.maximum(- random.uniform(brack_key, minval=0., maxval=w), left_bound)
    right = jnp.minimum(left + w, right_bound)
    if do_stepout:
        def step_out_body(state):
            (done, key, num_f_eval, left, f_left, right, f_right) = state

            key, direction_key = random.split(key, 2)

            def step_left(args):
                left, right, f_left, f_right = args
                left = jnp.maximum(left - (right - left), left_bound)
                return 1, left, right, log_likelihood_from_U(x + left * n), f_right

            def step_right(args):
                left, right, f_left, f_right = args
                right = jnp.minimum(right + (right - left), right_bound)
                return 1, left, right, f_left, log_likelihood_from_U(x + right * n)

            do_step_left = random.randint(direction_key, shape=(), minval=0, maxval=2, dtype=jnp.int_) == jnp.asarray(1)

            (n_f_eval, new_left, new_right, f_left, f_right) = cond(do_step_left,
                                                                    step_left,
                                                                    step_right,
                                                                    (left, right, f_left, f_right))

            done = ((f_left <= log_L_constraint) & (f_right <= log_L_constraint)) \
                   | ((jnp.abs(new_left - left_bound) < 1e-15) & (jnp.abs(new_right - right_bound) < 1e-15)) \
                   | ((left == new_left) & (right == new_right))
            return (done, key, num_f_eval + n_f_eval, new_left, f_left, new_right, f_right)

        f_left = log_likelihood_from_U(x + left * n)
        f_right = log_likelihood_from_U(x + right * n)
        num_f_eval = 2
        within_contour = (f_left > log_L_constraint) | (f_right > log_L_constraint)

        (_, key, num_f_eval, left, f_left, right, f_right) = while_loop(lambda state: ~state[0],
                                                                        step_out_body,
                                                                        (~within_contour, key, num_f_eval, left, f_left, right,
                                                                         f_right))
    else:
        num_f_eval = 0

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
        if midpoint_shrink:
            new_left = jnp.where(f_t < log_L_constraint - midpoint_shrink_thresh, 0.5*left, left)
            new_right = jnp.where(f_t < log_L_constraint - midpoint_shrink_thresh, 0.5*right, right)
            left = jnp.where(t < 0., new_left, left)
            right = jnp.where(t > 0., new_right, right)
        return (done, num_f_eval + 1, key, left, right, x_t, f_t)

    (done, num_f_eval, key, left, right, x_t, f_t) = while_loop(lambda state: ~state[0],
                                                                shrink_body,
                                                                (
                                                                    jnp.asarray(False), num_f_eval, key, left,
                                                                    right, x,
                                                                    log_L_constraint))

    return key, x_t, f_t, num_f_eval


def compute_init_interval_size(n, origin, u_current, mu, radii, rotation):
    # use the members of cluster_from_nn_dist to circularise the search.
    # let R map ellipsoidal to circular
    # let C map circular to ellipsoidal
    # if (x-f).C^T.C.(x-f) <= 1 then
    # select unit vector num_options distributed on ellipse:
    # num_options = C.n_circ / |C.n_circ|
    # Selecting x + num_options*t we ask how large can t be approximately and still remain inside the bounding ellipsoid.
    # Solve (x + num_options*t-f).R^T.R.(x + num_options*t-f) = 1
    # Solve (x-f + num_options*t).R^T.R.(x-f + num_options*t) = 1
    # Solve (x-f).R^T.R.(x-f) + 2*t*num_options.R^T.R.(x-f) + t**2 * num_options.R^T.R.num_options = 1
    def which_cluster(x):
        dist = vmap(lambda mu, radii, rotation: maha_ellipsoid(x, mu, radii, rotation))(mu, radii, rotation)
        return jnp.argmin(jnp.where(jnp.isnan(dist), jnp.inf, dist))

    k = which_cluster(u_current)
    mu_k = mu[k, :]
    radii_k = radii[k, :]
    rotation_k = rotation[k, :, :]
    # M
    # get w by circularising or getting
    Ln = ellipsoid_to_circle(n, origin, radii_k, rotation_k)
    Ldx = ellipsoid_to_circle(u_current - mu_k, origin, radii_k, rotation_k)
    a = Ln @ Ln
    b = 2. * Ln @ Ldx
    c = Ldx @ Ldx - 1.
    w = jnp.sqrt(b ** 2 - 4. * a * c) / a
    w = jnp.where(jnp.isnan(w), jnp.max(radii_k), w)
    return w