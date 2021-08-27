from jax import random, numpy as jnp, vmap
from jax.lax import while_loop, cond

from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_to_circle, maha_ellipsoid
from jax.scipy.linalg import solve_triangular
from jaxns.gaussian_process.kernels import M32
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize
from jax.scipy.special import erf

def prepare_bayesian_opt(X, Y):
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    #Remove infinities, which adhoc rule
    Y = jnp.where(jnp.isinf(Y), jnp.min(Y), Y)

    y_mean = jnp.mean(Y)
    y_scale = jnp.std(Y) + 1e-6
    Y -= y_mean
    Y /= y_scale

    x_mean = jnp.mean(X, axis=0, keepdims=True)
    x_scale = jnp.maximum(1e-6, jnp.max(X, axis=0, keepdims=True) - jnp.min(X, axis=0, keepdims=True))
    X -= x_mean
    X /= x_scale

    def log_likelihood(lengthscales, sigma):
        kernel = M32()
        K = kernel(X, X, lengthscales, sigma)
        data_cov = 1e-6 * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y)
        return log_normal(Y, mu, K + data_cov)

    init_param, unravel_func = ravel_pytree(dict(lengthscales=jnp.log(0.2 * jnp.ones(X.shape[-1])),
                                                 sigma=jnp.log(jnp.ones(())),
                                                 ))

    def loss(param):
        kwargs = unravel_func(jnp.exp(param))
        return -log_likelihood(**kwargs)

    result = minimize(loss, init_param, method='bfgs')

    def cdf(x, n, t, level):
        """
        P(f(x + n*t) < level)

        Args:
            x:
            n:
            t:
            level:

        Returns:

        """
        level -= y_mean
        level /= y_scale
        kernel = M32()
        kwargs = unravel_func(jnp.exp(result.x))
        K = kernel(X, X, kwargs['lengthscales'], kwargs['sigma'])
        data_cov = 1e-6 * jnp.eye(X.shape[0])
        dy = Y
        J = jnp.linalg.solve(K + data_cov, dy)
        K_star = kernel(x[None] + n[None] * t, X, kwargs['lengthscales'], kwargs['sigma'])
        mu = K_star @ J
        mu = mu[0]
        C = kwargs['sigma'] ** 2 - K_star @ jnp.linalg.solve(K + data_cov, K_star.T)
        C = C[0, 0]
        dx = level - mu
        dx /= jnp.sqrt(2. * C)
        return 0.5 * (1. + erf(dx))

    def interval_length(x, n, left_bound, right_bound, level):
        """
        Returns the fraction of line above a certain level.

        Args:
            x: ray origin
            n: ray direction
            left_bound: minimal ray affine paramter
            right_bound: maximal ray affine parameter
            level: slice level, log L contraint

        Returns: expectation of part of ray above `level`, marginalised over a Gaussian process constrained to the line.
        """
        def line_integral(y, t):
            return 1. - cdf(x, n, t, level)

        t = jnp.linspace(left_bound, right_bound, 2*X.shape[0])#can't get more resolution than this typically
        dt = t[1] - t[0]
        w = jnp.sum(vmap(lambda t: line_integral(None, t))(t)) * dt
        # w = odeint(line_integral, 0., jnp.asarray([left_bound, right_bound]))[-1]
        return w
    return interval_length

def slice_sample_1d(key, x, logL_current, n, w, log_L_constraint, log_likelihood_from_U,
                    live_points,
                    do_init_try_bracket=True, do_stepout=False, midpoint_shrink=False):
    """
    Perform slice sampling from a point which is inside the slice along a unit vector direction.
    The slice sampling is constrained to the unit-cube.

    Args:
        key: PRNG key
        x: initial point inside the slice
        logL_current: log-likelihood at x.
        n: unit-vector direction along which to slice sample
        w: initial estimate of interval size
        log_L_constraint: slice level
        log_likelihood_from_U: callable(U_compact)
        live_points: the current live points,
        do_init_try_bracket: bool, find the maximal support based on live-points.
        do_stepout: bool, if true then do stepout with doubling until bracketed
        midpoint_shrink: bool, if true the use midpoint shrinkage, at the cost of extra auto-correlation.

    Returns:
        key: new PRNG key
        x_t: accepted point
        f_t: log-likelihood at accepted point
        num_f_eval: number of likelihood evaluations used
    """
    left_bound, right_bound = slice_bounds(x, n)
    if do_init_try_bracket:
        #(n) @ (x + n*t - point) = 0
        #Take maximum positive as right and minimum negative as left
        def _orth_dist(point):
            t = (point - x) @ n
            return t
        t = vmap(_orth_dist)(live_points)
        left = jnp.maximum(left_bound, jnp.min(jnp.where(t < 0., t, 0.)))
        right = jnp.minimum(right_bound, jnp.max(jnp.where(t > 0., t, 0.)))
        # randomise out to boundary to ensure correctness
        key, key_left_shift, key_right_shift = random.split(key, 3)
        # left = jnp.where(random.uniform(key_left_shift) < 0.5, left, left_bound)
        # right = jnp.where(random.uniform(key_right_shift) < 0.5, right, right_bound)
        log_L_left = log_likelihood_from_U(x + n * left)
        log_L_right = log_likelihood_from_U(x + n * right)
        num_f_eval = jnp.asarray(2)
        left = jnp.where(log_L_left < log_L_constraint, left, left_bound)
        right = jnp.where(log_L_right < log_L_constraint, right, right_bound)
        if do_stepout:
            # w = interval_func(x, n, left_bound, right_bound, log_L_constraint)
            # key, left_key = random.split(key, 2)
            # left = random.uniform(left_key, (), minval=-w, maxval=0.)
            # right = left + w
            key, left, right, num_f_eval_stepout = stepout_1d(key, left, right,
                                                              left_bound, right_bound,
                                                              log_L_constraint,
                                                              log_likelihood_from_U, n, x,
                                                              f_left=log_L_left,
                                                              f_right=log_L_right)
            num_f_eval += num_f_eval_stepout
    else:
        left, right = left_bound, right_bound
        num_f_eval = jnp.asarray(0)

    # shrinkage step
    key, x_t, f_t, num_f_eval_shrink = shrink_1d(key, left, right, logL_current, log_L_constraint,
                                                 log_likelihood_from_U, x, n,
                                                 midpoint_shrink)
    num_f_eval += num_f_eval_shrink

    return key, x_t, f_t, num_f_eval

def slice_bounds(x, n):
    t1 = (1. - x) / n
    t1_right = jnp.min(jnp.where(t1 >= 0., t1, jnp.inf))
    t1_left = jnp.max(jnp.where(t1 <= 0., t1, -jnp.inf))
    t0 = -x / n
    t0_right = jnp.min(jnp.where(t0 >= 0., t0, jnp.inf))
    t0_left = jnp.max(jnp.where(t0 <= 0., t0, -jnp.inf))
    right_bound = jnp.minimum(t0_right, t1_right)
    left_bound = jnp.maximum(t0_left, t1_left)
    return left_bound, right_bound

def slice_initial_bounds_1d(key, w, x, n):
    """
    Determine initial slice bounds from point inside the slice along unit-vector direction.
    Constrains the initial bound to be inside the unit-cube.

    Args:
        key: PRNG key
        w: initial guess of interval
        x: point inside slice
        n: unit-vector direction to slice along

    Returns:
        key: new PRNG key
        left: left point of interval
        right: right point of interval
        left_bound: supreme left-most point
        right_bound: supreme right-most point
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
    return key, left, right, left_bound, right_bound


def stepout_1d(key, left, right, left_bound, right_bound,
               log_L_constraint, log_likelihood_from_U, n, x,
               f_left=None,f_right=None
               ):
    """
    Stepout along slice in the correct way guaranteeing detailed balance of proposals (interval should be equally
        probable from accepted point).

    Args:
        key: PRNG key
        left: initial left bound
        right: initial right boud
        left_bound: supreme left-most bound
        right_bound: supreme right-most boud
        log_L_constraint: slice level
        log_likelihood_from_U: callable(U_compact)
        n: unit-vector direction
        x: point inside slice

    Returns:
        key: new PRNG key
        left: bracketing left side of interval
        right: bracketing right side of interval
        num_f_eval: number of log-likelihood evaluations
    """

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

    num_f_eval = 0
    if f_left is None:
        f_left = log_likelihood_from_U(x + left * n)
        num_f_eval += 1
    if f_right is None:
        f_right = log_likelihood_from_U(x + right * n)
        num_f_eval += 1
    within_contour = (f_left > log_L_constraint) | (f_right > log_L_constraint)
    (_, key, num_f_eval, left, f_left, right, f_right) = while_loop(lambda state: ~state[0],
                                                                    step_out_body,
                                                                    (~within_contour, key, num_f_eval, left, f_left,
                                                                     right,
                                                                     f_right))
    return key, left, right, num_f_eval


def shrink_1d(key, left, right, logL_current, log_L_constraint, log_likelihood_from_U, x, n, midpoint_shrink):
    """
    Correctly shrink interval along slice.

    Args:
        key: PRNG key
        left: initial left point of interval
        right: initial right point of interval
        logL_current: log-likelihood at x
        log_L_constraint: slice level
        log_likelihood_from_U: callable(U_compact)
        x: point inside slice
        n: direction to slice along
        midpoint_shrink: bool, whether of shrink to mid-point from origin to rejected point

    Returns:
        key: ney PRNG key
        x_t: accepted point
        f_t: log-likelihood at accepted point
        num_f_eval: number of likelihood evaluations

    """

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
            # y(t) = m * t + b
            # y(0) = b
            # (y(t_R) - y(o))/t_R
            # y(t_R*alpha) = (y(t_R) - y(0))*alpha + y(0)
            key, alpha_key, beta_key = random.split(key, 3)
            alpha = random.uniform(alpha_key)
            beta = random.uniform(beta_key)
            logL_alpha = logL_current + alpha * (f_t - logL_current)
            logL_beta = f_t + beta * (log_L_constraint - f_t)
            do_mid_point_shrink = logL_alpha < logL_beta
            left = jnp.where((t < 0.) & do_mid_point_shrink, alpha * left, left)
            right = jnp.where((t > 0.) & do_mid_point_shrink, alpha * right, right)
        return (done, num_f_eval + 1, key, left, right, x_t, f_t)

    (done, num_f_eval, key, left, right, x_t, f_t) = while_loop(lambda state: ~state[0],
                                                                shrink_body,
                                                                (
                                                                    jnp.asarray(False), jnp.asarray(0), key, left,
                                                                    right,
                                                                    x,
                                                                    logL_current))
    return key, x_t, f_t, num_f_eval


def compute_init_interval_size(n, u_current, mu, radii, rotation):
    """
    Use a ellipsoidal decomposition to determine an initial bracket interval size `w`.

    Args:
        n: unit-vector direction
        u_current: point inside slice and ellipsoidal decomposition
        mu: means of ellipsoids
        radii: radii of ellipsoids
        rotation: rotation matrices of ellipsoids

    Returns:
        w: point of intersectin of ray and ellipsoid that u_current falls most.
    """

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
    Ln = ellipsoid_to_circle(n, jnp.zeros_like(n), radii_k, rotation_k)
    Ldx = ellipsoid_to_circle(u_current - mu_k, jnp.zeros_like(n), radii_k, rotation_k)
    a = Ln @ Ln
    b = 2. * Ln @ Ldx
    c = Ldx @ Ldx - 1.
    w = jnp.sqrt(b ** 2 - 4. * a * c) / a
    w = jnp.where(jnp.isnan(w), jnp.max(radii_k), w)
    return w
