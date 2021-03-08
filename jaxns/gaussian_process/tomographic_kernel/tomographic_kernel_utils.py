from jax import random, vmap, numpy as jnp, value_and_grad
from jax.scipy.special import logsumexp

from jaxns.gaussian_process.utils import squared_norm


def log_tomographic_weight_function_stochastic(key, u, x1, p1, x2, p2):
    """
    int w(x) f(x) dx = sum_i w(dx * i) f(dx * i) dx
    where,
    int w(x) dx = sum_i w(dx * i) dx = 1
    Args:
        key:
        u:
        x1: [N, 3]
        p1: [N, 3]
        x2: [M, 3]
        p2: [M, 3]

    Returns:
        w(dx*i) dx / sum_i w(dx * i) dx
        [N, M] shaped

    """
    n = u.size ** 2
    N = x1.shape[0]
    M = x2.shape[0]
    t1 = random.uniform(key, shape=(n, N, 1))
    t2 = random.uniform(key, shape=(n, M, 1))
    # L, N, M
    norm_squared = vmap(squared_norm)(x1 + t1 * p1, x2 + t2 * p2)
    bins = jnp.concatenate([u, u[-1:] + u[-1] - u[-2]])
    # N*M, U
    hist = vmap(lambda x: jnp.histogram(x, bins)[0])(jnp.reshape(norm_squared, (n, -1)).T)
    # N,M,U
    hist = jnp.reshape(hist, (x1.shape[0], x2.shape[0], u.size))
    log_hist = jnp.log(hist)
    log_du = jnp.diff(bins)
    log_w = log_hist + log_du
    # N,M,U
    log_w = log_w - logsumexp(log_w, axis=-1, keepdims=True)
    log_w = jnp.where(hist == 0., -jnp.inf, log_w)
    return log_w


def tomographic_weight_function_stochastic(key, u, x1, p1, x2, p2):
    return jnp.exp(log_tomographic_weight_function_stochastic(key, u, x1, p1, x2, p2))


def tomographic_weight_function_outer(gamma, x1, x2, p1, p2, S):
    def inner(x1, p1):
        # M
        return vmap(lambda x2, p2: tomographic_weight_function(gamma, x1, x2, p1, p2, S))(x2, p2)

    return vmap(inner)(x1, p1)  # N, M


def log_tomographic_weight_function_outer(gamma, x1, x2, p1, p2, S):
    return vmap(lambda x1,p1:
                vmap(lambda x2, p2: log_tomographic_weight_function(gamma, x1, x2, p1, p2, S)
                     )(x2, p2)
                )(x1, p1)  # N, M


def tomographic_weight_function(gamma, x1, x2, p1, p2=None, S=25):
    return jnp.exp(log_tomographic_weight_function(gamma, x1, x2, p1, p2=p2, S=S))


def log_tomographic_weight_function(gamma, x1, x2, p1, p2=None, S=25):
    parabolic = False
    if p2 is None:
        parabolic = True
        p2 = p1

    x12 = x1 - x2
    A = p1 @ p1
    C = p2 @ p2
    B = -2. * p1 @ p2
    D = 2. * x12 @ p1
    E = -2. * x12 @ p2
    F = x12 @ x12 - gamma

    t1 = jnp.linspace(0., 1., S)[:, None]
    H = (D ** 2 - 4. * A * F + (2. * B * D - 4. * A * E) * t1 + (B ** 2 - 4. * A * C) * t1 ** 2)
    u = (-D - B * t1)
    lower = jnp.clip(0.5 * (u - jnp.sqrt(H)) / A, 0., 1.)
    upper = jnp.clip(0.5 * (u + jnp.sqrt(H)) / A, 0., 1.)
    diff = (upper - lower) / (S - 1)
    if not parabolic:
        reg_valid = H >= 0.
        cdf = jnp.sum(jnp.where(reg_valid, diff, 0.), axis=0)
    else:
        cdf = jnp.sum(diff, axis=0)
    return jnp.log(jnp.diff(cdf)) - jnp.log(jnp.diff(gamma))


def cumulative_tomographic_weight_function(gamma, x1, x2, p1, p2=None, S=25):
    parabolic = False
    if p2 is None:
        parabolic = True
        p2 = p1

    x12 = x1 - x2
    A = p1 @ p1
    C = p2 @ p2
    B = -2. * p1 @ p2
    D = 2. * x12 @ p1
    E = -2. * x12 @ p2
    F = x12 @ x12 - gamma

    t1 = jnp.linspace(0., 1., S)[:, None]
    H = (D ** 2 - 4. * A * F + 2. * B * D * t1 - 4. * A * E * t1 + B ** 2 * t1 ** 2 - 4. * A * C * t1 ** 2)
    u = (-D - B * t1)
    lower = jnp.clip(0.5 * (u - jnp.sqrt(H)) / A, 0., 1.)
    upper = jnp.clip(0.5 * (u + jnp.sqrt(H)) / A, 0., 1.)
    diff = (upper - lower) / (S - 1)
    if not parabolic:
        reg_valid = H >= 0.
        cdf = jnp.sum(jnp.where(reg_valid, diff, 0.), axis=0)
    else:
        cdf = jnp.sum(diff, axis=0)
    return cdf


def _tomographic_weight_function(gamma, x1, x2, p1, p2=None, S=25):
    from jax import grad
    print(gamma.shape)
    return vmap(grad(lambda gamma: cumulative_tomographic_weight_function(gamma, x1, x2, p1, p2=p2, S=S)[0]))(gamma)


def cumulative_tomographic_weight_dimensionless_function(gamma_prime, n, w1, w2=None, S=25):
    parabolic = False
    if w2 is None:
        parabolic = True
        w2 = w1

    A = w1 @ w1
    C = w2 @ w2
    B = -2. * w1 @ w2
    D = 2. * n @ w1
    E = -2. * n @ w2
    F = 1. - gamma_prime

    t1 = jnp.linspace(0., 1., S)[:, None]
    H = (D ** 2 - 4. * A * F + 2. * B * D * t1 - 4. * A * E * t1 + B ** 2 * t1 ** 2 - 4. * A * C * t1 ** 2)
    u = (-D - B * t1)
    lower = jnp.clip(0.5 * (u - jnp.sqrt(H)) / A, 0., 1.)
    upper = jnp.clip(0.5 * (u + jnp.sqrt(H)) / A, 0., 1.)
    diff = (upper - lower) / (S - 1)
    if not parabolic:
        reg_valid = H >= 0.
        cdf = jnp.sum(jnp.where(reg_valid, diff, 0.), axis=0)
    else:
        cdf = jnp.sum(diff, axis=0)
    return cdf


def log_tomographic_weight_dimensionless_function(gamma_prime, n, w1, w2=None, S=25):
    cdf = cumulative_tomographic_weight_dimensionless_function(gamma_prime, n, w1, w2=w2, S=S)

    def density(gamma_prime):
        def cdf(gamma_prime):
            return cumulative_tomographic_weight_dimensionless_function(gamma_prime, n, w1, w2=w2, S=S)[0]

        f, grad = value_and_grad(cdf)(gamma_prime)
        return grad

    w = vmap(density)(0.5 * (gamma_prime[:-1] + gamma_prime[1:]))
    return jnp.log(w)

    return jnp.log(jnp.diff(cdf)) - jnp.log(jnp.diff(gamma_prime))


def cumulative_tomographic_weight_dimensionless_polynomial(Q, gamma_prime, n, w1, w2):
    """
    Computes log P(|x1-x2 + t1*p1 - t2*p2|^2 < lambda)

    Note, that this is invariant to scaling of the input vectors by a scalar,

        P(alpha*|x1-x2 + t1*p1 - t2*p2|^2 < alpha*lambda).

    Therefore a dimensionless form is ,

        P(|num_options + t1*w1 - t2*w2|^2 < lambda')

    where,

        num_options = x1-x2 / |x1-x2| is a unit vector.
        w1 = p1 / |x1-x2|
        w2 = p2 / |x1-x2|
        lambda' = lambda / |x1-x2|^2

    Args:
        Q:
        gamma:
        x1:
        x2:
        p1:
        p2:

    Returns:

    """
    parabolic = False
    if w2 is None:
        parabolic = True
        w2 = w1
    A = w1 @ w1
    C = w2 @ w2
    B = -2. * w1 @ w2
    D = 2. * n @ w1
    E = -2. * n @ w2
    F = 1. - gamma_prime
    param = jnp.asarray([1., A, C, B, D, E, F])
    Q = Q.reshape((-1, 7))
    coefficients = Q @ param
    return jnp.polyval(coefficients, gamma_prime)


def get_polynomial_form():
    """
    The polynomial form of log P(|num_options + t1*w1 - t2*w2|^2 < lambda') is assumed to be:

    c_i = Q_ij p_j
    log_cdf = c_i g_i = g_i Q_ij p_j = Tr(Q @ (p g))
    log_cdf_k = g_ki Q_ij p_kj
    Returns:

    """
    from jax.scipy.optimize import minimize
    from jax import jit, value_and_grad
    from jax.lax import scan
    import pylab as plt

    def generate_single_data(key):
        """
        Generate a physical set of:

        num_options = x1-x2/|x1-x2| is a unit vector.
        w1 = p1 / |x1-x2|
        w2 = p2 / |x1-x2|
        lambda' = lambda / |x1-x2|^2

        Args:
            key:

        Returns:

        """
        keys = random.split(key, 6)
        n = random.normal(keys[0], shape=(3,))
        n = n / jnp.linalg.norm(n)

        w1 = random.normal(keys[1], shape=(3,))
        w1 = w1 / jnp.linalg.norm(w1)
        w1 = w1 * random.uniform(keys[2], minval=0., maxval=10.)

        w2 = random.normal(keys[3], shape=(3,))
        w2 = w2 / jnp.linalg.norm(w2)
        w2 = w2 * random.uniform(keys[4], minval=0., maxval=10.)

        gamma_prime = jnp.linspace(0., 10., 100)  # random.uniform(keys[5],minval=0.,maxval=10.)**2

        cdf_ref = cumulative_tomographic_weight_dimensionless_function(gamma_prime, n, w1, w2, S=150)  # /h**2
        return n, w1, w2, gamma_prime, cdf_ref

    data = jit(vmap(generate_single_data))(random.split(random.PRNGKey(12340985), 100))

    # print(data[-1])
    def loss(Q):
        def single_loss(single_datum):
            n, w1, w2, gamma_prime, cdf_ref = single_datum
            return (vmap(
                lambda gamma_prime: cumulative_tomographic_weight_dimensionless_polynomial(Q, gamma_prime, n, w1, w2))(
                gamma_prime) - cdf_ref) ** 2

        return jnp.mean(vmap(single_loss)(data))

    K = 3
    Q0 = 0.01 * random.normal(random.PRNGKey(0), shape=(K * 7,))
    print(jit(loss)(Q0))

    @jit
    def do_minimize():
        results = minimize(loss, Q0, method='BFGS', options=dict(gtol=1e-8, line_search_maxiter=100))
        print(results.message)
        return results.x.reshape((K, 7)), results.status, results.fun, results.nfev, results.nit, results.jac

    @jit
    def do_sgd(key):
        def body(state, X):
            (Q,) = state
            (key,) = X
            n, w1, w2, gamma_prime, cdf_ref = generate_single_data(key)

            def loss(Q):
                return jnp.mean((vmap(
                    lambda gamma_prime: cumulative_tomographic_weight_dimensionless_polynomial(Q, gamma_prime, n, w1,
                                                                                               w2))(
                    gamma_prime) - cdf_ref) ** 2)  # + 0.1*jnp.mean(Q**2)

            f, g = value_and_grad(loss)(Q)
            Q = Q - 0.00000001 * g
            return (Q,), (f,)

        (Q,), (values,) = scan(body, (Q0,), (random.split(key, 1000),))
        return Q.reshape((-1, 7)), values

    # results = do_minimize()
    Q, values = vmap(do_sgd)(random.split(random.PRNGKey(12456), 100))
    print('Qmean', Q.mean(0))
    print('Qstd', Q.std(0))

    f = values.mean(0)
    fstd = values.std(0)
    plt.plot(jnp.percentile(values, 50, axis=0))
    plt.plot(jnp.percentile(values, 85, axis=0), ls='dotted', c='black')
    plt.plot(jnp.percentile(values, 15, axis=0), ls='dotted', c='black')
    plt.show()

    # print(results)
    return Q.mean(0)


def gamma_min_max(x1, p1, x2, p2):
    """
    Get the minimum and maximum separation squared between two line segments.
    |(x1-x2 + k1 t1 - k2 t2)|^2 smallest to largest in (0,1)x(0,1).
    |(x1-x2 + k1 t1 - k2 t2)|^2
    = |x1-x2|^2 + |k1 t1 - k2 t2|^2 + 2 (x1-x2).(k1 t1 - k2 t2)
    = |x1-x2|^2 + |k1|^2 t1^2 + |k2|^2 t2^2 - 2 t1 t2 k1.k2 + 2 (x1-x2).(k1 t1 - k2 t2)
    =
    Args:
        x1:
        k1:
        x2:
        k2:

    Returns:

    """
    x12 = x1 - x2
    A = p1 @ p1
    C = p2 @ p2
    B = -2. * p1 @ p2
    D = 2. * x12 @ p1
    E = -2. * x12 @ p2
    F = x12 @ x12
    disc = B ** 2 - 4. * A * C

    end_point_dist = vmap(lambda t1, t2: jnp.sum(jnp.square(x12 + t1 * p1 - t2 * p2)))(
        jnp.array([0.,0.,1.,1.]), jnp.array([0.,1.,0.,1.]))

    parabolic = disc == 0.
    t1 = (2. * C * D - B * E) / disc
    t2 = (2. * A * E - B * D) / disc
    closest_within_segments = (t1 > 0.) & (t1 < 1.) & (t2 > 0.) & (t2 < 1.)

    gamma_min = jnp.where(parabolic | (~closest_within_segments),
                          jnp.min(end_point_dist),
                          (C * D ** 2 - B * D * E + A * E ** 2) / disc + F)
    gamma_max = jnp.max(end_point_dist)

    return gamma_min, gamma_max

def test_gamma_min_max():
    x1 = jnp.array([0., 0.])
    x2 = jnp.array([1., 0.])
    p1 = jnp.array([0., 1.])
    p2 = jnp.array([0., 1.])
    assert gamma_min_max(x1, p1, x2, p2) == (1., 2.)

    x1 = jnp.array([0., 0.])
    x2 = jnp.array([1., 0.])
    p1 = jnp.array([0., 1.])
    p2 = jnp.array([1., 1.])
    assert gamma_min_max(x1, p1, x2, p2) == (1., 5.)

