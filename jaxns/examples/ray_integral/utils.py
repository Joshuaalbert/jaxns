from jax import random, vmap, numpy as jnp
from jax.scipy.special import logsumexp


def ellpe(x):
    """
    Complete elliptic integral of the second kind,
        int_0^pi/2 sqrt(1 - x * sin(t)^2) dt
    Args:
        a:

    Returns:

    """

    P = jnp.array([1.53552577301013293365E-4, 2.50888492163602060990E-3,
                   8.68786816565889628429E-3, 1.07350949056076193403E-2,
                   7.77395492516787092951E-3, 7.58395289413514708519E-3,
                   1.15688436810574127319E-2, 2.18317996015557253103E-2,
                   5.68051945617860553470E-2, 4.43147180560990850618E-1,
                   1.00000000000000000299E0])

    Q = jnp.array([3.27954898576485872656E-5, 1.00962792679356715133E-3,
                   6.50609489976927491433E-3, 1.68862163993311317300E-2,
                   2.61769742454493659583E-2, 3.34833904888224918614E-2,
                   4.27180926518931511717E-2, 5.85936634471101055642E-2,
                   9.37499997197644278445E-2, 2.49999999999888314361E-1])

    return jnp.where(x == 0., jnp.ones_like(x),
                     jnp.polyval(P, x) - jnp.log(x) * (x * jnp.polyval(Q, x))
                     )


def ellipe(m):
    return ellpe(1. - m)


def test_ellpe():
    from scipy.special import ellipe as ellipe_scipy
    test_x = jnp.linspace(0.1, 1.)
    assert jnp.all(jnp.isclose(ellipe_scipy(test_x), vmap(ellipe)(test_x)))


def ellpk(x):
    """
    Complete elliptic integral of the first kind,
        int_0^pi/2 1/sqrt(1 - x * sin(t)^2) dt
    Args:
        x:

    Returns: same type as x

    """
    machep = jnp.finfo(x.dtype).eps

    P = jnp.array([1.37982864606273237150E-4, 2.28025724005875567385E-3,
                   7.97404013220415179367E-3, 9.85821379021226008714E-3,
                   6.87489687449949877925E-3, 6.18901033637687613229E-3,
                   8.79078273952743772254E-3, 1.49380448916805252718E-2,
                   3.08851465246711995998E-2, 9.65735902811690126535E-2,
                   1.38629436111989062502E0])

    Q = jnp.array([2.94078955048598507511E-5, 9.14184723865917226571E-4,
                   5.94058303753167793257E-3, 1.54850516649762399335E-2,
                   2.39089602715924892727E-2, 3.01204715227604046988E-2,
                   3.73774314173823228969E-2, 4.88280347570998239232E-2,
                   7.03124996963957469739E-2, 1.24999999999870820058E-1,
                   4.99999999999999999821E-1])
    C1 = jnp.log(4.)
    return jnp.where(x > machep, jnp.polyval(P, x) - jnp.log(x) * jnp.polyval(Q, x),
                     jnp.where(x == 0., jnp.inf, C1 - 0.5 * jnp.log(x)))


def ellipk(m):
    return ellpk(1. - m)


def test_ellpk():
    from scipy.special import ellipk as ellipk_scipy
    test_x = jnp.linspace(0.1, 1.)
    assert jnp.all(jnp.isclose(ellipk_scipy(test_x), vmap(ellipk)(test_x)))


def ellie(phi, m):
    """
    Calculates E(phi, m) = int_0^phi sqrt(1 - m * sin(t)^2) dt
    using the arithmetic geometric mean approx as in Cephes.

    Args:
        phi:
        m:

    Returns: same type as phi

    """

    pio2 = jnp.pi / 2.
    machep = jnp.finfo(phi.dtype).eps

    if m == 0.:
        return phi

    lphi = phi
    npio2 = jnp.floor(lphi / pio2)
    npio2 = jnp.where(npio2 == 1., npio2 + 1., npio2)
    lphi = lphi - npio2 * pio2
    sign = jnp.where(lphi < 0.0, -1., 1.)
    lphi = jnp.where(lphi < 0.0, -lphi, lphi)
    a = 1. - m
    E = ellpe(a)
    if a == 0.0:
        temp = jnp.sin(lphi)
        temp = sign * temp
        temp += npio2 * E
        return temp

    t = jnp.tan(lphi)
    b = jnp.sqrt(a)

    if jnp.fabs(t) > 10.0:
        e = 1.0 / (b * t)
        if jnp.fabs(e) < 10.0:
            e = jnp.arctan(e)
            temp = E + m * jnp.sin(lphi) * jnp.sin(e) - ellie(e, m)
            temp = sign * temp
            temp += npio2 * E
            return temp

    c = jnp.sqrt(m)
    a = 1.0
    d = 1
    e = 0.0
    mod = 0

    while (jnp.fabs(c / a) > machep):
        temp = b / a
        lphi = lphi + jnp.arctan(t * temp) + mod * jnp.pi
        mod = (lphi + pio2) / jnp.pi
        t = t * (1.0 + temp) / (1.0 - temp * t * t)
        c = (a - b) / 2.0
        temp = jnp.sqrt(a * b)
        a = (a + b) / 2.0
        b = temp
        d += d
        e += c * jnp.sin(lphi)

    temp = E / ellpk(1.0 - m)
    temp *= (jnp.arctan(t) + mod * jnp.pi) / (d * a)
    temp += e

    temp = sign * temp
    temp += npio2 * E
    return temp


def ellipeinc(phi, m):
    return ellie(phi, m)


def test_ellipeinc():
    from scipy.special import ellipeinc as ellipeinc_scipy
    for i in range(1):
        phi = random.uniform(random.PRNGKey(i), (), minval=0, maxval=jnp.pi * 2)
        m = random.uniform(random.PRNGKey(i + 100), (), minval=0., maxval=1.)
        print(ellipeinc_scipy(phi, m))
        print(ellipeinc(phi, m))


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
    n = u.size**2
    N = x1.shape[0]
    M = x2.shape[0]
    t1 = random.uniform(key, shape=(n, N, 1))
    t2 = random.uniform(key, shape=(n, M, 1))
    #L, N, M
    norm_squared = vmap(squared_norm) (x1 + t1 * p1, x2 + t2 * p2)
    bins = jnp.concatenate([u, u[-1:] + u[-1] - u[-2]])
    #N*M, U
    hist = vmap(lambda x: jnp.histogram(x, bins)[0])(jnp.reshape(norm_squared, (n, -1)).T)
    #N,M,U
    hist = jnp.reshape(hist, (x1.shape[0], x2.shape[0], u.size))
    log_hist = jnp.log(hist)
    log_du = jnp.diff(bins)
    log_w = log_hist + log_du
    #N,M,U
    log_w = log_w - logsumexp(log_w, axis=-1, keepdims=True)
    log_w = jnp.where(hist == 0., -jnp.inf, log_w)
    return log_w

def tomographic_weight_function_stochastic(key, u, x1, p1, x2, p2):
    return jnp.exp(log_tomographic_weight_function_stochastic(key, u, x1, p1, x2, p2))


def test_tomographic_weight_function_stochastic():
    import pylab as plt
    u = jnp.linspace(0., 4., 200)**2
    x1 = random.normal(random.PRNGKey(0), shape=(1,2,))
    p1 = random.normal(random.PRNGKey(1), shape=(1,2,))
    x2 = random.normal(random.PRNGKey(2), shape=(1,2,))
    p2 = random.normal(random.PRNGKey(3), shape=(1,2,))
    w = tomographic_weight_function_stochastic(random.PRNGKey(4),
                                               u,
                                               x1, p1, x2, p2)
    plt.plot(u, w[0,0,:])
    plt.show()


def squared_norm(x1, x2):
    # r2_ij = sum_k (x_ik - x_jk)^2
    #       = sum_k x_ik^2 - 2 x_jk x_ik + x_jk^2
    #       = sum_k x_ik^2 + x_jk^2 - 2 X X^T
    # r2_ij = sum_k (x_ik - y_jk)^2
    #       = sum_k x_ik^2 - 2 y_jk x_ik + y_jk^2
    #       = sum_k x_ik^2 + y_jk^2 - 2 X Y^T
    x1 = x1
    x2 = x2
    r2 = jnp.sum(jnp.square(x1), axis=1)[:, None] + jnp.sum(jnp.square(x2), axis=1)[None, :]
    r2 = r2 - 2. * (x1 @ x2.T)
    return jnp.maximum(r2, 1e-36)

def test_squared_norm():
    x = jnp.linspace(0., 1., 100)[:, None]
    y = jnp.linspace(1., 2., 50)[:, None]
    assert jnp.all(jnp.isclose(squared_norm(x, x), jnp.sum(jnp.square(x[:, None, :] - x[None, :, :]), axis=-1)))
    assert jnp.all(jnp.isclose(squared_norm(x, y), jnp.sum(jnp.square(x[:, None, :] - y[None, :, :]), axis=-1)))


def msqrt(A):
    U, s, Vh = jnp.linalg.svd(A)
    L = U * s
    return L


def make_coord_array(*X, flat=True, coord_map=None):
    """
    Create the design matrix from a list of coordinates
    :param X: list of length p of float, array [Ni, D]
        Ni can be different for each coordinate array, but D must be the same.
    :param flat: bool
        Whether to return a flattened representation
    :param coord_map: callable(coordinates), optional
            If not None then get mapped over the coordinates
    :return: float, array [N0,...,Np, D] if flat=False else [N0*...*Np, D]
        The coordinate design matrix
    """

    if coord_map is not None:
        X = [coord_map(x) for x in X]

    def add_dims(x, where, sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return jnp.tile(jnp.reshape(x, shape), tiles)

    N = [x.shape[0] for x in X]
    X_ = []

    for i, x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:, dim], [i], N))
    X = jnp.stack(X_, axis=-1)
    if not flat:
        return X
    return jnp.reshape(X, (-1, X.shape[-1]))

def tomographic_weight_function_outer(gamma, x1, x2, p1, p2, S):
    def inner(x1,p1):
        #M
        return vmap(lambda x2,p2: tomographic_weight_function(gamma, x1,x2,p1,p2,S))(x2,p2)
    return vmap(inner)(x1,p1) #N, M

def log_tomographic_weight_function_outer(gamma, x1, x2, p1, p2, S):
    def inner(x1,p1):
        #M
        return vmap(lambda x2,p2: log_tomographic_weight_function(gamma, x1,x2,p1,p2,S))(x2,p2)
    return vmap(inner)(x1,p1) #N, M

def test_tomographic_weight_function_outer():
    keys = random.split(random.PRNGKey(0),6)
    x1 = random.normal(keys[0], shape=(5,2,))
    p1 = random.normal(keys[1], shape=(5, 2,))
    x2 = random.normal(keys[2], shape=(6,2,))
    p2 = random.normal(keys[3], shape=(6,2,))

    gamma = jnp.linspace(0., 20., 100)
    assert tomographic_weight_function_outer(gamma,x1,x2,p1,p2,S=15).shape == (x1.shape[0],x2.shape[0],gamma.size-1)

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
    E = -2.* x12 @ p2
    F = x12 @ x12 - gamma

    t1 = jnp.linspace(0., 1., S)[:, None]
    H = (D**2 - 4.*A * F + 2. * B * D * t1 - 4. * A * E * t1 + B**2 * t1**2 - 4.*A * C * t1**2)
    u = (-D - B * t1)
    lower = jnp.clip(0.5 * (u - jnp.sqrt(H)) / A, 0., 1.)
    upper = jnp.clip(0.5 * (u + jnp.sqrt(H)) / A, 0., 1.)
    diff = (upper - lower)/(S-1)
    if not parabolic:
        reg_valid = H >= 0.
        cdf = jnp.sum(jnp.where(reg_valid,diff, 0.), axis=0)
    else:
        cdf = jnp.sum(diff, axis=0)
    return jnp.log(jnp.diff(cdf)) - jnp.log(jnp.diff(gamma))


def test_tomographic_weight():
    import pylab as plt
    from jax import jit

    @jit
    def tomo_weight(gamma, x1,x2,p1,p2):
        return tomographic_weight_function(gamma, x1, x2,p1,p2,S=15)

    @jit
    def tomo_weight_ref(gamma, x1, x2, p1, p2):
        return tomographic_weight_function(gamma, x1, x2, p1, p2, S=150)

    for i in range(10):
        keys = random.split(random.PRNGKey(i),6)
        x1 = jnp.concatenate([10. * random.uniform(keys[0], shape=( 2,)), jnp.zeros((1,))], axis=-1)
        p1 = jnp.concatenate(
            [4. * jnp.pi / 180. * random.uniform(keys[1], shape=(2,), minval=-1, maxval=1),
             jnp.ones((1,))], axis=-1)
        p1 = 4*p1 / jnp.linalg.norm(p1, axis=-1, keepdims=True)

        x2 = jnp.concatenate([4. * random.uniform(keys[2], shape=(2,)), jnp.zeros((1,))], axis=-1)
        p2 = jnp.concatenate(
            [4. * jnp.pi / 180. * random.uniform(keys[3], shape=(2,), minval=-1, maxval=1),
             jnp.ones((1,))], axis=-1)
        p2 = 4*p2 / jnp.linalg.norm(p2, axis=-1, keepdims=True)

        t1 = random.uniform(keys[4], shape=(10000,))
        t2 = random.uniform(keys[5], shape=(10000,))
        u1 = x1 + t1[:, None]*p1
        u2 = x2 + t2[:, None]*p2
        gamma = jnp.linalg.norm(u1-u2,axis=1)**2
        plt.hist(gamma.flatten(),bins=100, density=True, label='histogram')
        hist, bins = jnp.histogram(gamma.flatten(), density=True, bins=100)
        gamma = 0.5*(bins[:-1]+bins[1:])
        w = tomo_weight(bins, x1, x2, p1, p2)
        plt.plot(gamma, w, label='analytic')
        w_ref = tomo_weight_ref(bins, x1, x2, p1, p2)
        plt.plot(gamma, w_ref, label='analytic ref')
        plt.legend()
        plt.show()

def test_tomographic_weight_rel_err():
    import pylab as plt
    from jax import jit

    for S in range(5,30,5):
        @jit
        def tomo_weight(gamma, x1,x2,p1,p2):
            return tomographic_weight_function(gamma, x1, x2,p1,p2,S=S)

        @jit
        def tomo_weight_ref(gamma, x1, x2, p1, p2):
            return tomographic_weight_function(gamma, x1, x2, p1, p2, S=150)

        rel_error = []
        for i in range(400):
            keys = random.split(random.PRNGKey(i),6)
            x1 = jnp.concatenate([4. * random.uniform(keys[0], shape=( 2,)), jnp.zeros((1,))], axis=-1)
            p1 = jnp.concatenate(
                [4. * jnp.pi / 180. * random.uniform(keys[1], shape=(2,), minval=-1, maxval=1),
                 jnp.ones((1,))], axis=-1)
            p1 = 4*p1 / jnp.linalg.norm(p1, axis=-1, keepdims=True)

            x2 = jnp.concatenate([4. * random.uniform(keys[2], shape=(2,)), jnp.zeros((1,))], axis=-1)
            p2 = jnp.concatenate(
                [4. * jnp.pi / 180. * random.uniform(keys[3], shape=(2,), minval=-1, maxval=1),
                 jnp.ones((1,))], axis=-1)
            p2 = 4*p2 / jnp.linalg.norm(p2, axis=-1, keepdims=True)

            # x1 = random.normal(keys[0], shape=(2,))
            # p1 = random.normal(keys[1], shape=(2,))
            # x2 = random.normal(keys[2], shape=(2,))
            # p2 = random.normal(keys[3], shape=(2,))

            t1 = random.uniform(keys[4], shape=(10000,))
            t2 = random.uniform(keys[5], shape=(10000,))
            u1 = x1 + t1[:, None]*p1
            u2 = x2 + t2[:, None]*p2
            gamma = jnp.linalg.norm(u1-u2,axis=1)**2
            hist, bins = jnp.histogram(gamma.flatten(), density=True, bins=100)
            w = tomo_weight(bins, x1, x2, p1, p2)
            w_ref = tomo_weight_ref(bins, x1, x2, p1, p2)
            rel_error.append(jnp.max(jnp.abs(w - w_ref)) / jnp.max(w_ref))
        rel_error = jnp.array(rel_error)
        plt.hist(rel_error,bins ='auto')
        plt.title("{} : {:.2f}|{:.2f}|{:.2f}".format(S, *jnp.percentile(rel_error, [5,50,95])))
        plt.show()
