from jax import random, vmap, numpy as jnp
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

            # x1 = random.normal(keys[0], shape_dict=(2,))
            # p1 = random.normal(keys[1], shape_dict=(2,))
            # x2 = random.normal(keys[2], shape_dict=(2,))
            # p2 = random.normal(keys[3], shape_dict=(2,))

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


def log_tomographic_weight_function_exact(gamma, x1, x2, p1, p2=None, S=25):

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
    # a x^2 + (b y + d) x +  (c y^2 + e y + f) < 0
    y = jnp.linspace(0., 1., 100)
    A1 = A
    B1 = (B * y + D)
    C1 = C * y**2 + E * y + F
    h1 = 0.5 * B1 / A1
    h2 = (0.5 / A1) * jnp.sqrt(B1**2 - 4. * A1 * C1)
    import pylab as plt
    for r in jnp.linspace(0., 1., 20):
        mask = h1 + h2 > r
        print(r,mask)


def test_exact():
    import pylab as plt
    keys = random.split(random.PRNGKey(1), 6)
    x1 = jnp.concatenate([2. * random.uniform(keys[0], shape=(2,)), jnp.zeros((1,))], axis=-1)
    p1 = jnp.concatenate(
        [4. * jnp.pi / 180. * random.uniform(keys[1], shape=(2,), minval=-1, maxval=1),
         jnp.ones((1,))], axis=-1)
    p1 = 4 * p1 / jnp.linalg.norm(p1, axis=-1, keepdims=True)

    x2 = jnp.concatenate([4. * random.uniform(keys[2], shape=(2,)), jnp.zeros((1,))], axis=-1)
    p2 = jnp.concatenate(
        [4. * jnp.pi / 180. * random.uniform(keys[3], shape=(2,), minval=-1, maxval=1),
         jnp.ones((1,))], axis=-1)
    p2 = 4 * p2 / jnp.linalg.norm(p2, axis=-1, keepdims=True)

    t1 = random.uniform(keys[4], shape=(10000,))
    t2 = random.uniform(keys[5], shape=(10000,))
    u1 = x1 + t1[:, None] * p1
    u2 = x2 + t2[:, None] * p2
    gamma = jnp.linalg.norm(u1 - u2, axis=1) ** 2
    plt.hist(gamma.flatten(), bins=100, density=True, label='histogram')
    plt.show()
    hist, bins = jnp.histogram(gamma.flatten(), density=True, bins=100)
    gamma = 0.5 * (bins[:-1] + bins[1:])

    log_tomographic_weight_function_exact(gamma.mean(), x1, x2, p1, p2)