import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import vmap, random, jit
from jax.lax import while_loop
from functools import partial


def boxes_intersect_volume(a1, a2, b1, b2):
    return jnp.prod(jnp.maximum(jnp.minimum(a2, b2) - jnp.maximum(a1, b1), 0.))


def boxes_intersect(a1, a2, b1, b2):
    return jnp.all(jnp.minimum(a2, b2) - jnp.maximum(a1, b1) > 0.)


def log_cubes_intersect_volume(x, y, width, unit_restriction=False):
    if unit_restriction:
        l = width / 2.
        return jnp.sum(
            jnp.log(jnp.maximum(jnp.minimum(jnp.minimum(x, y) + l, 1.) - jnp.maximum(jnp.maximum(x, y) - l, 0.), 0.)))
    else:
        return jnp.sum(jnp.log(jnp.maximum(jnp.minimum(x, y) - jnp.maximum(x, y) + width, 0.)))


def cubes_intersect_volume(x, y, width, unit_restriction=False):
    return jnp.exp(log_cubes_intersect_volume(x, y, width, unit_restriction=unit_restriction))


def cubes_intersect(x, y, width):
    return jnp.all(jnp.minimum(x, y) - jnp.maximum(x, y) + width > 0.)


def test_cubes_intersect_volume():
    from jax import random
    import pylab as plt
    x = jnp.array([0., 0.])
    y = jnp.array([1., 1.])
    l = 1.
    assert cubes_intersect_volume(x, y, 2 * l) == boxes_intersect_volume(x - l, x + l, y - l, y + l)

    x = jnp.array([0., 0.])
    y = jnp.array([1., 1.])
    l = 1.5
    assert cubes_intersect_volume(x, y, 2 * l) == boxes_intersect_volume(x - l, x + l, y - l, y + l)

    def stoch_volume(x, y, l):
        X = random.uniform(random.PRNGKey(0), shape=(1000, 2), minval=x[None, :] - l, maxval=x[None, :] + l)
        vol = ((2. * l) ** x.size) * jnp.mean(vmap(lambda x: point_in_cube(x, y, l))(X))
        return vol

    for i in range(5):
        U = random.uniform(random.PRNGKey(i), shape=(2, 2))
        x = U[0, :]
        y = U[1, :]
        l = 0.25
        assert jnp.abs(cubes_intersect_volume(x, y, 2 * l) - stoch_volume(x, y, l)) < 0.01
        assert cubes_intersect_volume(x, x, 2 * l) == (2. * l) ** 2
        # plt.plot([x[ 0] - l,
        #           x[ 0] + l,
        #           x[ 0] + l,
        #           x[ 0] - l,
        #           x[ 0] - l],
        #          [x[ 1] - l,
        #           x[ 1] - l,
        #           x[ 1] + l,
        #           x[ 1] + l,
        #           x[ 1] - l],
        #          c='black'
        #          )
        # plt.plot([y[0] - l,
        #           y[0] + l,
        #           y[0] + l,
        #           y[0] - l,
        #           y[0] - l],
        #          [y[1] - l,
        #           y[1] - l,
        #           y[1] + l,
        #           y[1] + l,
        #           y[1] - l],
        #          c='black'
        #          )
        # plt.show()


def test_boxes_intersect_volume():
    a1 = jnp.array([0., 0.])
    a2 = a1 + 1.
    b1 = jnp.array([1., 1.])
    b2 = b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 0.
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)

    a1 = jnp.array([0., 0.])
    a2 = a1 + 2.
    b1 = jnp.array([1., 1.])
    b2 = b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)

    a1 = jnp.array([0., 0.])
    a2 = a1 + 1.
    b1 = jnp.array([0.5, 0.5])
    b2 = b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 0.25
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)

    a1 = jnp.array([0., 0.])
    a2 = a1 + 1.
    b1 = jnp.array([0.5, -0.5])
    b2 = b1 + 1.
    assert boxes_intersect_volume(a1, a2, b1, b2) == 0.25
    assert boxes_intersect_volume(a1, a2, b1, b2) == boxes_intersect_volume(b1, b2, a1, a2)


def points_in_box(x, y_lower, y_upper):
    return jnp.all((x <= y_upper) & (x >= y_lower))


def point_in_cube(x, y, half_width):
    return points_in_box(x, y - half_width, y + half_width)


def log_all_cubes_intersect_volume(points, width):
    log_I = vmap(
        lambda x: vmap(
            lambda y: log_cubes_intersect_volume(x, y, width))(points))(points)
    return logsumexp(log_I)


def log_all_cubes_unique_intersect_volume(points, width):
    log_I = vmap(
        lambda x: vmap(
            lambda y: log_cubes_intersect_volume(x, y, width))(points))(points)
    return logsumexp(jnp.where(jnp.tri(points.shape[0], k=-1), log_I, -jnp.inf))


def all_cubes_intersect_volume(points, width):
    return jnp.exp(log_all_cubes_intersect_volume(points, width))


def all_cubes_unique_intersect_volume(points, width):
    return jnp.exp(log_all_cubes_unique_intersect_volume(points, width))

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

def log_klee_measure(key, points, w, eps=0.01, gamma=0.9, restricted=False):
    """
    Compute approximation to Vol(Union(Cubes))

    From https://arxiv.org/pdf/0809.0835.pdf

    Result U is such that Prob((1-eps) Vol(Union(Cubes)) < U < (1+eps) Vol(Union(Cubes))) <= gamma
    or
    Prob(log(1-eps) + log(Vol(Union(Cubes))) < log(U) < log(1+eps) + log(Vol(Union(Cubes)))) <= gamma

    Args:
        points: [N, D]
        w: float

    Returns:

    """
    alpha = (jnp.log(1. - gamma) - jnp.log(2.)) / (-3. * jnp.log(2.))
    # print(alpha)
    # alpha = 1.
    N, D = points.shape

    points_lower = points - 0.5 * w
    points_upper = points + 0.5 * w
    if restricted:
        points_lower = jnp.maximum(points_lower, 0.)
        points_upper = jnp.minimum(points_upper, 1.)

    log_T = jnp.log(alpha) + jnp.log(24.) + jnp.log(jnp.log(2)) + jnp.log(1. + eps) + jnp.log(N) - 2. * jnp.log(eps)
    # volume query
    log_Vp_i = jnp.sum(jnp.log(points_upper - points_lower), axis=1)
    log_Vp = logsumexp(log_Vp_i)

    def body(state):
        (key, _done, log_t_cum, log_M) = state
        key, choose_key, sample_key = random.split(key, 3)
        i = random.categorical(choose_key, logits=log_Vp_i - log_Vp)
        # sample query
        x = random.uniform(sample_key, shape=(D,), minval=points_lower[i, :], maxval=points_upper[i, :])

        def inner_body(inner_state):
            (key, _done, _completed, log_t_M,) = inner_state
            completed = jnp.logaddexp(log_t_cum, log_t_M) >= log_T
            # if not completed then increment t_M and test if point in cube
            log_t_M = jnp.where(completed, log_t_M, jnp.logaddexp(log_t_M, 0.))
            key, inner_choose_key = random.split(key, 2)
            j = random.randint(inner_choose_key, shape=(), minval=0, maxval=N + 1)
            # point query
            in_j = points_in_box(x, points_lower[j, :], points_upper[j, :])
            done = in_j | completed
            return key, done, completed, log_t_M

        (key, _done, completed, log_t_M) = while_loop(lambda inner_state: ~inner_state[1],
                                                      inner_body,
                                                      (key, log_t_cum >= log_T, log_t_cum >= log_T, -jnp.inf))
        done = completed
        log_t_cum = jnp.logaddexp(log_t_cum, log_t_M)
        return (key, done, log_t_cum, jnp.logaddexp(log_M, 0.))

    (key, _done, log_t_cum, log_M) = while_loop(lambda state: ~state[1],
                                                body,
                                                (key, jnp.array(False), -jnp.inf, -jnp.inf))
    return log_T + log_Vp - jnp.log(N) - log_M


def test_klee_measure():
    from jax import random, jit, disable_jit
    import pylab as plt
    N, D = 2, 2
    points = random.uniform(random.PRNGKey(1), shape=(N, D))
    # points = jnp.array([[0., 1.],[0., 0.]])
    eps = 0.1
    gamma = 0.90
    for w in jnp.linspace(0., 1., 10):
        true_volume = 2. * w ** 2 - cubes_intersect_volume(points[0, :], points[1, :], w)
        vol = jnp.exp(jit(vmap(lambda key: log_klee_measure(key, points, w, eps=eps, gamma=gamma)))(
            random.split(random.PRNGKey(0), 100)))
        print(jnp.mean(vol), jnp.std(vol), true_volume)
        eps_bound = jnp.mean((vol <= true_volume * (1. + eps)) & (vol >= true_volume * (1. - eps)))
        l = w / 2.
        plt.scatter(points[:, 0], points[:, 1])
        for i in range(N):
            plt.plot([points[i, 0] - l,
                      points[i, 0] + l,
                      points[i, 0] + l,
                      points[i, 0] - l,
                      points[i, 0] - l],
                     [points[i, 1] - l,
                      points[i, 1] - l,
                      points[i, 1] + l,
                      points[i, 1] + l,
                      points[i, 1] - l],
                     c='black'
                     )
        plt.title("prob_bound {}".format(eps_bound, true_volume))
        plt.show()


def test_all_cubes_unique_intersect_volume():
    from jax import random, disable_jit
    import pylab as plt
    N, D = 4, 2
    points = random.uniform(random.PRNGKey(0), shape=(N, D))
    w = 0.5
    vol = 0.
    for i in range(N):
        for j in range(i + 1, N):
            vol += cubes_intersect_volume(points[i, :], points[j, :], w)
    assert jnp.isclose(vol, all_cubes_unique_intersect_volume(points, w))


def test_all_cubes_intersect_volume():
    from jax import random, disable_jit
    import pylab as plt
    N, D = 4, 2
    points = random.uniform(random.PRNGKey(0), shape=(N, D))
    w = 0.5
    vol = 0.
    for i in range(N):
        for j in range(N):
            vol += cubes_intersect_volume(points[i, :], points[j, :], w)
    assert jnp.isclose(vol, all_cubes_intersect_volume(points, w))


def determine_log_cube_width(key, points, log_X, log_init_cube_width=None, tol=0.1, shrink_amount=0.5, grow_amount=1.5, log_vol_samples=1):
    N, D = points.shape

    def body(state):
        (key, done, log_cube_width,) = state
        cube_width = jnp.exp(log_cube_width)
        key, klee_key = random.split(key, 2)
        log_vol_union = jnp.mean(vmap(lambda key:
                                      log_klee_measure(key, points, cube_width, eps=0.1, gamma=0.9, restricted=True))(
            random.split(klee_key, log_vol_samples)))
        too_big = log_vol_union > log_X + jnp.log(1. + tol)
        too_small = log_vol_union < log_X + jnp.log(1. - tol)
        done = (~too_big) & (~too_small)
        log_cube_width = jnp.where(too_big, log_cube_width + jnp.log(shrink_amount), log_cube_width)
        log_cube_width = jnp.where(too_small, log_cube_width + jnp.log(grow_amount), log_cube_width)
        return (key, done, log_cube_width)

    log_ideal_cube_volume = log_X - jnp.log(N)
    log_ideal_cube_width = log_ideal_cube_volume / D
    if log_init_cube_width is None:
        log_init_cube_width = log_ideal_cube_width
    (_key, _done, log_cube_width) = while_loop(lambda state: ~state[1],
                                               body,
                                               (key, jnp.array(False), log_init_cube_width))
    return log_cube_width


def test_determine_log_cube_width():
    from jax import random, disable_jit, jit
    import pylab as plt
    N, D = 100, 5
    points = random.uniform(random.PRNGKey(0), shape=(N, D))
    # for w in jnp.linspace(0., 2., 20):
    #     plt.scatter(w, jnp.mean(jit(vmap(lambda key:
    #                                  log_klee_measure(key, points, w, 0.1, 0.9, True)))(
    #         random.split(random.PRNGKey(1), 1))))
    # plt.show()
    log_cube_width = determine_log_cube_width(random.PRNGKey(1), points, jnp.log(0.5))
    print(jnp.exp(log_cube_width))
    l = jnp.exp(log_cube_width)/ 2.
    points_lower = jnp.maximum(points - l, 0.)
    points_upper = jnp.minimum(points + l, 1.)
    plt.scatter(points[:, 0], points[:, 1])
    for i in range(N):
        plt.plot([points_lower[i, 0],
                  points_upper[i,0],
                  points_upper[i,0],
                  points_lower[i,0],
                  points_lower[i,0]
                  ],
                 [points_lower[i,1],
                  points_lower[i,1],
                  points_upper[i,1],
                  points_upper[i,1],
                  points_lower[i,1]],
                 c='black')
    plt.show()
