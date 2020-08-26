from jax import numpy as jnp, vmap, random
from jax.lax import while_loop

def minimum_volume_enclosing_ellipsoid(points, tol, init_u=None, return_u=False):
    """
    Performs the algorithm of
    MINIMUM VOLUME ENCLOSING ELLIPSOIDS
    NIMA MOSHTAGH
    psuedo-code here:
    https://stackoverflow.com/questions/1768197/bounding-ellipse

    Args:
        points: [N, D]
    """
    N, D = points.shape
    Q = jnp.concatenate([points, jnp.ones([N, 1])], axis=1)  # N,D+1

    def body(state):
        (count, err, u) = state
        V = Q.T @ jnp.diag(u) @ Q#D+1, D+1
        # g[i] = Q[i,j].V^-1_jk.Q[i,k]
        g = vmap(lambda q: q @ jnp.linalg.solve(V, q))(Q)#difference
        # jnp.diag(Q @ jnp.linalg.solve(V, Q.T))
        j = jnp.argmax(g)
        g_max = g[j]

        step_size = \
            (g_max - D - 1) / ((D + 1) * (g_max - 1))
        search_direction = jnp.where(jnp.arange(N) == j, 1. - u, -u)
        new_u = u + step_size * search_direction
        # new_u = (1. - step_size)*u
        new_u = jnp.where(jnp.arange(N) == j, u + step_size*(1. - u), u * (1.-step_size))
        new_err = jnp.linalg.norm(u - new_u)
        return (count + 1, new_err, new_u)

    if init_u is None:
        init_u = jnp.ones(N) / N
    (count, err, u) = while_loop(lambda state: state[1] > tol * jnp.linalg.norm(init_u),
                                 body,
                                 (0, jnp.inf, init_u))
    U = jnp.diag(u)
    PU = (points.T @ u)  # D, N
    A = jnp.reciprocal(D) * jnp.linalg.pinv(points.T @ U @ points - PU[:, None] @ PU[None, :])
    c = PU
    W, Q, Vh = jnp.linalg.svd(A)
    radii = jnp.reciprocal(jnp.sqrt(Q))
    rotation = Vh.conj().T
    if return_u:
        return c, radii, rotation, u
    return c, radii, rotation


def sample_ellipsoid(key, center, radii, rotation):
    """
    Sample uniformly inside an ellipsoid.
    Args:
        key:
        center: [D]
        radii: [D]
        rotation: [D,D]

    Returns: [D]

    """
    direction_key, radii_key = random.split(key, 2)
    direction = random.normal(direction_key, shape=radii.shape)
    log_norm = jnp.log(jnp.linalg.norm(direction))
    log_radius = jnp.log(random.uniform(radii_key)) / radii.size
    # x = direction * (radius/norm)
    x = direction * jnp.exp(log_radius - log_norm)
    return circle_to_ellipsoid(x, center, radii, rotation)


def ellipsoid_to_circle(points, center, radii, rotation):
    """
    Scale and rotate an ellipse to a circle.
    Args:
        points: [N, D] or [D]
        center: [D]
        radii: [D]
        rotation: [D,D]

    Returns: [N, D] or [D]

    """
    if len(points.shape) == 1:
        return (rotation.T / radii[:, None]) @ (points - center)
    else:
        return vmap(lambda point: ellipsoid_to_circle(point, center, radii, rotation))(points)


def circle_to_ellipsoid(points, center, radii, rotation):
    """
    Scale and rotate an ellipse to a circle.
    Args:
        points: [N, D] or [D]
        center: [D]
        radii: [D]
        rotation: [D,D]

    Returns: [N, D] or [D]

    """
    if len(points.shape) == 1:
        return (rotation * radii[None, :]) @ points + center
    else:
        return vmap(lambda point: circle_to_ellipsoid(point, center, radii, rotation))(points)


def test_mvee():
    import pylab as plt
    from jax import disable_jit
    ndims = 2
    N = 10 * ndims
    points = random.normal(random.PRNGKey(0), shape=(N, ndims))
    center, radii, rotation, next_u = minimum_volume_enclosing_ellipsoid(points, 0.01, return_u=True)
    points = random.normal(random.PRNGKey(1), shape=(N, ndims))
    with disable_jit():
        center, radii, rotation = minimum_volume_enclosing_ellipsoid(points, 0.01, init_u=next_u)

    circ_points = ellipsoid_to_circle(points[:-1,:], center, radii, rotation)
    # plt.scatter(circ_points[:,0], circ_points[:,1])
    plt.scatter(points[:,0], points[:,1])
    # plt.show()
    print(jnp.max(jnp.linalg.norm(circ_points, axis=-1)**2 - 1.))
    # assert jnp.all(jnp.linalg.norm(circ_points, axis=-1) <= 1.03)

    theta = jnp.linspace(0., jnp.pi*2,100)
    circle = jnp.stack([jnp.cos(theta), jnp.sin(theta)],axis=1)
    ellipse = circle_to_ellipsoid(circle, center, radii, rotation)
    plt.plot(ellipse[:,0], ellipse[:,1])
    plt.show()