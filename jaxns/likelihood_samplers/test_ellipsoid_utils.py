from jax import numpy as jnp, random

from jaxns.likelihood_samplers.ellipsoid_utils import bounding_ellipsoid, ellipsoid_params, kmeans, \
    rank_one_update_matrix_inv, log_ellipsoid_volume, cluster_split, ellipsoid_clustering, sample_multi_ellipsoid, \
    minimum_volume_enclosing_ellipsoid, ellipsoid_to_circle, circle_to_ellipsoid, generic_kmeans, hierarchical_clustering


def test_binary_tree_bf_walk():
    """
    Test
    0:    0 | # 2^0
    1:    1 2 | # 2^1
    2:    3 4 5 6
    3:    7 8 9 10 11 12 13 14

    index(r,c) = index(r,0) + c
    index(r,0) = sum_p=0^(r-1) 2^p = 2^r - 1
    index(r,c) = 2^r - 1 + c

    index(r,c) = index(r-1, 2^(r-1) - 1) + c
    child0(r,c) = index(r+1, 2*c)
    child1(r,c) = index(r+1, 2*c+1)
    parent(r,c) = index(r-1, c//2)
    Returns:

    """

    def index(r, c):
        return 2 ** r - 1 + c

    def child0(r, c):
        return index(r + 1, 2 * c)

    def child1(r, c):
        return index(r + 1, 2 * c + 1)

    def parent(r, c):
        return index(r - 1, c // 2)

    def row(i):
        return jnp.ceil(jnp.log2(i + 1))

    depth = 3
    for r in range(depth):
        for c in range(2 ** r):
            print("({},{}), row={}, index={}, child0={}, child1={}, parent={}".format(r, c,
                                                                                      row(index(r, c)),
                                                                                      index(r, c),
                                                                                      child0(r, c),
                                                                                      child1(r, c),
                                                                                      parent(r, c)))


def test_bounding_ellipsoid():
    points = random.normal(random.PRNGKey(0), shape=(10, 2))
    mu, C = bounding_ellipsoid(points, jnp.ones(points.shape[0]))
    radii, rotation = ellipsoid_params(C)
    print(mu, C, radii, rotation)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = mu[:, None] + rotation @ jnp.diag(radii) @ jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    import pylab as plt
    plt.scatter(points[:, 0], points[:, 1])
    plt.plot(x[0, :], x[1, :])
    plt.show()


def test_kmeans():
    points = jnp.concatenate([random.normal(random.PRNGKey(0), shape=(30, 2)),
                              3. + random.normal(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)

    cluster_id, centers = kmeans(random.PRNGKey(0), points, jnp.ones(points.shape[0], dtype=jnp.bool_), K=2)

    mu, C = bounding_ellipsoid(points, jnp.ones(points.shape[0]))
    radii, rotation = ellipsoid_params(C)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = mu[:, None] + rotation @ jnp.diag(radii) @ jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    import pylab as plt
    mask = cluster_id == 0
    plt.scatter(points[mask, 0], points[mask, 1])
    mask = cluster_id == 1
    plt.scatter(points[mask, 0], points[mask, 1])
    plt.plot(x[0, :], x[1, :])
    plt.show()


def test_inverse_update():
    A = random.normal(random.PRNGKey(2), shape=(3, 3))
    A = A @ A.T
    u = random.normal(random.PRNGKey(7), shape=(3,))
    v = random.normal(random.PRNGKey(6), shape=(3,))
    B = u[:, None] * v
    Ainv = jnp.linalg.inv(A)
    detAinv = jnp.linalg.det(Ainv)
    # C1, detC1 = rank_one_update_matrix_inv(Ainv, detAinv, u, v, add=True)
    # print(jnp.linalg.det(jnp.linalg.inv(A + B)), detC1)
    # assert jnp.isclose(jnp.linalg.inv(A + B), C1).all() & jnp.isclose(jnp.linalg.det(jnp.linalg.inv(A + B)), detC1)
    # C1, detC1 = rank_one_update_matrix_inv(Ainv, detAinv, u, v, add=False)
    # print(jnp.linalg.det(jnp.linalg.inv(A - B)), detC1)
    # assert jnp.isclose(jnp.linalg.inv(A - B), C1).all() & jnp.isclose(jnp.linalg.det(jnp.linalg.inv(A - B)), detC1)

    C1, logdetC1 = rank_one_update_matrix_inv(Ainv, jnp.log(detAinv), u, v, add=True)
    # print(jnp.log(jnp.linalg.det(jnp.linalg.inv(A + B))), logdetC1)
    assert jnp.isclose(jnp.linalg.inv(A + B), C1).all()
    assert jnp.isclose(jnp.log(jnp.linalg.det(jnp.linalg.inv(A + B))), logdetC1)
    C2, logdetC2 = rank_one_update_matrix_inv(Ainv, jnp.log(detAinv), u, v, add=False)
    print(jnp.log(jnp.linalg.det(jnp.linalg.inv(A - B))), logdetC2)
    assert jnp.isclose(jnp.linalg.inv(A - B), C2).all()
    assert jnp.isclose(jnp.log(jnp.linalg.det(jnp.linalg.inv(A - B))), logdetC2)


def test_ellipsoid_params_update():
    import pylab as plt
    N = 4
    points = random.normal(random.PRNGKey(43532), shape=(N, 3,))
    mu = jnp.mean(points, axis=0)
    C = jnp.linalg.inv(jnp.sum((points - mu)[:, :, None] * (points - mu)[:, None, :], axis=0))
    detC = jnp.linalg.det(C)
    n = N
    for i in range(100):
        x_n = random.normal(random.PRNGKey(i), shape=(3,))
        mu_next = mu + (x_n - mu) / (n + 1)
        C_next, detC_next = rank_one_update_matrix_inv(C,
                                                       detC,
                                                       x_n - mu,
                                                       x_n - mu_next,
                                                       add=True)
        n += 1
        points = jnp.concatenate([points, x_n[None, :]], axis=0)
        mu_com = jnp.mean(points, axis=0)
        C_com = jnp.linalg.inv(jnp.sum((points - mu_com)[:, :, None] * (points - mu_com)[:, None, :], axis=0))
        detC_com = jnp.linalg.det(C_com)
        assert jnp.isclose(detC_com, detC_next)
        assert jnp.isclose(mu_next, mu_com).all()
        assert jnp.isclose(C_next, C_com).all()
        mu, C, detC = mu_next, C_next, detC_next
        print(detC_next)


def test_cluster_split():
    import pylab as plt
    from jax import disable_jit
    points = jnp.concatenate([random.uniform(random.PRNGKey(0), shape=(30, 2)),
                              1.25 + random.uniform(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    mask = jnp.zeros(points.shape[0], jnp.bool_)
    mu, C = bounding_ellipsoid(points, jnp.ones(points.shape[0], jnp.bool_))
    radii, rotation = ellipsoid_params(C)
    y = mu[:, None] + rotation @ jnp.diag(radii) @ x
    plt.plot(y[0, :], y[1, :])
    log_VS = log_ellipsoid_volume(radii) - jnp.log(5)
    with disable_jit():
        cluster_id, log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split = \
            cluster_split(random.PRNGKey(0), points, mask, log_VS, log_ellipsoid_volume(radii), kmeans_init=True)
        print(jnp.logaddexp(log_ellipsoid_volume(radii1), log_ellipsoid_volume(radii2)), log_ellipsoid_volume(radii))
        print(log_VS1, mu1, radii1, rotation1, log_VS2, mu2, radii2, rotation2, do_split)
        print(cluster_id)

    y = mu1[:, None] + rotation1 @ jnp.diag(radii1) @ x
    plt.plot(y[0, :], y[1, :])

    y = mu2[:, None] + rotation2 @ jnp.diag(radii2) @ x
    plt.plot(y[0, :], y[1, :])

    mask = cluster_id == 0
    plt.scatter(points[mask, 0], points[mask, 1])
    mask = cluster_id == 1
    plt.scatter(points[mask, 0], points[mask, 1])

    plt.show()


def test_accept_order():
    def constraint(u):
        return u ** 2 > 0.5

    def accept_prob(u):
        if u > 0.9:
            return 1
        if u > 0.8:
            return 0.5
        if u > 0.7:
            return 0.25
        return 0.

    def f1(key):
        while True:
            key, u_key = random.split(key, 2)
            u = random.uniform(u_key)
            if constraint(u):
                key, a_key = random.split(key, 2)
                if random.uniform(a_key) < accept_prob(u):
                    return u

    def f2(key):
        while True:
            key, u_key, a_key = random.split(key, 3)
            u = random.uniform(u_key)
            if random.uniform(a_key) < accept_prob(u):
                if constraint(u):
                    return u

    from jax import vmap, disable_jit
    import pylab as plt
    keys = random.split(random.PRNGKey(0), 1000)
    with disable_jit():
        u1 = jnp.array([f1(key) for key in keys])
        u2 = jnp.array([f2(key) for key in keys])

    print(u1)

    plt.hist(u1, bins='auto', alpha=0.5)
    plt.hist(u2, bins='auto', alpha=0.5)
    plt.show()


def test_sample_multi_ellipsoid():
    import pylab as plt
    from jax import disable_jit, jit, vmap
    points = jnp.concatenate([random.uniform(random.PRNGKey(0), shape=(30, 2)),
                              1.25 + random.uniform(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    mask = jnp.ones(points.shape[0], jnp.bool_)
    mu, C = bounding_ellipsoid(points, mask)
    radii, rotation = ellipsoid_params(C)
    y = mu[:, None] + rotation @ jnp.diag(radii) @ x
    # plt.plot(y[0, :], y[1, :])
    log_VS = log_ellipsoid_volume(radii) - jnp.log(5)

    with disable_jit():
        cluster_id, ellipsoid_parameters = \
            jit(lambda key, points, log_VS: ellipsoid_clustering(random.PRNGKey(0), points, 4, log_VS)
                )(random.PRNGKey(0), points, log_VS)

        mu, radii, rotation = ellipsoid_parameters
        # print(mu, radii, rotation)
        u = vmap(lambda key: sample_multi_ellipsoid(key, mu, radii, rotation, unit_cube_constraint=True)[1])(random.split(random.PRNGKey(0),1000))
    plt.scatter(u[:, 0], u[:, 1], marker='+')
    for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
        y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        plt.plot(y[0, :], y[1, :])
        mask = cluster_id == i
        # plt.scatter(points[mask, 0], points[mask, 1], c=plt.cm.jet(i / len(ellipsoid_parameters)))
    plt.show()


def test_ellipsoid_clustering():
    import pylab as plt
    from jax import disable_jit, jit
    points = jnp.concatenate([random.uniform(random.PRNGKey(0), shape=(30, 2)),
                              1.25 + random.uniform(random.PRNGKey(0), shape=(10, 2))],
                             axis=0)
    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)
    mask = jnp.ones(points.shape[0], jnp.bool_)
    mu, C = bounding_ellipsoid(points, mask)
    radii, rotation = ellipsoid_params(C)
    # plt.plot(y[0, :], y[1, :])
    log_VS = log_ellipsoid_volume(radii) - jnp.log(5)

    with disable_jit():
        cluster_id, ellipsoid_parameters = \
            jit(lambda key, points, log_VS: ellipsoid_clustering(random.PRNGKey(0), points, 4, log_VS)
                )(random.PRNGKey(0), points, log_VS)
        mu, radii, rotation = ellipsoid_parameters
        print(mu, radii, rotation, jnp.bincount(cluster_id, minlength=0, length=4))

    for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
        y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        plt.plot(y[0, :], y[1, :])
        mask = cluster_id == i
        plt.scatter(points[mask, 0], points[mask, 1], c=plt.cm.jet(i / len(ellipsoid_parameters)))

    plt.show()


def test_log_ellipsoid_volume():
    radii = jnp.ones(2)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(jnp.pi))
    radii = jnp.ones(3)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(4. * jnp.pi / 3.))


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

    circ_points = ellipsoid_to_circle(points[:-1, :], center, radii, rotation)
    # plt.scatter(circ_points[:,0], circ_points[:,1])
    plt.scatter(points[:, 0], points[:, 1])
    # plt.show()
    print(jnp.max(jnp.linalg.norm(circ_points, axis=-1) ** 2 - 1.))
    # assert jnp.all(jnp.linalg.norm(circ_points, axis=-1) <= 1.03)

    theta = jnp.linspace(0., jnp.pi * 2, 100)
    circle = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    ellipse = circle_to_ellipsoid(circle, center, radii, rotation)
    plt.plot(ellipse[:, 0], ellipse[:, 1])
    plt.show()

def test_generic_kmeans():
    from jaxns.prior_transforms import PriorChain, UniformPrior
    from jax import vmap, disable_jit, jit
    import pylab as plt

    data = 'shells'
    if data == 'eggbox':
        def log_likelihood(theta, **kwargs):
            return (2. + jnp.prod(jnp.cos(0.5 * theta))) ** 5

        prior_chain = PriorChain() \
            .push(UniformPrior('theta', low=jnp.zeros(2), high=jnp.pi * 10. * jnp.ones(2)))

        U = vmap(lambda key: random.uniform(key, (prior_chain.U_ndims,)))(random.split(random.PRNGKey(0), 1000))
        theta = vmap(lambda u: prior_chain(u))(U)
        lik = vmap(lambda theta: log_likelihood(**theta))(theta)
        select = lik > 100.

    if data == 'shells':

        def log_likelihood(theta, **kwargs):
            def log_circ(theta, c, r, w):
                return -0.5*(jnp.linalg.norm(theta - c) - r)**2/w**2 - jnp.log(jnp.sqrt(2*jnp.pi*w**2))
            w1=w2=jnp.array(0.1)
            r1=r2=jnp.array(2.)
            c1 = jnp.array([0., -4.])
            c2 = jnp.array([0., 4.])
            return jnp.logaddexp(log_circ(theta, c1,r1,w1) , log_circ(theta,c2,r2,w2))


        prior_chain = PriorChain() \
            .push(UniformPrior('theta', low=-12.*jnp.ones(2), high=12.*jnp.ones(2)))

        U = vmap(lambda key: random.uniform(key, (prior_chain.U_ndims,)))(random.split(random.PRNGKey(0), 40000))
        theta = vmap(lambda u: prior_chain(u))(U)
        lik = vmap(lambda theta: log_likelihood(**theta))(theta)
        select = lik > 1.

    print("Selecting", jnp.sum(select))
    log_VS = jnp.log(jnp.sum(select)/select.size)
    print("V(S)",jnp.exp(log_VS))

    points = U[select, :]
    sc = plt.scatter(U[:,0], U[:,1],c=jnp.exp(lik))
    plt.colorbar(sc)
    plt.show()
    mask = jnp.ones(points.shape[0], dtype=jnp.bool_)
    K = 18
    with disable_jit():
        # state = generic_kmeans(random.PRNGKey(0), points, mask, method='ellipsoid',K=K,meta=dict(log_VS=log_VS))
        # state = generic_kmeans(random.PRNGKey(0), points, mask, method='mahalanobis',K=K)
        # state = generic_kmeans(random.PRNGKey(0), points, mask, method='euclidean',K=K)
        # cluster_id, log_cluster_VS = hierarchical_clustering(random.PRNGKey(0), points, 7, log_VS)
        cluster_id, ellipsoid_parameters = \
            jit(lambda key, points, log_VS: ellipsoid_clustering(random.PRNGKey(0), points, 7, log_VS)
                )(random.PRNGKey(0), points, log_VS)
        # mu, radii, rotation = ellipsoid_parameters
        K = int(jnp.max(cluster_id)+1)

    mu, C = vmap(lambda k: bounding_ellipsoid(points, cluster_id == k))(jnp.arange(K))
    radii, rotation = vmap(ellipsoid_params)(C)

    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)

    for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
        y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        plt.plot(y[0, :], y[1, :], c=plt.cm.jet(i / K))
        mask = cluster_id == i
        plt.scatter(points[mask, 0], points[mask, 1], c=jnp.atleast_2d(plt.cm.jet(i / K)))
    plt.xlim(-1,2)
    plt.ylim(-1,2)
    plt.show()