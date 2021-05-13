from jax import numpy as jnp, random, jit

from jaxns.likelihood_samplers.ellipsoid_utils import rank_one_update_matrix_inv, log_ellipsoid_volume
# from jaxns.utils import signed_logaddexp
# from jaxns.likelihood_samplers.klee_utils import union_volume_approximation_pistons, sphere_log_volume, piston_volume, cylinder_log_volume


def test_inverse_update():
    A = random.normal(random.PRNGKey(2), shape=(3, 3))
    A = A @ A.T
    u = random.normal(random.PRNGKey(7), shape=(3,))
    v = random.normal(random.PRNGKey(6), shape=(3,))
    B = u[:, None] * v
    Ainv = jnp.linalg.inv(A)
    detAinv = jnp.linalg.det(Ainv)
    C1, logdetC1 = rank_one_update_matrix_inv(Ainv, jnp.log(detAinv), u, v, add=True)
    assert jnp.isclose(jnp.linalg.inv(A + B), C1).all()
    assert jnp.isclose(jnp.log(jnp.linalg.det(jnp.linalg.inv(A + B))), logdetC1)
    C2, logdetC2 = rank_one_update_matrix_inv(Ainv, jnp.log(detAinv), u, v, add=False)
    print(jnp.log(jnp.linalg.det(jnp.linalg.inv(A - B))), logdetC2)
    assert jnp.isclose(jnp.linalg.inv(A - B), C2).all()
    assert jnp.isclose(jnp.log(jnp.linalg.det(jnp.linalg.inv(A - B))), logdetC2)


def test_log_ellipsoid_volume():
    radii = jnp.ones(2)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(jnp.pi))
    radii = jnp.ones(3)
    assert jnp.isclose(log_ellipsoid_volume(radii), jnp.log(4. * jnp.pi / 3.))

# def test_union_volume_approximation_pistons():
#     """
#     O
#     ||
#     O
#     ||
#     O
#     Returns:
#
#     """
#
#     f_unconstrained = jit(
#         lambda radius, y, z: union_volume_approximation_pistons(random.PRNGKey(3453252), y, z, jnp.log(radius), 100,
#                                                                 constraint_unit_cube=False))
#     f_constrained = jit(
#         lambda radius, y, z: union_volume_approximation_pistons(random.PRNGKey(3453252), y, z, jnp.log(radius), 100,
#                                                                 constraint_unit_cube=True))
#
#     for n in [10, 100]:
#         for d in [2, 10, 100]:
#
#             radius = 1 / (3 * n)
#             points = jnp.stack(
#                 [jnp.zeros(n), jnp.arange(n) * (2 * radius) + radius] + [0.5 * jnp.ones(n)] * (d - 2), axis=1)
#             y = points[:-1, :]
#             z = points[1:, :]
#             if d == 2:
#                 pass
#                 # samples = vmap(lambda key: sample_union_pistons(key,y, z, jnp.log(radius)))(random.split(random.PRNGKey(3245), 1000))
#                 # depth = vmap(lambda point: compute_depth_pistons(point,y, z, jnp.log(radius), constraint_unit_cube=True))(samples)
#                 #
#                 # sc=plt.scatter(samples[:, 0], samples[:, 1], c=depth, marker='.')
#                 # plt.colorbar(sc)
#                 # for _x, _y in points:
#                 #     rect = patches.Circle([_x, _y], radius=radius, linewidth=1, edgecolor='r', facecolor='none')
#                 #     plt.gca().add_patch(rect)
#                 # plt.show()
#
#             # (n-1) exp(log(piston volume)) - (n-2) exp(log(ball volume))
#             log_piston_V = piston_volume(y[0], z[0], jnp.log(radius))
#             log_ball_V = sphere_log_volume(d, jnp.log(radius))
#             true_log_V, _ = signed_logaddexp(jnp.log(n - 1) + log_piston_V, 1., jnp.log(n - 2) + log_ball_V, -1.)
#             log_V = f_unconstrained(radius, y, z)
#             # print(log_V, true_log_V)
#             assert jnp.isclose(log_V, true_log_V, atol=1e-1)
#
#             true_log_V = jnp.log(0.5) + true_log_V
#             log_V = f_constrained(radius, y, z)
#             # print(log_V, true_log_V)
#             assert jnp.isclose(log_V, true_log_V, atol=1e-1)
#
# def test_piston_volume():
#     y = jnp.asarray([0., 0.])
#     z = jnp.asarray([0., 2.])
#     radius = 1.
#     ball_V = jnp.pi * radius ** 2
#     cylinder_V = 2 * radius * jnp.linalg.norm(y - z)
#     true_V = jnp.log(ball_V + cylinder_V)
#     assert jnp.isclose(true_V, piston_volume(y, z, jnp.log(radius)))
#
# def test_sphere_log_volume():
#     radius = 2.
#     assert jnp.isclose(jnp.log(jnp.pi * radius ** 2), sphere_log_volume(2, jnp.log(radius)))
#     assert jnp.isclose(jnp.log(4. / 3 * jnp.pi * radius ** 3), sphere_log_volume(3, jnp.log(radius)))
#
# def test_cylinder_log_volume():
#     y = jnp.asarray([0., 0.])
#     z = jnp.asarray([0., 2.])
#     radius = 1.
#     length = jnp.linalg.norm(y - z)
#     assert jnp.isclose(jnp.log(length * 2 * radius), cylinder_log_volume(2, y, z, jnp.log(radius)))