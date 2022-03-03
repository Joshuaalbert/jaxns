import numpy as np
from jax import random, numpy as jnp

from jaxns.internals.linalg import msqrt
from jaxns.internals.log_semiring import LogSpace, logaddexp, is_complex, signed_logaddexp, cumulative_logsumexp
from jaxns.internals.random import random_ortho_matrix, latin_hypercube
from jaxns.internals.shapes import tuple_prod, broadcast_dtypes, broadcast_shapes
from jaxns.internals.stats import density_estimation, linear_to_log_stats
from jaxns.internals.maps import prepare_func_args


def test_prepare_func_args():
    import inspect

    def f(a, b=1):
        return a + b

    g = prepare_func_args(f)
    kwargs = dict(a=1, b=2, c=3)
    assert g(**kwargs) == f(kwargs['a'], b=kwargs['b'])
    kwargs = dict(a=1, c=3)
    assert g(**kwargs) == f(kwargs['a'])

    def f(a, b=2, *, c, d=4):
        return a + b + c + d

    g = prepare_func_args(f)
    kwargs = dict(a=5, b=6, c=7, d=8)
    assert g(**kwargs) == f(kwargs['a'], b=kwargs['b'], c=kwargs['c'], d=kwargs['d'])
    kwargs = dict(a=9, c=11)
    assert g(**kwargs) == f(kwargs['a'], c=kwargs['c'])


def test_random_ortho_matrix():
    M = random_ortho_matrix(random.PRNGKey(42), 5)
    assert jnp.allclose(M.T @ M, M @ M.T)
    assert jnp.allclose(M.T @ M, jnp.eye(5))
    assert jnp.allclose(jnp.linalg.norm(M, axis=0), jnp.linalg.norm(M, axis=1))


def test_density_estimation():
    np.random.seed(42)
    x = jnp.asarray(np.random.standard_gamma(1., 100))[:, None]
    xstar = jnp.linspace(0., 20., 1000)[:, None]
    assert density_estimation(xstar, x).size == 1000
    #
    # import pylab as plt
    #
    # plt.plot(xstar, density_estimation(xstar, x))
    # plt.hist(np.random.standard_gamma(1., 10000), bins=np.linspace(0, 20, 100), density=True, alpha=0.5)
    # plt.hist(x[:, 0], bins=np.linspace(0., 20, 100), density=True, alpha=0.5)
    # plt.show()


def test_log_space():
    #
    a = 5.
    b = 10.

    _a = LogSpace(jnp.log(a))
    _b = LogSpace(jnp.log(b))

    assert jnp.isclose((_a + _b).value, a + b)
    assert jnp.isclose((_a - _b).value, a - b)
    assert jnp.isclose((_a * _b).value, a * b)
    assert jnp.isclose((_a / _b).value, a / b)
    assert jnp.isclose((_a ** 2).value, a ** 2)
    assert jnp.isclose((_a ** 3).value, a ** 3)
    assert jnp.isclose((_a ** 0.5).value, a ** 0.5)

    #
    a = 0.
    b = 10.

    _a = LogSpace(jnp.log(a))
    _b = LogSpace(jnp.log(b))

    assert jnp.isclose((_a + _b).value, a + b)
    assert jnp.isclose((_a - _b).value, a - b)
    assert jnp.isclose((_a * _b).value, a * b)
    assert jnp.isclose((_a / _b).value, a / b)
    assert jnp.isclose((_a ** 2).value, a ** 2)
    assert jnp.isclose((_a ** 3).value, a ** 3)

    a = -5.
    b = 10.

    _a = LogSpace(jnp.log(jnp.abs(a)), jnp.sign(a))
    _b = LogSpace(jnp.log(jnp.abs(b)), jnp.sign(b))

    assert jnp.isclose((_a + _b).value, a + b)
    assert jnp.isclose((_a - _b).value, a - b)
    assert jnp.isclose((_a * _b).value, a * b)
    assert jnp.isclose((_a / _b).value, a / b)
    assert jnp.isclose((_a ** 2).value, a ** 2)
    assert jnp.isclose((_a ** 3).value, a ** 3)

    a = 0.
    b = 10.

    _a = LogSpace(jnp.log(jnp.abs(a)), jnp.sign(a))
    _b = LogSpace(jnp.log(jnp.abs(b)), jnp.sign(b))

    assert jnp.isclose((_a + _b).value, a + b)
    assert jnp.isclose((_a - _b).value, a - b)
    assert jnp.isclose((_a * _b).value, a * b)
    assert jnp.isclose((_a / _b).value, a / b)
    assert jnp.isclose((_a ** 2).value, a ** 2)
    assert jnp.isclose((_a ** 3).value, a ** 3)

    a = jnp.arange(5)
    sign = jnp.sign(a)
    _a = LogSpace(jnp.log(a), sign)
    assert jnp.allclose(_a[:2].value, a[:2])


def test_tuple_prod():
    assert tuple_prod(()) == 1
    assert tuple_prod((1, 2, 3)) == 6
    assert tuple_prod((4,)) == 4


def test_logaddexp():
    a = jnp.log(1.)
    b = jnp.log(1.)
    assert logaddexp(a, b) == jnp.log(2.)
    a = jnp.log(1.)
    b = jnp.log(-2. + 0j)
    assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, -1.)

    a = jnp.log(-1. + 0j)
    b = jnp.log(2. + 0j)
    assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, 1.)

    for i in range(100):
        u = random.uniform(random.PRNGKey(i), shape=(2,)) * 20. - 10.
        a = jnp.log(u[0] + 0j)
        b = jnp.log(u[1] + 0j)
        assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, u[0] + u[1])


def test_is_complex():
    assert is_complex(jnp.ones(1, dtype=jnp.complex_))


def test_signed_logaddexp():
    for i in range(100):
        u = random.uniform(random.PRNGKey(i), shape=(2,)) * 20. - 10.
        a = jnp.log(jnp.abs(u[0]))
        b = jnp.log(jnp.abs(u[1]))
        sign1 = jnp.sign(u[0])
        sign2 = jnp.sign(u[1])
        ans = u[0] + u[1]
        ans_sign = jnp.sign(ans)
        log_abs_ans = jnp.log(jnp.abs(ans))
        log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
        assert sign_c == ans_sign
        assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [1., 1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [1., -1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    # assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [-1., 1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    # assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [0., 0.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [0., 1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)

    u = [0., -1.]
    a = jnp.log(jnp.abs(u[0]))
    b = jnp.log(jnp.abs(u[1]))
    sign1 = jnp.sign(u[0])
    sign2 = jnp.sign(u[1])
    ans = u[0] + u[1]
    ans_sign = jnp.sign(ans)
    log_abs_ans = jnp.log(jnp.abs(ans))
    log_abs_c, sign_c = signed_logaddexp(a, sign1, b, sign2)
    assert sign_c == ans_sign
    assert jnp.isclose(log_abs_c, log_abs_ans)


def test_cumulative_logsumexp():
    a = jnp.linspace(-1., 1., 100)
    v1 = jnp.log(jnp.cumsum(jnp.exp(a)))
    v2 = cumulative_logsumexp(a)
    assert jnp.allclose(v1, v2)


def test_random_ortho_normal_matrix():
    for i in range(100):
        H = random_ortho_matrix(random.PRNGKey(0), 3)
        assert jnp.all(jnp.isclose(H @ H.conj().T, jnp.eye(3), atol=1e-7))


def test_latin_hyper_cube():
    num_samples = 50
    ndim = 2
    samples = latin_hypercube(random.PRNGKey(442525), num_samples, ndim, 0.)
    s = jnp.sort(samples, axis=0) * num_samples
    assert jnp.all(s < jnp.arange(1, num_samples + 1)[:, None])
    assert jnp.all(s > jnp.arange(0, num_samples)[:, None])

    num_samples = 50
    ndim = 2
    samples = latin_hypercube(random.PRNGKey(442525), num_samples, ndim, 1.)
    s = jnp.sort(samples, axis=0) * num_samples
    assert jnp.all(s < jnp.arange(1, num_samples + 1)[:, None])
    assert jnp.all(s > jnp.arange(0, num_samples)[:, None])


def test_linear_to_log_stats():
    Z = jnp.exp(np.random.normal(size=1000000))
    Z_mu = jnp.mean(Z)
    Z_var = jnp.var(Z)
    log_mu1, log_var1 = linear_to_log_stats(jnp.log(Z_mu), log_f_var=jnp.log(Z_var))
    log_mu2, log_var2 = linear_to_log_stats(jnp.log(Z_mu), log_f2_mean=jnp.log(Z_var + Z_mu ** 2))
    assert jnp.isclose(log_mu1, 0., atol=1e-2)
    assert jnp.isclose(log_var1, 1., atol=1e-2)
    assert log_mu1 == log_mu2
    assert log_var1 == log_var2


def test_msqrt():
    for i in range(10):
        A = random.normal(random.PRNGKey(i), shape=(30, 30))
        A = A @ A.T
        L = msqrt(A)
        assert jnp.all(jnp.isclose(A, L @ L.T))


def test_broadcast_dtypes():
    assert broadcast_dtypes(jnp.array(True).dtype, jnp.int32) == jnp.int32


def test_broadcast_shapes():
    assert broadcast_shapes(1, 1) == (1,)
    assert broadcast_shapes(1, 2) == (2,)
    assert broadcast_shapes(1, (2, 2)) == (2, 2)
    assert broadcast_shapes((2, 2), 1) == (2, 2)
    assert broadcast_shapes((1, 1), (2, 2)) == (2, 2)
    assert broadcast_shapes((1, 2), (2, 2)) == (2, 2)
    assert broadcast_shapes((2, 1), (2, 2)) == (2, 2)
    assert broadcast_shapes((2, 1), (2, 1)) == (2, 1)
    assert broadcast_shapes((2, 1), (1, 1)) == (2, 1)
    assert broadcast_shapes((1, 1), (1, 1)) == (1, 1)
    assert broadcast_shapes((1,), (1, 1)) == (1, 1)
    assert broadcast_shapes((1, 1, 2), (1, 1)) == (1, 1, 2)
    assert broadcast_shapes((1, 2, 1), (1, 3)) == (1, 2, 3)
    assert broadcast_shapes((1, 2, 1), ()) == (1, 2, 1)