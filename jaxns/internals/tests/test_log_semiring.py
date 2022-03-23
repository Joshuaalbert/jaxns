from jax import numpy as jnp, random

from jaxns.internals.log_semiring import LogSpace, signed_logaddexp, cumulative_logsumexp, logaddexp, is_complex


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
        assert jnp.isclose(jnp.exp(logaddexp(a, b)).real, u[0] + u[1], atol=5e-5)


def test_is_complex():
    assert is_complex(jnp.ones(1, dtype=jnp.complex_))