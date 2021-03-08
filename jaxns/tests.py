from jax import numpy as jnp, random

from jaxns.utils import broadcast_shapes, tuple_prod, msqrt, \
    logaddexp, signed_logaddexp, cumulative_logsumexp, resample, random_ortho_matrix, \
    iterative_topological_sort, is_complex


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


def test_tuple_prod():
    assert tuple_prod(()) == 1
    assert tuple_prod((1, 2, 3)) == 6
    assert tuple_prod((4,)) == 4


def test_msqrt():
    for i in range(10):
        A = random.normal(random.PRNGKey(i), shape=(30, 30))
        A = A @ A.T
        L = msqrt(A)
        assert jnp.all(jnp.isclose(A, L @ L.T))


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
    print(v1)
    print(v2)
    assert jnp.isclose(v1, v2).all()


def test_resample():
    x = random.normal(key=random.PRNGKey(0), shape=(50,))
    logits = -jnp.ones(50)
    samples = {'x': x}
    assert jnp.all(resample(random.PRNGKey(0), samples, logits)['x'] == resample(random.PRNGKey(0), x, logits))


def test_random_ortho_normal_matrix():
    for i in range(100):
        H = random_ortho_matrix(random.PRNGKey(0), 3)
        print(jnp.linalg.eigvals(H))
        assert jnp.all(jnp.isclose(H @ H.conj().T, jnp.eye(3), atol=1e-7))


def test_iterative_topological_sort():
    dsk = {'a': [],
           'b': ['a'],
           'c': ['a', 'b']}
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a']
    dsk = {'a': [],
           'b': ['a', 'd'],
           'c': ['a', 'b']}
    # print(iterative_topological_sort(dsk, ['a', 'b', 'c']))
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a', 'd']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a', 'd']