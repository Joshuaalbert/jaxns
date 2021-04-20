import numpy as np
from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import PriorChain, MVNPrior, GammaPrior
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln
from jax import random, jit, tree_multimap, test_util
from jax import numpy as jnp

from jaxns.utils import broadcast_shapes, tuple_prod, msqrt, \
    logaddexp, signed_logaddexp, cumulative_logsumexp, resample, random_ortho_matrix, \
    iterative_topological_sort, is_complex, latin_hypercube


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


def test_nested_sampling():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 4
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    prior_transform = PriorChain().push(MVNPrior('x', prior_mu, prior_cov))

    def param_mean(x, **args):
        return x

    for sampler in ['slice', 'multi_slice']:
        ns = NestedSampler(log_likelihood, prior_transform, sampler_name=sampler,
                           num_live_points=5000,
                           max_samples=1e6,
                           collect_samples=True,
                           num_parallel_samplers=2,
                           sampler_kwargs=dict(depth=5, num_slices=5 * ndims),
                           marginalised=dict(x_mean=param_mean)
                           )

        results = jit(ns)(key=random.PRNGKey(42), termination_frac=0.001)

        assert jnp.allclose(results.marginalised['x_mean'], post_mu, atol=0.02)
        assert jnp.abs(results.logZ - true_logZ) < 2. * results.logZerr


def test_gh21():
    num_samples = 10
    true_k = 1.
    true_theta = 0.5

    _gamma = np.random.gamma(true_k, true_theta, size=num_samples)
    samples = jnp.asarray(np.random.poisson(_gamma, size=num_samples))

    prior_k = 5.
    prior_theta = 0.3

    true_post_k = prior_k + jnp.sum(samples)
    true_post_theta = prior_theta / (num_samples * prior_theta + 1.)

    def log_likelihood(gamma, **kwargs):
        """
        Poisson likelihood.
        """
        return jnp.sum(samples * jnp.log(gamma) - gamma - gammaln(samples + 1))

    gamma = GammaPrior('gamma', prior_k, prior_theta)
    prior_chain = gamma.prior_chain()

    ns = NestedSampler(loglikelihood=log_likelihood, prior_chain=prior_chain,
                       sampler_name='slice', num_parallel_samplers=1,
                       sampler_kwargs=dict(depth=5, num_slices=prior_chain.U_ndims * 5),
                       num_live_points=5000, max_samples=1e6, collect_samples=True,
                       collect_diagnostics=True)
    results = jit(ns)(random.PRNGKey(32564), termination_frac=0.001)

    samples = resample(random.PRNGKey(43083245), results.samples, results.log_p, S=int(results.ESS))

    sample_mean = jnp.mean(samples['gamma'], axis=0)

    true_mean = true_post_k * true_post_theta

    assert jnp.allclose(sample_mean, true_mean, atol=0.05)


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
