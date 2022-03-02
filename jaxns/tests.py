import numpy as np

from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln
from jax import random, jit, vmap, numpy as jnp
from jax import numpy as jnp

from jaxns.parallel_sampling import _parallel_sampling
from jaxns.types import Reservoir, SampleCollection
from jaxns.log_math import LogSpace
from jaxns.utils import broadcast_shapes, tuple_prod, msqrt, \
    resample, random_ortho_matrix, \
    iterative_topological_sort, latin_hypercube, combine_sample_collections, _bit_mask, linear_to_log_stats
from jaxns.log_math import logaddexp, signed_logaddexp, cumulative_logsumexp, is_complex
from jaxns.nested_sampler.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, MVNPrior, GammaPrior

def test_combine_live_points():


    def _single_test(log_L_thread, num_live_points_thread, log_L_main, num_live_points_main,
                     true_merged_log_L, true_merged_num_live_points):
        points_U_thread = log_L_thread[:,None]
        points_X_thread = dict(x=points_U_thread)
        num_likelihood_evaluations_thread = log_L_thread
        sample_collection_thread = SampleCollection(points_U=points_U_thread,
                                                    points_X=points_X_thread,
                                                    log_L=log_L_thread,
                                                    num_likelihood_evaluations=num_likelihood_evaluations_thread,log_dZ_mean=None,
                                                    log_dZ_var=None,
                                                    log_X_mean=None,
                                                    log_X_var=None,
                                                    num_live_points=num_live_points_thread)
        points_U_main = log_L_main[:, None]
        points_X_main = dict(x=points_U_main)
        num_likelihood_evaluations_main = log_L_main
        sample_collection_main = SampleCollection(points_U=points_U_main,
                                                    points_X=points_X_main,
                                                    log_L=log_L_main,
                                                    num_likelihood_evaluations=num_likelihood_evaluations_main,
                                                    log_dZ_mean=None,
                                                    log_dZ_var=None,
                                                    log_X_mean=None,
                                                    log_X_var=None,
                                                    num_live_points=num_live_points_main)

        merged_sample_collection = combine_sample_collections(
            sample_collection_main=sample_collection_main,
            sample_collection_thread=sample_collection_thread,
            n_main=jnp.sum(~jnp.isinf(log_L_main)),
            n_thread=log_L_thread.size,
            head_main_start=0)

        assert jnp.allclose(merged_sample_collection.points_U, merged_sample_collection.log_L[:,None])
        assert jnp.allclose(merged_sample_collection.points_X['x'], merged_sample_collection.points_U)
        assert jnp.allclose(merged_sample_collection.num_likelihood_evaluations, merged_sample_collection.log_L)
        assert jnp.allclose(merged_sample_collection.log_L, true_merged_log_L)
        assert jnp.allclose(merged_sample_collection.num_live_points, true_merged_num_live_points)

    log_L_thread = jnp.asarray([0., 1., 2.])
    num_live_points_thread = jnp.asarray([3, 2, 1])
    log_L_main = jnp.asarray([0.5, 1.5, 2.5, 3., -jnp.inf, -jnp.inf, -jnp.inf])
    num_live_points_main = jnp.asarray([4, 3, 2, 1, 0, 0, 0])
    true_merged_log_L = jnp.asarray([0., 0.5, 1., 1.5, 2., 2.5, 3.])
    true_merged_num_live_points = jnp.asarray([7, 6, 5, 4, 3, 2, 1])

    _single_test(log_L_thread, num_live_points_thread, log_L_main, num_live_points_main,
                 true_merged_log_L, true_merged_num_live_points)

    log_L_thread = jnp.asarray([0., 1.])
    num_live_points_thread = jnp.asarray([2, 1])
    log_L_main = jnp.asarray([0.5, 0.5, 1.5, -jnp.inf, -jnp.inf])
    num_live_points_main = jnp.asarray([3, 2, 1, 0, 0])
    true_merged_log_L = jnp.asarray([0., 0.5, 0.5, 1., 1.5])
    true_merged_num_live_points = jnp.asarray([5, 4, 3, 2, 1])
    _single_test(log_L_thread, num_live_points_thread, log_L_main, num_live_points_main,
                 true_merged_log_L, true_merged_num_live_points)


    log_L_thread = jnp.asarray([0.5, 0.5, 1.])
    num_live_points_thread = jnp.asarray([3, 2, 1])
    log_L_main = jnp.asarray([0.5, 0.5, 1.5, -jnp.inf, -jnp.inf, -jnp.inf])
    num_live_points_main = jnp.asarray([3, 2, 1, 0, 0, 0])
    true_merged_log_L = jnp.asarray([0.5, 0.5, 0.5, 0.5, 1., 1.5])
    true_merged_num_live_points = jnp.asarray([6, 5, 4, 3, 2, 1])
    _single_test(log_L_thread, num_live_points_thread, log_L_main, num_live_points_main,
                 true_merged_log_L, true_merged_num_live_points)



    log_L_thread = jnp.asarray([0.5, 0.5, 1.])
    num_live_points_thread = jnp.asarray([3, 2, 1])
    log_L_main = jnp.asarray([0.5, 0.5, -jnp.inf, -jnp.inf, -jnp.inf])
    num_live_points_main = jnp.asarray([2, 1, 0, 0, 0])
    true_merged_log_L = jnp.asarray([0.5, 0.5, 0.5, 0.5, 1.])
    true_merged_num_live_points = jnp.asarray([5, 4, 3, 2, 1])
    _single_test(log_L_thread, num_live_points_thread, log_L_main, num_live_points_main,
                 true_merged_log_L, true_merged_num_live_points)




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

    for sampler in ['slice', 'multi_ellipsoid']:
        ns = NestedSampler(log_likelihood, prior_transform, sampler_name=sampler,
                           num_live_points=5000,
                           max_samples=1e6,
                           collect_samples=True,
                           num_parallel_samplers=2,
                           sampler_kwargs=dict(depth=5, num_slices=5 * ndims),
                           marginalised_funcs=dict(x_mean=param_mean)
                           )

        results = jit(ns)(key=random.PRNGKey(42), termination_frac=0.001)

        assert jnp.allclose(results.marginalised['x_mean'], post_mu, atol=0.02)
        assert jnp.abs(results.log_Z_mean - true_logZ) < 3. * results.log_Z_uncert


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

    samples = resample(random.PRNGKey(43083245), results.samples, results.log_p_mean, S=int(results.ESS))

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


def test_compute_log_relative_error():
    log_dZ = -jnp.inf * jnp.ones((5,), dtype=jnp.float64)
    r_tol = 1e-4
    relative_error = compute_log_relative_error(log_dZ)
    relative_error_low_enough = relative_error <= jnp.log(r_tol)
    assert relative_error_low_enough

    log_dZ = jnp.ones((5,), dtype=jnp.float64)
    r_tol = 1e-4
    relative_error = compute_log_relative_error(log_dZ)
    relative_error_low_enough = relative_error <= jnp.log(r_tol)
    assert relative_error_low_enough


def test_integer_order_statistic():
    n = 10
    k = n // 2
    m = 2

    def _sample(key, k, n, m):
        return jnp.sort(random.randint(key, (n,), 0, m))[k - 1]

    # uncomment to plot histograms up top of each other
    # import pylab as plt
    # keys = random.split(random.PRNGKey(42), 10000)
    # values1 = vmap(lambda key:integer_order_statistic(key, k, n, m))(keys)
    # plt.hist(values1, bins = np.arange(m+1), alpha=0.5)
    # values2 = vmap(lambda key: _sample(key, k, n, m))(keys)
    #
    # plt.hist(values2, bins=np.arange(m + 1), alpha=0.5)
    # plt.show()

    keys = random.split(random.PRNGKey(42), 10000)
    values1 = vmap(lambda key: integer_order_statistic(key, k, n, m))(keys)
    values2 = vmap(lambda key: _sample(key, k, n, m))(keys)
    freq1, _ = np.histogram(values1, bins=np.arange(m + 1), normed=True)
    freq2, _ = np.histogram(values2, bins=np.arange(m + 1), normed=True)
    ks = np.max(np.abs(np.cumsum(freq1) - np.cumsum(freq2)))
    assert ks < 1e-2

    # uncomment to see KS-statistic
    # import pylab as plt
    # for S in [50,100,1000,10000,50000,100000,500000,1000000]:
    #     keys = random.split(random.PRNGKey(42), S)
    #     values1 = vmap(lambda key: integer_order_statistic(key, k, n, m))(keys)
    #     values2 = vmap(lambda key: _sample(key, k, n, m))(keys)
    #     freq1, _ = np.histogram(values1, bins = np.arange(m+1), normed=True)
    #     freq2, _ = np.histogram(values2, bins = np.arange(m+1), normed=True)
    #     ks = np.max(np.abs(np.cumsum(freq1)-np.cumsum(freq2)))
    #     plt.scatter(S, ks)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()


def test_get_idx_contour():
    reservoir_state_log_L = jnp.asarray([5., 3., 0.])
    log_L_contour = -1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 3
    log_L_contour = 0.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 2
    log_L_contour = 1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 2
    log_L_contour = 3.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 1
    log_L_contour = 5.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0
    log_L_contour = 5.5
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0

    reservoir_state_log_L = jnp.asarray([5., 3., 3., 0.])
    log_L_contour = -1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 4
    log_L_contour = 0.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 3
    log_L_contour = 1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 3
    log_L_contour = 3.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 1
    log_L_contour = 5.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0
    log_L_contour = 5.5
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0

    reservoir_state_log_L = jnp.asarray([5., 3., 0., 0.])
    log_L_contour = -1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 4
    log_L_contour = 0.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 2
    log_L_contour = 1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 2
    log_L_contour = 3.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 1
    log_L_contour = 5.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0
    log_L_contour = 5.5
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0

    reservoir_state_log_L = jnp.asarray([5., 5., 3., 0.])
    log_L_contour = -1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 4
    log_L_contour = 0.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 3
    log_L_contour = 1.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 3
    log_L_contour = 3.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 2
    log_L_contour = 5.
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0
    log_L_contour = 5.5
    assert get_idx_contour(reservoir_state_log_L, log_L_contour) == 0


def test_parallel_sampling():
    def loglikelihood_from_U(U_flat):
        return -jnp.sum((U_flat - 0.5) ** 2 / 0.1 ** 2)

    num_chains = 3
    reservoir_size = 100
    U_size = 2

    key = random.PRNGKey(42)

    perform_refill = jnp.asarray(True)
    points_U = random.uniform(random.PRNGKey(43), (reservoir_size, U_size))

    log_L_reservoir = vmap(loglikelihood_from_U)(points_U)
    argsort = jnp.argsort(log_L_reservoir)[::-1]
    log_L_reservoir = log_L_reservoir[argsort]
    points_U = points_U[argsort]
    points_X = dict(X=points_U)

    log_L_contour = log_L_reservoir[1]
    chains_state = ChainsState(log_X=jnp.zeros(num_chains),
                              log_L=log_L_contour,
                              idx_in_reservoir=jnp.full((num_chains,), reservoir_size))

    reservoir_state = Reservoir(points_U, points_X,
                                log_L=log_L_reservoir,
                                num_likelihood_evaluations=jnp.ones(log_L_reservoir.shape, jnp.int_))

    num_slices = 10
    midpoint_shrink = True
    num_parallel_samplers = 10
    from jax import jit
    f = jit(lambda key: _parallel_sampling(loglikelihood_from_U, chains_state, key, log_L_contour, perform_refill,
                                           reservoir_state, num_slices, midpoint_shrink,
                                           num_parallel_samplers))

    # with disable_jit():
    next_chains_state, next_reservoir_state = f(key)
    next_reservoir_state.num_likelihood_evaluations.block_until_ready()

    # import pylab as plt
    # plt.scatter(next_reservoir_state.points_U[:, 0],
    #             next_reservoir_state.points_U[:, 1],
    #             c=next_reservoir_state.log_L)
    # plt.show()
    #
    # next_reservoir_state.num_likelihood_evaluations.block_until_ready()
    # from timeit import default_timer
    # t0 = default_timer()
    # for _ in range(100):
    #     next_chains_state, next_reservoir_state = f(key)
    #     next_reservoir_state.num_likelihood_evaluations.block_until_ready()
    # print((default_timer() - t0)/100.)
    # print(next_reservoir_state.num_likelihood_evaluations)


def test_bit_mask():
    assert _bit_mask(1, width=2) == [1,0]
    assert _bit_mask(2, width=2) == [0,1]
    assert _bit_mask(3, width=2) == [1,1]


def test_linear_to_log_stats():
    Z = jnp.exp(np.random.normal(size=1000000))
    Z_mu = jnp.mean(Z)
    Z_var = jnp.var(Z)
    log_mu1, log_var1 = linear_to_log_stats(jnp.log(Z_mu), log_f_var=jnp.log(Z_var))
    log_mu2, log_var2 = linear_to_log_stats(jnp.log(Z_mu), log_f2_mean=jnp.log(Z_var + Z_mu**2))
    assert jnp.isclose(log_mu1, 0., atol=1e-2)
    assert jnp.isclose(log_var1, 1.,atol=1e-2)
    assert log_mu1 == log_mu2
    assert log_var1 == log_var2