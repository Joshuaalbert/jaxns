

from jaxns.nested_sampling import NestedSampler
from jaxns.utils import minimum_volume_enclosing_ellipsoid, sample_ellipsoid
from jax import numpy as jnp, random, vmap
from jaxns.prior_transforms import MVNDiagPrior, UniformPrior
from jaxns.utils import cluster

def debug_triangular_solve():
    from jax.lax_linalg import triangular_solve
    from scipy.linalg import solve_triangular
    from jax.scipy.linalg import solve_triangular as solve_triangular_jax
    import jax.numpy as jnp
    ndims = 2
    A = jnp.diag(jnp.ones(ndims))
    A = jnp.where(A == 0., 0.95, A)
    b = jnp.ones(ndims)
    L = jnp.linalg.cholesky(A)
    assert jnp.all(L @ L.T == A)

    x = jnp.linalg.solve(L, b)
    print("Solving L.x  = b with scipy")
    print("x should be {}".format(x))
    scipy_x = solve_triangular(L, b, lower=True)
    assert jnp.all(scipy_x == x)
    print("Works as expected!")

    print("Now note JAX's solution to L^T.x = b corresponds to scipy's L.x = b")
    jax_x = triangular_solve(L, b, lower=True, transpose_a=True)
    assert jnp.all(jax_x==scipy_x)
    print("Likewise, JAX's solution to L.x = b corresponds to scipy's L^T.x = b")
    assert jnp.all(triangular_solve(L, b, lower=True) == solve_triangular(L, b, lower=True, trans=1))

    print("Note, I have not tested for the L^H.x=b case.")

    jax_x = solve_triangular_jax(L, b, lower=True)
    assert jnp.all(scipy_x == jax_x)

def debug_nestest_sampler():
    from jax.scipy.linalg import solve_triangular
    from jax import random, jit, disable_jit
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx


    ndims = 2
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.ones(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu
    post_cov = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_cov

    log_likelihood = lambda x: log_normal(x, data_mu, data_cov)

    prior_transform = MVNDiagPrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    # prior_transform = LaplacePrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    # prior_transform = UniformPrior(-20.*jnp.ones(ndims), 20.*jnp.ones(ndims))
    ns = NestedSampler(log_likelihood, prior_transform, sampler_name='whitened_box')

    def run_with_n(n):
        @jit
        def run():
            return ns(key=random.PRNGKey(0),
                      num_live_points=n * ndims,
                      max_samples=1e6,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=True)

        # with disable_jit():
        results = run()
        return results

    import pylab as plt
    for n in [350]:
        # with disable_jit():
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.hlines(true_logZ, 10,320)
    plt.show()

    print(results)


    print(results.logZ, results.param_mean, results.param_covariance, results.efficiency)

    print("True logZ={}".format(true_logZ))
    print("True posterior m={}\nCov={}".format(post_mu, post_cov))

    ###
    import pylab as plt

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    axs[0].plot(-results.log_X, results.n_per_sample)
    axs[0].set_ylabel(r'$n(X)$')
    axs[1].plot(-results.log_X, jnp.exp(results.log_L_samples))
    axs[1].set_ylabel(r'$L(X)$')
    axs[2].plot(-results.log_X, jnp.exp(results.log_p))
    axs[2].set_ylabel(r'$Z^{-1}L(X) dX$')
    axs[3].plot(-results.log_X, jnp.exp(results.logZ) * jnp.cumsum(jnp.exp(results.log_p)))
    axs[3].set_ylabel(r'$Z(x > X)$')
    axs[3].set_xlabel(r'$-\log X$')
    plt.show()

    fig, axs = plt.subplots(ndims, 1, sharex=True, figsize=(8, 8))
    for i in range(ndims):
        axs[i].scatter(-results.log_X, results.samples['x'][:, i])
        axs[i].set_ylabel(r'$x[{}](X)$'.format(i))
    axs[-1].set_xlabel(r'$-\log X$')
    plt.show()

    # #sum(
    weights = jnp.exp(results.log_p)
    # print(jnp.sum(weights), results.logZ, logsumexp(results.log_p))
    # m = jnp.average(results.samples, weights=weights, axis=0)
    # print(m)
    # cov = jnp.sum(weights[:, None,None]*((results.samples[:, :, None]-m) * (results.samples[:, None,:]-m)), axis=0)
    # print(cov)

    plt.hist2d(results.samples['x'][:, 0], results.samples['x'][:, 1], bins=50,
               weights=weights)  # ,range=((-10., 10.),(-10., 10.)))
    plt.show()
    #P(A|B)=(1-P(not A or not B))/P(B)
    # =(1-(P(not A) + P(not B) - P(not A | not B) P(not B)))/P(B)
    # =(P(A) - P(not B) + P(not A | not B) P(not B))/P(B)
    # =(P(A) - P(not B)( 1 - P(not A | not B)))/P(B)
    # P(A|B)=(P(A) - P(not B) P(A | not B))/P(B)
    # P(B|A)=(P(B) - P(not A) P(B | not A))/P(A)
    #   P(A | not B) = (P(A)-P(B) + P(not A) P(B | not A))/ P(not B)
    #P(B|A)=(1-P(not B or not A))/P(A)
    #
    # => P(not B or not A) = 1 - P(B|A) P(A)
    # import dynesty
    # import dynesty.plotting as dyplot
    # # initialize our "static" nested sampler
    # sampler = dynesty.NestedSampler(lambda x: log_normal(x, data_mu, data_cov),
    #                                 prior_transform,
    #                                 ndims,
    #                                 nlive=500*ndims)
    #
    # # sample from the distribution
    # sampler.run_nested(dlogz=0.01)
    #
    # # grab our results
    # res = sampler.results
    # dyplot.cornerplot(res, show_titles=True)
    # plt.show()


def generate_data():
    T = 1
    tec = jnp.cumsum(10. * random.normal(random.PRNGKey(0), shape=(T,)))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV + 0.2  # + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * random.normal(random.PRNGKey(1), shape=Y.shape)
    # Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape)
    Sigma = 0.25 ** 2 * jnp.eye(48)
    amp = jnp.ones_like(phase)
    return Sigma, T, Y_obs, amp, tec, freqs


def debug_tec_clock_inference():
    from jax.lax_linalg import triangular_solve
    from jax import random, jit
    from born_rime.nested_sampling.prior_transforms import UniformUncertNormalParam

    Sigma, T, Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = triangular_solve(L, dx, lower=True, transpose_a=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    def log_likelihood(x, uncert):
        tec = x[0]  # [:, 0]
        clock = x[1]  # [:, 1]
        phase = tec * (TEC_CONV / freqs) + clock
        Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        return log_normal(Y, Y_obs[0, :], uncert ** 2 * jnp.eye(2 * freqs.size))

    prior_mu = jnp.array([0., 0.])
    prior_cov = jnp.diag(jnp.array([20., 1.])) ** 2

    # import pylab as plt
    # X = random.uniform(random.PRNGKey(0),shape=(10000,2),minval=jnp.array([-100., -jnp.pi])[None,:], maxval=jnp.array([100., jnp.pi])[None,:])
    # plt.scatter(X[:,0], X[:,1], c=vmap(log_likelihood)(X))
    # plt.show()
    # return

    # prior_transform = MVNDiagPrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    prior_transform = UniformUncertNormalParam(jnp.array([0.01]), jnp.array([1.]), prior_mu,
                                               jnp.sqrt(jnp.diag(prior_cov)))
    print(prior_transform.to_shapes)
    ndims = prior_transform.U_ndims
    # prior_transform = HMMPrior(0. * jnp.array([-100., -jnp.pi]), jnp.array([100., 1.]), jnp.array([10., 0.1]), T, 2)

    # prior_transform = UniformPrior(-10.*jnp.ones(ndims), 10.*jnp.ones(ndims))
    ns = NestedSampler(log_likelihood, prior_transform)

    @jit
    def run():
        return ns(key=random.PRNGKey(1),
                  num_live_points=25 * ndims,
                  max_samples=1e6,
                  collect_samples=True,
                  termination_frac=0.05,
                  stoachastic_uncertainty=True)

    # with disable_jit():
    results = run()

    print(results)

    print(tec)

    ###
    import pylab as plt

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    axs[0].plot(-results.log_X, results.n_per_sample)
    axs[0].set_ylabel(r'$n(X)$')
    axs[1].plot(-results.log_X, jnp.exp(results.log_L_samples))
    axs[1].set_ylabel(r'$L(X)$')
    axs[2].plot(-results.log_X, jnp.exp(results.log_p))
    axs[2].set_ylabel(r'$Z^{-1}L(X) dX$')
    axs[3].plot(-results.log_X, jnp.exp(results.logZ) * jnp.cumsum(jnp.exp(results.log_p)))
    axs[3].set_ylabel(r'$Z(x > X)$')
    axs[3].set_xlabel(r'$-\log X$')
    plt.show()

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    weights = jnp.exp(results.log_p)

    ax = axs[0][0]
    samples = results.samples['x'][:, 0]
    ax.hist(samples, weights=weights, bins=int(jnp.sqrt(samples.shape[0])))
    ax = axs[1][1]
    samples = results.samples['x'][:, 1]
    ax.hist(samples, weights=weights, bins=int(jnp.sqrt(samples.shape[0])))
    ax = axs[2][2]
    samples = results.samples['uncert'][:, 0]
    ax.hist(samples, weights=weights, bins=int(jnp.sqrt(samples.shape[0])))
    ax = axs[1][0]
    samples1 = results.samples['x'][:, 1]
    samples2 = results.samples['x'][:, 0]
    ax.hist2d(samples1, samples2, weights=weights, bins=int(jnp.sqrt(samples1.shape[0])))
    ax = axs[2][0]
    samples1 = results.samples['uncert'][:, 0]
    samples2 = results.samples['x'][:, 0]
    ax.hist2d(samples1, samples2, weights=weights, bins=int(jnp.sqrt(samples1.shape[0])))
    ax = axs[2][1]
    samples1 = results.samples['uncert'][:, 0]
    samples2 = results.samples['x'][:, 1]
    ax.hist2d(samples1, samples2, weights=weights, bins=int(jnp.sqrt(samples1.shape[0])))
    plt.show()

    # import dynesty
    # import dynesty.plotting as dyplot
    # # initialize our "static" nested sampler
    # sampler = dynesty.NestedSampler(lambda x: log_normal(x, data_mu, data_cov),
    #                                 prior_transform,
    #                                 ndims,
    #                                 nlive=500*ndims)
    #
    # # sample from the distribution
    # sampler.run_nested(dlogz=0.01)
    #
    # # grab our results
    # res = sampler.results
    # dyplot.cornerplot(res, show_titles=True)
    # plt.show()


def debug_tec_clock_inference_hmm():
    from jax.lax_linalg import triangular_solve
    from jax import random, jit
    from born_rime.nested_sampling.prior_transforms import HMMPrior

    Sigma, T, Y_obs, amp, tec, freqs = generate_data()
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = triangular_solve(L, dx, lower=True, transpose_a=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    def log_likelihood(x):
        tec = x[:, 0]
        clock = x[:, 1]
        phase = tec[:, None] * (TEC_CONV / freqs) + clock[:, None]
        Y = jnp.concatenate([amp[0, :] * jnp.cos(phase), amp[0, :] * jnp.sin(phase)], axis=1)
        return jnp.sum(vmap(lambda Y, Y_obs: log_normal(Y, Y_obs, Sigma))(Y, Y_obs))

    ndims = 6
    prior_mu = jnp.zeros(ndims)
    prior_cov = jnp.diag(jnp.array([100., 1.])) ** 2

    # import pylab as plt
    # X = random.uniform(random.PRNGKey(0),shape=(10000,2),minval=jnp.array([-100., -jnp.pi])[None,:], maxval=jnp.array([100., jnp.pi])[None,:])
    # plt.scatter(X[:,0], X[:,1], c=vmap(log_likelihood)(X))
    # plt.show()
    # return

    # prior_transform = MVNDiagPrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    prior_transform = HMMPrior(0. * jnp.array([-100., -jnp.pi]), jnp.array([100., 1.]), jnp.array([10., 0.1]), T, 2)

    # prior_transform = UniformPrior(-10.*jnp.ones(ndims), 10.*jnp.ones(ndims))
    ns = NestedSampler(log_likelihood, prior_transform)

    @jit
    def run():
        return ns(key=random.PRNGKey(1),
                  num_live_points=25 * ndims,
                  max_samples=5e3,
                  collect_samples=True,
                  termination_frac=0.05,
                  stoachastic_uncertainty=True)

    # with disable_jit():
    results = run()

    print(results)

    print(tec)

    ###
    import pylab as plt

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    axs[0].plot(-results.log_X, results.n_per_sample)
    axs[0].set_ylabel(r'$n(X)$')
    axs[1].plot(-results.log_X, jnp.exp(results.log_L_samples))
    axs[1].set_ylabel(r'$L(X)$')
    axs[2].plot(-results.log_X, jnp.exp(results.log_p))
    axs[2].set_ylabel(r'$Z^{-1}L(X) dX$')
    axs[3].plot(-results.log_X, jnp.exp(results.logZ) * jnp.cumsum(jnp.exp(results.log_p)))
    axs[3].set_ylabel(r'$Z(x > X)$')
    axs[3].set_xlabel(r'$-\log X$')
    plt.show()

    fig, axs = plt.subplots(ndims, 1, sharex=True, figsize=(8, 8))
    for i in range(ndims):
        axs[i].scatter(-results.log_X, results.samples['x'].reshape((-1, ndims))[:, i])
        axs[i].set_ylabel(r'$x[{}](X)$'.format(i))
    axs[-1].set_xlabel(r'$-\log X$')
    plt.show()

    return
    weights = jnp.exp(results.log_p)

    plt.hist2d(results.samples[:, 0], results.samples[:, 1], bins=100,
               weights=weights)  # ,range=((-10., 10.),(-10., 10.)))
    plt.show()

    # import dynesty
    # import dynesty.plotting as dyplot
    # # initialize our "static" nested sampler
    # sampler = dynesty.NestedSampler(lambda x: log_normal(x, data_mu, data_cov),
    #                                 prior_transform,
    #                                 ndims,
    #                                 nlive=500*ndims)
    #
    # # sample from the distribution
    # sampler.run_nested(dlogz=0.01)
    #
    # # grab our results
    # res = sampler.results
    # dyplot.cornerplot(res, show_titles=True)
    # plt.show()


def debug_cluster(width=1.1):
    import jax.numpy as jnp
    from jax import random, disable_jit
    import numpy as np
    import pylab as plt

    true_num_clusters = 4
    X = jnp.array(np.stack([np.random.normal((i % 4) * 5., width, size=(2,)) for i in range(20)], axis=0))
    with disable_jit():
        key, cluster_centers, K, sillohettes = cluster(key=random.PRNGKey(1),
                                                        points=X,
                                                        max_K=jnp.array(10))

    cluster_id = masked_cluster_id(X, cluster_centers, K)
    print(cluster_centers, K, cluster_id)
    plt.plot(list(range(1, len(sillohettes) + 1)), sillohettes)

    plt.show()
    print('Found {} clusters'.format(K))

    for m in range(K):
        plt.scatter(X[m == cluster_id, 0], X[m == cluster_id, 1])
        plt.scatter(cluster_centers[:K, 0], cluster_centers[:K, 1], c='red')
    print(cluster_centers)
    plt.show()

    assert K == true_num_clusters


def debug_masked_cluster_id():
    points = random.normal(random.PRNGKey(0), shape=(100, 3))
    centers = random.normal(random.PRNGKey(0), shape=(10, 3))
    K = 5
    cluster_id = masked_cluster_id(points, centers, K)
    dist = jnp.linalg.norm(centers[:K, None, :] - points[None, :, :], axis=-1)
    cluster_id_ = jnp.argmin(dist, axis=0)
    assert jnp.all(cluster_id == cluster_id_)

def debug_mvee():
    import pylab as plt

    n = random.normal(random.PRNGKey(0), (10000,2))
    n = n /jnp.linalg.norm(n, axis=1, keepdims=True)
    angle = jnp.arctan2(n[:,1], n[:,0])
    plt.hist(angle, bins=100)
    plt.show()
    N = 120
    D = 2
    points = random.uniform(random.PRNGKey(0), (N, D))

    from jax import disable_jit
    with disable_jit():
        center, radii, rotation = minimum_volume_enclosing_ellipsoid(points, 0.01)

    plt.hist(jnp.linalg.norm((rotation.T @ (points.T - center[:, None])) / radii[:, None], axis=0))
    plt.show()
    print(center, radii, rotation)
    plt.scatter(points[:, 0], points[:, 1])
    theta = jnp.linspace(0., jnp.pi*2, 100)
    ellipsis = center[:, None] + rotation @ jnp.stack([radii[0]*jnp.cos(theta), radii[1]*jnp.sin(theta)], axis=0)
    plt.plot(ellipsis[0,:], ellipsis[1,:])

    for i in range(1000):
        y = sample_ellipsoid(random.PRNGKey(i), center, radii, rotation)
        plt.scatter(y[0], y[1])



    C = jnp.linalg.pinv(jnp.cov(points, rowvar=False, bias=True))
    p = (N - D - 1)/N
    def q(p):
        return p + p**2/(4.*(D-1))
    C = C / q(p)
    c = jnp.mean(points, axis=0)
    W, Q, Vh = jnp.linalg.svd(C)
    radii = jnp.reciprocal(jnp.sqrt(Q))
    rotation = Vh.conj().T
    ellipsis = c[:, None] + rotation @ jnp.stack([radii[0] * jnp.cos(theta), radii[1] * jnp.sin(theta)], axis=0)
    plt.plot(ellipsis[0, :], ellipsis[1, :])

    plt.show()

def debug_riddle():
    import numpy as np
    import pylab as plt

    t = np.linspace(0., 12., 10000)
    m = (t - np.floor(t))*2*np.pi
    h = t/12.*2.*np.pi




    plt.plot(t, h)
    plt.show()
    plt.plot(t, m)
    plt.show()

    def time(m, h):
        return h/(2.*np.pi)*12.

    plt.plot(t, time(m,h))
    plt.plot(t, time(h,m))
    plt.show()

if __name__ == '__main__':
    # debug_triangular_solve()
    debug_nestest_sampler()
    # test_tec_clock_inference()
    # debug_riddle()
    # debug_mvee()