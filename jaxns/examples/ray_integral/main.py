from jaxns.examples.ray_integral.build_prior import build_prior
from jaxns.gaussian_process.kernels import RBF
from jaxns.examples.ray_integral.generate_data import rbf_dtec
from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax import random, jit
from jax.scipy.linalg import solve_triangular
from jax import numpy as jnp, vmap
import pylab as plt
from timeit import default_timer


def main(kernel):
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean

        dx = solve_triangular(L, dx, lower=True)
        # maha = dx @ jnp.linalg.solve(cov, dx)
        maha = dx @ dx
        # logdet = jnp.log(jnp.linalg.det(cov))
        logdet = jnp.sum(jnp.log(jnp.diag(L)))
        log_prob = -0.5 * x.size * jnp.log(2. * jnp.pi) - logdet - 0.5 * maha
        return log_prob

    true_height, true_width, true_sigma, true_l, true_uncert = 200., 100., 1., 10., 2.5
    nant = 5
    ndir = 5
    X, Y, Y_obs = rbf_dtec(nant, ndir, true_height, true_width, true_sigma, true_l, true_uncert)
    a = X[:, 0:3]
    k = X[:, 3:6]
    x0 = a[0, :]

    def log_likelihood(dtec, uncert, **kwargs):
        """
        P(Y|sigma, half_width) = N[Y, f, K]
        Args:
            sigma:
            l:

        Returns:

        """
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        return log_normal(Y_obs, dtec, data_cov)

    def predict_f(dtec, uncert, **kwargs):
        return dtec

    def predict_fvar(dtec, uncert, **kwargs):
        return dtec ** 2

    def tec_to_dtec(tec):
        tec = tec.reshape((nant, ndir))
        dtec = jnp.reshape(tec - tec[0, :], (-1,))
        return dtec

    prior_chain = build_prior(X, kernel, tec_to_dtec, x0)
    print(prior_chain)

    U_test = jnp.array([random.uniform(key, shape=(prior_chain.U_ndims,)) for key in random.split(random.PRNGKey(4325),1000)])
    log_lik = jnp.array([log_likelihood(**prior_chain(U)) for U in U_test])
    print(jnp.sum(jnp.isnan(log_lik)))
    print(U_test[jnp.isnan(log_lik)])
    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice', predict_f=predict_f,
                       predict_fvar=predict_fvar)

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                      sampler_kwargs=dict(depth=7, num_slices=1))

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print("Efficiency", results.efficiency)
        print("Time to run (including compile)", default_timer() - t0)
        t0 = default_timer()
        results = run(random.PRNGKey(1))
        print("Efficiency",results.efficiency)
        print("Time to run (no compile)", default_timer() - t0)
        print("Time efficiency normalised", (default_timer() - t0)*results.efficiency)
        return results

    for n in [1000]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)

    # #
    # K = GaussianProcessKernelPrior('K',
    #                                TomographicKernel(x0, kernel, S=20), X,
    #                                MVNPrior('height', results.param_mean['height'], results.param_covariance['height']),#UniformPrior('height', 100., 300.),
    #                                MVNPrior('width', results.param_mean['width'], results.param_covariance['width']),#UniformPrior('width', 50., 150.),
    #                                MVNPrior('l', results.param_mean['l'], results.param_covariance['l']),#UniformPrior('l', 7., 20.),
    #                                MVNPrior('sigma', results.param_mean['sigma'], results.param_covariance['sigma']),#UniformPrior('sigma', 0.3, 2.),
    #                                tracked=False)
    # tec = MVNPrior('tec', jnp.zeros((X.shape[0],)), K, ill_cond=True, tracked=False)
    # dtec = DeterministicTransformPrior('dtec', tec_to_dtec, tec.to_shape, tec, tracked=False)
    # prior_chain = PriorChain() \
    #     .push(dtec) \
    #     .push(UniformPrior('uncert', 0., 5.))
    #
    # ns = NestedSampler(log_likelihood, prior_chain, sampler_name='multi_ellipsoid', predict_f=predict_f,
    #                    predict_fvar=predict_fvar)
    #
    # def run_with_n(n):
    #     @jit
    #     def run(key):
    #         return ns(key=key,
    #                   num_live_points=n,
    #                   max_samples=1e5,
    #                   collect_samples=True,
    #                   termination_frac=0.01,
    #                   stoachastic_uncertainty=False,
    #                   sampler_kwargs=dict(depth=4))
    #
    #     t0 = default_timer()
    #     results = run(random.PRNGKey(0))
    #     print("Efficiency", results.efficiency)
    #     print("Time to run (including compile)", default_timer() - t0)
    #     t0 = default_timer()
    #     results = run(random.PRNGKey(1))
    #     print(results.efficiency)
    #     print("Time to run (no compile)", default_timer() - t0)
    #     return results
    #
    # for n in [100]:
    #     results = run_with_n(n)
    #     plt.scatter(n, results.logZ)
    #     plt.errorbar(n, results.logZ, yerr=results.logZerr)


    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.ylabel('log Z')
    plt.show()

    fstd = jnp.sqrt(results.marginalised['predict_fvar'] - results.marginalised['predict_f'] ** 2)
    plt.scatter(jnp.arange(Y.size),Y_obs, marker='+', label='data')
    plt.scatter(jnp.arange(Y.size),Y, marker="o", label='underlying')
    plt.scatter(jnp.arange(Y.size), results.marginalised['predict_f'], marker=".", label='underlying')
    plt.errorbar(jnp.arange(Y.size), results.marginalised['predict_f'], yerr=fstd, label='marginalised')
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.legend()
    plt.show()

    # plot_samples_development(results,save_name='./ray_integral_solution.mp4')
    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr


if __name__ == '__main__':
    logZ_rbf, logZerr_rbf = main(RBF())
    # logZ_m12, logZerr_m12 = main(M12())
    # plt.errorbar(['rbf', 'm12'],
    #              [logZ_rbf, logZ_m12],
    #              [logZerr_rbf, logZerr_m12])
    # plt.ylabel("log Z")
    # plt.show()
