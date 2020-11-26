from jaxns.examples.ray_integral_from_jones.build_prior import build_prior
from jaxns.gaussian_process.kernels import RBF, M12
from jaxns.examples.ray_integral_from_jones.generate_data import rbf_dtec
from jaxns.nested_sampling import NestedSampler
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax import random, jit
from jax.scipy.linalg import solve_triangular
from jax import numpy as jnp
import pylab as plt
from timeit import default_timer


def main(kernel):
    def log_mvnormal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        # maha = dx @ jnp.linalg.solve(cov, dx)
        maha = dx @ dx
        # logdet = jnp.log(jnp.linalg.det(cov))
        logdet = jnp.sum(jnp.diag(L))
        log_prob = -0.5 * x.size * jnp.log(2. * jnp.pi) - logdet - 0.5 * maha
        return log_prob

    def log_normal(x, mean, uncert):

        dx = (x - mean)/uncert
        # maha = dx @ jnp.linalg.solve(cov, dx)
        maha = dx @ dx
        # logdet = jnp.log(jnp.linalg.det(cov))
        logdet = x.size * jnp.log(uncert)
        log_prob = -0.5 * x.size * jnp.log(2. * jnp.pi) - logdet - 0.5 * maha
        return log_prob

    true_height, true_width, true_sigma, true_l, true_uncert = 200., 100., 1., 10., 0.5
    nant = 2
    ndir = 10
    X, Y_dtec, Y, Y_obs, tec_conv = rbf_dtec(nant, ndir, true_height, true_width, true_sigma, true_l, true_uncert)
    a = X[:, 0:3]
    k = X[:, 3:6]
    x0 = a[0, :]

    def log_likelihood(Y, uncert, **kwargs):
        """
        P(Y|sigma, half_width) = N[Y, mu, K]
        Args:
            sigma:
            l:

        Returns:

        """
        return jnp.sum(log_normal(Y_obs.reshape((-1,)), Y.reshape((-1,)), uncert))

    def predict_f(dtec, **kwargs):
        return dtec

    def predict_fvar(dtec, **kwargs):
        return dtec ** 2

    def tec_to_dtec(tec):
        tec = tec.reshape((nant, ndir))
        dtec = jnp.reshape(tec - tec[0, :], (-1,))
        return dtec

    prior_chain = build_prior(X, kernel, tec_to_dtec, x0, tec_conv)
    # print(prior_chain)
    # prior_chain.test_prior(random.PRNGKey(876136),1000, log_likelihood=log_likelihood)

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='multi_ellipsoid', predict_f=predict_f,
                       predict_fvar=predict_fvar)

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stochastic_uncertainty=False,
                      sampler_kwargs=dict(depth=4))

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print("Efficiency", results.efficiency)
        print("Time to run (including compile)", default_timer() - t0)
        print("Time efficiency normalised", (default_timer() - t0) * results.efficiency)
        # t0 = default_timer()
        # results = run(random.PRNGKey(1))
        # print("Efficiency",results.efficiency)
        # print("Time to run (no compile)", default_timer() - t0)
        # print("Time efficiency normalised", (default_timer() - t0)*results.efficiency)
        return results

    for n in [100]:
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
    #                   stochastic_uncertainty=False,
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
    # plt.scatter(jnp.arange(Y.size),Y_obs, marker='+', label='data')
    plt.scatter(jnp.arange(Y_dtec.size),Y_dtec, marker="o", label='underlying')
    plt.errorbar(jnp.arange(Y_dtec.size), results.marginalised['predict_f'], yerr=fstd, label='marginalised')
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.legend()
    plt.show()

    # plot_samples_development(results,save_name='./ray_integral_solution.mp4')
    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr


if __name__ == '__main__':
    logZ_rbf, logZerr_rbf = main(RBF())
    logZ_m12, logZerr_m12 = main(M12())
    plt.errorbar(['rbf', 'm12'],
                 [logZ_rbf, logZ_m12],
                 [logZerr_rbf, logZerr_m12])
    plt.ylabel("log Z")
    plt.show()
