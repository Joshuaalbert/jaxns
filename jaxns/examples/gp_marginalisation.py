from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, MVNDiagPrior, UniformPrior, DeltaPrior, GMMDiagPrior, \
    ForcedIdentifiabilityPrior, GaussianProcessKernelPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.gaussian_process.kernels import RBF, M12
from jax.scipy.linalg import solve_triangular
from jax import random, jit, disable_jit
from jax import numpy as jnp
import pylab as plt


def main(kernel):
    print(("Working on Kernel: {}".format(kernel.__class__.__name__)))
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        log_likelihood = -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx
        print(log_likelihood)
        return log_likelihood

    N = 100
    X = jnp.linspace(-2., 2., N)[:, None]
    true_sigma, true_l, true_uncert = 1., 0.2, 0.2
    data_mu = jnp.zeros((N,))
    prior_cov = RBF()(X, X, true_l, true_sigma) + 1e-13*jnp.eye(N)
    # print(jnp.linalg.cholesky(prior_cov), jnp.linalg.eigvals(prior_cov))
    # return
    Y = jnp.linalg.cholesky(prior_cov) @ random.normal(random.PRNGKey(0), shape=(N,)) + data_mu
    Y_obs = Y + true_uncert * random.normal(random.PRNGKey(1), shape=(N,))
    Y_obs = jnp.where((jnp.arange(N) > 50) & (jnp.arange(N) < 60),
                      random.normal(random.PRNGKey(1), shape=(N,)),
                      Y_obs)

    # plt.scatter(X[:, 0], Y_obs, label='data')
    # plt.plot(X[:, 0], Y, label='underlying')
    # plt.legend()
    # plt.show()

    def log_likelihood(K, uncert, **kwargs):
        """
        P(Y|sigma, half_width) = N[Y, mu, K]
        Args:
            sigma:
            l:

        Returns:

        """
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return log_normal(Y_obs, mu, K + data_cov)

    def predict_f(K, uncert, **kwargs):
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(K, uncert, **kwargs):
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    # l_loc = ForcedIdentifiabilityPrior('l_loc', 2, 0., 2., tracked=False)
    # l_std = UniformPrior('l_std', jnp.array([[0., 0.]]).T, jnp.array([[2., 2.]]).T, tracked=False)
    # l_pi = UniformPrior('l_pi', jnp.array([0., 0.]), jnp.array([1., 1.]), tracked=False)
    # l = GMMDiagPrior('l', l_pi, l_loc, l_std)
    l = UniformPrior('l', 0., 2.)
    uncert = UniformPrior('uncert', 0., 2.)
    sigma = UniformPrior('sigma', 0., 2.)
    cov = GaussianProcessKernelPrior('K',kernel, X, l, sigma)
    prior_chain = PriorChain().push(uncert).push(cov)

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='multi_ellipsoid', predict_f=predict_f,
                       predict_fvar=predict_fvar)

    def run_with_n(n):
        @jit
        def run():
            return ns(key=random.PRNGKey(0),
                      num_live_points=n,
                      max_samples=1e4,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False)

        results = run()
        return results

    for n in [100]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.ylabel('log Z')
    plt.show()

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.plot(X[:, 0], results.marginalised['predict_f'], label='marginalised')
    plt.plot(X[:, 0], results.marginalised['predict_f'] + jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted',
             c='black')
    plt.plot(X[:, 0], results.marginalised['predict_f'] - jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted',
             c='black')
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.legend()
    plt.show()
    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr


if __name__ == '__main__':
    logZ_rbf, logZerr_rbf = main(RBF())
    logZ_m12, logZerr_m12 = main(M12())
    plt.errorbar(['rbf', 'm12'], [logZ_rbf, logZ_m12], [logZerr_rbf, logZerr_m12])
    plt.ylabel("log Z")
    plt.legend()
    plt.show()
