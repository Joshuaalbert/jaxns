from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior, GaussianProcessKernelPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.gaussian_process.kernels import RBF, M12, M32
from jax.scipy.linalg import solve_triangular
from jax import random, jit
from jax import numpy as jnp
from timeit import default_timer
import pylab as plt


def main(kernel):
    print(("Working on Kernel: {}".format(kernel.__class__.__name__)))

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        # U, S, Vh = jnp.linalg.svd(cov)
        log_det = jnp.sum(jnp.log(jnp.diag(L)))  # jnp.sum(jnp.log(S))#
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        # U S Vh V 1/S Uh
        # pinv = (Vh.T.conj() * jnp.where(S!=0., jnp.reciprocal(S), 0.)) @ U.T.conj()
        maha = dx @ dx  # dx @ pinv @ dx#solve_triangular(L, dx, lower=True)
        log_likelihood = -0.5 * x.size * jnp.log(2. * jnp.pi) \
                         - log_det \
                         - 0.5 * maha
        # print(log_likelihood)
        return log_likelihood

    N = 100
    X = jnp.linspace(-2., 2., N)[:, None]
    true_sigma, true_l, true_uncert = 1., 0.2, 0.2
    data_mu = jnp.zeros((N,))
    prior_cov = RBF()(X, X, true_l, true_sigma) + 1e-13 * jnp.eye(N)

    Y = jnp.linalg.cholesky(prior_cov) @ random.normal(random.PRNGKey(0), shape=(N,)) + data_mu
    Y_obs = Y + true_uncert * random.normal(random.PRNGKey(1), shape=(N,))
    Y_obs = jnp.where((jnp.arange(N) > 25) & (jnp.arange(N) < 30),
                      random.normal(random.PRNGKey(1), shape=(N,)),
                      Y_obs)

    def log_likelihood(K, uncert, **kwargs):
        """
        P(Y|sigma, half_width) = N[Y, f, K]
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

    l = UniformPrior('l', 0., 2.)
    uncert = UniformPrior('uncert', 0., 2.)
    sigma = UniformPrior('sigma', 0., 2.)
    cov = GaussianProcessKernelPrior('K', kernel, X, l, sigma)
    prior_chain = PriorChain().push(uncert).push(cov)

    def run_with_n(n):
        ns = NestedSampler(log_likelihood, prior_chain, sampler_name='multi_ellipsoid', num_live_points=n,
                           max_samples=1e5,
                           collect_samples=True,
                           sampler_kwargs=dict(depth=3, num_slices=5),
                           predict_f=predict_f,
                           predict_fvar=predict_fvar)
        @jit
        def run(key):
            return ns(key=key, termination_frac=0.01)

        t0 = default_timer()
        # with disable_jit():
        results = run(random.PRNGKey(6))
        print(results.efficiency)
        print("Time to execute (including compile): {}".format(default_timer() - t0))
        t0 = default_timer()
        results = run(random.PRNGKey(6))
        print(results.efficiency)
        print("Time to execute (not including compile): {}".format((default_timer() - t0)))
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
    logZ_m32, logZerr_m32 = main(M32())
    plt.errorbar(['rbf', 'm12', 'm32'], [logZ_rbf, logZ_m12, logZ_m32], [logZerr_rbf, logZerr_m12, logZerr_m32])
    plt.ylabel("log Z")
    plt.legend()
    plt.show()
