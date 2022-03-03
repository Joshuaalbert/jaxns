from jaxns.nested_sampler.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior, GaussianProcessKernelPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.gaussian_process.kernels import RBF, M12, M32
from jaxns.utils import marginalise_dynamic, summary
from jax.scipy.linalg import solve_triangular
from jax import random, jit
from jax import numpy as jnp
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

    def predict_f(uncert, l, sigma):
        K = kernel(X, X, l, sigma)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(uncert, l, sigma):
        K = kernel(X, X, l, sigma)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    with PriorChain() as prior_chain:
        l = UniformPrior('l', 0., 2.)
        uncert = UniformPrior('uncert', 0., 2.)
        sigma = UniformPrior('sigma', 0., 2.)
        cov = GaussianProcessKernelPrior('K', kernel, X, l, sigma)

    ns = NestedSampler(log_likelihood,
                       prior_chain,
                       num_live_points=20)
    results = jit(ns)(key=random.PRNGKey(42))

    predict_f = marginalise_dynamic(random.PRNGKey(42), results.samples,results.log_dp_mean,
                                                  results.ESS, predict_f)

    predict_fvar = marginalise_dynamic(random.PRNGKey(42), results.samples, results.log_dp_mean,
                                    results.ESS, predict_fvar)

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.plot(X[:, 0], predict_f, label='marginalised')
    plt.plot(X[:, 0], predict_f + jnp.sqrt(predict_fvar), ls='dotted',
             c='black')
    plt.plot(X[:, 0], predict_f - jnp.sqrt(predict_fvar), ls='dotted',
             c='black')
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.legend()
    plt.show()
    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.log_Z_mean, results.log_Z_uncert


if __name__ == '__main__':
    logZ_rbf, logZerr_rbf = main(RBF())
    logZ_m12, logZerr_m12 = main(M12())
    logZ_m32, logZerr_m32 = main(M32())
    plt.errorbar(['rbf', 'm12', 'm32'], [logZ_rbf, logZ_m12, logZ_m32], [logZerr_rbf, logZerr_m12, logZerr_m32])
    plt.ylabel("log Z")
    plt.legend()
    plt.show()
