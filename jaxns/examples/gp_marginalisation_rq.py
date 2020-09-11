from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.gaussian_process.kernels import rational_quadratic
from jax.scipy.linalg import solve_triangular
from jax import random, jit
from jax import numpy as jnp
import pylab as plt

def main():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    N = 100
    X = jnp.linspace(-2., 2., N)[:, None]
    true_alpha, true_sigma, true_l, true_uncert = 1., 1., 0.2, 0.25
    data_mu = jnp.zeros((N, ))
    prior_cov = rational_quadratic(X, X, true_sigma, true_alpha, true_l)
    Y = jnp.linalg.cholesky(prior_cov) @ random.normal(random.PRNGKey(0), shape=(N, )) + data_mu
    Y_obs = Y + true_uncert * random.normal(random.PRNGKey(1), shape=(N, ))
    # Y_obs = jnp.where((jnp.arange(N) > 50) & (jnp.arange(N) < 60),
    #                   random.normal(random.PRNGKey(1), shape_dict=(N, )),
    #                   Y_obs)

    # plt.scatter(X[:, 0], Y_obs, label='data')
    # plt.plot(X[:, 0], Y, label='underlying')
    # plt.legend()
    # plt.show()

    def log_likelihood(sigma, l, alpha, uncert, **kwargs):
        """
        P(Y|sigma, half_width) = N[Y, mu, K]
        Args:
            sigma:
            l:

        Returns:

        """
        K = rational_quadratic(X, X, sigma, alpha, l)
        data_cov = jnp.square(uncert)*jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return log_normal(Y_obs, mu , K + data_cov)

    def predict_f(sigma, l, alpha,  uncert, **kwargs):
        K = rational_quadratic(X, X, sigma, alpha, l)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(sigma, l, alpha, uncert, **kwargs):
        K = rational_quadratic(X, X, sigma, alpha, l)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    prior_chain = PriorChain()\
        .push(UniformPrior('sigma', 0., 4.))\
        .push(UniformPrior('l', 0., 4.))\
        .push(UniformPrior('alpha', 0., 4.))\
        .push(UniformPrior('uncert',0., 2.))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='ellipsoid', predict_f=predict_f, predict_fvar=predict_fvar)

    def run_with_n(n):
        @jit
        def run():
            return ns(key=random.PRNGKey(0),
                      num_live_points=n,
                      max_samples=1e3,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=True)

        results = run()
        return results

    for n in [200]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.title("Kernel: {}".format(rational_quadratic.__name__))
    plt.ylabel('log Z')
    plt.show()

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.plot(X[:,0], results.marginalised['predict_f'], label='marginalised')
    plt.plot(X[:,0], results.marginalised['predict_f'] + jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    plt.plot(X[:,0], results.marginalised['predict_f'] - jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    plt.title("Kernel: {}".format(rational_quadratic.__name__))
    plt.legend()
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr



if __name__ == '__main__':
    main()
