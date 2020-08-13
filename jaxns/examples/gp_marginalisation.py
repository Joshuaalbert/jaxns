from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, MVNDiagPrior, UniformPrior, DeltaPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax.scipy.linalg import solve_triangular
from jax import random, jit, disable_jit
from jax import numpy as jnp
import pylab as plt

def rbf(x, sigma, l):
    # r2_ij = sum_k (x_ik - x_jk)^2
    #       = sum_k x_ik^2 - 2 x_jk x_ik + x_jk^2
    #       = sum_k x_ik^2 + x_jk^2 - 2 X X^T
    x = x/l
    x2 = jnp.sum(jnp.square(x), axis=1)
    x2 = (x2[:, None] + x2[None,:]) - 2.*(x @ x.T)
    x2 = jnp.maximum(x2, 1e-36)
    return jnp.square(sigma)*jnp.exp(-0.5*x2) + 1e-6*jnp.eye(x.shape[0])

def main():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    N = 100
    X = jnp.linspace(-2., 2., N)[:, None]
    true_sigma, true_l, true_uncert = 1., 0.2, 1.
    data_mu = jnp.zeros((N, ))
    prior_cov = rbf(X, true_sigma, true_l)
    Y = jnp.linalg.cholesky(prior_cov) @ random.normal(random.PRNGKey(0), shape=(N, )) + data_mu
    Y_obs = Y + true_uncert * random.normal(random.PRNGKey(1), shape=(N, ))
    Y_obs = jnp.where((jnp.arange(N) > 50) & (jnp.arange(N) < 60),
                      random.normal(random.PRNGKey(1), shape=(N, )),
                      Y_obs)

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.legend()
    plt.show()

    def log_likelihood(sigma, l, uncert, **kwargs):
        """
        P(Y|sigma, l) = N[Y, mu, K]
        Args:
            sigma:
            l:

        Returns:

        """
        K = rbf(X, sigma, l)
        data_cov = jnp.square(uncert)*jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return log_normal(Y_obs, mu , K + data_cov)

    def predict_f(sigma, l, uncert, **kwargs):
        K = rbf(X, sigma, l)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(sigma, l, uncert, **kwargs):
        K = rbf(X, sigma, l)
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    prior_chain = PriorChain()\
        .push(UniformPrior('sigma', 0., 2.))\
        .push(UniformPrior('l', 0., 2.))\
        .push(UniformPrior('uncert',0., 2.))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='whitened_ellipsoid', predict_f=predict_f, predict_fvar=predict_fvar)

    def run_with_n(n):
        @jit
        def run():
            return ns(key=random.PRNGKey(0),
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=True)

        results = run()
        return results

    for n in [100]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.show()

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.plot(X[:,0], results.marginalised['predict_f'], label='marginalised')
    plt.plot(X[:,0], results.marginalised['predict_f'] + jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    plt.plot(X[:,0], results.marginalised['predict_f'] - jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted', c='black')
    plt.legend()
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)



if __name__ == '__main__':
    main()
