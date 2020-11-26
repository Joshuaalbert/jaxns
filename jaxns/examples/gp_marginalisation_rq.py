from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior, GaussianProcessKernelPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jaxns.gaussian_process.kernels import RationalQuadratic
from jax.scipy.linalg import solve_triangular
from jax import random, jit, disable_jit
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
    data_mu = jnp.zeros((N,))
    prior_cov = RationalQuadratic()(X, X, true_l, true_sigma, true_alpha)
    Y = jnp.linalg.cholesky(prior_cov) @ random.normal(random.PRNGKey(0), shape=(N,)) + data_mu
    Y_obs = Y + true_uncert * random.normal(random.PRNGKey(1), shape=(N,))

    # Y_obs = jnp.where((jnp.arange(N) > 50) & (jnp.arange(N) < 60),
    #                   random.normal(random.PRNGKey(1), shape_dict=(N, )),
    #                   Y_obs)

    # plt.scatter(X[:, 0], Y_obs, label='data')
    # plt.plot(X[:, 0], Y, label='underlying')
    # plt.legend()
    # plt.show()

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
        log_prob = log_normal(Y_obs, mu, K + data_cov)
        # print(log_prob)
        return log_prob

    def predict_f(K, uncert, **kwargs):
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return mu + K @ jnp.linalg.solve(K + data_cov, Y_obs)

    def predict_fvar(K, uncert, **kwargs):
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        mu = jnp.zeros_like(Y_obs)
        return jnp.diag(K - K @ jnp.linalg.solve(K + data_cov, K))

    prior_chain = PriorChain() \
        .push(GaussianProcessKernelPrior('K', RationalQuadratic(), X,
                                         UniformPrior('l', 0., 4.),
                                         UniformPrior('sigma', 0., 4.),
                                         UniformPrior('alpha', 0., 4.))) \
        .push(UniformPrior('uncert', 0., 2.))

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
                      stoachastic_uncertainty=False,
                      sampler_kwargs=dict(depth=4))

        results = run()
        return results

    for n in [200]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.title("Kernel: {}".format(RationalQuadratic.__name__))
    plt.ylabel('log Z')
    plt.show()

    plt.scatter(X[:, 0], Y_obs, label='data')
    plt.plot(X[:, 0], Y, label='underlying')
    plt.plot(X[:, 0], results.marginalised['predict_f'], label='marginalised')
    plt.plot(X[:, 0], results.marginalised['predict_f'] + jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted',
             c='black')
    plt.plot(X[:, 0], results.marginalised['predict_f'] - jnp.sqrt(results.marginalised['predict_fvar']), ls='dotted',
             c='black')
    plt.title("Kernel: {}".format(RationalQuadratic.__name__))
    plt.legend()
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr


if __name__ == '__main__':
    main()
