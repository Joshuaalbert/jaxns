from jax import numpy as jnp, random, jit, vmap
from jax.scipy.special import gammaln
from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import UniformPrior, PriorChain, NormalPrior, GammaPrior
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, resample
import numpy as np
import pylab as plt

def log_gamma_prob(gamma, k, theta):
    return (k-1) * jnp.log(gamma)  - gamma / theta - gammaln(k) - k * jnp.log(theta)

def main(num_samples=10):
    true_k = 0.1
    true_theta = 0.1

    _gamma = np.random.gamma(true_k, true_theta, size=num_samples)
    samples = jnp.asarray(np.random.poisson(_gamma, size=num_samples))

    prior_k = 100.
    prior_theta = 100.

    true_post_k = prior_k + jnp.sum(samples)
    true_post_theta = prior_theta / (num_samples * prior_theta + 1.)

    print(f"True posterior k = {true_post_k}")
    print(f"True posterior theta = {true_post_theta}")

    def log_likelihood(gamma, **kwargs):
        """
        Poisson likelihood.
        """
        return jnp.sum(samples * jnp.log(gamma) - gamma - gammaln(samples + 1))

    true_logZ = log_likelihood(1.) + log_gamma_prob(1., prior_k, prior_theta) - log_gamma_prob(1., true_post_k, true_post_theta)
    print(f"True Evidence = {true_logZ}")

    gamma = GammaPrior('gamma', prior_k, prior_theta)
    prior_chain = gamma.prior_chain()

    ns = NestedSampler(loglikelihood=log_likelihood, prior_chain=prior_chain, num_live_points=100)

    f = jit(ns)

    results = f(random.PRNGKey(3452345), 0.001)

    summary(results)

    plot_diagnostics(results)
    plot_cornerplot(results)

    samples = resample(random.PRNGKey(43083245),results.samples, results.log_p, S=int(results.ESS))

    plt.hist(samples['gamma'], bins='auto', ec='blue', alpha=0.5, density=True, fc='none')

    _gamma = np.random.gamma(true_post_k, true_post_theta, size=100000)

    plt.hist(_gamma, bins='auto', ec='orange', alpha=0.5, density=True, fc='none')
    plt.show()



if __name__ == '__main__':
    main(num_samples=10)