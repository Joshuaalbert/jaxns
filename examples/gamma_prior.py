from jax import numpy as jnp, random, jit, vmap
from jax.scipy.special import gammaln
from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import UniformPrior, PriorChain, NormalPrior, GammaPrior
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, resample
import numpy as np
import pylab as plt

def main():

    true_k = 1.
    true_theta = 0.5

    def log_likelihood(gamma, **kwargs):
        """
        Unit likelihood
        """
        return 0.

    gamma = GammaPrior('gamma', true_k, true_theta)
    prior_chain = gamma.prior_chain()
    ns = NestedSampler(loglikelihood=log_likelihood,
                       prior_chain=prior_chain)

    results = ns(random.PRNGKey(32564))
    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)

    samples = resample(random.PRNGKey(43083245),results.samples, results.log_p, S=int(results.ESS))

    plt.hist(samples['gamma'], bins='auto', ec='blue', alpha=0.5, density=True, fc='none')

    _gamma = np.random.gamma(true_k, true_theta, size=100000)

    plt.hist(_gamma, bins='auto', ec='orange', alpha=0.5, density=True, fc='none')
    plt.show()



if __name__ == '__main__':
    main()