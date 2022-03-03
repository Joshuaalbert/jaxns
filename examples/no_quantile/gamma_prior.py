from jax import random
from jaxns.nested_sampler.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, GammaPrior
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, resample
import numpy as np
import pylab as plt

def main():

    true_k = 1.
    true_theta = 0.5

    print("True log(Z)=0")

    def log_likelihood(gamma, **kwargs):
        """
        Unit likelihood
        """
        return 0.

    with PriorChain() as prior_chain:
        gamma = GammaPrior('gamma', true_k, true_theta)

    ns = NestedSampler(loglikelihood=log_likelihood,
                       prior_chain=prior_chain)

    results = ns(random.PRNGKey(32564))
    summary(results)
    plot_diagnostics(results)
    plot_cornerplot(results)

    samples = resample(random.PRNGKey(43083245), results.samples, results.log_dp_mean, S=int(results.ESS))

    plt.hist(samples['gamma'], bins='auto', ec='blue', alpha=0.5, density=True, fc='none')

    _gamma = np.random.gamma(true_k, true_theta, size=100000)

    plt.hist(_gamma, bins='auto', ec='orange', alpha=0.5, density=True, fc='none')
    plt.show()



if __name__ == '__main__':
    main()