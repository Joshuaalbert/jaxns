import numpy as np
from jax import numpy as jnp, random, jit
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln

import jaxns.new_code.utils
from jaxns import resample, summary, plot_diagnostics, analytic_log_evidence, evidence_posterior_samples
from jaxns.nested_sampler.nested_sampling import compute_remaining_evidence



def test_gh21():
    num_samples = 10
    true_k = 1.
    true_theta = 0.5

    _gamma = np.random.gamma(true_k, true_theta, size=num_samples)
    samples = jnp.asarray(np.random.poisson(_gamma, size=num_samples))

    prior_k = 5.
    prior_theta = 0.3

    true_post_k = prior_k + jnp.sum(samples)
    true_post_theta = prior_theta / (num_samples * prior_theta + 1.)

    def log_likelihood(gamma, **kwargs):
        """
        Poisson likelihood.
        """
        return jnp.sum(samples * jnp.log(gamma) - gamma - gammaln(samples + 1))

    with PriorChain() as prior_chain:
        gamma = GammaPrior('gamma', prior_k, prior_theta)

    ns = NestedSampler(loglikelihood=log_likelihood, prior_chain=prior_chain)
    results = jit(ns)(random.PRNGKey(32564))

    samples = resample(random.PRNGKey(43083245), results.samples, results.log_dp_mean, S=int(results.ESS))

    sample_mean = jnp.mean(samples['gamma'], axis=0)

    true_mean = true_post_k * true_post_theta

    assert jnp.allclose(sample_mean, true_mean, atol=0.05)


def test_compute_remaining_evidence():
    # [a,b,-inf], 2 -> [a+b, b, -inf]
    log_dZ_mean = jnp.asarray([0., 1., -jnp.inf])
    sample_idx = 2
    expect = jnp.asarray([jnp.logaddexp(0, 1), 1, -jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)

    # [-inf, -inf,-inf], 0 -> [-inf, -inf, -inf]
    log_dZ_mean = jnp.asarray([-jnp.inf, -jnp.inf - jnp.inf])
    sample_idx = 0
    expect = jnp.asarray([-jnp.inf, -jnp.inf - jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)
