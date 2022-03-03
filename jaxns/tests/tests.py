import numpy as np

from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln
from jax import random, jit
from jax import numpy as jnp

from jaxns.utils import resample, _bit_mask
from jaxns.nested_sampler.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, MVNPrior, GammaPrior


def test_resample():
    x = random.normal(key=random.PRNGKey(0), shape=(50,))
    logits = -jnp.ones(50)
    samples = {'x': x}
    assert jnp.all(resample(random.PRNGKey(0), samples, logits)['x'] == resample(random.PRNGKey(0), x, logits))


def test_nested_sampling():
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = solve_triangular(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 4
    prior_mu = 2 * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    log_likelihood = lambda x, **kwargs: log_normal(x, data_mu, data_cov)

    with PriorChain() as prior_chain:
        MVNPrior('x', prior_mu, prior_cov)

    for sampler in ['slice']:
        ns = NestedSampler(log_likelihood, prior_chain, sampler_name=sampler)

        results = jit(ns)(key=random.PRNGKey(42))

        assert jnp.abs(results.log_Z_mean - true_logZ) < 3. * results.log_Z_uncert


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


def test_bit_mask():
    assert _bit_mask(1, width=2) == [1, 0]
    assert _bit_mask(2, width=2) == [0, 1]
    assert _bit_mask(3, width=2) == [1, 1]
