import logging

import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp

from jaxns.internals.types import float_type
from jaxns.nested_sampling.prior import PriorModelGen, Prior, parse_prior, compute_log_likelihood, InvalidDistribution, \
    InvalidPriorName, prepare_input, distribution_chain
from jaxns.nested_sampling.special_priors import Bernoulli, Categorical, Poisson, Beta, ForcedIdentifiability

logger = logging.getLogger('jaxns')
tfpd = tfp.distributions
tfpb = tfp.bijectors


def test_single_prior():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2), name='x')
        y = yield Prior(tfpd.Normal(loc=2, scale=x), name='x')
        z = x + y
        return z

    try:
        parse_prior(prior_model)
        assert False
    except InvalidPriorName:
        assert True

    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2), name='x')
        y = yield Prior(tfpd.Normal(loc=2, scale=x), name='y')
        z = x + y
        return z

    U, X = parse_prior(prior_model)
    U_ndims = U.shape[0]
    assert isinstance(X, dict)
    assert len(X) == 2
    assert U_ndims == 2

    U = random.uniform(random.PRNGKey(42), shape=(U_ndims,), dtype=float_type)

    def log_likelihood(Z):
        return Z - jnp.sum(Z ** 2)

    log_L = compute_log_likelihood(U, prior_model, log_likelihood=log_likelihood)
    assert not jnp.isnan(log_L)


def test_no_quantile_prior():
    def prior_model() -> PriorModelGen:
        z = yield Prior(tfpd.StudentT(df=1, loc=0., scale=1.))
        return z

    try:
        parse_prior(prior_model)
        assert False
    except InvalidDistribution:
        assert True


def test_prior_model_basic():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z, z ** 2

    U, X = parse_prior(prior_model)
    U_ndims = U.shape[0]
    assert isinstance(X, dict)
    assert len(X) == 0
    assert U_ndims == 2

    U = random.uniform(random.PRNGKey(42), shape=(U_ndims,), dtype=float_type)
    output = prepare_input(U, prior_model)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert jnp.allclose(output[1], output[0] ** 2)

    def log_likelihood(Z, Z2):
        return Z - jnp.sum(Z2)

    log_L = compute_log_likelihood(U, prior_model, log_likelihood=log_likelihood)
    assert not jnp.isnan(log_L)


def test_distribution_chain():
    d = tfpd.MultivariateNormalTriL(loc=jnp.zeros(5), scale_tril=jnp.eye(5))
    chain = distribution_chain(d)
    assert len(chain) == 2
    assert isinstance(chain[0], tfpd.Sample)
    assert isinstance(chain[1], tfpd.MultivariateNormalTriL)

    chain = distribution_chain(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    assert len(chain) == 1
    assert isinstance(chain[0], tfpd.Normal)


def test_priors():
    d = Prior(tfpd.Uniform(low=jnp.zeros(5), high=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Laplace(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Cauchy(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.HalfNormal(scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.HalfCauchy(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Gamma(concentration=jnp.ones(5), rate=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Gumbel(loc=jnp.ones(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == ()
    assert d.shape == (5,)


    d = Prior(tfpd.MultivariateNormalTriL(loc=jnp.zeros(5), scale_tril=jnp.eye(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)


def test_special_priors():
    d = Bernoulli(probs=jnp.ones(5), name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Categorical(parametrisation='gumbel_max', probs=jnp.ones(5), name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == ()

    d = Categorical(parametrisation='cdf', probs=jnp.ones(5), name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == ()
    assert d.shape == ()

    d = Poisson(rate=jnp.ones(5), name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Beta(concentration0=jnp.ones(5), concentration1=jnp.ones(5), name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (10, 5)
    assert d.shape == (10, 5)
