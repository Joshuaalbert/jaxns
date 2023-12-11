import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp, vmap

from jaxns.framework.bases import PriorModelGen
from jaxns.framework.distribution import InvalidDistribution, distribution_chain
from jaxns.framework.ops import parse_prior, prepare_input, compute_log_likelihood
from jaxns.framework.prior import Prior, InvalidPriorName
from jaxns.framework.special_priors import Bernoulli, Categorical, Poisson, Beta, ForcedIdentifiability, \
    UnnormalisedDirichlet
from jaxns.internals.types import float_type

tfpd = tfp.distributions


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
        z = yield Prior(tfpd.VonMises(loc=0., concentration=1.))
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
    d = Prior(jnp.zeros(5))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (0,)
    assert d.shape == (5,)

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

    d = Prior(tfpd.StudentT(df=1.5, loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Beta(concentration0=jnp.ones(5), concentration1=jnp.ones(5)))
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

    probs = jnp.asarray([1, 2, 3, 2, 1], float_type)
    d = Categorical(parametrisation='gumbel_max', probs=probs, name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == ()
    x = vmap(lambda key: d.forward(random.uniform(key, shape=d.base_shape)))(random.split(random.PRNGKey(42), 10000))
    assert jnp.all(x >= 0)
    assert jnp.all(x < 5)
    assert jnp.any(x == 0)
    assert jnp.any(x == 4)
    count = jnp.bincount(x)
    count /= jnp.sum(count)
    assert jnp.allclose(count, probs / jnp.sum(probs), atol=1e-2)

    probs = jnp.asarray([1, 2, 3, 2, 1], float_type)
    d = Categorical(parametrisation='cdf', probs=probs, name='x')
    print(d)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape
    assert d.base_shape == ()
    assert d.shape == ()
    u_array = jnp.linspace(0., 1., 10000)
    x = vmap(lambda u: d.forward(u))(u_array)
    assert jnp.all(x >= 0)
    assert jnp.all(x < 5)
    assert jnp.any(x == 0)
    assert jnp.any(x == 4)
    count = jnp.bincount(x)
    count /= jnp.sum(count)
    assert jnp.allclose(count, probs / jnp.sum(probs), atol=1e-2)

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

    u_input = vmap(lambda key: random.uniform(key, shape=d.base_shape))(random.split(random.PRNGKey(42), 1))
    x = vmap(lambda u: d.forward(u))(u_input)
    u = vmap(lambda x: d.inverse(x))(x)
    assert jnp.allclose(u, u_input)

    d = UnnormalisedDirichlet(concentration=jnp.ones(5), name='x')
    print(d)
    assert d.base_shape == (5,)
    assert d.shape == (5,)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape

    u_input = vmap(lambda key: random.uniform(key, shape=d.base_shape))(random.split(random.PRNGKey(42), 10))
    x = vmap(lambda u: d.forward(u))(u_input)
    assert jnp.all(x > 0.)
    u = vmap(lambda x: d.inverse(x))(x)
    assert jnp.allclose(u, u_input)

    d = UnnormalisedDirichlet(concentration=jnp.ones((3, 5)), name='x')
    print(d)
    assert d.base_shape == (3, 5)
    assert d.shape == (3, 5)
    assert d.forward(jnp.ones(d.base_shape, float_type)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, float_type)).shape == d.shape

    u_input = vmap(lambda key: random.uniform(key, shape=d.base_shape))(random.split(random.PRNGKey(42), 10))
    x = vmap(lambda u: d.forward(u))(u_input)
    assert jnp.all(x > 0.)
    u = vmap(lambda x: d.inverse(x))(x)
    assert jnp.allclose(u, u_input)
