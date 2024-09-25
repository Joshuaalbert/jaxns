import warnings
from typing import List, Tuple, Union

import jax.random
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp, vmap

from jaxns.framework.bases import PriorModelGen, BaseAbstractPrior
from jaxns.framework.ops import parse_prior, prepare_input, compute_log_likelihood
from jaxns.framework.prior import Prior, InvalidPriorName
from jaxns.framework.special_priors import Bernoulli, Categorical, Poisson, Beta, ForcedIdentifiability, \
    UnnormalisedDirichlet, _poisson_quantile_bisection, _poisson_quantile, Empirical, TruncationWrapper, \
    ExplicitDensityPrior
from jaxns.framework.wrapped_tfp_distribution import InvalidDistribution, distribution_chain
from jaxns.internals.mixed_precision import mp_policy

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

    U, X, W = parse_prior(prior_model)
    U_ndims = U.shape[0]
    assert isinstance(X, dict)
    assert len(X) == 2
    assert U_ndims == 2
    assert sum([np.size(w) for w in W]) == U_ndims

    def log_likelihood(Z):
        return Z - jnp.sum(Z ** 2)

    log_L = compute_log_likelihood(W, prior_model, log_likelihood=log_likelihood)
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

    U, X, W = parse_prior(prior_model)
    U_ndims = U.shape[0]
    assert isinstance(X, dict)
    assert len(X) == 0
    assert U_ndims == 2
    assert sum([np.size(w) for w in W]) == U_ndims

    output = prepare_input(W, prior_model)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert jnp.allclose(output[1], output[0] ** 2)

    def log_likelihood(Z, Z2):
        return Z - jnp.sum(Z2)

    log_L = compute_log_likelihood(W, prior_model, log_likelihood=log_likelihood)
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
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (0,)
    assert d.shape == (5,)

    d = Prior(tfpd.Uniform(low=jnp.zeros(5), high=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Laplace(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Cauchy(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.StudentT(df=1.5, loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Beta(concentration0=jnp.ones(5), concentration1=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.HalfNormal(scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.HalfCauchy(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Gamma(concentration=jnp.ones(5), rate=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Gumbel(loc=jnp.ones(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == ()
    assert d.shape == (5,)

    d = Prior(tfpd.MultivariateNormalTriL(loc=jnp.zeros(5), scale_tril=jnp.eye(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, mp_policy.measure_dtype)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)


@pytest.fixture(scope='module')
def mock_special_priors() -> List[
    Tuple[BaseAbstractPrior, Tuple[Union[float, int], Union[float, int]], Tuple[int, ...]]]:
    # prior, (min, max), shape
    return [
        (Bernoulli(probs=jnp.ones(5), name='x'), (0, 1), (5,)),
        (Categorical(parametrisation='gumbel_max', probs=jnp.ones(5), name='x'), (0, 4), ()),
        (Categorical(parametrisation='cdf', probs=jnp.ones(5), name='x'), (0, 4), ()),
        (Poisson(rate=jnp.ones(5), name='x'), (0, 100), (5,)),
        (Beta(concentration0=jnp.ones(5), concentration1=jnp.ones(5), name='x'), (0, 1), (5,)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), name='x'), (0, 1), (10, 5)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), fix_left=True, name='x'), (0, 1), (10, 5)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), fix_right=True, name='x'), (0, 1), (10, 5)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), fix_left=True, fix_right=True, name='x'),
         (0, 1), (10, 5)),
        (UnnormalisedDirichlet(concentration=jnp.ones(5), name='x'), (0, jnp.inf), (5,)),
        (UnnormalisedDirichlet(concentration=jnp.ones((3, 5)), name='x'), (0, jnp.inf), (3, 5)),
    ]


def test_special_priors(mock_special_priors: List[BaseAbstractPrior]):
    for prior, (vmin, vmax), shape in mock_special_priors:
        print(f"Testing {prior.__class__}")
        x = prior.forward(jnp.ones(prior.base_shape, mp_policy.measure_dtype))
        assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
        assert jnp.all(x >= vmin)
        assert jnp.all(x <= vmax)
        assert x.shape == shape
        assert x.shape == prior.shape
        x = prior.forward(jnp.zeros(prior.base_shape, mp_policy.measure_dtype))
        assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
        assert jnp.all(x >= vmin)
        assert jnp.all(x <= vmax)
        assert x.shape == shape
        assert x.shape == prior.shape

        u_input = vmap(lambda key: random.uniform(key, shape=prior.base_shape))(random.split(random.PRNGKey(42), 10))
        x = vmap(lambda u: prior.forward(u))(u_input)
        assert jnp.all(x >= vmin)
        assert jnp.all(x <= vmax)

        try:
            u = vmap(lambda x: prior.inverse(x))(x)
            assert u.shape[1:] == prior.base_shape

            if prior.dtype in [jnp.bool_, jnp.int32, jnp.int64]:
                continue
            print(u)
            print(u_input)
            assert jnp.allclose(u, u_input)
        except NotImplementedError:
            warnings.warn(f"Skipping inverse test for {prior.__class__}")
            pass


@pytest.mark.parametrize("rate, error", (
        [2.0, 1.],
        [10., 1.],
        [100., 1.],
        [1000., 1.],
        [10000., 1.]
)
                         )
def test_poisson_quantile_bisection(rate, error):
    U = jnp.linspace(0., 1. - np.spacing(1.), 1000)
    x, x_results = _poisson_quantile_bisection(U, rate, unroll=False)
    diff_last_two = jnp.abs(x_results[..., -1] - x_results[..., -2])

    # Make sure less than 1 apart
    assert jnp.all(diff_last_two <= error)


@pytest.mark.parametrize("rate", [2.0, 10., 100., 1000., 10000.])
def test_poisson_quantile(rate):
    U = jnp.linspace(0., 1. - np.spacing(1.), 10000)
    x = _poisson_quantile(U, rate)
    assert jnp.all(jnp.isfinite(x))


def test_forced_identifiability():
    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.1, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.1, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.11, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.1, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)


def test_empirical():
    samples = jax.random.normal(jax.random.PRNGKey(42), shape=(5, 2000), dtype=mp_policy.measure_dtype)
    prior = Empirical(samples=samples, resolution=100, name='x')
    assert prior._percentiles.shape == (101, 5)

    x = prior.forward(jnp.ones(prior.base_shape, mp_policy.measure_dtype))
    assert x.shape == (5,)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    x = prior.forward(jnp.zeros(prior.base_shape, mp_policy.measure_dtype))
    assert x.shape == (5,)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))

    x = prior.forward(0.5 * jnp.ones(prior.base_shape, mp_policy.measure_dtype))
    np.testing.assert_allclose(x, 0., atol=0.05)

    u_input = vmap(lambda key: random.uniform(key, shape=prior.base_shape, dtype=mp_policy.measure_dtype))(
        random.split(random.PRNGKey(42), 1000))
    x = vmap(lambda u: prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))

    u = vmap(lambda x: prior.inverse(x))(x)
    # print(u)
    np.testing.assert_allclose(u, u_input)
    assert u.shape[1:] == prior.base_shape


def test_truncation_wrapper():
    prior = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    trancated_prior = TruncationWrapper(prior=prior, low=0., high=1.)

    x = trancated_prior.forward(jnp.ones(trancated_prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    x = trancated_prior.forward(jnp.zeros(trancated_prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    u_input = vmap(lambda key: random.uniform(key, shape=trancated_prior.base_shape))(
        random.split(random.PRNGKey(42), 1000))
    x = vmap(lambda u: trancated_prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: trancated_prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)

    prior = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    trancated_prior = TruncationWrapper(prior=prior, low=-jnp.inf, high=1.)

    x = trancated_prior.forward(jnp.ones(trancated_prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= -jnp.inf)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    x = trancated_prior.forward(jnp.zeros(trancated_prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= -jnp.inf)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    u_input = vmap(lambda key: random.uniform(key, shape=trancated_prior.base_shape))(
        random.split(random.PRNGKey(42), 1000))
    x = vmap(lambda u: trancated_prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= -jnp.inf)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: trancated_prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)

    prior = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=0.01 * jnp.ones(5)))
    trancated_prior = TruncationWrapper(prior=prior, low=0., high=1.)

    x = trancated_prior.forward(jnp.ones(trancated_prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    x = trancated_prior.forward(jnp.zeros(trancated_prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    u_input = vmap(lambda key: random.uniform(key, shape=trancated_prior.base_shape))(
        random.split(random.PRNGKey(42), 1000))
    x = vmap(lambda u: trancated_prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: trancated_prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)


def test_explicit_density_prior():
    resolution = 10
    density = jnp.ones((resolution + 1, resolution))
    axes = (jnp.linspace(0, 1, resolution + 1), jnp.linspace(0, 1, resolution))
    prior = ExplicitDensityPrior(axes=axes, density=density, regular_grid=True)

    x = prior.forward(jnp.ones(prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x == 1.)
    assert x.shape == (2,)

    x = prior.forward(jnp.zeros(prior.base_shape, mp_policy.measure_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x == 0.)
    assert x.shape == (2,)

    u_input = vmap(lambda key: random.uniform(key, shape=prior.base_shape))(random.split(random.PRNGKey(42), 1000))
    x = vmap(lambda u: prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)

    assert jnp.all(jnp.isfinite(jax.vmap(prior.log_prob)(x)))
