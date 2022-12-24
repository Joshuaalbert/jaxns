import numpy as np
from jax import random, numpy as jnp, vmap
from tensorflow_probability.substrates.jax import distributions as tfpd

from jaxns.internals.random import resample_indicies
from jaxns.internals.types import float_type
from jaxns.new_code.initial_state import init_sample_collection, get_uniform_init_live_points, \
    get_live_points_from_samples
from jaxns.new_code.model import Model
from jaxns.new_code.nested_sampler import tfpd
from jaxns.new_code.prior import PriorModelGen, Prior, parse_prior, compute_log_likelihood, InvalidDistribution, \
    transform
from jaxns.new_code.static_uniform import StaticUniform
from jaxns.new_code.types import NestedSamplerState, SampleCollection


def test_get_live_points_from_samples():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -z ** 2

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    U = jnp.stack(list(
        map(lambda x: x.flatten(), jnp.meshgrid(jnp.linspace(0., 1., 100), jnp.linspace(0., 1., 100), indexing='ij'))),
                  axis=1)
    X, log_L = vmap(model.forward)(U)
    assert jnp.bitwise_not(jnp.any(jnp.isnan(log_L)))

    n = 100
    live_points = get_uniform_init_live_points(random.PRNGKey(43),
                                               num_live_points=n,
                                               model=model)
    init_state = NestedSamplerState(key=random.PRNGKey(42),
                                    sample_collection=SampleCollection(sample_idx=n,
                                                                       reservoir=live_points.reservoir))
    assert np.unique(init_state.sample_collection.reservoir.log_L).size == n
    state, live_points = get_live_points_from_samples(state=init_state,
                                                      log_L_constraint=-jnp.inf,
                                                      num_live_points=n - 1)

    unique = np.unique(live_points.reservoir.log_L)
    assert unique.size == n - 1

    assert np.unique(state.sample_collection.reservoir.log_L).size == 2  # 1 + empty type
    assert state.sample_collection.sample_idx == 1


def test_static_uniform():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -z ** 2

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    m = 1000
    sample_collection = init_sample_collection(size=m, model=model)

    state = NestedSamplerState(key=random.PRNGKey(42), sample_collection=sample_collection)

    n = 100
    live_points = get_uniform_init_live_points(random.PRNGKey(43),
                                               num_live_points=n,
                                               model=model)
    efficiency_threshold = 0.1
    ns = StaticUniform(model=model, num_live_points=n, efficiency_threshold=efficiency_threshold)
    state, live_points = ns(state=state, live_points=live_points)
    assert state.sample_collection.sample_idx > 0
    assert jnp.mean(live_points.reservoir.num_likelihood_evaluations) > 1 / efficiency_threshold


def test_resample_indicies():
    n = 100
    sample_key = random.PRNGKey(42)
    log_weights = jnp.zeros(n)
    indices = resample_indicies(key=sample_key,
                                log_weights=log_weights,
                                S=n,
                                replace=False)
    assert np.unique(indices).size == n


def test_single_prior():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    U, X = parse_prior(prior_model)
    U_ndims = U.shape[0]
    assert isinstance(X, tuple)
    assert len(X) == 1
    assert U_ndims == 2

    U = random.uniform(random.PRNGKey(42), shape=(U_ndims,), dtype=float_type)

    def log_likelihood(Z):
        return Z - jnp.sum(Z ** 2)

    X, log_L = compute_log_likelihood(U, prior_model, log_likelihood=log_likelihood)
    assert not jnp.isnan(log_L)
    assert not any(jnp.any(jnp.isnan(x)) for x in X)


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
    assert isinstance(X, tuple)
    assert len(X) == 2
    assert U_ndims == 2

    U = random.uniform(random.PRNGKey(42), shape=(U_ndims,), dtype=float_type)
    output = transform(U, prior_model)
    assert len(output) == 2
    assert jnp.allclose(output[1], output[0] ** 2)

    def log_likelihood(Z, Z2):
        return Z - jnp.sum(Z2)

    X, log_L = compute_log_likelihood(U, prior_model, log_likelihood=log_likelihood)
    assert not jnp.isnan(log_L)
    assert not any(jnp.any(jnp.isnan(x)) for x in X)
