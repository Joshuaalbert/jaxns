from typing import NamedTuple

import jax.random
import pytest
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns.framework.special_priors import ForcedIdentifiability

tfpd = tfp.distributions


def test_gh144():
    class Output(NamedTuple):
        x: jnp.ndarray
        y: jnp.ndarray

    def prior_model():
        x = yield Prior(dist_or_value=jnp.asarray(0.))
        y = yield Prior(dist_or_value=jnp.asarray(0.))
        return Output(x, y)

    def log_likelihood(z: Output):
        return z.x + z.y

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.U_ndims == 0
    model.sanity_check(key=jax.random.PRNGKey(0), S=10)


def test_gh95():
    def prior_model():
        x = yield Prior(dist_or_value=jnp.asarray(0.))
        return x

    def log_likelihood(x):
        return jnp.sum(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.U_ndims == 0


def test_parametrised_singular():
    def prior_model():
        x = yield Prior(dist_or_value=jnp.asarray(0.), name='x').parametrised()
        return x

    def log_likelihood(x):
        return jnp.sum(x)

    _ = Model(prior_model=prior_model, log_likelihood=log_likelihood)


def test_parametrised():
    def prior_model():
        x = yield Prior(tfpd.Uniform(), name='x').parametrised()
        return x

    def log_likelihood(x):
        return x

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.num_params == 1
    assert model.U_ndims == 0

    log_prob_joint = model.log_prob_joint(model.U_placeholder, allow_nan=True)
    assert log_prob_joint.shape == ()
    assert log_prob_joint == 0.5, "Didn't init at median of uniform"


def test_parametrised_randomised():
    def prior_model():
        x = yield Prior(tfpd.Uniform(), name='x').parametrised(random_init=True)
        y = yield Prior(tfpd.Uniform(), name='y')
        return x, y

    def log_likelihood(x, y):
        return x - y

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.num_params == 1
    assert model.U_ndims == 1

    log_prob_joint = model.log_prob_joint(model.U_placeholder, allow_nan=True)
    assert log_prob_joint.shape == ()

    model.sanity_check(key=jax.random.PRNGKey(0), S=10)

    model.prepare_input(model.U_placeholder)
    model.transform_parametrised(model.U_placeholder)
    model.forward(model.U_placeholder, allow_nan=True)
    model.sample_U(jax.random.PRNGKey(0))
    model.log_prob_prior(model.U_placeholder)
    model.log_prob_joint(model.U_placeholder, allow_nan=True)
    model.transform(model.U_placeholder)


def test_parametrised_randomised_special():
    def prior_model():
        x = yield ForcedIdentifiability(n=10, low=0., high=1., name='x').parametrised(random_init=True)
        y = yield ForcedIdentifiability(n=10, low=0., high=1., name='y')
        return x, y

    def log_likelihood(x, y):
        return jnp.sum(x - y)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    assert model.num_params == 10
    assert model.U_ndims == 10

    log_prob_joint = model.log_prob_joint(model.U_placeholder, allow_nan=True)
    assert log_prob_joint.shape == ()

    model.sanity_check(key=jax.random.PRNGKey(0), S=10)

    model.prepare_input(model.U_placeholder)
    model.transform_parametrised(model.U_placeholder)
    model.forward(model.U_placeholder, allow_nan=True)
    model.sample_U(jax.random.PRNGKey(0))
    model.log_prob_prior(model.U_placeholder)
    model.log_prob_joint(model.U_placeholder, allow_nan=True)
    model.transform(model.U_placeholder)
