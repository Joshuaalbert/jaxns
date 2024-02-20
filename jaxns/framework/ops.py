from typing import Tuple

import jax
from jax import numpy as jnp

from jaxns.framework.bases import PriorModelType, BaseAbstractPrior
from jaxns.framework.prior import InvalidPriorName, SingularPrior
from jaxns.internals.types import UType, XType, float_type, LikelihoodInputType, FloatArray, LikelihoodType, PRNGKey, \
    isinstance_namedtuple

__all__ = [
    'simulate_prior_model'
]


def compute_U_ndims(prior_model: PriorModelType) -> int:
    """
    Computes placeholders of model.

    Args:
        prior_model: a callable that produces a prior model generator

    Returns:
        number of U dims
    """
    U_ndims = 0
    gen = prior_model()
    prior_response = None
    names = set()
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            U_ndims += d
            u = jnp.zeros(prior.base_shape, float_type)
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
        except StopIteration:
            break
    return U_ndims


def simulate_prior_model(key: PRNGKey, prior_model: PriorModelType) -> Tuple[LikelihoodInputType, XType]:
    """
    Simulate a prior model.

    Args:
        key: PRNGKey
        prior_model: A prior model

    Returns:
        a tuple of the likelihood input variables, and dict of non-hidden (named) prior variables.
    """
    U_ndims = compute_U_ndims(prior_model=prior_model)
    U = jax.random.uniform(key, shape=(U_ndims,), dtype=float_type)
    return prepare_input(U=U, prior_model=prior_model), transform(U=U, prior_model=prior_model)


def parse_prior(prior_model: PriorModelType) -> Tuple[UType, XType]:
    """
    Computes placeholders of model.

    Args:
        prior_model: a callable that produces a prior model generator

    Returns:
        U placeholder, X placeholder
    """
    U_ndims = 0
    gen = prior_model()
    prior_response = None
    names = set()
    X_placeholder: XType = dict()
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            U_ndims += d
            u = jnp.zeros(prior.base_shape, float_type)
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if not isinstance(prior, SingularPrior):
                    X_placeholder[prior.name] = prior_response
        except StopIteration:
            break
    U_placeholder = jnp.zeros((U_ndims,), float_type)
    return U_placeholder, X_placeholder


def parse_joint(prior_model: PriorModelType, log_likelihood: LikelihoodType) -> Tuple[
    UType, XType, LikelihoodInputType, FloatArray]:
    """
    Computes placeholders of model.

    Args:
        prior_model: a callable that produces a prior model generator

    Returns:
        U placeholder, X placeholder
    """
    U_ndims = 0
    gen = prior_model()
    prior_response = None
    names = set()
    X_placeholder: XType = dict()
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            U_ndims += d
            u = jnp.zeros(prior.base_shape, float_type)
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if not isinstance(prior, SingularPrior):
                    X_placeholder[prior.name] = prior_response
        except StopIteration as e:
            output = e.value
            if (not isinstance(output, tuple)) and (not isinstance_namedtuple(output)):
                output = (output,)
            break
    likelihood_input_placeholder = output
    log_L_placeholder = jnp.asarray(log_likelihood(*output), float_type)
    U_placeholder = jnp.zeros((U_ndims,), float_type)
    return U_placeholder, X_placeholder, likelihood_input_placeholder, log_L_placeholder


def transform(U: UType, prior_model: PriorModelType) -> XType:
    """
    Transforms a flat array of `U_ndims` i.i.d. samples of U[0,1] into the target prior.

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator

    Returns:
        the prior variables
    """

    gen = prior_model()
    prior_response = None
    names = set()
    X_collection = dict()
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            u = jnp.reshape(U[idx:idx + d], prior.base_shape)
            idx += d
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if not isinstance(prior, SingularPrior):
                    X_collection[prior.name] = prior_response
        except StopIteration:
            break
    return X_collection


def transform_parametrised(U: UType, prior_model: PriorModelType) -> XType:
    """
    Transforms a flat array of `U_ndims` i.i.d. samples of U[0,1] into the the parametrised prior variables.

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator

    Returns:
        the parametrised prior variables
    """

    gen = prior_model()
    prior_response = None
    names = set()
    Y_collection = dict()
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            u = jnp.reshape(U[idx:idx + d], prior.base_shape)
            idx += d
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in names:
                    raise InvalidPriorName(name=prior.name)
                names.add(prior.name)
                if isinstance(prior, SingularPrior):
                    Y_collection[prior.name] = prior_response
        except StopIteration:
            break
    return Y_collection


def prepare_input(U: UType, prior_model: PriorModelType) -> LikelihoodInputType:
    """
    Transforms a flat array of `U_ndims` i.i.d. samples of U[0,1] into the likelihood conditional variables.

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator

    Returns:
        the conditional variables of likelihood model
    """

    gen = prior_model()
    prior_response = None
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            u = jnp.reshape(U[idx:idx + d], prior.base_shape)
            idx += d
            prior_response = prior.forward(u)
        except StopIteration as e:
            output = e.value
            if (not isinstance(output, tuple)) and (not isinstance_namedtuple(output)):
                output = (output,)
            break
    return output


def compute_log_prob_prior(U: UType, prior_model: PriorModelType) -> FloatArray:
    """
    Computes the prior log-density from a U-space sample.

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator

    Returns:
        prior log-density
    """

    gen = prior_model()
    prior_response = None
    log_prob = []
    idx = 0
    while True:
        try:
            prior: BaseAbstractPrior = gen.send(prior_response)
            d = prior.base_ndims
            u = jnp.reshape(U[idx:idx + d], prior.base_shape)
            idx += d
            prior_response = prior.forward(u)
            log_prob.append(prior.log_prob(prior_response))
        except StopIteration:
            break
    return sum(log_prob, jnp.asarray(0., float_type))


def compute_log_likelihood(U: UType, prior_model: PriorModelType, log_likelihood: LikelihoodType,
                           allow_nan: bool = False) -> FloatArray:
    """
    Computes the log likelihood from U-space sample.

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator
        log_likelihood: callable that takes arrays returned by the prior model and returns a scalar float
        allow_nan: whether to allow nans in likelihood

    Returns:
        log-likelihood
    """

    V = prepare_input(U=U, prior_model=prior_model)
    log_L = jnp.asarray(log_likelihood(*V), float_type)
    if not allow_nan:
        log_L = jnp.where(jnp.isnan(log_L), -jnp.inf, log_L)
    if log_L.size != 1:
        raise ValueError(f"Log likelihood should be scalar, but got {log_L.shape}.")
    if log_L.shape != ():
        log_L = jnp.reshape(log_L, ())
    return log_L
