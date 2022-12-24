import logging
from typing import Tuple, Generator, Callable, Optional

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfpd
from etils.array_types import FloatArray

from jaxns.internals.shapes import tuple_prod
from jaxns.internals.types import float_type

logger = logging.getLogger('jaxns')


class InvalidDistribution(Exception):
    def __init__(self, dist: Optional[tfpd.Distribution] = None):
        super(InvalidDistribution, self).__init__(f'Distribution {dist} is missing a quantile and variance.')


class Prior:
    def __init__(self, dist: tfpd.Distribution):
        self.has_quantile = '_quantile' in dist.__class__.__dict__
        if not self.has_quantile:
            # TODO(Joshuaalbert): we should numerically approximate it. This requires knowing the support of dist.
            # Repartitioning the prior also requires knowing the support and choosing a replacement, which is not
            # always easy from stats. E.g. StudentT variance doesn't exist but a numerial quantile can be made.
            raise InvalidDistribution(dist=dist)
        self.dist = dist

    def shape(self) -> Tuple[int, ...]:
        return self.dist.event_shape_tensor()

    def dtype(self):
        return self.dist.dtype

    def forward(self, U: FloatArray) -> FloatArray:
        return self.dist.quantile(U)

    def log_prob(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.dist.log_prob(value=X)


PriorModelGen = Generator[Prior, jnp.ndarray, Tuple[jnp.ndarray, ...]]
PriorModelType = Callable[[], PriorModelGen]
LikelihoodType = Callable[[jnp.ndarray, ...], FloatArray]
XType = Tuple[jnp.ndarray, ...]
UType = FloatArray


def parse_prior(prior_model: PriorModelType) -> Tuple[UType, XType]:
    """
    Computes the total dimension of the prior model.

    Args:
        prior_model: a callable that produces a prior model generator

    Returns:
        U placeholder, X placeholder
    """
    U_ndims = 0
    gen = prior_model()
    prior_response = None

    while True:
        try:
            prior: Prior = gen.send(prior_response)
            shape = prior.shape()
            d = tuple_prod(shape)
            U_ndims += d
            u = jnp.zeros(shape, float_type)
            prior_response = prior.forward(u)
        except StopIteration as e:
            X_placeholder = e.value
            if not isinstance(X_placeholder, tuple):
                X_placeholder = (X_placeholder,)
            break
    U_placeholder = jnp.zeros((U_ndims,), float_type)
    return U_placeholder, X_placeholder


def transform(U: UType, prior_model: PriorModelType) -> XType:
    """
    Transforms a flat array of `U_ndims` i.i.d. samples of U[0,1] into the target prior.

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator

    Returns:
        the prior model output, a tuple of ndarray
    """

    gen = prior_model()
    prior_response = None
    idx = 0
    while True:
        try:
            prior: Prior = gen.send(prior_response)
            shape = prior.shape()
            d = tuple_prod(shape)
            u = jnp.reshape(U[idx:idx + d], shape)
            idx += d
            prior_response = prior.forward(u)
        except StopIteration as e:
            output = e.value
            if not isinstance(output, tuple):
                output = (output,)
            break
    return output


def log_prob_prior(U: UType, prior_model: PriorModelType) -> FloatArray:
    """
    Transforms a flat array of `U_ndims` i.i.d. samples of U[0,1] into the target prior.

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator

    Returns:
        the prior model output, a tuple of ndarray
    """

    gen = prior_model()
    prior_response = None
    log_prob = []
    idx = 0
    while True:
        try:
            prior: Prior = gen.send(prior_response)
            shape = prior.shape()
            d = tuple_prod(shape)
            u = jnp.reshape(U[idx:idx + d], shape)
            idx += d
            prior_response = prior.forward(u)
            log_prob.append(prior.log_prob(prior_response))
        except StopIteration:
            break
    return sum(log_prob, jnp.asarray(0., float_type))

def compute_log_likelihood(U: UType, prior_model: PriorModelType, log_likelihood: LikelihoodType) -> Tuple[
    XType, FloatArray]:
    """
    Computes the log likelihood of a prior model

    Args:
        U: [U_ndims] a flat array of i.i.d. samples of U[0,1]
        prior_model: a callable that produces a prior model generator
        log_likelihood: callable that takes arrays returned by the prior model and returns a scalar float

    Returns:
        the prior model output, a tuple of ndarray
    """

    X = transform(U=U, prior_model=prior_model)
    log_L = log_likelihood(*X)
    log_L = jnp.where(jnp.isnan(log_L), -jnp.inf, jnp.asarray(log_L, float_type))
    if log_L.size != 1:
        raise ValueError(f"Log likelihood should be scalar, but got {log_L.shape}.")
    if len(log_L.shape) > 0:
        log_L = jnp.reshape(log_L, ())

    return X, log_L
