import logging
from abc import abstractmethod, ABC
from typing import Tuple, Generator, Callable, Optional, Union, List

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from etils.array_types import FloatArray, IntArray, BoolArray

from jaxns.internals.shapes import tuple_prod
from jaxns.types import float_type, LikelihoodType, LikelihoodInputType, UType, XType

logger = logging.getLogger('jaxns')
tfpd = tfp.distributions

__all__ = [
    "Prior",
    "PriorModelGen"
]


class InvalidDistribution(Exception):
    def __init__(self, dist: Optional[tfpd.Distribution] = None):
        super(InvalidDistribution, self).__init__(
            f'Distribution {dist} is missing a quantile. '
            f'Try checking if your desired prior exists in `jaxns.special_priors`.')


class InvalidPriorName(Exception):
    def __init__(self, name: Optional[str] = None):
        super(InvalidPriorName, self).__init__(f'Prior name {name} already taken by another prior.')


def distribution_chain(dist: tfpd.Distribution) -> List[Union[tfpd.TransformedDistribution,
tfpd.Sample,
tfpd.Distribution]]:
    chain = []
    while True:
        chain.append(dist)
        if isinstance(dist, tfpd.TransformedDistribution):
            dist = dist.distribution
            continue
        break
    return chain[::-1]


class AbstractDistribution(ABC):
    @abstractmethod
    def _dtype(self):
        ...

    @abstractmethod
    def _base_shape(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def _shape(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def _forward(self, U):
        ...

    @abstractmethod
    def _inverse(self, X):
        ...

    @abstractmethod
    def _log_prob(self, X):
        ...

    @property
    def dtype(self):
        return self._dtype()

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self._base_shape()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape()

    def forward(self, U):
        return self._forward(U)

    def inverse(self, X):
        return self._inverse(X)

    def log_prob(self, X):
        return self._log_prob(X)


class Distribution(AbstractDistribution):
    def __init__(self, dist: tfpd.Distribution):
        self.dist_chain = distribution_chain(dist)
        check_dist = self.dist_chain[0]
        if isinstance(self.dist_chain[0], tfpd.Sample):
            check_dist = self.dist_chain[0].distribution
        if '_quantile' not in check_dist.__class__.__dict__:
            # TODO(Joshuaalbert): we could numerically approximate it. This requires knowing the support of dist.
            # Repartitioning the prior also requires knowing the support and choosing a replacement, which is not
            # always easy from stats. E.g. StudentT variance doesn't exist but a numerial quantile can be made.
            raise InvalidDistribution(dist=dist)

    def __repr__(self):
        return " -> ".join(map(repr, self.dist_chain))

    def _dtype(self):
        return self.dist_chain[-1].dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return tuple(self.dist_chain[0].batch_shape_tensor()) + tuple(self.dist_chain[0].event_shape_tensor())

    def _shape(self) -> Tuple[int, ...]:
        return tuple(self.dist_chain[-1].batch_shape_tensor()) + tuple(self.dist_chain[-1].event_shape_tensor())

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        dist = self.dist_chain[0]
        if isinstance(dist, tfpd.Sample):
            dist = dist.distribution
        X = dist.quantile(U)
        for dist in self.dist_chain[1:]:
            X = dist.bijector.forward(X)
        return X

    def _inverse(self, X) -> FloatArray:
        for dist in reversed(self.dist_chain[1:]):
            X = dist.bijector.inverse(X)
        dist = self.dist_chain[0]
        if isinstance(dist, tfpd.Sample):
            dist = dist.distribution
        X = dist.cdf(X)
        return X

    def _log_prob(self, X):
        return self.dist_chain[-1].log_prob(X)


class AbstractPrior(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __repr__(self):
        return f"{self.name if self.name is not None else '*'}\t{self.base_shape} -> {self.shape} {self.dtype}"

    @abstractmethod
    def _dtype(self):
        ...

    @abstractmethod
    def _base_shape(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def _shape(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        ...

    @abstractmethod
    def _inverse(self, X) -> FloatArray:
        ...

    @abstractmethod
    def _log_prob(self, X) -> FloatArray:
        ...

    @property
    def dtype(self):
        return self._dtype()

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self._base_shape()

    @property
    def base_ndims(self):
        return tuple_prod(self.base_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape()

    def forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self._forward(U)

    def inverse(self, X) -> FloatArray:
        return self._inverse(X)

    def log_prob(self, X) -> FloatArray:
        log_prob = self._log_prob(X)
        if log_prob.size > 1:
            log_prob = jnp.sum(log_prob)
        if log_prob.shape != ():
            log_prob = log_prob.reshape(())
        return log_prob


class Prior(AbstractPrior):
    def __init__(self, dist_or_value: Union[tfpd.Distribution, jnp.ndarray], name: Optional[str] = None):
        super(Prior, self).__init__(name=name)
        if isinstance(dist_or_value, tfpd.Distribution):
            self._type = 'dist'
            self._dist = Distribution(dist_or_value)
        else:
            self._type = 'value'
            self._value = jnp.asarray(dist_or_value)
        self.name = name

    @property
    def dist(self) -> Distribution:
        if self._type != 'dist':
            raise ValueError(f"Wrong type, got {self._type}")
        return self._dist

    @property
    def value(self) -> jnp.ndarray:
        if self._type != 'value':
            raise ValueError(f"Wrong type, got {self._type}")
        return self._value

    def _base_shape(self) -> Tuple[int, ...]:
        if self._type == 'value':
            return (0,)
        elif self._type == 'dist':
            return self.dist.base_shape
        else:
            raise NotImplementedError()

    def _shape(self) -> Tuple[int, ...]:
        if self._type == 'value':
            return self.value.shape
        elif self._type == 'dist':
            return self.dist.shape
        else:
            raise NotImplementedError()

    def _dtype(self):
        if self._type == 'value':
            return self.value.dtype
        elif self._type == 'dist':
            return self.dist.dtype
        else:
            raise NotImplementedError()

    def _forward(self, U: FloatArray) -> Union[FloatArray, IntArray, BoolArray]:
        if self._type == 'value':
            return self.value
        elif self._type == 'dist':
            return self.dist.forward(U)
        else:
            raise NotImplementedError()

    def _inverse(self, X) -> FloatArray:
        if self._type == 'value':
            return jnp.asarray([], float_type)
        elif self._type == 'dist':
            return self.dist.inverse(X)
        else:
            raise NotImplementedError()

    def _log_prob(self, X: jnp.ndarray) -> FloatArray:
        if self._type == 'value':
            return jnp.asarray(0., float_type)
        elif self._type == 'dist':
            return self.dist.log_prob(X=X)
        else:
            raise NotImplementedError()


PriorModelGen = Generator[Prior, jnp.ndarray, Tuple[jnp.ndarray, ...]]
PriorModelType = Callable[[], PriorModelGen]


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
    X_placeholder = dict()
    while True:
        try:
            prior: Prior = gen.send(prior_response)
            d = prior.base_ndims
            U_ndims += d
            u = jnp.zeros(prior.base_shape, float_type)
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in X_placeholder:
                    raise InvalidPriorName(name=prior.name)
                X_placeholder[prior.name] = prior_response
        except StopIteration:
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
        the prior variables
    """

    gen = prior_model()
    prior_response = None
    collection = dict()
    idx = 0
    while True:
        try:
            prior: Prior = gen.send(prior_response)
            d = prior.base_ndims
            u = jnp.reshape(U[idx:idx + d], prior.base_shape)
            idx += d
            prior_response = prior.forward(u)
            if prior.name is not None:
                if prior.name in collection:
                    raise InvalidPriorName(name=prior.name)
                collection[prior.name] = prior_response
        except StopIteration:
            break
    return collection


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
            prior: Prior = gen.send(prior_response)
            d = prior.base_ndims
            u = jnp.reshape(U[idx:idx + d], prior.base_shape)
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
            prior: Prior = gen.send(prior_response)
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
