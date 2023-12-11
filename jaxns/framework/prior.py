from typing import Tuple, Optional, Union

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jaxns.framework.bases import BaseAbstractPrior, BaseAbstractDistribution
from jaxns.framework.distribution import Distribution
from jaxns.internals.types import FloatArray, IntArray, BoolArray, XType, UType, float_type

tfpd = tfp.distributions

__all__ = [
    "Prior",
    "InvalidPriorName"
]


class InvalidPriorName(Exception):
    """
    Raised when a prior name is already taken.
    """

    def __init__(self, name: Optional[str] = None):
        super(InvalidPriorName, self).__init__(f'Prior name {name} already taken by another prior.')


class Prior(BaseAbstractPrior):
    """
    Represents a generative prior.
    """

    def __init__(self, dist_or_value: Union[tfpd.Distribution, BaseAbstractDistribution, jnp.ndarray],
                 name: Optional[str] = None):
        super(Prior, self).__init__(name=name)
        if isinstance(dist_or_value, tfpd.Distribution):
            self._type = 'dist'
            self._dist = Distribution(dist_or_value)
        elif isinstance(dist_or_value, BaseAbstractDistribution):
            self._type = 'dist'
            self._dist = dist_or_value
        else:
            self._type = 'value'
            self._value = jnp.asarray(dist_or_value)
        self.name = name

    @property
    def dist(self) -> BaseAbstractDistribution:
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

    def _forward(self, U: UType) -> Union[FloatArray, IntArray, BoolArray]:
        if self._type == 'value':
            return self.value
        elif self._type == 'dist':
            return self.dist.forward(U)
        else:
            raise NotImplementedError()

    def _inverse(self, X: XType) -> FloatArray:
        if self._type == 'value':
            return jnp.asarray([], float_type)
        elif self._type == 'dist':
            return self.dist.inverse(X)
        else:
            raise NotImplementedError()

    def _log_prob(self, X: XType) -> FloatArray:
        if self._type == 'value':
            return jnp.asarray(0., float_type)
        elif self._type == 'dist':
            return self.dist.log_prob(X=X)
        else:
            raise NotImplementedError()
