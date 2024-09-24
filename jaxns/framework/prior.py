import warnings
from typing import Tuple, Optional, Union

import jax.nn
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

import jaxns.framework.context as ctx
from jaxns.framework.bases import BaseAbstractPrior, BaseAbstractDistribution
from jaxns.framework.wrapped_tfp_distribution import WrappedTFPDistribution
from jaxns.internals.constraint_bijections import quick_unit
from jaxns.internals.types import FloatArray, IntArray, BoolArray, XType, UType
from jaxns.internals.mixed_precision import float_type

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


class SingularPrior(BaseAbstractPrior):
    """
    Represents a singular prior, which has no inverse transformation, but does have a log_prob
        (at the singular value).
    """

    def __init__(self, value: jax.Array, base_prior: BaseAbstractPrior, name: Optional[str] = None):
        super().__init__(name=name)
        self.value = value
        self.base_prior = base_prior

    def __repr__(self):
        return f"{self.value} -> {self.base_prior}"

    def _dtype(self):
        return self.base_prior.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return (0,)  # Singular prior has no base shape

    def _shape(self) -> Tuple[int, ...]:
        return self.base_prior.shape

    def _forward(self, U: UType) -> Union[FloatArray, IntArray, BoolArray]:
        return self.value

    def _inverse(self, X: XType) -> UType:
        return jnp.asarray([], float_type)

    def _log_prob(self, X: XType) -> FloatArray:
        return self.base_prior.log_prob(X)


class Prior(BaseAbstractPrior):
    """
    Represents a generative prior.
    """

    def __init__(self, dist_or_value: Union[tfpd.Distribution, FloatArray, IntArray, BoolArray],
                 name: Optional[str] = None):
        super(Prior, self).__init__(name=name)
        if isinstance(dist_or_value, tfpd.Distribution):
            self._dist = WrappedTFPDistribution(dist_or_value)
            self._type = 'dist'
        else:
            try:
                self._value = jnp.asarray(dist_or_value)
            except TypeError:
                raise ValueError(f"Could not convert {dist_or_value} to array.")
            except Exception as e:
                raise e
            self._type = 'value'
        self.name = name

    @property
    def dist(self) -> BaseAbstractDistribution:
        if self._type != 'dist':
            raise ValueError(f"Wrong type, got {self._type}")
        return self._dist

    @property
    def value(self) -> jax.Array:
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

    def parametrised(self, random_init: bool = False) -> SingularPrior:
        """
        Convert this prior into a non-Bayesian parameter, that takes a single value in the model, but still has an
        associated log_prob. The parameter is registered as a `get_parameter` with added `_param` name suffix. Prior
        must have a name.

        Args:
            random_init: whether to initialise the parameter randomly or at the median of the distribution.

        Returns:
            A singular prior.

        Raises:
            ValueError: if the prior has no name.
        """
        return prior_to_parametrised_singular(self, random_init=random_init)


def prior_to_parametrised_singular(prior: BaseAbstractPrior, random_init: bool = False) -> SingularPrior:
    """
    Convert a prior into a non-Bayesian parameter, that takes a single value in the model, but still has an associated
    log_prob. The parameter is registered as a `jaxns.get_parameter` with added `_param` name suffix.

    To constrain the parameter we use a Normal parameter with centre on unit cube, and scale covering the whole cube,
    as the base representation. This base representation covers the whole real line and be reliably used with SGD, etc.

    Args:
        prior: any prior
        random_init: whether to initialise the parameter randomly or at the median of the distribution.

    Returns:
        A parameter representing the prior.
    """
    if prior.name is None:
        raise ValueError("Prior must have a name to be parametrised.")
    name = f"{prior.name}_param"
    # Initialises at median of distribution using zeros, else unit-normal.
    if random_init:
        initaliser = ctx.wrap_random(jax.random.normal)
    else:
        initaliser = jnp.zeros
    if prior.base_ndims == 0:
        warnings.warn(f"Creating a zero-sized parameter for {prior.name}. Probably unintended.")
    norm_U_base_param = ctx.get_parameter(
        name=name,
        shape=prior.base_shape,
        dtype=float_type,
        init=initaliser
    )
    # transform [-inf, inf] -> [0,1]
    U_base_param = quick_unit(norm_U_base_param)
    param = prior.forward(U_base_param)
    return SingularPrior(value=param, base_prior=prior, name=prior.name)
