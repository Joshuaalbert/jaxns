from functools import cached_property
from typing import Tuple, Optional, Generator, Callable

import jax.numpy as jnp

from jaxns.internals.types import LikelihoodInputType
from jaxns.internals.shapes import tuple_prod
from jaxns.framework.abc import AbstractModel, AbstractPrior, AbstractDistribution
from jaxns.internals.types import LikelihoodType, UType, XType, RandomVariableType, MeasureType

__all__ = [
    'BaseAbstractModel',
    "BaseAbstractPrior",
    "BaseAbstractDistribution",
    "PriorModelGen",
    "PriorModelType"
]


class BaseAbstractPrior(AbstractPrior):
    """
    The base prior class with public methods.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __repr__(self):
        return f"{self.name if self.name is not None else '*'}\t{self.base_shape} -> {self.shape} {self.dtype}"

    @property
    def dtype(self):
        """
        The dtype of the prior random variable in X-space.
        """
        return self._dtype()

    @property
    def base_shape(self) -> Tuple[int, ...]:
        """
        The base shape of the prior random variable in U-space.
        """
        return self._base_shape()

    @property
    def base_ndims(self):
        """
        The number of dimensions of the prior random variable in U-space.
        """
        return tuple_prod(self.base_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the prior random variable in X-space.
        """
        return self._shape()

    def forward(self, U: UType) -> RandomVariableType:
        """
        The forward transformation from U-space to X-space.

        Args:
            U: U-space representation

        Returns:
            X-space representation
        """
        return self._forward(U)

    def inverse(self, X: RandomVariableType) -> UType:
        """
        The inverse transformation from X-space to U-space.

        Args:
            X: X-space representation

        Returns:
            U-space representation
        """
        return self._inverse(X)

    def log_prob(self, X: RandomVariableType) -> MeasureType:
        """
        The log probability of the prior.

        Args:
            X: X-space representation

        Returns:
            log probability of the prior
        """
        log_prob = self._log_prob(X)
        if log_prob.size > 1:
            log_prob = jnp.sum(log_prob)
        if log_prob.shape != ():
            log_prob = log_prob.reshape(())
        return log_prob


PriorModelGen = Generator[BaseAbstractPrior, RandomVariableType, LikelihoodInputType]
PriorModelType = Callable[[], PriorModelGen]


class BaseAbstractModel(AbstractModel):
    """
    The base model class with public methods.
    """

    def __init__(self, prior_model: PriorModelType, log_likelihood: LikelihoodType):
        self._prior_model = prior_model
        self._log_likelihood = log_likelihood

    @property
    def prior_model(self) -> PriorModelType:
        """
        The prior model.
        """
        return self._prior_model

    @property
    def log_likelihood(self) -> LikelihoodType:
        """
        The log likelihood function.

        Returns:
            log likelihood function
        """
        return self._log_likelihood

    @property
    def U_placeholder(self) -> UType:
        """
        A placeholder for U-space sample.
        """
        return self.parsed_prior[0]

    @property
    def X_placeholder(self) -> XType:
        """
        A placeholder for X-space sample.
        """
        return self.parsed_prior[1]

    @property
    def U_ndims(self) -> int:
        """
        The prior dimensionality.
        """
        return self.U_placeholder.size

    @cached_property
    def parsed_prior(self) -> Tuple[UType, XType]:
        """
        The parsed prior.

        Returns:
            U-space sample, X-space sample
        """
        return self._parsed_prior()


# TODO(Joshuaalbert): distribution is too similar to prior, where we only need to be able to extract the log_prob and
#  potential tranformations from the underlying. Perhaps we should just define as ops to create priors. Try making
#  priors just use distribution functionality. Special priors will need treatment.
class BaseAbstractDistribution(AbstractDistribution):
    """
    The base distribution class with public methods.
    """

    @property
    def dtype(self):
        """
        The dtype of the distribution, in X-space.
        """
        return self._dtype()

    @property
    def base_shape(self) -> Tuple[int, ...]:
        """
        The base shape of the distribution, in U-space.
        """
        return self._base_shape()

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the distribution, in X-space.
        """
        return self._shape()

    def forward(self, U: UType) -> RandomVariableType:
        """
        The forward transformation from U-space to X-space.

        Args:
            U: U-space representation

        Returns:
            X-space representation
        """
        return self._forward(U)

    def inverse(self, X: RandomVariableType) -> UType:
        """
        The inverse transformation from X-space to U-space.

        Args:
            X: X-space representation

        Returns:
            U-space representation
        """
        return self._inverse(X)

    def log_prob(self, X: RandomVariableType) -> MeasureType:
        """
        The log probability of the distribution.

        Args:
            X: X-space representation

        Returns:
            log probability of the distribution
        """
        return self._log_prob(X)
