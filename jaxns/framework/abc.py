from abc import ABC, abstractmethod
from typing import Tuple

from jaxns.internals.types import UType, XType, PRNGKey, LikelihoodInputType, RandomVariableType, MeasureType

__all__ = [
    'AbstractPrior',
    'AbstractModel',
    'AbstractDistribution'
]


class AbstractPrior(ABC):
    """
    Represents a generative prior.
    """

    @abstractmethod
    def _dtype(self):
        """
        The dtype of the prior.
        """
        ...

    @abstractmethod
    def _base_shape(self) -> Tuple[int, ...]:
        """
        The base shape of the prior, in U-space.
        """
        ...

    @abstractmethod
    def _shape(self) -> Tuple[int, ...]:
        """
        The shape of the prior, in X-space.
        """
        ...

    @abstractmethod
    def _forward(self, U: UType) -> RandomVariableType:
        """
        The forward transformation from U-space to X-space.

        Args:
            U: U-space representation

        Returns:
            X-space representation
        """
        ...

    @abstractmethod
    def _inverse(self, X: RandomVariableType) -> UType:
        """
        The inverse transformation from X-space to U-space.

        Args:
            X: X-space representation

        Returns:
            U-space representation
        """
        ...

    @abstractmethod
    def _log_prob(self, X: RandomVariableType) -> MeasureType:
        """
        The log probability of the prior.

        Args:
            X: X-space representation

        Returns:
            log probability of the prior
        """

        ...


class AbstractModel(ABC):
    """
    Represents a Bayesian model in terms of a generative prior, and likelihood function.
    """


    @abstractmethod
    def __hash__(self):
        """
        Hash of the model.
        """
        ...

    @abstractmethod
    def sample_U(self, key: PRNGKey) -> UType:
        """
        Sample uniformly from the prior in U-space.

        Args:
            key: PRNGKey

        Returns:
            U-space sample
        """
        ...

    @abstractmethod
    def transform(self, U: UType) -> XType:
        """
        Compute the prior sample.

        Args:
            U: U-space sample

        Returns:
            prior sample
        """
        ...

    @abstractmethod
    def transform_parametrised(self, U: UType) -> XType:
        """
        Compute the parametrised prior variables.

        Args:
            U: U-space sample

        Returns:
            prior sample
        """
        ...

    @abstractmethod
    def forward(self, U: UType, allow_nan: bool = False) -> MeasureType:
        """
        Compute the log-likelihood.

        Args:
            U: U-space sample
            allow_nan: whether to allow nans in likelihood

        Returns:
            log likelihood at the sample
        """
        ...

    def log_prob_likelihood(self, U: UType, allow_nan: bool = False) -> MeasureType:
        """
        Compute the log-likelihood.

        Args:
            U: U-space sample
            allow_nan: whether to allow nans in likelihood

        Returns:
            log likelihood at the sample
        """
        return self.forward(U=U, allow_nan=allow_nan)

    @abstractmethod
    def log_prob_prior(self, U: UType) -> MeasureType:
        """
        Computes the log-probability of the prior.

        Args:
            U: The U-space sample

        Returns:
            the log probability of prior
        """
        ...

    def log_prob_joint(self, U: UType, allow_nan: bool) -> MeasureType:
        """
        Computes the log-joint probability of the model.

        Args:
            U: The U-space sample
            allow_nan: whether to allow nans in likelihood

        Returns:
            the log joint probability of the model
        """
        return self.log_prob_prior(U=U) + self.log_prob_likelihood(U=U, allow_nan=allow_nan)

    @abstractmethod
    def prepare_input(self, U: UType) -> LikelihoodInputType:
        """
        Prepares the input for the likelihood function.

        Args:
            U: The U-space sample

        Returns:
            the input to the likelihood function
        """
        ...

    @abstractmethod
    def sanity_check(self, key: PRNGKey, S: int):
        """
        Performs a sanity check on the model.

        Args:
            key: PRNGKey
            S: number of samples to check

        Raises:
            AssertionError: if any of the samples are nan.
        """
        ...


class AbstractDistribution(ABC):
    """
    Represents adistribution.
    """

    @abstractmethod
    def _dtype(self):
        """
        The dtype of the distribution, in X-space.
        """
        ...

    @abstractmethod
    def _base_shape(self) -> Tuple[int, ...]:
        """
        The base shape of the distribution, in U-space.
        """
        ...

    @abstractmethod
    def _shape(self) -> Tuple[int, ...]:
        """
        The shape of the distribution, in X-space.
        """
        ...

    @abstractmethod
    def _forward(self, U: UType) -> RandomVariableType:
        """
        The forward transformation from U-space to X-space.

        Args:
            U: U-space representation

        Returns:
            X-space representation
        """
        ...

    @abstractmethod
    def _inverse(self, X: RandomVariableType) -> UType:
        """
        The inverse transformation from X-space to U-space.

        Args:
            X: X-space representation

        Returns:
            U-space representation
        """
        ...

    @abstractmethod
    def _log_prob(self, X: RandomVariableType) -> MeasureType:
        """
        The log probability of the distribution.

        Args:
            X: X-space representation

        Returns:
            log probability of the distribution
        """
        ...
