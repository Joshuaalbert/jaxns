from abc import abstractmethod
from typing import Tuple, TypeVar, Generic

from jax import random

from jaxns.nested_samplers.types import Sample, PRNGKey, FloatArray, SeedPoint
from jaxns.samplers.abc import AbstractSampler

T = TypeVar('T')


class BaseAbstractRejectionSampler(AbstractSampler[T], Generic[T]):
    ...


class BaseAbstractMarkovSampler(AbstractSampler[T], Generic[T]):
    """
    A sampler that conditions off a known satisfying point, e.g. a seed point.
    """

    @abstractmethod
    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             sampler_state: T) -> tuple[Sample, Sample]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            seed_point: function that gets the next sample from a seed point
            log_L_constraint: the constraint to sample within
            sampler_state: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample, and batched phantom samples
        """
        ...

    @abstractmethod
    def get_seed_point(self, key: PRNGKey, sampler_state: T,
                       log_L_constraint: FloatArray) -> SeedPoint:
        """
        Samples a seed point from the live points.

        Args:
            key: PRNGKey
            sampler_state: the current sampler state
            log_L_constraint: a log-L constraint to sample within. Must always be at least one sample in front above
                this to avoid infinite loop.

        Returns:
            a seed point
        """
        ...

    def _get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, sampler_state: T) -> tuple[Sample, Sample]:
        sample_key, seed_key = random.split(key, 2)
        seed_point = self.get_seed_point(
            key=seed_key,
            sampler_state=sampler_state,
            log_L_constraint=log_L_constraint
        )
        return self.get_sample_from_seed(
            key=sample_key,
            seed_point=seed_point,
            log_L_constraint=log_L_constraint,
            sampler_state=sampler_state
        )
