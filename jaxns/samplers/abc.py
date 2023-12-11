from abc import ABC, abstractmethod
from typing import TypeVar, Tuple

from jaxns.internals.types import PRNGKey, FloatArray, Sample, StaticStandardNestedSamplerState

SamplerState = TypeVar('SamplerState')


class AbstractSampler(ABC):

    @abstractmethod
    def pre_process(self, state: StaticStandardNestedSamplerState) -> SamplerState:
        """
        Run this periodically on the state to produce a data pytree that can be used by the sampler, and
        updated quickly.

        Args:
            state: nested sampler state

        Returns:
            any valid pytree
        """
        ...

    @abstractmethod
    def post_process(self, state: StaticStandardNestedSamplerState, sampler_state: SamplerState) -> SamplerState:
        """
        Post process the sampler state, after the sampler has been run. Should be quick.

        Args:
            state: the state after successful sampling
            sampler_state: data pytree produced by the sampler

        Returns:
            the updated sampler state
        """
        ...

    @abstractmethod
    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, sampler_state: SamplerState) -> Tuple[
        Sample, Sample]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            sampler_state: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample, and batched phantom samples
        """
        ...

    @abstractmethod
    def num_phantom(self) -> int:
        """
        The number of phantom samples produced by the sampler.

        Returns:
            number of phantom samples
        """
        ...
