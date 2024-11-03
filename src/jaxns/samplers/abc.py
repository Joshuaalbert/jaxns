from abc import ABC, abstractmethod
from typing import TypeVar, Tuple, NamedTuple, Generic, Any

from jaxns.internals.types import PRNGKey, FloatArray
from jaxns.nested_samplers.common.types import Sample, TerminationRegister, LivePointCollection

SamplerState = TypeVar('SamplerState')


class EphemeralState(NamedTuple):
    # Contains all available ephemeral quantities that might be used for constructing a sampler state.
    # Excludes the full state, as this is too large oft times.
    key: PRNGKey
    live_points_collection: LivePointCollection
    termination_register: TerminationRegister


class AbstractSampler(ABC, Generic[SamplerState]):

    @abstractmethod
    def _pre_process(self, ephemeral_state: EphemeralState) -> Any:
        """
        Run this periodically on the state to produce a data pytree that can be used by the sampler, and
        updated quickly.

        Args:
            ephemeral_state: the current state of the sampler

        Returns:
            any valid pytree
        """
        ...

    def pre_process(self, ephemeral_state: EphemeralState) -> SamplerState:
        """
        Run this periodically on the state to produce a data pytree that can be used by the sampler, and
        updated quickly.

        Args:
            ephemeral_state: the current state of the sampler

        Returns:
            any valid pytree
        """
        return self._pre_process(ephemeral_state)

    @abstractmethod
    def _post_process(self, ephemeral_state: EphemeralState,
                      sampler_state: Any) -> Any:
        """
        Post process the sampler state, after the sampler has been run. Should be quick.

        Args:
            ephemeral_state: a sample collection post sample step
            sampler_state: data pytree produced by the sampler

        Returns:
            the updated sampler state
        """
        ...

    def post_process(self, ephemeral_state: EphemeralState,
                     sampler_state: SamplerState) -> SamplerState:
        """
        Post process the sampler state, after the sampler has been run. Should be quick.

        Args:
            ephemeral_state: a sample collection post sample step
            sampler_state: data pytree produced by the sampler

        Returns:
            the updated sampler state
        """
        return self._post_process(ephemeral_state, sampler_state)

    @abstractmethod
    def _get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, sampler_state: Any) -> Tuple[
        Sample, Sample]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            sampler_state: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample, and phantom samples
        """
        ...

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, sampler_state: SamplerState) -> Tuple[
        Sample, Sample]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            sampler_state: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample, and phantom samples
        """
        return self._get_sample(key, log_L_constraint, sampler_state)

    @abstractmethod
    def num_phantom(self) -> int:
        """
        The number of phantom samples produced by the sampler.

        Returns:
            number of phantom samples
        """
        ...
