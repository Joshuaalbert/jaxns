from abc import ABC, abstractmethod
from typing import Tuple

from jaxns.internals.types import PRNGKey, IntArray
from jaxns.nested_samplers.common.types import TerminationCondition, NestedSamplerResults, NestedSamplerState, \
    TerminationRegister


class AbstractNestedSampler(ABC):
    """
    The abstract base class for nested samplers.
    """

    @abstractmethod
    def _run(self, key: PRNGKey, term_cond: TerminationCondition) -> Tuple[IntArray, TerminationRegister, NestedSamplerState]:
        """
        Run the nested sampler.

        Args:
            key: PRNGKey
            term_cond: termination condition

        Returns:
            termination reason, termination register and the final sampler state
        """
        ...

    @abstractmethod
    def _to_results(self, termination_reason: IntArray, state: NestedSamplerState,
                    trim: bool) -> NestedSamplerResults:
        """
        Convert the sampler state to results.

        Args:
            termination_reason: termination reason
            state: sampler state
            trim: whether to trim the results

        Returns:
            Results of the nested sampling run
        """
        ...
