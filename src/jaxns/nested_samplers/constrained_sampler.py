from abc import ABC, abstractmethod
from typing import Tuple

from jaxns.nested_samplers.samples import UType
from jaxns.nested_samplers.types import FloatArray, IntArray


class AbstractSampler(ABC):
    """
    Performs sampling from the prior within a likelihood constraint, to produce i.i.d. samples for nested sampling.
    The sampler is assumed to be stateless and pure.
    """

    @abstractmethod
    def get_sample(self, key, log_L_constraint: FloatArray, seed_point: UType) -> Tuple[UType, FloatArray, IntArray]:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            seed_point: a seed point to begin sampling from

        Returns:
            U_sample: an i.i.d. sample within the constraint
            log_L: the log-likelihood of the sample
            num_likelihood_evaluations: number of likelihood evaluations used to produce the sample
            TODO: consider returning phantom points.
        """
        ...
