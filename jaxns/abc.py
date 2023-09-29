from abc import ABC, abstractmethod
from typing import TypeVar, NamedTuple, Optional, Union, Tuple

from jax import numpy as jnp, random

from jaxns.initial_state import find_first_true_indices
from jaxns.model import Model
from jaxns.types import FloatArray, float_type, PRNGKey, IntArray, NestedSamplerState, LivePoints, Sample, \
    TerminationCondition

PreProcessType = TypeVar('PreProcessType')


class SeedPoint(NamedTuple):
    U0: FloatArray
    log_L0: FloatArray


class AbstractSampler(ABC):
    def __init__(self, model: Model, efficiency_threshold: Optional[FloatArray] = None):
        self.model = model
        if efficiency_threshold is None:
            efficiency_threshold = 0.
        if efficiency_threshold < 0. or efficiency_threshold >= 1.:
            raise ValueError(f"{efficiency_threshold} must be in [0., 1.), got {efficiency_threshold}.")
        efficiency_threshold = jnp.asarray(efficiency_threshold, float_type)
        self.efficiency_threshold = efficiency_threshold

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def preprocess(self, state: NestedSamplerState, live_points: Union[LivePoints, None] = None) -> PreProcessType:
        """
        Produces a data structure that is necessary for sampling to run.
        Typically this is where clustering happens.

        Args:
            state: nested sampler state

        Returns:
            any valid pytree
        """
        ...

    @abstractmethod
    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, live_points: LivePoints,
                   preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            live_points: the current live points reservoir
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        ...


class AbstractRejectionSampler(AbstractSampler):
    """
    Samplers that are based on rejection sampling. They usually first-lines of attack, and are stopped once efficiency
    gets too low.
    """
    pass


class AbstractMarkovSampler(AbstractSampler):
    """
    A sampler that conditions off a known satisfying point, e.g. a seed point.
    """

    def get_seed_point(self, key: PRNGKey, live_points: LivePoints, log_L_constraint: FloatArray) -> SeedPoint:
        """
        Samples a seed point from the live points.

        Args:
            key: PRNGKey
            live_points: the current live point set. All points satisfy the log-L constraint
            log_L_constraint: a log-L constraint to sample within. Note: Currently, redundant because we assume live
                points satisfies the constraint, but in the future, some points may not and this will be used.

        Returns:
            a seed point
        """
        select_mask = live_points.reservoir.log_L > log_L_constraint
        sample_idx = find_first_true_indices(select_mask, N=1)[0]
        sample_idx = random.randint(key, (), minval=0, maxval=live_points.reservoir.log_L.size)
        return SeedPoint(
            U0=live_points.reservoir.point_U[sample_idx],
            log_L0=live_points.reservoir.log_L[sample_idx]
        )

    @abstractmethod
    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            seed_point: function that gets the next sample from a seed point
            log_L_constraint: the constraint to sample within
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        ...

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, live_points: LivePoints,
                   preprocess_data: PreProcessType) -> Sample:
        """
        Produce a single i.i.d. sample from the model within the log_L_constraint.

        Args:
            key: PRNGkey
            log_L_constraint: the constraint to sample within
            live_points: the current live points reservoir
            preprocess_data: the data pytree needed and produced by the sampler

        Returns:
            an i.i.d. sample
        """
        key, seed_key = random.split(key, 2)
        seed_point = self.get_seed_point(key=seed_key, live_points=live_points, log_L_constraint=log_L_constraint)
        return self.get_sample_from_seed(key=key, seed_point=seed_point, log_L_constraint=log_L_constraint,
                                         preprocess_data=preprocess_data)


class AbstractNestedSampler(ABC):
    @abstractmethod
    def __call__(self, key: PRNGKey, term_cond: TerminationCondition, *,
                 init_state: Optional[NestedSamplerState] = None) -> Tuple[IntArray, NestedSamplerState]:
        """
        Performs approximate nested sampling followed by adaptive refinement.

        Args:
            key: PRNGKey
            term_cond: termination condition
            init_state: optional initial state

        Returns:
            termination reason, and exact state
        """
        ...
