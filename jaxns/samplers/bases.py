from abc import abstractmethod
from typing import NamedTuple, Tuple

from jax import numpy as jnp
from jax import random

from jaxns.model.bases import BaseAbstractModel
from jaxns.samplers.abc import AbstractSampler, SamplerState
from jaxns.types import FloatArray, float_type, Sample, StaticStandardNestedSamplerState
from jaxns.types import PRNGKey


class BaseAbstractSampler(AbstractSampler):
    def __init__(self, model: BaseAbstractModel):
        self.model = model

    def __repr__(self):
        return f"{self.__class__.__name__}"


class SeedPoint(NamedTuple):
    U0: FloatArray
    log_L0: FloatArray


class BaseAbstractRejectionSampler(BaseAbstractSampler):
    """
    Samplers that are based on rejection sampling. They usually first-lines of attack, and are stopped once efficiency
    gets too low.
    """
    pass


class BaseAbstractMarkovSampler(BaseAbstractSampler):
    """
    A sampler that conditions off a known satisfying point, e.g. a seed point.
    """

    @abstractmethod
    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             sampler_state: SamplerState) -> Tuple[Sample, Sample]:
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

    def get_seed_point(self, key: PRNGKey, state: StaticStandardNestedSamplerState,
                       log_L_constraint: FloatArray) -> SeedPoint:
        """
        Samples a seed point from the live points.

        Args:
            key: PRNGKey
            state: the current nested sampler state
            log_L_constraint: a log-L constraint to sample within. Must always be at least one sample in front above
                this to avoid infinite loop.

        Returns:
            a seed point
        """

        unnorm_select_prob = (state.sample_collection.log_L[state.front_idx] > log_L_constraint).astype(float_type)
        # Choose randomly where mask is True
        g = random.gumbel(key, shape=unnorm_select_prob.shape)
        sample_idx = state.front_idx[jnp.argmax(g + jnp.log(unnorm_select_prob))]

        return SeedPoint(
            U0=state.sample_collection.U_samples[sample_idx],
            log_L0=state.sample_collection.log_L[sample_idx]
        )

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, sampler_state: SamplerState) -> Tuple[
        Sample, Sample]:
        sample_key, seed_key = random.split(key, 2)
        state: StaticStandardNestedSamplerState
        (state,) = sampler_state
        seed_point = self.get_seed_point(
            key=seed_key,
            state=state,
            log_L_constraint=log_L_constraint
        )
        return self.get_sample_from_seed(
            key=sample_key,
            seed_point=seed_point,
            log_L_constraint=log_L_constraint,
            sampler_state=sampler_state
        )
