from jax import random

from jaxns.abc import AbstractMarkovSampler, SeedPoint, PreProcessType
from jaxns.initial_state import find_first_true_indices
from jaxns.types import FloatArray, PRNGKey, LivePoints, Sample


class BaseMarkovSampler(AbstractMarkovSampler):
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
