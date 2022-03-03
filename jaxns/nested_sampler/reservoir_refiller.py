from typing import Callable, Dict, Any

from jax import numpy as jnp, random

from jaxns.likelihood_samplers.parallel_slice_sampling import _parallel_sampling
from jaxns.prior_transforms import PriorChain
from jaxns.types import NestedSamplerState


def increase_reservoir(prior_chain: PriorChain,
                       loglikelihood_from_U: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray],
                       state: NestedSamplerState,
                       goal_num_live_points: jnp.ndarray,
                       strict_contour: jnp.ndarray,
                       sampler_name: str,
                       sampler_kwargs: Dict[str, Any]) -> NestedSamplerState:
    """
    Sample points and add to reservoir until there are enough points.
    """

    num_samples = jnp.minimum(goal_num_live_points, state.reservoir.available.size) - jnp.sum(
        state.reservoir.available)
    key, sample_key = random.split(state.key, 2)
    if sampler_name == 'slice':
        next_reservoir = _parallel_sampling(loglikelihood_from_U=loglikelihood_from_U,
                                            prior_chain=prior_chain,
                                            key=sample_key,
                                            log_L_contour=state.log_L_contour,
                                            num_samples=num_samples,
                                            reservoir_state=state.reservoir,
                                            num_slices=sampler_kwargs['num_slices'],
                                            midpoint_shrink=sampler_kwargs['midpoint_shrink'],
                                            num_parallel_samplers=sampler_kwargs['num_parallel_samplers'],
                                            strict_contour=strict_contour)
    else:
        raise ValueError(f"Sampler type {sampler_name} is not implemented in parallel form.")
    # technically not needed to filter
    # next_reservoir = next_reservoir._replace(points_X=_filter_prior_chain(next_reservoir.points_X))
    state = state._replace(key=key, reservoir=next_reservoir)

    return state


class ReservoirRefiller(object):
    """
    Controls refilling reservoir, by applying sampling.
    """
    def __init__(self, prior_chain: PriorChain, loglikelihood_from_U, goal_num_live_points,
                 sampler_name, sampler_kwargs):
        self.prior_chain = prior_chain
        self.loglikelihood_from_U = loglikelihood_from_U
        self.goal_num_live_points = goal_num_live_points
        self.sampler_name = sampler_name
        self.sampler_kwargs = sampler_kwargs

    def __call__(self, state: NestedSamplerState, refill_thread, strict_contour) -> NestedSamplerState:
        """
        Does the refill, optionally, if needed, using a while_loop instead of cond for control flow.

        Args:
            state: NestedSamplerState
            refill_thread: whether to move toward empty reservoir.
            strict_contour: whether to accept points off the contour, needed in some edge cases.

        Returns:
            NestedSamplerState
        """
        return increase_reservoir(prior_chain=self.prior_chain,
                                  loglikelihood_from_U=self.loglikelihood_from_U,
                                  state=state,
                                  goal_num_live_points=jnp.where(refill_thread, self.goal_num_live_points,
                                                                 jnp.asarray(0, jnp.int_)),
                                  strict_contour=strict_contour,
                                  sampler_name=self.sampler_name,
                                  sampler_kwargs=self.sampler_kwargs)