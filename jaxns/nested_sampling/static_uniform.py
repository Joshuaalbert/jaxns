from typing import Tuple

from etils.array_types import BoolArray
from jax import random, numpy as jnp, tree_map
from jax.lax import while_loop

from jaxns.internals.maps import replace_index
from jaxns.nested_sampling.model import Model
from jaxns.nested_sampling.types import Sample, NestedSamplerState, LivePoints, Reservoir
from jaxns.nested_sampling.uniform_sampler import UniformSampler

__all__ = ['StaticUniform']


class StaticUniform:
    def __init__(self, model: Model, num_live_points: int, efficiency_threshold: float):
        self.uniform_sampler = UniformSampler(model=model)
        self.model = model
        self.num_live_points = num_live_points
        if efficiency_threshold <= 0.:
            raise ValueError(f"Efficiency threshold should be > 0, got {efficiency_threshold}.")
        self.efficiency_threshold = efficiency_threshold

    def __call__(self, state: NestedSamplerState, live_points: LivePoints) -> Tuple[NestedSamplerState, LivePoints]:
        if live_points.reservoir.log_L.size != self.num_live_points:
            raise ValueError(
                f"live points reservoir is the wrong size. "
                f"Got {live_points.reservoir.log_L.size} by expected {self.num_live_points}.")

        CarryType = Tuple[BoolArray, NestedSamplerState, LivePoints]

        def body(carry: CarryType) -> CarryType:
            (_, state, live_points) = carry
            idx_min = jnp.argmin(live_points.reservoir.log_L)
            dead_point: Sample = tree_map(lambda x: x[idx_min], live_points.reservoir)

            sample_collection_reservoir = tree_map(
                lambda old, update: replace_index(old, update, state.sample_collection.sample_idx),
                state.sample_collection.reservoir, dead_point)

            sample_collection = state.sample_collection._replace(reservoir=sample_collection_reservoir,
                                                                 sample_idx=state.sample_collection.sample_idx + 1)

            log_L_dead = dead_point.log_L

            # contour becomes log_L_dead if log_L_dead is not supremum of live-points, else we choose the original
            # constraint of dead point. Note: we are at liberty to choose any log_L level as a contour so long as we
            # can sample within it uniformly.
            on_supremum = jnp.equal(log_L_dead, jnp.max(live_points.reservoir.log_L))
            log_L_contour = jnp.where(on_supremum, dead_point.log_L_constraint, log_L_dead)

            # replace dead point with a new sample about contour
            key, sample_key = random.split(state.key, 2)
            sample = self.uniform_sampler.single_sample(key=sample_key, log_L_constraint=log_L_contour)
            live_points_reservoir = tree_map(lambda old, update: replace_index(old, update, idx_min),
                                             live_points.reservoir, Reservoir(*sample))
            live_points = live_points._replace(reservoir=live_points_reservoir)
            state = state._replace(key=key,
                                   sample_collection=sample_collection)
            # done = sample.num_likelihood_evaluations >= 1. / self.efficiency_threshold
            # done = jnp.mean(live_points.reservoir.num_likelihood_evaluations) >= 1. / self.efficiency_threshold
            done = sample_collection.sample_idx > self.num_live_points / self.efficiency_threshold  # log(X) = -N/n = 1/eff
            return (done, state, live_points)

        (_, state, live_points) = while_loop(lambda carry: jnp.bitwise_not(carry[0]),
                                             body, (jnp.asarray(False, jnp.bool_), state, live_points))
        return (state, live_points)
