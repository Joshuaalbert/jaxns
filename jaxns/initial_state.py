import logging
from typing import Tuple, NamedTuple

from etils.array_types import PRNGKey, FloatArray, BoolArray, IntArray
from jax import tree_map, numpy as jnp, random, core
from jax._src.lax.control_flow import scan, while_loop

from jaxns.model import Model
from jaxns.random import resample_indicies
from jaxns.types import Reservoir, SampleCollection, LivePoints, NestedSamplerState, float_type, \
    int_type, Sample
from jaxns.utils import sort_samples

logger = logging.getLogger('jaxns')

__all__ = ['get_live_points_from_samples',
           'get_uniform_init_live_points']


def init_sample_collection(size: int, model: Model) -> SampleCollection:
    """
    Return an initial sample collection, that will be incremented by the sampler.

    Args:
        size: the size of the sample collection.

    Returns:
        sample collection
    """

    def _repeat(a):
        return jnp.repeat(a[None], repeats=size, axis=0, total_repeat_length=size)

    reservoir = Reservoir(
        point_U=_repeat(model.U_placeholder),
        log_L_constraint=jnp.full((size,), jnp.inf, dtype=float_type),
        log_L=jnp.full((size,), jnp.inf, dtype=float_type),
        num_likelihood_evaluations=jnp.full((size,), 0, dtype=int_type),
        num_slices=jnp.full((size,), 0, dtype=int_type),
        iid=jnp.full((size,), False, dtype=jnp.bool_)
    )

    sample_collection = SampleCollection(
        sample_idx=jnp.zeros((), int_type),
        reservoir=reservoir
    )
    return sample_collection


def _single_uniform_sample(key: PRNGKey, model: Model) -> Sample:
    """
    Gets a single sample strictly within -inf bound (the entire prior), i.e. all returned samples will have non-zero
    likeihood.

    Args:
        key: PRNGKey
        model: the model to use.

    Returns:
        a sample
    """

    class CarryState(NamedTuple):
        done: BoolArray
        key: PRNGKey
        U: FloatArray
        log_L: FloatArray
        num_likelihood_evals: IntArray

    def body(carry_state: CarryState):
        key, sample_key = random.split(carry_state.key, 2)
        log_L = model.forward(U=carry_state.U)
        num_likelihood_evals = carry_state.num_likelihood_evals + jnp.asarray(1, int_type)
        done = log_L > -jnp.inf
        U = jnp.where(done, carry_state.U, model.sample_U(key=sample_key))
        return CarryState(done=done, key=key, U=U, log_L=log_L, num_likelihood_evals=num_likelihood_evals)

    key, sample_key = random.split(key, 2)
    init_carry_state = CarryState(done=jnp.asarray(False),
                                  key=key,
                                  U=model.sample_U(key=sample_key),
                                  log_L=-jnp.inf,
                                  num_likelihood_evals=jnp.asarray(0, int_type))

    carry_state = while_loop(lambda s: jnp.bitwise_not(s.done), body, init_carry_state)

    sample = Sample(point_U=carry_state.U,
                    log_L_constraint=-jnp.inf,
                    log_L=carry_state.log_L,
                    num_likelihood_evaluations=carry_state.num_likelihood_evals,
                    num_slices=jnp.asarray(0, int_type),
                    iid=jnp.asarray(True, jnp.bool_))
    return sample


def get_uniform_init_live_points(key: PRNGKey, num_live_points: int, model: Model) -> LivePoints:
    """
    Get initial live points from uniformly sampling the entire prior.

    Args:
        key: PRNGKey
        num_live_points: the number of live points
        model: the model

    Returns:
        live points
    """
    _, samples = scan(lambda s, key: (s, _single_uniform_sample(key=key, model=model)),
                      (),
                      random.split(key, num_live_points))
    reservoir = Reservoir(*samples)
    return LivePoints(reservoir=reservoir)


def sort_sample_collection(sample_collection: SampleCollection) -> SampleCollection:
    """
    Sort a sample collection lexigraphically.

    Args:
        sample_collection: sample collections

    Returns:
        sample collection sorted
    """
    idx_sort = jnp.lexsort((sample_collection.reservoir.log_L_constraint,
                            sample_collection.reservoir.log_L))
    sample_collection = sample_collection._replace(
        reservoir=tree_map(lambda x: x[idx_sort], sample_collection.reservoir)
    )
    return sample_collection


def get_live_points_from_samples(state: NestedSamplerState, log_L_constraint: FloatArray, num_live_points: int,
                                 sorted_collection: bool = True) \
        -> Tuple[NestedSamplerState, LivePoints]:
    """
    Extract live points from the samples already collected.

    Args:
        state: the current state
        log_L_constraint: the contour to sample above
        num_live_points: the number of unique live points to sample
        sorted_collection: whether the sample collection is already sorted

    Returns:
        a new state, and live points
    """
    if isinstance(state.sample_collection.reservoir.iid, core.Tracer):
        raise RuntimeError("Tracer detected, but expected imperative context.")
    ###
    # We select from samples where sample.log_L_constraint <= log_L_constraint
    # 1. sort samples
    # 2. find the largest index, s, where sample.log_L_constraint[s] <= log_L_constraint
    # 3. select without replacement `num_live_points` indices from 0..s (inclusive)
    # 4. replace those indices with an empty sample

    sample_collection = state.sample_collection
    if not sorted_collection:
        sample_collection = sort_sample_collection(sample_collection)

    # find the largest index, s, where sample.log_L_constraint[s] <= log_L_constraint
    idx_supremum = jnp.searchsorted(sample_collection.reservoir.log_L_constraint, log_L_constraint, side='right') - 1
    idx_supremum = int(idx_supremum)
    key, sample_key = random.split(state.key)
    take_indices = resample_indicies(key=sample_key,
                                     log_weights=None,
                                     S=num_live_points,
                                     replace=False,
                                     num_total=idx_supremum + 1)
    reservoir = tree_map(lambda x: x[take_indices], sample_collection.reservoir)
    live_points = LivePoints(reservoir=reservoir)

    # replace the taken indicies
    empty_sample: Reservoir = tree_map(lambda x: jnp.zeros_like(x[-1]), sample_collection.reservoir)
    empty_sample = empty_sample._replace(log_L_constraint=jnp.inf, log_L=jnp.inf)

    sample_collection = sample_collection._replace(
        reservoir=tree_map(lambda x, y: x.at[take_indices].set(y), sample_collection.reservoir, empty_sample),
        sample_idx=sample_collection.sample_idx - num_live_points
    )
    # sort the return
    sample_collection = sort_samples(sample_collection=sample_collection)
    state = state._replace(key=key, sample_collection=sample_collection)
    return state, live_points
