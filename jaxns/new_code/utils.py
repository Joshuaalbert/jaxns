from jax import numpy as jnp, tree_map

from jaxns.internals.maps import replace_index
from jaxns.new_code.types import SampleCollection, NestedSamplerState, Reservoir


def sort_samples(sample_collection:SampleCollection):
    idx_sort = jnp.lexsort((sample_collection.reservoir.log_L_constraint,
                            sample_collection.reservoir.log_L))
    sample_collection = sample_collection._replace(
        reservoir=tree_map(lambda x: x[idx_sort], sample_collection.reservoir)
    )
    return sample_collection


def collect_samples(state: NestedSamplerState, new_reservoir: Reservoir) -> NestedSamplerState:
    """
    Merge samples from new reservoir into sample collection

    Args:
        state: state
        new_reservoir: reservoir to add to sample collection

    Returns:
        state with sorted sample collection
    """
    old_reservoir = state.sample_collection.reservoir
    # Insert the new samples with a slice update
    reservoir = tree_map(lambda old, new: replace_index(old, new, state.sample_collection.sample_idx),
                         old_reservoir, new_reservoir)
    # Update the sample collection
    sample_collection = state.sample_collection._replace(
        reservoir=reservoir,
        sample_idx=state.sample_collection.sample_idx + new_reservoir.log_L.size
    )
    # Sort
    sample_collection = sort_samples(sample_collection)
    state = state._replace(sample_collection=sample_collection)
    return state
