from jax import numpy as jnp, tree_map

from jaxns.internals.types import Reservoir


def sort_reservoir(reservoir: Reservoir) -> Reservoir:
    idx_sort = jnp.lexsort((reservoir.log_L_constraint, reservoir.log_L_samples))
    reservoir = tree_map(lambda x: x[idx_sort], reservoir)
    return reservoir
