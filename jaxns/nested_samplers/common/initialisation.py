from functools import partial
from typing import Optional, Tuple

import jax
from jax import numpy as jnp, random
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.maps import tree_device_put
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.shrinkage_statistics import create_init_evidence_calc
from jaxns.internals.types import PRNGKey
from jaxns.nested_samplers.common.types import StaticStandardSampleCollection, \
    NestedSamplerState, TerminationRegister, LivePointCollection
from jaxns.nested_samplers.common.uniform_sample import draw_uniform_samples


def create_init_state(key: PRNGKey, num_live_points: int, max_samples: int, model: BaseAbstractModel,
                      mesh: Optional[Mesh] = None) -> Tuple[LivePointCollection, NestedSamplerState]:
    """
    Return an initial sample collection, that will be incremented by the sampler.

    Args:
        key: PRNGKey
        num_live_points: the number of live points
        max_samples: the maximum number of samples
        model: the model to use.

    Returns:
        live_point_collection: the initial live point collection
        state: the initial state
    """

    def _repeat(a):
        return jnp.repeat(a[None], repeats=max_samples, axis=0, total_repeat_length=max_samples)

    sample_collection = StaticStandardSampleCollection(
        sender_node_idx=jnp.zeros(max_samples, dtype=mp_policy.index_dtype),
        log_L=jnp.full((max_samples,), jnp.inf, dtype=mp_policy.measure_dtype),
        U_samples=_repeat(model.U_placeholder),
        num_likelihood_evaluations=jnp.full((max_samples,), 0, dtype=mp_policy.count_dtype),
        phantom=jnp.full((max_samples,), False, dtype=jnp.bool_)
    )

    key, sample_key = random.split(key, 2)
    if mesh is not None:
        sharded_keys = tree_device_put(jax.random.split(sample_key, num_live_points), mesh, ('shard',))

        @partial(shard_map, mesh=mesh, in_specs=PartitionSpec('shard', ), out_specs=PartitionSpec('shard', ),
                 check_rep=False)
        def get_init_samples(local_keys):
            return draw_uniform_samples(
                keys=local_keys,
                model=model,
                method='scan'
            )

        live_point_samples = get_init_samples(sharded_keys)
    else:
        keys = jax.random.split(sample_key, num_live_points)
        live_point_samples = draw_uniform_samples(
            keys=keys,
            model=model,
            method='vmap'
        )
    live_point_collection = LivePointCollection(
        sender_node_idx=jnp.full((num_live_points,), 0, dtype=mp_policy.index_dtype),
        U_sample=live_point_samples.U_sample,
        log_L=live_point_samples.log_L,
        log_L_constraint=live_point_samples.log_L_constraint,
        num_likelihood_evaluations=live_point_samples.num_likelihood_evaluations
    )
    sort_indices = jnp.argsort(live_point_samples.log_L)
    live_point_collection = jax.tree_map(lambda x: x[sort_indices], live_point_collection)

    state = NestedSamplerState(
        key=key,
        next_sample_idx=jnp.asarray(0, mp_policy.index_dtype),
        num_samples=jnp.asarray(0, mp_policy.index_dtype),
        sample_collection=sample_collection
    )
    return live_point_collection, state


def create_init_termination_register() -> TerminationRegister:
    """
    Initialise the termination register.

    Returns:
        The initial termination register.
    """
    zero_count = jnp.asarray(0, mp_policy.count_dtype)
    init_evidence_calc = create_init_evidence_calc()
    return TerminationRegister(
        num_samples_used=zero_count,
        evidence_calc=init_evidence_calc,
        evidence_calc_with_remaining=init_evidence_calc,
        num_likelihood_evaluations=zero_count,
        log_L_contour=jnp.asarray(-jnp.inf, mp_policy.measure_dtype),
        efficiency=jnp.asarray(0., mp_policy.measure_dtype),
        plateau=jnp.asarray(False, jnp.bool_),
        no_seed_points=jnp.asarray(False, jnp.bool_),
        relative_spread=jnp.asarray(jnp.inf, mp_policy.measure_dtype),
        absolute_spread=jnp.asarray(jnp.inf, mp_policy.measure_dtype)
    )
