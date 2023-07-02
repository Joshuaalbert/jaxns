import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, vmap, random

from jaxns import StaticNestedSampler, collect_samples
from jaxns.initial_state import init_sample_collection, get_uniform_init_live_points, \
    get_live_points_from_samples, find_first_true_indices
from jaxns.types import NestedSamplerState, SampleCollection, TerminationCondition
from jaxns.uniform_samplers import UniformSampler

tfpd = tfp.distributions


def test_init_sample_collection(basic3_model):
    model = basic3_model
    sample_collection = init_sample_collection(size=10, model=model)
    assert sample_collection.sample_idx == 0
    assert sample_collection.reservoir.log_L.size == 10


def test_get_live_points_from_samples(basic3_model):
    model = basic3_model

    U = jnp.stack(list(
        map(lambda x: x.flatten(), jnp.meshgrid(jnp.linspace(0., 1., 100), jnp.linspace(0., 1., 100), indexing='ij'))),
        axis=1)
    log_L = vmap(model.forward)(U)
    assert jnp.bitwise_not(jnp.any(jnp.isnan(log_L)))

    n = 100
    live_points = get_uniform_init_live_points(random.PRNGKey(43),
                                               num_live_points=n,
                                               model=model)
    init_state = NestedSamplerState(key=random.PRNGKey(42),
                                    sample_collection=SampleCollection(sample_idx=n,
                                                                       reservoir=live_points.reservoir))
    assert np.unique(init_state.sample_collection.reservoir.log_L).size == n
    state, live_points = get_live_points_from_samples(state=init_state,
                                                      log_L_constraint=-jnp.inf,
                                                      num_live_points=n - 1)

    unique = np.unique(live_points.reservoir.log_L)
    assert unique.size == n - 1

    assert np.unique(state.sample_collection.reservoir.log_L).size == 2  # 1 + empty type
    assert state.sample_collection.sample_idx == 1


def test_static_uniform(basic3_model):
    model = basic3_model

    m = 1000
    sample_collection = init_sample_collection(size=m, model=model)

    state = NestedSamplerState(key=random.PRNGKey(42), sample_collection=sample_collection)

    n = 100
    live_points = get_uniform_init_live_points(random.PRNGKey(43),
                                               num_live_points=n,
                                               model=model)
    efficiency_threshold = 0.1
    ns = StaticNestedSampler(
        samplers=[UniformSampler(model=model, efficiency_threshold=efficiency_threshold)],
        num_live_points=n, num_parallel_samplers=1)
    _, state, live_points = ns(state=state, live_points=live_points,
                               termination_cond=TerminationCondition(max_samples=0))
    state = collect_samples(state=state, new_reservoir=live_points.reservoir)
    assert state.sample_collection.sample_idx > 0
    assert jnp.mean(1. / live_points.reservoir.num_likelihood_evaluations) > efficiency_threshold


def test_find_first_true_indices():
    # Usage:
    arr = jnp.array([False, True, False, True, True, False, True, False])
    assert find_first_true_indices(arr, 3).tolist() == [1, 3, 4]
    assert find_first_true_indices(arr, 1).tolist() == [1]
