from timeit import default_timer

import jax
import numpy as np
from jax import tree_map, numpy as jnp, random, tree_leaves

from jaxns.internals.tree_structure import SampleTreeGraph, SampleLivePointCounts, count_crossed_edges, count_intervals_naive, \
    plot_tree, count_old, count_crossed_edges_less_fast, concatenate_sample_trees, unbatch_state
from jaxns.internals.types import StaticStandardNestedSamplerState, StaticStandardSampleCollection


def test_naive():
    S = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 0, 0, 1, 2, 3]),
        log_L=jnp.asarray([1, 2, 3, 4, 5, 6])
    )
    plot_tree(S)
    expected = SampleLivePointCounts(
        samples_indices=jnp.asarray([0, 1, 2, 3, 4, 5]),
        num_live_points=jnp.asarray([3, 3, 3, 3, 2, 1])
    )
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_intervals_naive(S), expected))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_intervals_naive(S), count_old(S)))


def test_basic():
    S = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 0, 0, 1, 2, 3]),
        log_L=jnp.asarray([1, 2, 3, 4, 5, 6])
    )
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_intervals_naive(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_old(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_crossed_edges_less_fast(S)))

    S = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 0, 0, 1, 3, 2]),
        log_L=jnp.asarray([1, 2, 3, 4, 6, 5])
    )

    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_intervals_naive(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_old(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_crossed_edges_less_fast(S)))


def test_with_num_samples():
    S1 = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 0, 0, 1, 2, 3, 4, 5, 0]),
        log_L=jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, -jnp.inf])
    )
    num_samples = 8

    S2 = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 0, 0, 1, 2, 3, 4, 5]),
        log_L=jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    )

    assert all(tree_map(lambda x, y: np.array_equal(x[:num_samples], y),
                        count_crossed_edges(S1, num_samples),
                        count_crossed_edges(S2)))


def test_random_tree():
    np.random.seed(42)
    log_L = [0]
    parent_idx = []

    for idx in range(10):
        new_log_L = log_L[idx] + np.random.uniform(low=0, high=1 - log_L[idx]) ** 4
        log_L.append(new_log_L)
        parent_idx.append(idx)

    for idx in range(10):
        for _ in range(5):
            new_log_L = np.random.uniform(low=log_L[idx], high=1.)
            log_L.append(new_log_L)
            parent_idx.append(idx)

    S = SampleTreeGraph(
        sender_node_idx=jnp.asarray(parent_idx),
        log_L=jnp.asarray(log_L)[1:]
    )

    plot_tree(S)

    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_intervals_naive(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_old(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_crossed_edges_less_fast(S)))

    T = count_crossed_edges_less_fast(S)
    import pylab as plt
    plt.plot(S.log_L[T.samples_indices], T.num_live_points)
    plt.show()


def test_speed():
    np.random.seed(42)
    log_L = [0]
    parent_idx = []

    for _ in range(100000):
        idx = np.random.choice(len(log_L))
        new_log_L = log_L[idx] + np.random.uniform(low=0.5, high=1.)
        log_L.append(new_log_L)
        parent_idx.append(idx)

    S = SampleTreeGraph(
        sender_node_idx=jnp.asarray(parent_idx),
        log_L=jnp.asarray(log_L)[1:]
    )

    def time_method(f):
        f_jit = jax.jit(f)
        T = f_jit(S)
        T.samples_indices.block_until_ready()
        t0 = default_timer()
        for _ in range(100):
            T = f_jit(S)
            T.samples_indices.block_until_ready()
        dt = (default_timer() - t0) / 100
        print(f.__name__, dt)
        return dt

    # time_method(count_intervals_naive)
    time_method(count_crossed_edges)
    time_method(count_crossed_edges_less_fast)
    time_method(count_old)


def test_concatenate_sample_trees():
    S1 = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 1]),
        log_L=jnp.asarray([1, 2])
    )
    expect = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 1, 0, 3]),
        log_L=jnp.asarray([1, 2, 1, 2])
    )
    S = concatenate_sample_trees([S1, S1])
    assert all(S.sender_node_idx == expect.sender_node_idx)
    assert all(S.log_L == expect.log_L)

    # Three graphs
    S = concatenate_sample_trees([S1, S1, S1])
    expect = SampleTreeGraph(
        sender_node_idx=jnp.asarray([0, 1, 0, 3, 0, 5]),
        log_L=jnp.asarray([1, 2, 1, 2, 1, 2])
    )
    assert all(S.sender_node_idx == expect.sender_node_idx)
    assert all(S.log_L == expect.log_L)


def test_unbatch_state():
    # Example of two batched states, first dimension is has 1 sample, second has 2 samples. Max samples is 2 for both.
    # num_live_points=1 for both.
    single_state_1 = StaticStandardNestedSamplerState(
        key=random.PRNGKey(42),
        next_sample_idx=jnp.asarray(2),
        sample_collection=StaticStandardSampleCollection(
            sender_node_idx=jnp.asarray([0, 1, 0]),
            log_L=jnp.asarray([1., 2., jnp.inf]),
            U_samples=jnp.asarray([1., 1., 0.])[:, None],
            num_likelihood_evaluations=jnp.asarray([1, 2, 0]),
            phantom=jnp.asarray([True, False, False])
        ),
        front_idx=jnp.asarray([1])
    )

    single_state_2 = StaticStandardNestedSamplerState(
        key=random.PRNGKey(42),
        next_sample_idx=jnp.asarray(2),
        sample_collection=StaticStandardSampleCollection(
            sender_node_idx=jnp.asarray([0, 1, 0]),
            log_L=jnp.asarray([1.1, 2.1, jnp.inf]),
            U_samples=jnp.asarray([1., 1., 0.])[:, None],
            num_likelihood_evaluations=jnp.asarray([1, 2, 0]),
            phantom=jnp.asarray([True, False, False])
        ),
        front_idx=jnp.asarray([1])
    )

    batched_state = tree_map(lambda x, y: jnp.stack([x, y], axis=0), single_state_1, single_state_2)

    unbatched_state = unbatch_state(batched_state)

    print(unbatched_state)

    # The second element of first batch is not measure so it should be at end of unbatched
    expected_state = StaticStandardNestedSamplerState(
        key=random.PRNGKey(42),
        next_sample_idx=jnp.asarray(4),
        sample_collection=StaticStandardSampleCollection(
            sender_node_idx=jnp.asarray([0, 1, 0, 0, 4, 0]),
            log_L=jnp.asarray([1., 2., jnp.inf, 1.1, 2.1, jnp.inf]),
            U_samples=jnp.asarray([1., 1., 0., 1., 1., 0.])[:, None],
            num_likelihood_evaluations=jnp.asarray([1, 2, 0, 1, 2, 0]),
            phantom=jnp.asarray([True, False, False, True, False, False])
        ),
        front_idx=jnp.asarray([1, 4])
    )

    expected_state = StaticStandardNestedSamplerState(
        key=random.PRNGKey(42),
        next_sample_idx=jnp.asarray(4),
        sample_collection=StaticStandardSampleCollection(
            sender_node_idx=jnp.asarray([0, 0, 1, 2, 0, 0]),
            log_L=jnp.asarray([1., 1.1, 2., 2.1, jnp.inf, jnp.inf]),
            U_samples=jnp.asarray([1., 1., 1., 1., 0., 0.])[:, None],
            num_likelihood_evaluations=jnp.asarray([1, 1, 2, 2, 0, 0]),
            phantom=jnp.asarray([True, True, False, False, False, False])
        ),
        front_idx=jnp.asarray([2, 3])
    )

    # compare pytrees
    for a, b in zip(tree_leaves(unbatched_state), tree_leaves(expected_state)):
        assert jnp.all(a == b)
