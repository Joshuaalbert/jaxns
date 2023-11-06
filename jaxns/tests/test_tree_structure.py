from timeit import default_timer

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_map

from jaxns.tree_structure import SampleTreeGraph, SampleLivePointCounts, count_crossed_edges, count_intervals_naive, \
    plot_tree, count_old, count_crossed_edges_less_fast


def test_naive():
    S = SampleTreeGraph(
        sender_idx=jnp.asarray([0, 0, 0, 1, 2, 3]),
        log_L=jnp.asarray([-1, 1, 2, 3, 4, 5, 6])
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
        sender_idx=jnp.asarray([0, 0, 0, 1, 2, 3]),
        log_L=jnp.asarray([-1, 1, 2, 3, 4, 5, 6])
    )
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_intervals_naive(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_old(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_crossed_edges_less_fast(S)))

    S = SampleTreeGraph(
        sender_idx=jnp.asarray([0, 0, 0, 1, 3, 2]),
        log_L=jnp.asarray([-1, 1, 2, 3, 4, 6, 5])
    )
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_intervals_naive(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_old(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_crossed_edges_less_fast(S)))


def test_random_tree():
    np.random.seed(42)
    log_L = [0]
    parent_idx = []

    for _ in range(100):
        idx = np.random.choice(len(log_L))
        new_log_L = log_L[idx] + np.random.uniform(low=0.5, high=1.)
        log_L.append(new_log_L)
        parent_idx.append(idx)

    S = SampleTreeGraph(
        sender_idx=jnp.asarray(parent_idx),
        log_L=jnp.asarray(log_L)
    )

    plot_tree(S)

    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_intervals_naive(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_old(S)))
    assert all(tree_map(lambda x, y: np.array_equal(x, y), count_crossed_edges(S), count_crossed_edges_less_fast(S)))

    T = count_crossed_edges_less_fast(S)
    import pylab as plt
    plt.plot(S.log_L[1:][T.samples_indices], T.num_live_points)
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
        sender_idx=jnp.asarray(parent_idx),
        log_L=jnp.asarray(log_L)
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
