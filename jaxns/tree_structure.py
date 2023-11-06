from typing import NamedTuple

from jax import numpy as jnp
from jax._src.lax.control_flow import scan

from jaxns.statistics import compute_num_live_points_from_unit_threads


class SampleTreeGraph(NamedTuple):
    """
    Represents tree structure of samples.
    There are N+1 nodes, and N edges.
    Each node has exactly 1 sender (except the root node).
    Each node has zero or more receivers.
    The root is always node 0.
    """
    sender_idx: jnp.ndarray  # [N] with values in [0, N]
    log_L: jnp.ndarray  # [1+N]


class SampleLivePointCounts(NamedTuple):
    samples_indices: jnp.ndarray  # [N] with values in [0, N], points to the sample that the live point represents.
    num_live_points: jnp.ndarray  # [N] with values in [0, N], number of live points that the sample represents.


def count_crossed_edges(S: SampleTreeGraph) -> SampleLivePointCounts:
    # Construct N edges from N+1 nodes
    N = S.sender_idx.size
    sender = S.sender_idx  # [N]
    sort_idx = jnp.argsort(S.log_L)  # [N+1]

    # Count out-degree of each node, how many nodes have parent_idx==idx.
    # At least one node will have zero, but we don't know which.
    # Could just use sender (unsorted)
    out_degree = jnp.bincount(sender, length=N + 1)  # [N+1]

    def body(crossed_edges, last_node):
        # init = 1
        # delta = degree(nodes[last_node]) - 1
        crossed_edges += out_degree[last_node] - 1
        return crossed_edges, crossed_edges

    _, crossed_edges_sorted = scan(
        body,
        1,
        sort_idx
    )
    # Since the root node is always 0, we need to slice and subtract 1 to get the sample index.
    samples_indices = sort_idx[1:] - 1  # [N]

    # The last node is the accumulation, which is always 0, so we drop it.
    num_live_points = crossed_edges_sorted[:-1]  # [N]
    return SampleLivePointCounts(
        samples_indices=samples_indices,
        num_live_points=num_live_points
    )


def count_crossed_edges_less_fast(S: SampleTreeGraph) -> SampleLivePointCounts:
    # Construct N edges from N+1 nodes
    N = S.sender_idx.size
    sender = S.sender_idx  # [N]
    sort_idx = jnp.argsort(S.log_L)  # [N+1]

    # Count out-degree of each node, how many nodes have parent_idx==idx.
    # At least one node will have zero, but we don't know which.
    # Could just use sender (unsorted)
    out_degree = jnp.bincount(sender, length=N + 1)  # [N+1]

    crossed_edges_sorted = 1 + jnp.cumsum(out_degree[sort_idx] - 1)

    # Since the root node is always 0, we need to slice and subtract 1 to get the sample index.
    samples_indices = sort_idx[1:] - 1  # [N]

    # The last node is the accumulation, which is always 0, so we drop it.
    num_live_points = crossed_edges_sorted[:-1]  # [N]
    return SampleLivePointCounts(
        samples_indices=samples_indices,
        num_live_points=num_live_points
    )


def count_intervals_naive(S: SampleTreeGraph) -> SampleLivePointCounts:
    # We use the simple method, of counting the number that satisfy the selection condition
    log_L_constraints = S.log_L[S.sender_idx]  # [N]
    sort_idx = jnp.argsort(S.log_L[1:])  # [N]
    N = S.sender_idx.size
    available = jnp.ones(N, dtype=jnp.bool_)
    contour = S.log_L[0]  # Root is always 0 index
    num_live_points = jnp.zeros(N, dtype=jnp.int32)
    for i in range(N):
        mask = (log_L_constraints[sort_idx] <= contour) & (S.log_L[1:][sort_idx] > contour) & available[sort_idx]  # [N]
        num_live_points = num_live_points.at[i].set(jnp.sum(mask))
        contour = S.log_L[sort_idx[i] + 1]
        available = available.at[sort_idx[i]].set(False)

    return SampleLivePointCounts(
        samples_indices=sort_idx,
        num_live_points=num_live_points
    )


def count_old(S: SampleTreeGraph) -> SampleLivePointCounts:
    # 3x slower than new method
    log_L_constraints = S.log_L[S.sender_idx]  # [N]
    log_L_samples = S.log_L[1:]
    num_live_points, sort_idx = compute_num_live_points_from_unit_threads(
        log_L_constraints=log_L_constraints,
        log_L_samples=log_L_samples,
        num_samples=None,
        sorted_collection=False
    )
    return SampleLivePointCounts(
        samples_indices=sort_idx,
        num_live_points=num_live_points
    )


def plot_tree(S: SampleTreeGraph):
    """
    Plots the tree where x-position is log_L and y-position is a unique integer for each branch such that no edges
    cross. The y-position should be the same as it's sender's y-position, unless that would make an edge cross,
    in which case, an addition should be made so that no edges cross.

    e.g.

    For the tree:

    S = SampleTree(
        sender_idx=jnp.asarray([0, 0, 0, 1, 2, 3]),
        log_L=jnp.asarray([-jnp.inf, 1, 2, 3, 4, 5, 6])
    )

    The root node connects to nodes 1, 2, 3, and then it's straight lines from 1, 2, 3 to 4, 5, 6.

    The ASCII plot is:

    0 -- 1 -- 4
     \-- 2 -- 5
      \- 3 -- 6

    If we add a branch from 2 to 7 then we get:

    0 --- 1 -- 4
     \--- 2 -- 5
      \   \ -- 7
       \-- 3 -- 6

    See how 2-7 edge doesn't cross any other edges.

    Note the in-degree of each node is 1, except the root node which has in-degree 0.
    The out-degree can be anything.

    Args:
        S: SampleTree
    """
    import networkx as nx
    import pylab as plt

    # Initialize graph
    G = nx.DiGraph()

    # Add edges and nodes to the graph
    for idx, sender in enumerate(S.sender_idx):
        node = idx + 1
        sender = int(sender)
        G.add_node(node, x=float(S.log_L[node]), sender=sender)
        G.add_edge(sender, node)

    G.nodes[0]['x'] = float(S.log_L[0])
    G.nodes[0]['y'] = 0
    G.nodes[0]['sender'] = -1

    out_degree = jnp.bincount(S.sender_idx, length=S.log_L.size)  # [N+1]

    # Dictionary to store the positions of each node

    visited = []
    branch = 0
    for node in nx.traversal.dfs_tree(G, 0):
        if node == 0:
            continue
        # print(G.nodes[node]['sender'], node, visited.count(G.nodes[node]['sender']))
        G.nodes[node]['y'] = G.nodes[G.nodes[node]['sender']]['y']  # + visited.count(G.nodes[node]['sender'])
        # if out_degree[G.nodes[node]['sender']] > 1:
        if visited.count(G.nodes[node]['sender']) > 0:
            branch += visited.count(G.nodes[node]['sender'])
            G.nodes[node]['y'] = branch
        visited.append(G.nodes[node]['sender'])

    pos = dict((node, (G.nodes[node]['x'], G.nodes[node]['y'])) for node in G.nodes)
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', linewidths=0.5, font_size=10, arrows=True)

    # Display the plot
    plt.show()
