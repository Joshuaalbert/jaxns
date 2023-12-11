from typing import NamedTuple, List, Tuple, Union, Optional

from jax import numpy as jnp, lax, tree_map

from jaxns.internals.maps import remove_chunk_dim
from jaxns.internals.types import MeasureType, IntArray, float_type, FloatArray, StaticStandardNestedSamplerState, int_type


class SampleTreeGraph(NamedTuple):
    """
    Represents tree structure of samples.
    There are N+1 nodes, and N edges.
    Each node has exactly 1 sender (except the root node).
    Each node has zero or more receivers.
    The root is always node 0.
    """
    sender_node_idx: IntArray  # [N] with values in [0, N]
    log_L: MeasureType  # [N]


class SampleLivePointCounts(NamedTuple):
    samples_indices: IntArray  # [N] with values in [0, N], points to the sample that the live point represents.
    num_live_points: IntArray  # [N] with values in [0, N], number of live points that the sample represents.


def count_crossed_edges(sample_tree: SampleTreeGraph, num_samples: Optional[IntArray] = None) -> SampleLivePointCounts:
    N = sample_tree.sender_node_idx.size

    if num_samples is not None:
        # We put edges from root to +inf for the indices that are not used.
        # Since all lines will cross these injected edges, we subtract them from the total.

        mask = jnp.arange(N) < num_samples
        sample_tree = SampleTreeGraph(
            sender_node_idx=jnp.where(mask, sample_tree.sender_node_idx, 0),
            log_L=jnp.where(mask, sample_tree.log_L, jnp.inf)
        )

    # Construct N edges from N+1 nodes

    log_L_nodes = jnp.concatenate([jnp.asarray([-jnp.inf], float_type), sample_tree.log_L])  # [N+1]

    sender = sample_tree.sender_node_idx  # [N]
    sort_idx = jnp.argsort(log_L_nodes)  # [N+1]

    # Count out-degree of each node, how many nodes have parent_idx==idx.
    # At least one node will have zero, but we don't know which.
    # Could just use sender (unsorted)
    out_degree = jnp.bincount(sender, length=N + 1)  # [N+1]

    def body(crossed_edges, last_node):
        # init = 1
        # delta = degree(nodes[last_node]) - 1
        crossed_edges += out_degree[last_node] - 1
        return crossed_edges, crossed_edges

    _, crossed_edges_sorted = lax.scan(
        body,
        jnp.asarray(1, out_degree.dtype),
        sort_idx
    )

    if num_samples is not None:
        substract_edges = N - num_samples
        crossed_edges_sorted -= substract_edges

    # Since the root node is always 0, we need to slice and subtract 1 to get the sample index.
    samples_indices = sort_idx[1:] - 1  # [N]

    # The last node is the accumulation, which is always 0, so we drop it.
    num_live_points = crossed_edges_sorted[:-1]  # [N]
    return SampleLivePointCounts(
        samples_indices=samples_indices,
        num_live_points=num_live_points
    )


def count_crossed_edges_less_fast(S: SampleTreeGraph) -> SampleLivePointCounts:
    log_L = jnp.concatenate([-jnp.inf * jnp.ones(1), S.log_L])  # [N+1]
    # Construct N edges from N+1 nodes
    N = S.sender_node_idx.size
    sender = S.sender_node_idx  # [N]
    sort_idx = jnp.argsort(log_L)  # [N+1]

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
    log_L = jnp.concatenate([-jnp.inf * jnp.ones(1), S.log_L])  # [N+1]
    log_L_constraints = log_L[S.sender_node_idx]  # [N]
    sort_idx = jnp.argsort(log_L[1:])  # [N]
    N = S.sender_node_idx.size
    available = jnp.ones(N, dtype=jnp.bool_)
    contour = log_L[0]  # Root is always 0 index
    num_live_points = jnp.zeros(N, dtype=jnp.int32)
    for i in range(N):
        mask = (log_L_constraints[sort_idx] <= contour) & (log_L[1:][sort_idx] > contour) & available[sort_idx]  # [N]
        num_live_points = num_live_points.at[i].set(jnp.sum(mask))
        contour = log_L[sort_idx[i] + 1]
        available = available.at[sort_idx[i]].set(False)

    return SampleLivePointCounts(
        samples_indices=sort_idx,
        num_live_points=num_live_points
    )


def fast_perfect_live_point_computation_jax(log_L_constraints: jnp.ndarray, log_L_samples: jnp.ndarray,
                                            num_samples: Union[jnp.ndarray, None] = None):
    # log_L_constraints has shape [N]
    # log_L_samples has shape [N]
    sort_idx = jnp.lexsort((log_L_constraints, log_L_samples))
    log_L_samples = log_L_samples[sort_idx]
    log_L_constraints = log_L_constraints[sort_idx]
    log_L_contour = log_L_constraints[0]
    search_contours = jnp.concatenate([log_L_contour[None], log_L_samples], axis=0)

    contour_map_idx = jnp.searchsorted(search_contours, log_L_samples, side='left') - 1
    log_L_contours = search_contours[contour_map_idx]
    diag_i = jnp.arange(log_L_samples.size)
    right_most_idx = jnp.searchsorted(jnp.sort(log_L_constraints), log_L_contours, side='right') - 1
    left_most_idx = jnp.maximum(diag_i, jnp.searchsorted(log_L_samples, log_L_contours, side='right') - 1)
    num_live_points = jnp.maximum(0, right_most_idx - left_most_idx + 1)

    if num_samples is not None:
        empty_mask = jnp.greater_equal(jnp.arange(log_L_samples.size), num_samples)
        num_live_points = jnp.where(empty_mask, jnp.asarray(0., log_L_samples.dtype), num_live_points)

    return num_live_points, sort_idx


def compute_num_live_points_from_unit_threads(log_L_constraints: FloatArray, log_L_samples: FloatArray,
                                              num_samples: IntArray = None, sorted_collection: bool = True) \
        -> Union[FloatArray, Tuple[FloatArray, IntArray]]:
    """
    Compute the number of live points of shrinkage distribution, from an arbitrary list of samples with
    corresponding sampling constraints.

    Args:
        log_L_constraints: [N] likelihood constraint that sample was uniformly sampled within
        log_L_samples: [N] likelihood of the sample
        sorted_collection: bool, whether the sample collection was already sorted.

    Returns:
        if sorted_collection is true:
            num_live_points for shrinkage distribution
        otherwise:
            num_live_points for shrinkage distribution, and sort indicies
    """
    num_live_points, sort_idx = fast_perfect_live_point_computation_jax(log_L_constraints=log_L_constraints,
                                                                        log_L_samples=log_L_samples,
                                                                        num_samples=num_samples)

    if not sorted_collection:
        return num_live_points, sort_idx

    return num_live_points


def count_old(S: SampleTreeGraph) -> SampleLivePointCounts:
    log_L = jnp.concatenate([-jnp.inf * jnp.ones(1), S.log_L])  # [N+1]
    # 3x slower than new method
    log_L_constraints = log_L[S.sender_node_idx]  # [N]
    log_L_samples = S.log_L
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
    r"""
    Plots the tree where x-position is log_L and y-position is a unique integer for each branch such that no edges
    cross. The y-position should be the same as it's sender's y-position, unless that would make an edge cross,
    in which case, an addition should be made so that no edges cross.

    e.g.

    For the tree:

    S = SampleTree(
        sender_node_idx=jnp.asarray([0, 0, 0, 1, 2, 3]),
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
    for idx, sender in enumerate(S.sender_node_idx):
        node = idx + 1
        sender = int(sender)
        G.add_node(node, x=float(S.log_L[node]), sender=sender)
        G.add_edge(sender, node)

    G.nodes[0]['x'] = float(jnp.min(S.log_L) - 1.)
    G.nodes[0]['y'] = 0
    G.nodes[0]['sender'] = -1

    out_degree = jnp.bincount(S.sender_node_idx, length=S.log_L.size)  # [N+1]

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


def concatenate_sample_trees(trees: List[SampleTreeGraph],
                             num_samples: Optional[List[IntArray]] = None) -> SampleTreeGraph:
    """
    Concatenates a list of SampleTreeGraphs into a single SampleTreeGraph.

    The root nodes of each tree must be the same, as this is equivalent of adding each tree as a branch from the same
    root node, 0 in each tree.

    Args:
        trees: list of SampleTreeGraph's

    Returns:
        a single tree
    """
    # To do this all sender_node_idx must point to proper new nodes:
    # Example:
    # 0 -- 1 -- 2
    # 0 -- 1 -- 2

    # Becomes:
    # 0 -- 1 -- 2
    #  \-- 3 -- 4

    if num_samples is None:
        num_samples = [t.sender_node_idx.size for t in trees]

    if len(num_samples) != len(trees):
        raise ValueError("num_samples must be same length as trees.")

    offset = 0
    shifted_trees = []
    for s, t in zip(num_samples, trees):
        shifted_trees.append(
            SampleTreeGraph(
                sender_node_idx=jnp.where(t.sender_node_idx.astype(jnp.bool_), t.sender_node_idx + offset,
                                          t.sender_node_idx),
                log_L=t.log_L
            )
        )
        offset += s

    output = SampleTreeGraph(
        sender_node_idx=jnp.concatenate([t.sender_node_idx for t in shifted_trees]),
        log_L=jnp.concatenate([t.log_L for t in shifted_trees])
    )
    return output


def unbatch_state(batched_state: StaticStandardNestedSamplerState) -> StaticStandardNestedSamplerState:
    """
    Remove the batch dimension from the state. The returned samples will be sorted by log_L,
    so assumes,

        log_L[i]==+inf ==> i is not a sample

    Args:
        batched_state: the state with batch dimension

    Returns:
        the state without batch dimension
    """
    if len(batched_state.sample_collection.log_L.shape) == 1:
        # Already unbatched
        return batched_state

    if batched_state.sample_collection.log_L.shape[0] == 1:
        # Remove batch dimension is all that's needed
        return remove_chunk_dim(batched_state)

    key = batched_state.key[0]  # Take first key
    next_sample_idx = jnp.sum(batched_state.next_sample_idx)  # Next insert will be sum
    # Shifts are the cumulative sum of the number of samples per batch dimension
    shifts = [0]
    for i in range(len(batched_state.next_sample_idx) - 1):
        shifts.append(shifts[-1] + batched_state.sample_collection.log_L.shape[1])
    shifts = jnp.asarray(shifts, int_type)

    # shifts = jnp.concatenate([jnp.asarray([0], int_type), jnp.cumsum(batched_state.next_sample_idx[:-1])])
    sender_node_idx = jnp.where(
        batched_state.sample_collection.sender_node_idx.astype(jnp.bool_),
        batched_state.sample_collection.sender_node_idx + shifts[:, None],
        batched_state.sample_collection.sender_node_idx
    )
    # Front indices are shifted like senders
    front_idx = remove_chunk_dim(
        batched_state.front_idx + shifts[:, None]
    )

    unbatched_state = StaticStandardNestedSamplerState(
        key=key,
        next_sample_idx=next_sample_idx,
        sample_collection=remove_chunk_dim(
            batched_state.sample_collection._replace(sender_node_idx=sender_node_idx)
        ),
        front_idx=front_idx
    )

    # Some non-samples will interleave samples, so we sort by log_L, carefully adjusting sender_node_idx to match.

    sort_idx = jnp.argsort(unbatched_state.sample_collection.log_L)
    inverse_idx = jnp.argsort(sort_idx)

    # Shift the front_idx and sender idx
    front_idx = inverse_idx[unbatched_state.front_idx]
    sender_node_idx = jnp.where(
        unbatched_state.sample_collection.sender_node_idx.astype(jnp.bool_),
        inverse_idx[unbatched_state.sample_collection.sender_node_idx - 1] + 1,
        jnp.zeros_like(unbatched_state.sample_collection.sender_node_idx)
    )
    unbatched_state = unbatched_state._replace(
        sample_collection=unbatched_state.sample_collection._replace(sender_node_idx=sender_node_idx)
    )

    # Rearrange the samples
    unbatched_state = unbatched_state._replace(
        sample_collection=tree_map(lambda x: x[sort_idx], unbatched_state.sample_collection),
        front_idx=front_idx
    )
    return unbatched_state
