from typing import TypeVar, Callable, Optional

import jax
import jax.tree
from jax import numpy as jnp
from jax._src.pjit import auto_axes
from jax._src.sharding_impls import canonicalize_sharding

from jaxns.mixed_precision import mp_policy
from jaxns.types import IntArray

X = TypeVar('X')
V = TypeVar('V')
Y = TypeVar('Y')


def cumulative_op_static(op: Callable[[V, Y], V], init: V, xs: Y, pre_op: bool = False, unroll: int = 1) -> tuple[
    V, V]:
    """
    Compute a cumulative operation on a list of values.

    Args:
        op: the operation to perform
        init: the initial value
        xs: the list of values
        pre_op: if True, the operation is applied before the accumulation, so the first value is the initial value.
        unroll: how many iterations to unroll the loop at a time

    Returns:
        the final accumulated value, and the result of the cumulative operation applied on input
    """

    def body(accumulate: V, y: Y):
        next_accumulate = op(accumulate, y)
        if pre_op:
            return next_accumulate, accumulate
        return next_accumulate, next_accumulate

    final_accumulate, result = jax.lax.scan(
        f=body,
        init=init,
        xs=xs,
        unroll=unroll
    )

    return final_accumulate, result


def cumulative_op_dynamic(op: Callable[[V, Y], V], init: V, xs: Y, stop_idx: IntArray, pre_op: bool = False,
                          empty_fill: Optional[V] = None) -> tuple[
    V, V]:
    """
    Compute a cumulative operation on a list of values with a dynamic stop index.

    Args:
        op: the operation to perform
        init: the initial value
        xs: the list of values
        stop_idx: how many accumulations to perform
        pre_op: if True, the operation is applied before the accumulation, so the first value is the initial value.
        empty_fill: the value to fill the output with if the stop_idx is provided, else uses `init`

    Returns:
        the final accumulated value, and the result of the cumulative operation applied on input
    """

    def cond(carry: tuple[V, IntArray, V]):
        (accumulate, i, output) = carry
        return jnp.less(i, stop_idx)

    def body(carry: tuple[V, IntArray, V]):
        (accumulate, i, output) = carry
        y = jax.tree.map(lambda x: x[i], xs)
        next_accumulate = op(accumulate, y)
        next_i = i + jnp.ones_like(i)
        if pre_op:
            next_output = jax.tree.map(lambda a, b: a.at[i].set(b), output, accumulate)
            return (next_accumulate, next_i, next_output)
        next_output = jax.tree.map(lambda a, b: a.at[i].set(b), output, next_accumulate)
        return (next_accumulate, next_i, next_output)

    length = jax.tree.flatten(xs)[0][0].shape[0]

    output = jax.tree.map(
        lambda x: jnp.tile(x[None], [length] + [1] * len(x.shape)),
        empty_fill if empty_fill is not None else init
    )

    w_init = (init, jnp.asarray(0, mp_policy.count_dtype), output)

    (final_accumulate, _, final_output) = jax.lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=w_init
    )

    return final_accumulate, final_output


def scan_associative_cumulative_op(op: Callable[[V, Y], V], init: V, xs: Y, pre_op: bool = False) -> tuple[V, V]:
    """
    Compute cumulative operation via ``jax.lax.associative_scan``.
    """

    stacked_xs = jax.tree.map(lambda x, i: jnp.concatenate([i[None, ...], x], axis=0), xs, init)
    try:
        cumulative = jax.lax.associative_scan(op, stacked_xs)
    except Exception:
        return cumulative_op_static(op=op, init=init, xs=xs, pre_op=pre_op)
    final_accumulate = jax.tree.map(lambda x: x[-1], cumulative)
    if pre_op:
        result = jax.tree.map(lambda x: x[:-1], cumulative)
    else:
        result = jax.tree.map(lambda x: x[1:], cumulative)
    return final_accumulate, result


def scan_or_while_loop(scan_fn, carry_init, xs, length: IntArray | None = None, unroll: int = 1) -> tuple:
    # if length is None or static use scan, other wise use while_loop
    if length is None or isinstance(length, int):
        return jax.jax.lax.scan(scan_fn, carry_init, xs, length=length, unroll=unroll)

    def cond_fn(carry):
        i, _, _ = carry
        return i < length

    def body_fn(carry):
        i, carry_inner, ys = carry
        x = jax.tree.map(lambda x: x[i], xs)
        carry_inner, y = scan_fn(carry_inner, x)
        ys = jax.tree.map(lambda y, y_new: y.at[i, ...].set(y_new), ys, y)
        return (i + 1, carry_inner, ys)

    # aeval to build ys structure
    max_length = jax.tree.leaves(xs)[0].shape[0]
    carry_struct, ys_struct = jax.eval_shape(scan_fn, carry_init, jax.tree.map(lambda x: x[0], xs))
    ys_init = jax.tree.map(lambda y: jnp.zeros((max_length,) + y.shape, dtype=y.dtype), ys_struct)
    carry = (0, carry_init, ys_init)
    _, carry_inner, ys = jax.jax.lax.while_loop(cond_fn, body_fn, carry)
    return carry_inner, ys


def unzip2(pairs):
    xs, ys = zip(*pairs) if pairs else ((), ())
    return tuple(xs), tuple(ys)


def _scan_leaf(leaf, batch_elems, num_batches, batch_size):
    def f(l):
        return l[:batch_elems].reshape(num_batches, batch_size, *l.shape[1:])

    aval = jax.typeof(leaf)
    if aval.sharding.spec[0] is not None:
        raise ValueError(
            '0th dimension of leaf passed to `jax.jax.lax.map` should be replicated.'
            f' Got {aval.str_short(True, True)}')
    out_s = aval.sharding.update(spec=(None, None, *aval.sharding.spec[1:]))
    out_s = canonicalize_sharding(out_s, 'jax.lax.map')
    if out_s is not None and out_s.mesh._any_axis_explicit:
        return auto_axes(f, out_sharding=out_s, axes=out_s.mesh.explicit_axes)(leaf)
    return f(leaf)


def _remainder_leaf(leaf, batch_elems):
    def f(l):
        return l[batch_elems:]

    sharding = canonicalize_sharding(jax.typeof(leaf).sharding, 'jax.lax.map')
    if sharding is not None and sharding.mesh._any_axis_explicit:
        return auto_axes(f, out_sharding=sharding,
                         axes=sharding.mesh.explicit_axes)(leaf)
    return f(leaf)


def _batch_and_remainder(x, batch_size: int):
    leaves, treedef = jax.tree.flatten(x)
    if not leaves:
        return x, None
    num_batches, remainder = divmod(leaves[0].shape[0], batch_size)
    batch_elems = num_batches * batch_size
    if num_batches == 0:
        remainder_leaves = [_remainder_leaf(leaf, batch_elems) for leaf in leaves]
        return None, treedef.unflatten(remainder_leaves)
    if remainder:
        scan_leaves, remainder_leaves = unzip2([
            (_scan_leaf(leaf, batch_elems, num_batches, batch_size),
             _remainder_leaf(leaf, batch_elems))
            for leaf in leaves
        ])
        return treedef.unflatten(scan_leaves), treedef.unflatten(remainder_leaves)
    else:
        scan_leaves = tuple(_scan_leaf(leaf, batch_elems, num_batches, batch_size)
                            for leaf in leaves)
        return treedef.unflatten(scan_leaves), None


def _reduce_tree_axis0(reduce_fn, tree):
    return jax.tree.map(lambda leaf: reduce_fn(leaf, axis=0), tree)


def _merge_two(a, b, reduce_fn):
    # Fast path for jnp.sum-equivalent reduction to avoid stacking
    if reduce_fn is jnp.sum:
        return jax.tree.map(lambda x, y: x + y, a, b)
    elif reduce_fn is jnp.mean:
        return jax.tree.map(lambda x, y: (x + y) / 2, a, b)
    elif reduce_fn is jnp.prod:
        return jax.tree.map(lambda x, y: x * y, a, b)
    elif reduce_fn is jnp.max:
        return jax.tree.map(lambda x, y: jnp.maximum(x, y), a, b)
    elif reduce_fn is jnp.min:
        return jax.tree.map(lambda x, y: jnp.minimum(x, y), a, b)
    else:
        return jax.tree.map(
            lambda x, y: reduce_fn(jnp.stack([x, y], axis=0), axis=0),
            a, b
        )


def _tree_add(a, b):
    return jax.tree.map(lambda x, y: x + y, a, b)


def batch_reduce(
        f,
        xs,
        *,
        reduce_fn=jnp.sum,
        batch_size: int | None = None,
        vectorised_kernel: bool = False,
):
    """
    Reduce over leading axis of xs by applying `f` and then reducing along axis 0
    of the mapped outputs.

    Semantics:
      - If batch_size is None: behave like jax.lax.map over the leading axis of xs.
          * If vectorised_kernel=False: call f(x_i) for each element x_i
          * If vectorised_kernel=True: call f(x_i[None,...]) for each element (batch of 1)
        Then reduce across the mapped outputs.
      - If batch_size is not None: chunk xs into [num_batches, batch_size, ...] plus remainder.
          * If vectorised_kernel=False: vmap(f) over each chunk
          * If vectorised_kernel=True: call f(chunk) directly
        Reduce within each chunk (axis 0), then merge the partial reductions.

    Note: `reduce_fn=jnp.mean` is handled as a weighted mean across unequal chunk sizes.
    """

    # --------------------
    # jax.lax.map-style (no batching) case
    # --------------------
    if batch_size is None:
        leaves, _ = jax.tree.flatten(xs)
        if not leaves or leaves[0].shape[0] == 0:
            raise ValueError("batch_reduce: cannot reduce empty input.")

        # Important: even for vectorised kernels, batch_size=None means "map",
        # but we call the kernel with an explicit batch axis of size 1.
        batch_size = 1
        # Fall through to batched logic.

    # --------------------
    # batched case
    # --------------------
    scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)

    # Special-case: weighted mean (mean is not associative under chunking).
    if reduce_fn is jnp.mean:
        total_sum = None
        total_count = 0

        if scan_xs is not None:
            scan_leaves, _ = jax.tree.flatten(scan_xs)
            num_batches = scan_leaves[0].shape[0]
            count_scan = num_batches * batch_size

            def per_batch(_, batch_chunk):
                if vectorised_kernel:
                    batch_ys = f(batch_chunk)  # [B,...]
                else:
                    batch_ys = jax.vmap(f)(batch_chunk)  # [B,...]
                batch_sum = _reduce_tree_axis0(jnp.sum, batch_ys)
                return (), batch_sum

            _, batch_sums = jax.jax.lax.scan(per_batch, (), scan_xs)  # [num_batches,...]
            sum_scan = _reduce_tree_axis0(jnp.sum, batch_sums)  # [...]

            total_sum = sum_scan
            total_count += count_scan

        if remainder_xs is not None:
            rem_leaves, _ = jax.tree.flatten(remainder_xs)
            R = rem_leaves[0].shape[0]
            if R > 0:
                if vectorised_kernel:
                    rem_ys = f(remainder_xs)
                else:
                    rem_ys = jax.vmap(f)(remainder_xs)
                rem_sum = _reduce_tree_axis0(jnp.sum, rem_ys)

                total_sum = rem_sum if total_sum is None else _tree_add(total_sum, rem_sum)
                total_count += R

        if total_sum is None or total_count == 0:
            raise ValueError("batch_reduce: cannot reduce empty input.")

        return jax.tree.map(lambda s: s / total_count, total_sum)

    # Generic associative-ish reductions (sum/prod/min/max/other via stacking merge).
    partial_reduced_chunks = []

    if scan_xs is not None:
        def per_batch(_, batch_chunk):
            if vectorised_kernel:
                batch_ys = f(batch_chunk)
            else:
                batch_ys = jax.vmap(f)(batch_chunk)
            batch_red = _reduce_tree_axis0(reduce_fn, batch_ys)  # [...]
            return (), batch_red

        _, batch_reduced = jax.jax.lax.scan(per_batch, (), scan_xs)  # [num_batches,...]
        batch_total = _reduce_tree_axis0(reduce_fn, batch_reduced)  # [...]
        partial_reduced_chunks.append(batch_total)

    if remainder_xs is not None:
        if vectorised_kernel:
            remainder_ys = f(remainder_xs)
        else:
            remainder_ys = jax.vmap(f)(remainder_xs)
        remainder_red = _reduce_tree_axis0(reduce_fn, remainder_ys)
        partial_reduced_chunks.append(remainder_red)

    if not partial_reduced_chunks:
        raise ValueError("batch_reduce: cannot reduce empty input.")

    out = partial_reduced_chunks[0]
    for pr in partial_reduced_chunks[1:]:
        out = _merge_two(out, pr, reduce_fn)
    return out
