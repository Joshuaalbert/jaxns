import functools
from typing import TypeVar, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def _slice_along_axis(x, start=0, stop=None, step=1, axis=0):
    """Slices a Tensor along the given axis."""

    def _slice(x):
        if axis >= 0:
            slices = [slice(None)] * axis + [slice(start, stop, step)]
        else:
            slices = [Ellipsis, slice(start, stop, step)] + [slice(None)] * (-1 - axis)
        return x[tuple(slices)]

    return jax.tree.map(_slice, x)


def _interleave(a, b, axis):
    """Interleaves two `Tensor`s along the given axis."""

    # [a b c ...] [d e f ...] -> [a d b e c f ...]

    def _op(a, b):
        if np.shape(a) != np.shape(b):
            raise ValueError('`a` and `b` must have the same shape.')

        stacked = jnp.stack([a, b], axis=axis + 1)

        output_shape = np.shape(a)[:axis] + (2 * np.shape(a)[axis],) + np.shape(a)[axis + 1:]
        return lax.reshape(stacked, output_shape)

    return jax.tree.map(_op, a, b)


X = TypeVar('X')


def scan_associative(fn: Callable[[X, X], X], elems: X, axis: int = 0) -> X:
    """
    Perform a scan with an associative binary operation, in parallel.

    Suitable for fn: (X, X) -> X where (f o f) o f == f o (f o f)

    The associative scan operation computes the cumulative sum, or
    [all-prefix sum](https://en.wikipedia.org/wiki/Prefix_sum), of a set of
    elements under an associative binary operation [1]. For example, using the
    ordinary addition operator `fn = lambda a, b: a + b`, this is equivalent to
    the ordinary cumulative sum.

    The associative structure allows the computation to be decomposed
    and executed by parallel reduction. Where a naive sequential
    implementation would loop over all `N` elements, this method requires
    only a logarithmic number (`2 * ceil(log_2 N)`) of sequential steps, and
    can thus yield substantial performance speedups from hardware-accelerated
    vectorization. The total number of invocations of the binary operation
    (including those performed in parallel) is
    `2 * (N / 2 + N / 4 + ... + 1) = 2N - 2`
    --- i.e., approximately twice as many as a naive approach.

    [1] Blelloch, Guy E.
        [Prefix sums and their applications](
        https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
        Technical Report CMU-CS-90-190,
        School of Computer Science,
        Carnegie Mellon University, 1990.

    Args:
        fn: the associative binary operation to perform.
        elems: [..., N, ...] the input elements to scan over.
        axis: the axis to scan over.

    Returns:
        [..., N, ...] cumulative operation applied on input.
    """

    # This is a rewrite of tfp.math.scan_associative, which relies on broadcasting, to better handle vectorisation.
    # This tends to be more performant for smaller ops,
    # and TFP's version tends to be more performant for larger ops by about 10%.
    # Equivalent performance for 100 matmuls of 100x100 matrices
    # (See test_performance for more on performance.)

    _num_elements = np.shape(jax.tree.leaves(elems)[0])[axis]
    if _num_elements < 2:
        raise ValueError(f"Must be at least 2 elements, got {_num_elements}.")

    odd_slice = functools.partial(_slice_along_axis, start=1, stop=None, step=2, axis=axis)
    even_slice = functools.partial(_slice_along_axis, start=0, stop=None, step=2, axis=axis)
    trim_slice = functools.partial(_slice_along_axis, start=0, stop=-1, step=1, axis=axis)
    pretrim_slice = functools.partial(_slice_along_axis, start=1, stop=None, step=1, axis=axis)
    zero_slice = functools.partial(_slice_along_axis, start=0, stop=1, step=1, axis=axis)
    last_slice = functools.partial(_slice_along_axis, start=-1, stop=None, step=1, axis=axis)

    # The basic idea of the algorithm is as follows, using "sum" to represent the associative operation:
    # 1. The cumulative sum is: y0=x0, y1=x0+x1, y2=x0+x1+x2, y3=x0+x1+x2+x3
    # 2. Thus odd indices are equivalent to a two part process:
    #       a. Sum even+odd inputs: (x0+x1), (x2+x3)
    #       b. recursively apply `_scan` on the result to give: y1=x0+x1, y3=x0+x1+x2+x3
    # 3. The even indices are then found by:
    #       a. adding even indices (ignoring x0 in this step): y2=y1+x2
    #       b. preconcatenating x0: y0=x0
    # 4. The result of a single `_scan` is then interleaving the above: _interleave([y0, y1, y2], [y1, y2, y3])

    # TODO: If the associative identity, `id`, was provided then we could combine 3b into 3a via: y0=id+x0

    def _scan(elems):
        """Perform scan on `elems`."""
        # Compute _scan(op(elem_odds, elem_evens)) -> odd indicies
        num_elements = np.shape(jax.tree.leaves(elems)[0])[axis]
        if num_elements == 0:
            raise RuntimeError(f"Got zero length `elems`")
        elif num_elements == 1:
            return elems
        elif num_elements == 2:
            elems_0 = jax.tree.map(zero_slice, elems)
            elems_1 = jax.tree.map(last_slice, elems)
            consecutive_sum = jax.vmap(fn, in_axes=axis, out_axes=axis)(
                elems_0,
                elems_1
            )  # x0+x1
            _results = [
                elems_0,  # x0
                consecutive_sum  # x0+x1
            ]
            results = jax.tree.map(
                lambda *x: jnp.concatenate(x, axis=axis),
                *_results
            )
            return results

        is_even = num_elements % 2 == 0
        if is_even:
            # Want odd number of elements
            last_elem = jax.tree.map(last_slice, elems)
            elems = jax.tree.map(lambda *x: jnp.concatenate(x, axis=axis), elems, last_elem)
            # elems = jax.tree.map(trim_slice, elems)  # x0, x1, x2, x3, x4, x5, x6 (+ x7)
        elems_odd = jax.tree.map(odd_slice, elems)  # x1, x3, x5
        elems_even = jax.tree.map(even_slice, elems)  # x0, x2, x4, x6
        consecutive_sum = jax.vmap(fn, in_axes=axis, out_axes=axis)(
            jax.tree.map(trim_slice, elems_even),  # x0, x2, x4
            elems_odd  # x1, x3, x5
        )  # x0+x1, x2+x3, x4+x5
        result_odd = _scan(consecutive_sum)  # x0+x1, x0+x1+x2+x3, x0+x1+x2+x3+x4+x5

        result_even = jax.vmap(fn, in_axes=axis, out_axes=axis)(
            result_odd,
            jax.tree.map(pretrim_slice, elems_even)
        )  # x0+x1+x2, x0+x1+x2+x3+x4, x0+x1+x2+x3+x4+x5+x6

        partial_results = _interleave(
            result_odd,
            result_even,
            axis=axis
        )
        # x0+x1, x0+x1+x2,
        # x0+x1+x2+x3, x0+x1+x2+x3+x4,
        # x0+x1+x2+x3+x4+x5, x0+x1+x2+x3+x4+x5+x6

        _results = [
            jax.tree.map(zero_slice, elems),  # x0
            partial_results
        ]

        results = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=axis),
            *_results
        )

        if is_even:
            results = jax.tree.map(trim_slice, results)

        return results

    result = _scan(elems)

    return result
