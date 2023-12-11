from typing import TypeVar, Callable, Tuple, Optional

import jax.numpy as jnp
from jax import tree_map, tree_util, lax

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.tree_structure import SampleTreeGraph, count_crossed_edges
from jaxns.internals.types import MeasureType, EvidenceCalculation, float_type, IntArray, FloatArray, int_type

V = TypeVar('V')
Y = TypeVar('Y')


def _cumulative_op_static(op: Callable[[V, Y], V], init: V, xs: Y, pre_op: bool = False, unroll: int = 1) -> Tuple[
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

    final_accumulate, result = lax.scan(
        f=body,
        init=init,
        xs=xs,
        unroll=unroll
    )

    return final_accumulate, result


def test_cumulative_op_static():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, float_type)
    xs = jnp.asarray([1, 2, 3], float_type)
    final_accumulate, result = _cumulative_op_static(op=op, init=init, xs=xs)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], float_type))

    final_accumulate, result = _cumulative_op_static(op=op, init=init, xs=xs, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], float_type))


def _cumulative_op_dynamic(op: Callable[[V, Y], V], init: V, xs: Y, stop_idx: IntArray, pre_op: bool = False) -> Tuple[
    V, V]:
    """
    Compute a cumulative operation on a list of values with a dynamic stop index.

    Args:
        op: the operation to perform
        init: the initial value
        xs: the list of values
        stop_idx: how many accumulations to perform
        pre_op: if True, the operation is applied before the accumulation, so the first value is the initial value.

    Returns:
        the final accumulated value, and the result of the cumulative operation applied on input
    """

    def cond(carry: Tuple[V, IntArray, V]):
        (accumulate, i, output) = carry
        return jnp.less(i, stop_idx)

    def body(carry: Tuple[V, IntArray, V]):
        (accumulate, i, output) = carry
        y = tree_map(lambda x: x[i], xs)
        next_accumulate = op(accumulate, y)
        next_i = i + jnp.ones_like(i)
        if pre_op:
            next_output = tree_map(lambda a, b: a.at[i].set(b), output, accumulate)
            return (next_accumulate, next_i, next_output)
        next_output = tree_map(lambda a, b: a.at[i].set(b), output, next_accumulate)
        return (next_accumulate, next_i, next_output)

    length = tree_util.tree_flatten(xs)[0][0].shape[0]

    output = tree_map(lambda x: jnp.tile(x[None], [length] + [1] * len(x.shape)), init)

    w_init = (init, jnp.asarray(0, int_type), output)

    (final_accumulate, _, final_output) = lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=w_init
    )

    return final_accumulate, final_output


def test_cumulative_op_dynamic():
    def op(accumulate, y):
        return accumulate + y

    init = jnp.asarray(0, float_type)
    xs = jnp.asarray([1, 2, 3], float_type)
    stop_idx = jnp.asarray(3, jnp.int_)
    final_accumulate, result = _cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([1, 3, 6], float_type))

    final_accumulate, result = _cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 6
    assert all(result == jnp.asarray([0, 1, 3], float_type))

    stop_idx = jnp.asarray(2, jnp.int_)
    final_accumulate, result = _cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([1, 3, 0], float_type))

    final_accumulate, result = _cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx, pre_op=True)
    assert final_accumulate == 3
    assert all(result == jnp.asarray([0, 1, 0], float_type))


def compute_enclosed_prior_volume(sample_tree: SampleTreeGraph) -> MeasureType:
    """
    Compute the enclosed prior volume of the likelihood constraint.

    Args:
        sample_tree: The sample tree graph.

    Returns:
        The log enclosed prior volume.
    """
    live_point_counts = count_crossed_edges(sample_tree=sample_tree)

    def op(log_X, num_live_points):
        X_mean = LogSpace(log_X)
        # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
        # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
        T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
        next_X_mean = X_mean * T_mean
        return next_X_mean.log_abs_val

    _, log_X = _cumulative_op_static(op=op, init=jnp.asarray(-jnp.inf, float_type),
                                     xs=live_point_counts.num_live_points)
    return log_X


def compute_evidence_stats(log_L: MeasureType, num_live_points: FloatArray, num_samples: Optional[IntArray] = None) -> \
Tuple[EvidenceCalculation, EvidenceCalculation]:
    """
    Compute the evidence statistics along the shrinkage process.

    Args:
        log_L: The log likelihoods of the samples.
        num_live_points: The number of live points at each sample.
        num_samples: The number of samples to use. If None, all samples are used.

    Returns:
        The final evidence statistics, and the evidence statistics for each sample.
    """

    def op(carry: EvidenceCalculation, y) -> EvidenceCalculation:
        (num_live_points, log_next_L) = y

        # num_live_points = num_live_points.astype(float_type)
        next_L = LogSpace(log_next_L)
        L_contour = LogSpace(carry.log_L)
        midL = LogSpace(jnp.log(0.5)) * (next_L + L_contour)
        X_mean = LogSpace(carry.log_X_mean)
        X2_mean = LogSpace(carry.log_X2_mean)
        Z_mean = LogSpace(carry.log_Z_mean)
        ZX_mean = LogSpace(carry.log_ZX_mean)
        Z2_mean = LogSpace(carry.log_Z2_mean)
        dZ2_mean = LogSpace(carry.log_dZ2_mean)

        # T_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.))
        # T_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points))
        T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
        # T_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)))
        t_mean = LogSpace(- jnp.log(num_live_points + 1.))
        # T2_mean = LogSpace(jnp.log(num_live_points) - jnp.log( num_live_points + 2.))
        # T2_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 2./num_live_points))
        T2_mean = LogSpace(- jnp.logaddexp((0.), jnp.log(2.) - jnp.log(num_live_points)))
        # T2_mean = LogSpace(- jnp.logaddexp(jnp.log(2.), -jnp.log(num_live_points)))
        t2_mean = LogSpace(jnp.log(2.) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
        # tT_mean = LogSpace(jnp.log(num_live_points) - jnp.log(num_live_points + 1.) - jnp.log(num_live_points + 2.))
        # tT_mean = LogSpace(jnp.log(1.) - jnp.log(1. + 1./num_live_points) - jnp.log(num_live_points + 2.))
        tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))
        # tT_mean = LogSpace(- jnp.logaddexp(0., -jnp.log(num_live_points)) - jnp.log(num_live_points + 2.))

        dZ_mean = X_mean * t_mean * midL
        next_X_mean = X_mean * T_mean
        next_X2_mean = X2_mean * T2_mean
        next_Z_mean = Z_mean + dZ_mean
        next_ZX_mean = ZX_mean * T_mean + X2_mean * tT_mean * midL
        next_Z2_mean = Z2_mean + LogSpace(jnp.log(2.)) * ZX_mean * t_mean * midL + (X2_mean * t2_mean * midL ** 2)
        next_dZ2_mean = dZ2_mean + (X2_mean * t2_mean * midL ** 2)

        next_evidence_calculation = EvidenceCalculation(
            log_L=log_next_L.astype(float_type),
            log_X_mean=next_X_mean.log_abs_val.astype(float_type),
            log_X2_mean=next_X2_mean.log_abs_val.astype(float_type),
            log_Z_mean=next_Z_mean.log_abs_val.astype(float_type),
            log_Z2_mean=next_Z2_mean.log_abs_val.astype(float_type),
            log_ZX_mean=next_ZX_mean.log_abs_val.astype(float_type),
            log_dZ_mean=dZ_mean.log_abs_val.astype(float_type),
            log_dZ2_mean=next_dZ2_mean.log_abs_val.astype(float_type)
        )

        return next_evidence_calculation

    evidence_calculation = EvidenceCalculation(
        log_L=jnp.asarray(-jnp.inf, float_type),
        log_X_mean=jnp.asarray(0., float_type),
        log_X2_mean=jnp.asarray(0., float_type),
        log_Z_mean=jnp.asarray(-jnp.inf, float_type),
        log_ZX_mean=jnp.asarray(-jnp.inf, float_type),
        log_Z2_mean=jnp.asarray(-jnp.inf, float_type),
        log_dZ_mean=jnp.asarray(-jnp.inf, float_type),
        log_dZ2_mean=jnp.asarray(-jnp.inf, float_type)
    )

    init = evidence_calculation
    xs = (num_live_points, log_L)
    if num_samples is not None:
        stop_idx = num_samples
        final_accumulate, result = _cumulative_op_dynamic(op=op, init=init, xs=xs, stop_idx=stop_idx)
    else:
        final_accumulate, result = _cumulative_op_static(op=op, init=init, xs=xs)
    final_evidence_calculation = final_accumulate
    per_sample_evidence_calculation = result
    return final_evidence_calculation, per_sample_evidence_calculation
