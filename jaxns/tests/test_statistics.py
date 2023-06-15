from time import monotonic_ns
from typing import Union

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from etils.array_types import PRNGKey
from jax import numpy as jnp, random, tree_map
from jax import vmap
from jax.lax import dynamic_update_slice

from jaxns import TerminationCondition, Reservoir, ApproximateNestedSampler, Model, Prior, PriorModelGen
from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.maps import replace_index
from jaxns.static_uniform import BadUniformSampler
from jaxns.statistics import compute_num_live_points_from_unit_threads, compute_remaining_evidence, \
    perfect_live_point_computation_jax, fast_perfect_live_point_computation_jax, fast_triu_rowsum, combine_reservoirs

tfpd = tfp.distributions


def _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples, num_samples=None, debug=False):
    if num_samples is None:
        num_samples = log_L_samples.size

    empty_mask = jnp.arange(log_L_samples.size) >= num_samples
    log_L_samples = jnp.where(empty_mask, jnp.inf, log_L_samples)
    idx_sort = jnp.lexsort((log_L_constraints, log_L_samples))
    empty_mask = empty_mask[idx_sort]
    num_live_points = []

    available = jnp.bitwise_not(empty_mask)  # jnp.ones(log_L_samples.shape, jnp.bool_)
    log_L_samples = log_L_samples[idx_sort]
    log_L_constraints = log_L_constraints[idx_sort]
    contour = log_L_constraints[0]

    if debug:
        print("###")
        print(f"log_L_constraints: {log_L_constraints}")
        print(f"log_L_samples: {log_L_samples}")
    for idx in range(log_L_samples.size):
        # by definition the number of live points is count of points where all three conditions are true:
        # 1) not yet consumed (once consumed no longer valid)
        # 2) sampling constraint of sample are <= contour (uniform sampling within a smaller contour also uniformly samples within a larger contour)
        # 3) log(L) of sample >= current contour (we shrink upwards in likelihood, the equality makes this work for plateaus)
        mask = available & (log_L_samples >= contour) & (log_L_constraints <= contour)
        n = jnp.sum(mask)
        # n = jnp.sum(available & (log_L_constraints <= contour)) # also valid because we sort samples
        if debug:
            print("---")
            print(f"L*: {contour} n:{n}")
            for j in range(log_L_samples.size):
                s = f"{'*' if mask[j] else ' '}\t{j}: available: [{'x' if available[j] else ''}] " \
                    f"log(L)>contour ({log_L_samples[j]},{contour}): [{'x' if log_L_samples[j] > contour else ''}] " \
                    f"L*<=contour ({log_L_constraints[j]},{contour}): [{'x' if log_L_constraints[j] <= contour else ''}]"
                print(s)

        contour = log_L_samples[idx]
        available = dynamic_update_slice(available, jnp.asarray([False]), [idx])
        num_live_points.append(n)

    num_live_points = jnp.stack(num_live_points)

    return num_live_points


def perfect_live_point_computation(log_L_constraints, log_L_samples, num_samples: Union[int, None] = None, debug=True):
    log_L_constraints = np.array(log_L_constraints)
    log_L_samples = np.array(log_L_samples)
    sort_idx = np.lexsort((log_L_constraints, log_L_samples))

    log_L_contour = log_L_constraints[sort_idx[0]]

    num_live_points = []
    idx = 0
    while idx < log_L_samples.size:
        log_L_dead = log_L_samples[sort_idx[idx]]
        while idx < log_L_samples.size:
            if log_L_samples[sort_idx[idx]] != log_L_dead:
                break
            count = np.sum(np.bitwise_and(log_L_contour < log_L_samples, log_L_constraints <= log_L_contour))
            if debug:
                print('index', sort_idx[idx], 'dead point', log_L_dead, count, ' points within', log_L_contour)
            log_L_samples[sort_idx[idx]] = -np.inf
            log_L_constraints[sort_idx[idx]] = np.inf
            num_live_points.append(count)
            idx += 1
        log_L_contour = log_L_dead
    if debug:
        print(num_live_points)
    num_live_points = jnp.asarray(num_live_points)
    if num_samples is not None:
        empty_mask = np.greater_equal(np.arange(log_L_samples.size), num_samples)
        num_live_points = np.where(empty_mask, 0, num_live_points)
    return num_live_points, sort_idx


def test_perfect_live_point_computation_jax():
    log_L_constraints = random.uniform(random.PRNGKey(42), shape=(100,))
    log_L_samples = random.uniform(random.PRNGKey(43), shape=(100,), minval=log_L_constraints)
    num_live_points, sort_idx = perfect_live_point_computation(log_L_constraints, log_L_samples)
    num_live_points_jax, sort_idx_jax = perfect_live_point_computation_jax(log_L_constraints, log_L_samples)
    assert jnp.all(num_live_points_jax == num_live_points)


def test_fast_perfect_live_point_computation_jax():
    for i in range(10):
        log_L_constraints = random.uniform(random.PRNGKey(i), shape=(100,))
        log_L_samples = random.uniform(random.PRNGKey(i + 100000), shape=(100,), minval=log_L_constraints)
        num_live_points, sort_idx = perfect_live_point_computation(log_L_constraints, log_L_samples, debug=False)
        num_live_points_jax, sort_idx_jax = perfect_live_point_computation_jax(log_L_constraints, log_L_samples)
        num_live_points_fast, sort_idx = fast_perfect_live_point_computation_jax(log_L_constraints, log_L_samples)
        assert jnp.all(num_live_points_jax == num_live_points)
        assert jnp.all(num_live_points_fast == num_live_points)

    for n in [1e3, 1e4, 2e4]:
        f1 = jax.jit(perfect_live_point_computation_jax)
        f2 = jax.jit(fast_perfect_live_point_computation_jax)
        log_L_constraints = random.uniform(random.PRNGKey(42), shape=(int(n),))
        log_L_samples = random.uniform(random.PRNGKey(420), shape=(int(n),), minval=log_L_constraints)
        num_live_points, _ = f1(log_L_constraints, log_L_samples)
        num_live_points.block_until_ready()
        t0 = monotonic_ns()
        for _ in range(10):
            num_live_points, _ = f1(log_L_constraints, log_L_samples)
            num_live_points.block_until_ready()
        dt_normal = (monotonic_ns() - t0) / 1e9 / 10
        print(f"Normal {n}: {dt_normal} sec")

        num_live_points, _ = f2(log_L_constraints, log_L_samples)
        num_live_points.block_until_ready()
        t0 = monotonic_ns()
        for _ in range(10):
            num_live_points, _ = f2(log_L_constraints, log_L_samples)
            num_live_points.block_until_ready()
        dt_fast = (monotonic_ns() - t0) / 1e9 / 10
        print(f"Fast {n}: {dt_fast} sec")

        assert dt_fast < dt_normal

    # ensure scaling law is sub linear
    n_array = [1e3, 1e4, 1e5, 1e6]
    dt_array = []
    for n in n_array:
        num_live_points, _ = f2(log_L_constraints, log_L_samples)
        num_live_points.block_until_ready()
        t0 = monotonic_ns()
        for _ in range(10):
            num_live_points, _ = f2(log_L_constraints, log_L_samples)
            num_live_points.block_until_ready()
        dt_fast = (monotonic_ns() - t0) / 1e9 / 10
        print(f"Fast {n}: {dt_fast} sec")
        dt_array.append(dt_fast)

    def is_sub_linear(n_array, dt_array):
        if len(n_array) != len(dt_array):
            raise ValueError("Both arrays must have the same length.")

        if len(n_array) < 2:
            raise ValueError("At least two data points are needed to determine the scaling law.")

        sub_linear = True

        for i in range(1, len(n_array)):
            n_ratio = n_array[i] / n_array[i - 1]
            dt_ratio = dt_array[i] / dt_array[i - 1]
            print(n_ratio, dt_ratio)

            # If the ratio of execution times is greater than or equal to the ratio of input sizes,
            # it is not sub-linear.
            if dt_ratio >= n_ratio:
                sub_linear = False
                break

        return sub_linear

    assert is_sub_linear(n_array, dt_array)


def _sure_infimum_constraint(log_L_constraints, log_L_samples, sort_idx):
    """
    For a single sample find `i` such that log_L_contours[i] is the greatest strict lower bound of log_L_sample.

    Args:
        log_L_constraints:
        log_L_samples:
        sort_idx:
    """
    log_L_constraints = log_L_constraints[sort_idx]
    log_L_samples = log_L_samples[sort_idx]
    log_L_contours = jnp.concatenate([log_L_constraints[0:1], log_L_samples[:-1]], axis=0)
    print("###")
    print(f'log_L_contours:{log_L_contours}')
    print(f'log_L_samples:{log_L_samples}')
    indices = []
    indices = jnp.arange(log_L_contours.size)
    for _ in range(log_L_samples.size):
        # indices = jnp.where((log_L_contours[indices]==log_L_contours[indices-1]) | (log_L_contours[indices] == log_L_samples), indices-1,indices)
        indices = jnp.clip(jnp.where((log_L_contours[indices] == log_L_samples), indices - 1, indices), -1,
                           log_L_samples.size - 1)

    print(f"indices:{indices}")
    return jnp.asarray(indices)


def test_fast_triu_rowsum():
    def exact(a, b):
        M = a[:, None] * b[None, :]
        res = []
        for i in range(a.size):
            res.append(np.sum(M[i, i:]))
        return np.asarray(res)

    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])

    assert np.allclose(fast_triu_rowsum(a, b), exact(a, b))


def test_fast_sum_inequality():
    def exact(a, b):
        res = []
        for i in range(a.size):
            res.append(jnp.sum(a[i] <= b[i:]))
        return jnp.asarray(res)

    def upper_triangle_sum(a, b):
        b_minus_cumsum_a = b - np.cumsum(a)
        count_positive = np.cumsum(np.concatenate(([0], b_minus_cumsum_a[:-1])) >= 0)
        result = count_positive - np.concatenate(([0], count_positive[:-1]))
        return result

    a = np.array([1, 2, 4, 5])
    b = np.array([3, 4, 5, 6])

    print(exact(a, b))
    print(upper_triangle_sum(a, b))


def test_compute_num_live_points():
    from jax import random

    def run_test(log_L_constraints, log_L_samples, num_live_points, num_samples=None, debug=False):
        print("log_L_samples", log_L_samples, 'log_L_constraints', log_L_constraints)
        num_live_points_perfect, sort_idx_perfect = perfect_live_point_computation(log_L_constraints, log_L_samples,
                                                                                   num_samples=num_samples)
        assert np.all(num_live_points_perfect == num_live_points)
        print('1 passed')

        num_live_points_perfect_jax, sort_idx_perfect_jax = perfect_live_point_computation_jax(log_L_constraints,
                                                                                               log_L_samples,
                                                                                               num_samples=num_samples)
        # assert np.all(num_live_points_perfect_jax == num_live_points)
        # print('2 passed')

        num_live_points_perfect_jax_fast, sort_idx_perfect_jax_fast = fast_perfect_live_point_computation_jax(
            log_L_constraints,
            log_L_samples,
            num_samples=num_samples)
        assert np.all(num_live_points_perfect_jax_fast == num_live_points)
        print('3 passed')

        # assert jnp.allclose(num_live_points,
        #                     _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples,
        #                                                                     num_samples, debug=False))

        # test_num_live_points, sort_idx = compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples,
        #                                                                            num_samples, sorted_collection=False)
        # assert jnp.all(num_live_points_perfect == test_num_live_points)
        # assert jnp.all(infimum_constraint(log_L_constraints, log_L_samples, sort_idx) == _sure_infimum_constraint(
        #     log_L_constraints, log_L_samples, sort_idx))

    # constraints are previous samples
    log_L_constraints = jnp.asarray([0., 0.5, 2.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([1, 1, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are previous samples, with extra unfilled slots
    log_L_constraints = jnp.asarray([0., 0.5, 2., jnp.inf])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, jnp.inf])
    num_live_points = jnp.asarray([1, 1, 1, 0])
    run_test(log_L_constraints, log_L_samples, num_live_points, num_samples=3, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0.45, 0., 2.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([2, 1, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0.45, 0., 2., jnp.inf, jnp.inf])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, jnp.inf, jnp.inf])
    num_live_points = jnp.asarray([2, 1, 1, 0, 0])
    run_test(log_L_constraints, log_L_samples, num_live_points, num_samples=3, debug=True)

    log_L_constraints = jnp.asarray([0.45, 0.45, 0.5, 2., jnp.inf, jnp.inf])
    log_L_samples = jnp.asarray([0.5, 2., 2., 2.1, jnp.inf, jnp.inf])
    num_live_points = jnp.asarray([2, 2, 1, 1, 0, 0])
    run_test(log_L_constraints, log_L_samples, num_live_points, num_samples=4, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0., 0.45, 0., 2.])
    log_L_samples = jnp.asarray([0.5, 0.55, 2., 2.1])
    num_live_points = jnp.asarray([2, 2, 1, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0., 0., 0., 0.])
    log_L_samples = jnp.asarray([0.5, 0.5, 0.5, 0.5])
    num_live_points = jnp.asarray([4, 3, 2, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0., 0.45, 0.45, 0.])
    log_L_samples = jnp.asarray([0.5, 0.5, 0.5, 0.5])
    num_live_points = jnp.asarray([2, 1, 0, 0])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # All one common constraint, no plateau in samples
    log_L_constraints = jnp.asarray([0., 0., 0.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([3, 2, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # All one common constraint, no plateau in samples
    log_L_constraints = jnp.asarray([0., 0., 0., jnp.inf])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, jnp.inf])
    num_live_points = jnp.asarray([3, 2, 1, 0])
    run_test(log_L_constraints, log_L_samples, num_live_points, num_samples=3, debug=True)

    # Growing
    log_L_constraints = jnp.asarray([0., 0.5, 0.5])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([1, 2, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # Growing with plateau
    log_L_constraints = jnp.asarray([0., 0.5, 0.5, 2.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, 2.1])
    num_live_points = jnp.asarray([1, 2, 2, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # All one common constraint, a plateau in samples
    log_L_constraints = jnp.asarray([0., 0., 0., 0.])
    log_L_samples = jnp.asarray([0.5, 2., 2., 2.1])
    num_live_points = jnp.asarray([4, 3, 2, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # # one plateau -- broken assumption
    # log_L_constraints = jnp.asarray([0., 0., 0.])
    # log_L_samples = jnp.asarray([0., 0., 0.])
    # num_live_points = jnp.asarray([0, 0, 0])
    # run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # one plateau
    log_L_constraints = jnp.asarray([0., 0., 0.])
    log_L_samples = jnp.asarray([0.1, 0.1, 0.1])
    num_live_points = jnp.asarray([3, 2, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are not other samples
    # 0. (1) 0.5 (1) 1.5 (1) 2.5
    log_L_constraints = jnp.asarray([0., 0.25, 0.75])
    log_L_samples = jnp.asarray([0.5, 1.5, 2.5])
    num_live_points = jnp.asarray([1, 1, 1])
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # random generated example, no plateaus
    log_L_constraints = random.uniform(random.PRNGKey(42), shape=(100,))
    log_L_samples = random.uniform(random.PRNGKey(43), shape=(100,), minval=log_L_constraints)
    num_live_points = _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples)
    run_test(log_L_constraints, log_L_samples, num_live_points)

    # random generated example, no plateaus
    log_L_constraints = jnp.concatenate([jnp.zeros(5), 0.5 * jnp.arange(5) / 5 + 0.5])
    log_L_samples = random.uniform(random.PRNGKey(43), shape=(10,), minval=log_L_constraints)
    num_live_points = _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples)
    run_test(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # print(num_live_points)
    # import pylab as plt
    # plt.plot(num_live_points)
    # plt.show()


def test_standard_nested_sampling():
    n = 0.5

    def log_likelihood(x):
        return 1. - x ** n

    def sample(key, log_L):
        # sample from 1 - x**n > log_L
        upper = (1. - log_L) ** (1. / n)
        return (1. - random.uniform(key)) * upper

    num_live_points = 10
    key = random.PRNGKey(42)
    num_samples = 100
    log_L_constraints = []
    log_L_samples = []
    U_samples = []

    key, sample_key = random.split(key, 2)

    U_samples_live = random.uniform(sample_key, (num_live_points,))
    log_L_samples_live = vmap(log_likelihood)(U_samples_live)
    log_L_constraints_live = jnp.asarray([0.] * num_live_points)

    while len(log_L_samples) < num_samples:
        idx_min = jnp.argmin(log_L_samples_live)
        log_L_samples.append(log_L_samples_live[idx_min])
        log_L_constraints.append(log_L_constraints_live[idx_min])
        U_samples.append(U_samples_live[idx_min])

        key, sample_key = random.split(key, 2)
        U_sample = sample(sample_key, log_L_samples[-1])
        log_L_sample = log_likelihood(U_sample)

        assert log_L_sample > log_L_samples[-1], "Numerical precision problem with unittest, don't shrink so much."

        log_L_samples_live = replace_index(log_L_samples_live, log_L_sample, idx_min)
        U_samples_live = replace_index(U_samples_live, U_sample, idx_min)
        log_L_constraints_live = replace_index(log_L_constraints_live, log_L_samples[-1], idx_min)

    for idx_min in jnp.argsort(log_L_samples_live):
        log_L_samples.append(log_L_samples_live[idx_min])
        log_L_constraints.append(log_L_constraints_live[idx_min])
        U_samples.append(U_samples_live[idx_min])

    n = compute_num_live_points_from_unit_threads(jnp.asarray(log_L_constraints), jnp.asarray(log_L_samples))
    for ni, log_L_constraint, log_L_sample in zip(n, jnp.asarray(log_L_constraints), jnp.asarray(log_L_samples)):
        # should be all `num_live_points` except last bit which decreases to 1
        print(ni, log_L_constraint, log_L_sample)

    assert jnp.allclose(n[:num_samples], num_live_points)

    n_check = _sure_compute_num_live_points_from_unit_threads(jnp.asarray(log_L_constraints),
                                                              jnp.asarray(log_L_samples))
    assert jnp.allclose(n, n_check)


def infimum_constraint(log_L_constraints, log_L_samples, sort_idx=None, return_contours: bool = False):
    """
    For a single sample find `i` such that log_L_contours[i] is the greatest strict lower bound of log_L_sample.
    E.g.

    Args:
        log_L_constraints:
        log_L_samples:
        sort_idx:
        return_contours: if true also return the value of the contour at `i`, i.e. the constraint.
    """
    if sort_idx is not None:
        log_L_constraints = log_L_constraints[sort_idx]
        log_L_samples = log_L_samples[sort_idx]
    # mask the non-samples, already done since they should be inf.
    # if num_samples is None:
    #       num_samples = log_L_samples.size
    # empty_mask = jnp.arange(log_L_samples.size) >= num_samples
    # log_L_constraints = jnp.where(empty_mask, jnp.inf, log_L_constraints)
    # log_L_samples = jnp.where(empty_mask, jnp.inf, log_L_samples)
    log_L_contours = jnp.concatenate([log_L_constraints[0:1], log_L_samples[:-1]], axis=0)
    contour_idx = jnp.searchsorted(log_L_contours, log_L_samples, side='left') - 1
    if return_contours:
        # todo: consider clamping to (0, n-1) and avoid the where op
        constraints = jnp.where(contour_idx < 0,
                                -jnp.inf,
                                log_L_contours[contour_idx])
        return contour_idx, constraints
    return contour_idx


def test_compute_remaining_evidence():
    # [a,b,-inf], 2 -> [a+b, b, -inf]
    log_dZ_mean = jnp.asarray([0., 1., -jnp.inf])
    sample_idx = 2
    expect = jnp.asarray([jnp.logaddexp(0, 1), 1, -jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)

    # [-inf, -inf,-inf], 0 -> [-inf, -inf, -inf]
    log_dZ_mean = jnp.asarray([-jnp.inf, -jnp.inf - jnp.inf])
    sample_idx = 0
    expect = jnp.asarray([-jnp.inf, -jnp.inf - jnp.inf])
    assert jnp.allclose(compute_remaining_evidence(sample_idx, log_dZ_mean), expect)


def _create_test_reservoir(key: PRNGKey, num_live_points: int) -> Reservoir:
    n = 2

    # Prior is uniform in U[0,1]
    # Likelihood is 1 - x**n
    # Z = 1 - 1/n+1

    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=1))
        return x

    def log_likelihood(x):
        return (LogSpace(0.) - LogSpace(jnp.log(x)) ** n).log_abs_val

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)
    ns = ApproximateNestedSampler(
        model=model,
        num_live_points=num_live_points,
        num_parallel_samplers=1,
        max_samples=5,
        sampler_chain=[
            BadUniformSampler(mis_fraction=0., model=model)
        ]
    )

    termination_reason, state = ns(key, term_cond=TerminationCondition(live_evidence_frac=1e-4))
    return tree_map(lambda x: x[:state.sample_collection.sample_idx], state.sample_collection.reservoir)


def _exact_reservoir_num_live_points(reservoir: Reservoir) -> jnp.ndarray:
    num_live_points, _ = perfect_live_point_computation(reservoir.log_L_constraint, log_L_samples=reservoir.log_L)
    return num_live_points


def test_combine_reservoirs():
    reservoir_1 = _create_test_reservoir(random.PRNGKey(42), num_live_points=2)
    reservoir_2 = _create_test_reservoir(random.PRNGKey(43), num_live_points=2)
    print(reservoir_1)

    combined_reservoirs, num_live_points = combine_reservoirs(reservoir_1, reservoir_2)

    # print(combined_reservoirs)
    print(num_live_points)

    def make_contours(log_L_constraints, log_L_samples):
        sort_idx = jnp.lexsort((log_L_constraints, log_L_samples))
        log_L_samples = log_L_samples[sort_idx]
        log_L_constraints = log_L_constraints[sort_idx]
        log_L_contour = log_L_constraints[0]
        search_contours = jnp.concatenate([log_L_contour[None], log_L_samples], axis=0)

        contour_map_idx = jnp.searchsorted(search_contours, log_L_samples, side='left') - 1
        log_L_contours = search_contours[contour_map_idx]
        return log_L_contours

    print(compute_num_live_points_from_unit_threads(
        make_contours(combined_reservoirs.log_L_constraint, combined_reservoirs.log_L),
        combined_reservoirs.log_L,
        sorted_collection=True))
