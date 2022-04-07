from jax import numpy as jnp, random
from jax.lax import dynamic_update_slice

from jaxns.nested_sampler.live_points import supremum_contour_idx, compute_num_live_points_from_unit_threads, \
    infimum_constraint
from jaxns.nested_sampler.nested_sampling import sample_goal_distribution


def _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples, num_samples=None, debug=False):
    if num_samples is None:
        num_samples = log_L_samples.size

    empty_mask = jnp.arange(log_L_samples.size) >= num_samples
    log_L_samples = jnp.where(empty_mask, jnp.inf,log_L_samples)
    idx_sort = jnp.lexsort((log_L_constraints, log_L_samples))
    empty_mask = empty_mask[idx_sort]
    num_live_points = []

    available = jnp.bitwise_not(empty_mask)#jnp.ones(log_L_samples.shape, jnp.bool_)
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
                s = f"{'*' if mask[j] else ' '}\t{j}: available: [{'x'if available[j] else ''}] " \
                    f"log(L)>contour ({log_L_samples[j]},{contour}): [{'x' if log_L_samples[j]>contour else ''}] " \
                    f"L*<=contour ({log_L_constraints[j]},{contour}): [{'x' if log_L_constraints[j]<=contour else ''}]"
                print(s)


        contour = log_L_samples[idx]
        available = dynamic_update_slice(available, jnp.asarray([False]), [idx])
        num_live_points.append(n)

    num_live_points = jnp.stack(num_live_points)
    return num_live_points


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
        indices = jnp.clip(jnp.where((log_L_contours[indices] == log_L_samples), indices-1,indices), -1, log_L_samples.size-1)

    print(f"indices:{indices}")
    return jnp.asarray(indices)


def test_supremum_contour_idx():
    def test_example(log_L_contour,log_L_samples, expect):
        assert jnp.all(supremum_contour_idx(log_L_contour,log_L_samples,sort_idx=None)==expect)

    log_L_contour = jnp.asarray([-jnp.inf, 1., 2.])
    log_L_samples = jnp.asarray([0., 1., 1., 2., 2., 3., 4., jnp.inf])
    expect = jnp.asarray([0, 3, 5])
    test_example(log_L_contour,log_L_samples, expect)


def test_compute_num_live_points():
    from jax import random

    def test_example(log_L_constraints, log_L_samples, num_live_points, num_samples=None, debug=False):
        assert jnp.allclose(num_live_points, _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples, num_samples, debug=debug))
        test_num_live_points, sort_idx = compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples, num_samples, return_sort_idx=True)
        assert jnp.all(test_num_live_points==num_live_points)
        assert jnp.all(infimum_constraint(log_L_constraints, log_L_samples, sort_idx) == _sure_infimum_constraint(log_L_constraints, log_L_samples, sort_idx))

    # constraints are previous samples
    log_L_constraints = jnp.asarray([0., 0.5, 2.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([1,1,1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are previous samples, with extra unfilled slots
    log_L_constraints = jnp.asarray([0., 0.5, 2., jnp.inf])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, jnp.inf])
    num_live_points = jnp.asarray([1, 1, 1, 0])
    test_example(log_L_constraints, log_L_samples, num_live_points, num_samples=3, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0.45, 0., 2.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([2, 1, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0.45, 0., 2., jnp.inf, jnp.inf])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, jnp.inf, jnp.inf])
    num_live_points = jnp.asarray([2, 1, 1, 0, 0])
    test_example(log_L_constraints, log_L_samples, num_live_points, num_samples=3, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0., 0.45, 0., 2.])
    log_L_samples = jnp.asarray([0.5, 0.55, 2., 2.1])
    num_live_points = jnp.asarray([2, 2, 1, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are not ordered like samples (contour starts at non-minimum of constraints)
    log_L_constraints = jnp.asarray([0., 0.45, 0.45, 0.])
    log_L_samples = jnp.asarray([0.5, 0.5, 0.5, 0.5])
    num_live_points = jnp.asarray([2,3,2,1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # All one common constraint, no plateau in samples
    log_L_constraints = jnp.asarray([0., 0., 0.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([3, 2, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # All one common constraint, no plateau in samples
    log_L_constraints = jnp.asarray([0., 0., 0., jnp.inf])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, jnp.inf])
    num_live_points = jnp.asarray([3, 2, 1, 0])
    test_example(log_L_constraints, log_L_samples, num_live_points, num_samples=3, debug=True)

    # Growing
    log_L_constraints = jnp.asarray([0., 0.5, 0.5])
    log_L_samples = jnp.asarray([0.5, 2., 2.1])
    num_live_points = jnp.asarray([1, 2, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # Growing with plateau
    log_L_constraints = jnp.asarray([0., 0.5, 0.5, 2.])
    log_L_samples = jnp.asarray([0.5, 2., 2.1, 2.1])
    num_live_points = jnp.asarray([1, 2, 2, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    #All one common constraint, a plateau in samples
    log_L_constraints = jnp.asarray([0., 0., 0., 0.])
    log_L_samples = jnp.asarray([0.5, 2., 2., 2.1])
    num_live_points = jnp.asarray([4, 3, 2, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # one plateau
    log_L_constraints = jnp.asarray([0., 0., 0.])
    log_L_samples = jnp.asarray([0.,0.,0.])
    num_live_points = jnp.asarray([3, 2, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # one plateau
    log_L_constraints = jnp.asarray([0., 0., 0.])
    log_L_samples = jnp.asarray([0.1, 0.1, 0.1])
    num_live_points = jnp.asarray([3, 2, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # constraints are not other samples
    # 0. (1) 0.5 (1) 1.5 (1) 2.5
    log_L_constraints = jnp.asarray([0., 0.25, 0.75])
    log_L_samples = jnp.asarray([0.5, 1.5, 2.5])
    num_live_points = jnp.asarray([1, 1, 1])
    test_example(log_L_constraints, log_L_samples, num_live_points, debug=True)

    # random generated example, no plateaus
    log_L_constraints = random.uniform(random.PRNGKey(42),shape=(100,))
    log_L_samples = random.uniform(random.PRNGKey(43),shape=(100,),minval=log_L_constraints)
    num_live_points = _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples)
    test_example(log_L_constraints, log_L_samples, num_live_points)

    # random generated example, no plateaus
    log_L_constraints = jnp.concatenate([jnp.zeros(5),0.5*jnp.arange(5)/5+0.5])
    log_L_samples = random.uniform(random.PRNGKey(43), shape=(10,), minval=log_L_constraints)
    num_live_points = _sure_compute_num_live_points_from_unit_threads(log_L_constraints, log_L_samples)
    test_example(log_L_constraints, log_L_samples, num_live_points,debug=True)

    # print(num_live_points)
    # import pylab as plt
    # plt.plot(num_live_points)
    # plt.show()


def test_sample_goal_distribution():
    key = random.PRNGKey(42)
    log_goal_weights = jnp.asarray([-jnp.inf, 0., 1., 2.])
    S = 400
    samples = sample_goal_distribution(key, log_goal_weights, S, replace = True)
    f, _ = jnp.histogram(samples, bins=jnp.arange(log_goal_weights.size))
    assert f[0] == 0
    S = log_goal_weights.size
    assert jnp.all(jnp.isin(sample_goal_distribution(key, log_goal_weights, S, replace=False), jnp.arange(S)))


def test_standard_nested_sampling():
    from jax import vmap
    from jaxns.internals.maps import replace_index

    n = 0.5

    def log_likelihood(x):
        return 1. - x**n

    def sample(key, log_L):
        # sample from 1 - x**n > log_L
        upper = (1. - log_L)**(1./n)
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
    log_L_constraints_live = jnp.asarray([0.]*num_live_points)

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

    n_check = _sure_compute_num_live_points_from_unit_threads(jnp.asarray(log_L_constraints), jnp.asarray(log_L_samples))
    assert jnp.allclose(n, n_check)
