"""Focused controller, fixed-kernel, chronological feedback and JAX-boundary gates."""

from dataclasses import replace
import pickle

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from benchmarks.issue_312.online_controller import OnlineController, observe_completed
from benchmarks.issue_312.online_policy import (
    COMPRESSION_KNOTS,
    choose_parameters,
    initial_policy,
    predict_table,
)
from benchmarks.issue_312.online_sampler import OnlineReflectionSampler, PreparedContours
from benchmarks.issue_312.reflection_sampler import ReflectionSampler, reflection_trajectory
from benchmarks.issue_312.tuning_common import exact_draws, make_model
from jaxns.pytree import pytree_ravel
from jaxns.samples import SeedPoint


def test_policy_fixed_shape_values_do_not_retrace_and_bounds_hold():
    traces = []

    @jax.jit
    def choose(policy, key, c):
        traces.append(policy.log_parameters.shape)
        return choose_parameters(policy, key, c)

    policy = initial_policy()
    for index, logs in enumerate((np.log([.003, .03]), np.log([1e-10, 100.]),
                                  np.log([100., 1e-10]))):
        variant = replace(policy, log_parameters=jnp.asarray(np.tile(logs, (65, 1))),
                          version=jnp.asarray(index, policy.version.dtype))
        result = choose(variant, jax.random.PRNGKey(index), jnp.asarray(15.))
        assert 1 <= int(result.num_bounces) <= 100
        assert .003 <= float(result.nominal_duration) <= .1
        assert not bool(result.duration_capped)
        assert float(result.step_size) > 0
        np.testing.assert_allclose(result.step_size * result.num_bounces,
                                   result.nominal_duration, rtol=2e-15)
    assert traces == [(65, 2)]
    fixed = jax.jit(lambda p: choose_parameters(
        p, jax.random.PRNGKey(1), jnp.asarray(12.), explore=False,
    ))(policy)
    assert int(fixed.num_bounces) == 10 and float(fixed.step_size) == .003
    assert float(fixed.nominal_duration) == .03
    with pytest.raises(ValueError, match='Expected policy'):
        choose(replace(policy, log_parameters=jnp.zeros((66, 2))), jax.random.PRNGKey(0), 2.)


def test_predictor_interpolation_and_bounded_extrapolation():
    anchors = np.zeros((64, 2))
    learned = np.zeros(64, bool)
    constant, support = predict_table(anchors, learned)
    assert np.isnan(support).all()
    np.testing.assert_array_equal(constant, np.tile(np.log([.003, .03]), (65, 1)))
    anchors[4], anchors[8] = np.log([.003, .03]), np.log([.03, .003])
    learned[[4, 8]] = True
    table, support = predict_table(anchors, learned)
    np.testing.assert_array_equal(support, [9., 17.])
    # Beyond two e-folds, the predictor holds its bounded extrapolated value.
    np.testing.assert_array_equal(table[0], table[1])
    np.testing.assert_array_equal(table[10], table[-1])
    assert np.max(np.abs(table[0] - anchors[4])) <= 2 * np.log(1.25) + 1e-15
    np.testing.assert_allclose(table[6], anchors[4] + (anchors[8] - anchors[4]) * 3 / 8)
    assert COMPRESSION_KNOTS.shape == (65,)


def test_ratio_of_sums_clock_revisits_and_exactly_once_feedback():
    initial = OnlineController.initialise(150)
    choices = np.repeat(np.arange(5), [128, 16, 16, 16, 16])
    size = len(choices)
    work = np.where(choices == 2, 1000, 100).astype(np.int64)
    jumps = np.where(choices == 2, 4., 1.)
    # h+ has more absolute movement, but worse movement per total work.
    kwargs = dict(start=150, seed_indices=np.zeros(size, np.int64),
                  compression=np.full(size, 11.), incoming_k=np.full(size, 150),
                  choices=choices, squared_jumps=jumps, work=work, policy_version=0)
    updated, receipt = observe_completed(initial, **kwargs)
    assert len(receipt['changes']) == 1 and receipt['changes'][0]['dimensions'] == [0, 1]
    assert receipt['real_parameter_change']
    assert updated.anchors[5, 0] < np.log(.003)
    assert updated.anchors[5, 1] == np.log(.03)
    assert updated.total_exposure[5] == pytest.approx(size * np.log1p(1 / 150))
    assert updated.deepest_compression == 11.
    assert initial.counts.sum() == 0 and int(initial.policy.version) == 0
    restored = pickle.loads(pickle.dumps(updated))
    with pytest.raises(ValueError, match='contiguous'):
        observe_completed(restored, **kwargs)
    kwargs.update(start=updated.next_sample, policy_version=1, compression=np.full(size, 9.))
    revisited, _ = observe_completed(restored, **kwargs)
    assert revisited.deepest_compression == 11.  # revisits are exposure, not frontier advance
    assert revisited.total_exposure[4] > 1.


def test_new_roots_do_not_enter_controller_clock_or_objective():
    initial = OnlineController.initialise(150)
    updated, receipt = observe_completed(
        initial, start=150, seed_indices=np.full(6, -1), compression=np.full(6, np.nan),
        incoming_k=np.zeros(6), choices=np.full(6, -1), squared_jumps=np.zeros(6),
        work=np.ones(6, np.int64), policy_version=0,
    )
    assert updated.next_sample == 156 and updated.counts.sum() == 0
    assert updated.total_exposure.sum() == 0 and receipt['roots'] == 6
    with pytest.raises(ValueError, match='policy version'):
        observe_completed(updated, start=156, seed_indices=np.full(6, -1),
                          compression=np.zeros(6), incoming_k=np.ones(6),
                          choices=np.zeros(6), squared_jumps=np.zeros(6),
                          work=np.ones(6), policy_version=2)


def test_original_fixed_kernel_exact_sampler_and_phantom_parity():
    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122940)))

    def likelihood(u):
        return model.log_likelihood(unravel(u))

    points, _ = exact_draws(np.random.default_rng(3122940), 32, 15.)
    contour = likelihood(jnp.full(18, .5)) - 15.
    kernel = ReflectionSampler(num_cycles=9, slices_per_cycle=1, num_bounces=10,
                               step_size=.003, num_phantom_samples=8)
    context = PreparedContours(jnp.full(150, contour), jnp.full(150, 11.), jnp.full(150, 150))
    online = OnlineReflectionSampler(kernel, initial_policy(), context, explore=False)
    keys = jax.random.split(jax.random.PRNGKey(3122941), 32)
    outputs = []
    for sampler in (kernel, online):
        @jax.jit
        @jax.vmap
        def sample(key, point):
            return sampler.get_sample(key, contour, SeedPoint(unravel(point), likelihood(point)),
                                      model=model)
        outputs.append(jax.block_until_ready(sample(keys, jnp.asarray(points))))
    for reference, actual in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]),
                                 strict=True):
        np.testing.assert_array_equal(reference, actual)


def test_frozen_heterogeneous_resolution_duration_reversal():
    model = make_model()
    _, unravel = pytree_ravel(model.sample_U(jax.random.PRNGKey(3122942)))

    def likelihood(u):
        return model.log_likelihood(unravel(u))

    points, _ = exact_draws(np.random.default_rng(3122942), 32, 15.)
    rng = np.random.default_rng(3122943)
    velocities = rng.normal(size=points.shape)
    counts = jnp.tile(jnp.array([4, 10, 37, 100]), 8)
    durations = jnp.tile(jnp.array([.003000000001, .03, .06, .1]), 8)
    steps = durations / counts * rng.uniform(.8, 1.2, 32)
    contour = likelihood(jnp.full(18, .5)) - 15.

    @jax.jit
    @jax.vmap
    def advance(point, velocity, count, step):
        return reflection_trajectory(point, velocity, step, count, contour,
                                     jax.value_and_grad(likelihood))

    endpoints, v = advance(jnp.asarray(points), jnp.asarray(velocities), counts, steps)
    restored, reverse_v = advance(endpoints, -v, counts, steps)
    np.testing.assert_allclose(restored, points, rtol=0, atol=2e-9)
    np.testing.assert_allclose(reverse_v, -velocities, rtol=0, atol=2e-8)


@pytest.mark.parametrize('allocation', ['uniform', 'evidence_improving'])
@pytest.mark.parametrize('frozen', [False, True])
def test_observer_fixed_policy_core_parity_with_growth_and_checkpoint(allocation, frozen):
    from benchmarks.issue_312.allocation_observer import run_with_observer
    from benchmarks.issue_312.online_session import OnlineSession
    from jaxns.core import NestedSampler

    model = make_model()
    sampler = ReflectionSampler(num_cycles=9, slices_per_cycle=1, num_bounces=10, step_size=.003)
    runner = NestedSampler(model=model, root_allocation_degree=12, replacement_width=4,
                           delta_K=12, initial_capacity=32, unlimited_samples=True,
                           allocation_target=allocation, sampler=sampler,
                           collect_phantom_samples=True)
    initial = runner.initialise(jax.random.PRNGKey(3122944))
    baseline = run_with_observer(
        runner, initial, lambda before, after, work: int(after.depth_loop_iter) >= 16,
        lambda state: False,
    )
    from benchmarks.issue_312.online_frozen import FrozenLikelihoodPolicy

    frozen_policy = None
    if frozen:
        contours = np.full(268, np.inf)
        contours[:2] = [-1e20, 100.]
        frozen_policy = FrozenLikelihoodPolicy(
            jnp.asarray(contours), jnp.asarray(np.tile(np.log([.003, .03]), (268, 1))),
            jnp.asarray(2),
        )
    session = OnlineSession(12, adapt=False, explore=False, frozen_policy=frozen_policy)

    def observe(before, after, work):
        session.observe(before, after, work)
        return int(after.depth_loop_iter) >= 3

    first = run_with_observer(runner, initial, observe, lambda state: False,
                              prepare_sampler=session.prepare)
    # Full experiment continuation includes the controller, not only the race tree.
    packed = pickle.dumps(session.checkpoint(
        first, budgets=(30000, 60000), next_budget=0, snapshots=(),
        sampling_elapsed_seconds=1.25, source_sha256={'test': 'test-source'},
    ))
    del session
    checkpoint = pickle.loads(packed)
    session = OnlineSession.from_checkpoint(checkpoint)
    state = checkpoint.state
    assert checkpoint.budgets == (30000, 60000) and checkpoint.next_budget == 0
    assert checkpoint.sampling_elapsed_seconds == 1.25
    assert checkpoint.source_sha256 == {'test': 'test-source'}
    assert len(session.rows) == 3 and len(session.receipts) == 3

    def finish(before, after, work):
        session.observe(before, after, work)
        return int(after.depth_loop_iter) >= 16

    actual = run_with_observer(runner, state, finish, lambda state: False,
                               prepare_sampler=session.prepare)
    assert actual.samples.log_likelihoods.shape[0] > 32
    for a, b in zip(jax.tree.leaves(baseline), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)
    assert actual.samples.phantom_samples.U_samples is None
    arrays = session.arrays()
    assert len(arrays['work']) == int(actual.num_samples) - 12
    np.testing.assert_array_equal(
        arrays['work'], np.asarray(actual.samples.num_likelihood_evaluations)[
            12:int(actual.num_samples)
        ],
    )


def test_frozen_likelihood_policy_has_no_evaluation_population_dependence():
    from benchmarks.issue_312.online_frozen import TrainingProfile, freeze_training_profiles

    profiles = tuple(TrainingProfile(
        np.linspace(-1000 + run, 20 + run, 101), np.linspace(.01, 30 + run, 101),
        np.column_stack((np.log(.003) - COMPRESSION_KNOTS / (100 + run),
                         np.log(.03) + COMPRESSION_KNOTS / (200 + run))),
    ) for run in range(4))
    frozen = freeze_training_profiles(profiles)
    assert 2 < int(frozen.num_knots) <= 268
    context = PreparedContours(jnp.linspace(-1000., 25., 150),
                               jnp.linspace(0., 100., 150), jnp.full(150, 150))
    kernel = ReflectionSampler(num_cycles=9, slices_per_cycle=1, num_bounces=10, step_size=.003)
    sampler = OnlineReflectionSampler(kernel, initial_policy(), context, frozen_policy=frozen)
    changed = replace(sampler, prepared=replace(context, compression=jnp.full(150, 120.)),
                      policy=replace(initial_policy(), log_parameters=jnp.zeros((65, 2))))
    keys = jax.random.split(jax.random.PRNGKey(3122945), 100)
    contours = jnp.linspace(-2000., 100., 100)
    first = jax.jit(jax.vmap(sampler.parameters))(keys, contours)
    second = jax.jit(jax.vmap(changed.parameters))(keys, contours)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    # Fixed tails are determined from training, independent of any evaluation tree.
    low = frozen.choose(keys[0], jnp.asarray(-1e20), explore=False)
    edge = frozen.choose(keys[0], frozen.contours[0], explore=False)
    assert low.step_size == edge.step_size and low.nominal_duration == edge.nominal_duration


def test_zero_movement_consumption_is_not_reported_as_learning():
    controller = OnlineController.initialise(150)
    choices = np.repeat(np.arange(5), [128, 16, 16, 16, 16])
    size = len(choices)
    after, receipt = observe_completed(
        controller, start=150, seed_indices=np.zeros(size, np.int64),
        compression=np.full(size, 5.), incoming_k=np.full(size, 150), choices=choices,
        squared_jumps=np.zeros(size), work=np.ones(size, np.int64), policy_version=0,
    )
    assert int(after.policy.version) == 1 and len(receipt['changes']) == 1
    assert not receipt['real_parameter_change']
    np.testing.assert_array_equal(after.policy.log_parameters, controller.policy.log_parameters)


def test_ceil_boundary_can_reduce_effective_h_below_desired_bound():
    from benchmarks.issue_312.online_policy import choose_log_parameters

    actual = jax.jit(lambda logs: choose_log_parameters(
        logs, jax.random.PRNGKey(3122947), explore=False,
    ))(jnp.log(jnp.array([.001, .003000000001])))
    assert int(actual.num_bounces) == 4
    assert .00075 < float(actual.step_size) < .001
    assert not bool(actual.duration_capped)
    np.testing.assert_allclose(actual.step_size * actual.num_bounces,
                               actual.nominal_duration, rtol=2e-15)


def test_classic_prefix_audit_has_no_future_edges_or_atom_at_singletons():
    from benchmarks.issue_312.online_audit import classic_volume_profile

    values = np.array([-4., -3., -2., -1.])
    parents = np.array([-1, -1, 0, 1])
    levels, compression, incoming, roots = classic_volume_profile(values, parents)
    np.testing.assert_array_equal(levels, values)
    np.testing.assert_array_equal(incoming, [2, 2, 2, 1])
    np.testing.assert_allclose(compression, -np.cumsum(np.log([2/3, 2/3, 2/3, 1/2])))
    assert roots == 2
    # Truncating the prefix removes the later edge; a future child cannot affect c/K.
    _, prefix, prefix_k, _ = classic_volume_profile(values[:3], parents[:3])
    np.testing.assert_array_equal(prefix_k, [2, 2, 1])
    assert prefix[-1] > compression[2]
    with pytest.raises(ValueError, match='future'):
        classic_volume_profile(values, np.array([-1, -1, 3, 1]))


def test_training_entrypoint_requires_reviewed_release(tmp_path, monkeypatch):
    import hashlib
    import json

    from benchmarks.issue_312 import online_common

    protocol = tmp_path / 'benchmarks/issue_312/REFLECTION_ONLINE_PROTOCOL.md'
    protocol.parent.mkdir(parents=True)
    protocol.write_text('unit-test protocol; never a real experiment release')
    freeze = dict(
        source_commit='test-only-commit', baseline_commit=online_common.BASELINE,
        source_sha256={}, protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
    )
    (tmp_path / 'SOURCE_FREEZE.json').write_text(json.dumps(freeze))
    monkeypatch.setattr(online_common, 'SOURCE_ROOT', tmp_path)
    monkeypatch.setattr(online_common, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(online_common.subprocess, 'check_output',
                        lambda *a, **k: 'test-only-commit')
    with pytest.raises(FileNotFoundError):
        online_common.load_online_freeze('training', 3122910, 'online', None)
    (tmp_path / 'TRAINING_RELEASE.json').write_text(json.dumps(dict(approved=False)))
    with pytest.raises(ValueError, match='coordinator release'):
        online_common.load_online_freeze('training', 3122910, 'online', None)


@pytest.mark.parametrize('allocation', ['uniform', 'evidence_improving'])
def test_online_feedback_is_exactly_independent_of_phantom_collection(allocation):
    from benchmarks.issue_312.allocation_observer import run_with_observer
    from benchmarks.issue_312.online_session import OnlineSession
    from jaxns.core import NestedSampler
    from jaxns.samples import PhantomSamples

    states, ledgers, histories = [], [], []
    for retain in (False, True):
        session = OnlineSession(12)
        runner = NestedSampler(
            model=make_model(), root_allocation_degree=12, replacement_width=4, delta_K=12,
            initial_capacity=32, unlimited_samples=True, allocation_target=allocation,
            sampler=session.kernel, collect_phantom_samples=retain,
        )

        def observe(before, after, work):
            session.observe(before, after, work)
            return int(after.depth_loop_iter) >= 16

        initial = runner.initialise(jax.random.PRNGKey(3122948))
        state = run_with_observer(runner, initial, observe, lambda current: False,
                                  prepare_sampler=session.prepare)
        assert state.samples.phantom_samples.U_samples is None
        empty = PhantomSamples(None, jnp.zeros(0, bool), jnp.empty((0, 0)))
        states.append(replace(state, samples=replace(state.samples, phantom_samples=empty)))
        ledgers.append(session.arrays())
        histories.append(session.receipts)
    for left, right in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    for name in ledgers[0]:
        np.testing.assert_array_equal(ledgers[0][name], ledgers[1][name])
    assert len(np.unique(ledgers[0]['num_bounces'])) > 1
    assert histories[0] == histories[1]
