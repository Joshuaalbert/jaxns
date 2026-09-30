"""Scientific boundary regressions identified in the develop audit (#307)."""

import dataclasses
import json
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfpd

from cicd.tests.core_fixtures import make_state
from cicd.tests.distributed_support import make_toy_model
from jaxns.checkpoint import CheckpointManager
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.distributed_core import DistributedNestedSampler, DistributedState
from jaxns.log_semiring import LogSpace
from jaxns.model import Model
from jaxns.priors import Prior
from jaxns.pytree import TreeField


def _offset_prior_model(offset=0.0):
    x = Prior(tfpd.Uniform(0.0, 1.0), name="x").realise()
    return -x + offset


def test_merge_compacts_valid_rows_and_preserves_results():
    runner = NestedSampler(
        make_toy_model(), root_allocation_degree=4,
        initial_capacity=16, max_samples=32,
    )
    left = runner.initialise(jax.random.PRNGKey(0))
    right = runner.initialise(jax.random.PRNGKey(1))
    actual = left.merge(right)
    expected = left.trim().merge(right.trim())
    actual.ensure_consistency()
    np.testing.assert_array_equal(
        actual.samples.log_likelihoods[:8],
        expected.samples.log_likelihoods[:8],
    )
    np.testing.assert_allclose(
        actual.to_result().log_Z_mean, expected.to_result().log_Z_mean,
    )


@pytest.mark.parametrize("field", ["args", "params"])
def test_merge_checks_input_values_at_host_boundary(field):
    state = NestedSampler(make_toy_model(), root_allocation_degree=2).initialise()
    left = dataclasses.replace(state, **{field: (jnp.asarray(0.0),)})
    right = dataclasses.replace(state, **{field: (jnp.asarray(10.0),)})
    with pytest.raises(AssertionError, match=field):
        left.merge(right)


@pytest.mark.parametrize("batch_size", [None, 3])
@pytest.mark.parametrize("semi_positive", [False, True])
def test_posterior_integration_ignores_undefined_zero_mass_rows(
        batch_size, semi_positive,
):
    runner = NestedSampler(
        make_toy_model(), root_allocation_degree=4, initial_capacity=16,
    )
    results = runner.initialise().to_result()

    def integrand(x):
        return {"inverse": 1 / x, "log_square": jnp.log(x) ** 2}

    expected = results.trim().integrate_fn_over_posterior(
        integrand, semi_positive=semi_positive,
    )
    actual = results.integrate_fn_over_posterior(
        integrand, semi_positive=semi_positive, batch_size=batch_size,
    )
    jax.tree.map(np.testing.assert_allclose, actual, expected)


def test_posterior_resampling_rejects_without_replacement():
    results = NestedSampler(
        make_toy_model(), root_allocation_degree=2,
    ).initialise().to_result()
    with pytest.raises((TypeError, ValueError), match="replace|replacement"):
        results.resample(2, key=jax.random.PRNGKey(1), replace=False)


@pytest.mark.parametrize("permutation", [(0, 1, 2, 3), (3, 1, 0, 2)])
def test_parent_graph_respects_contours_degrees_and_storage_indices(permutation):
    state = make_state(
        root_out_degree=2, log_likelihoods=(1.0, 2.0, 3.0, 4.0),
        out_degree=(0, 2, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, 2.0, 2.0), max_samples=4,
    )
    state = dataclasses.replace(
        state, samples=state.samples[jnp.asarray(permutation)],
    ).resize(8)
    edges = np.asarray(state.determine_parent_graph())
    assert edges.shape == (4, 2)
    np.testing.assert_array_equal(np.sort(edges[:, 1]), np.arange(4))
    parents, children = edges.T
    np.testing.assert_array_equal(
        np.bincount(parents + 1, minlength=5),
        np.r_[2, np.asarray(state.samples.out_degree[:4])],
    )
    selected = parents >= 0
    np.testing.assert_array_equal(
        state.samples.log_likelihoods[parents[selected]],
        state.samples.log_L_constraints[children[selected]],
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_signed_logspace_cumsum_preserves_sign_changes(reverse, axis):
    values = np.asarray([[-2.0, 1.0, 3.0, -2.0], [1.0, -1.0, 0.0, 2.0]])
    ordered = np.flip(values, axis=axis) if reverse else values
    expected = np.cumsum(ordered, axis=axis)
    if reverse:
        expected = np.flip(expected, axis=axis)
    actual = jax.jit(lambda x: LogSpace.from_signed_value(x).cumsum(
        axis=axis, reverse=reverse,
    ).value)(jnp.asarray(values))
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_json_round_trip_restores_nested_numpy_arrays():
    original = TreeField({
        "nested": (TreeField(jnp.asarray([1.0, np.nan])), None),
        "complex": np.asarray([1 + 2j], dtype=np.complex128),
        "empty": np.empty((0, 3), dtype=np.int32),
    })
    restored = TreeField.from_json(json.loads(json.dumps(original.to_json())))
    assert jax.tree.structure(original) == jax.tree.structure(restored)
    for before, after in zip(
            jax.tree.leaves(original), jax.tree.leaves(restored), strict=True,
    ):
        assert type(after) is np.ndarray
        np.testing.assert_array_equal(after, before)
        assert after.dtype == before.dtype


class UnserializablePayload:
    def __reduce__(self):
        raise AttributeError("payload serialization failed")


def test_serialization_failure_does_not_publish_checkpoint(tmp_path):
    with (
        CheckpointManager(tmp_path) as manager,
        pytest.raises(AttributeError, match="payload serialization failed"),
    ):
        manager.save(TreeField(UnserializablePayload()))
    assert not (tmp_path / "CHECKPOINT").exists()


@pytest.mark.parametrize("wrap", [partial, jax.jit])
def test_models_accept_partial_and_jit_wrappers(wrap):
    expected = Model(_offset_prior_model)
    actual = Model(wrap(_offset_prior_model))
    for offset in (0.0, 3.0):
        key = jax.random.PRNGKey(2)
        u = actual.sample_U(key, args=(offset,))
        jax.tree.map(
            np.testing.assert_array_equal, u,
            expected.sample_U(key, args=(offset,)),
        )
        np.testing.assert_allclose(
            actual.log_likelihood(u, args=(offset,)),
            expected.log_likelihood(u, args=(offset,)),
        )


def test_runner_owns_phantom_collection_with_explicit_sampler():
    runner = NestedSampler(
        make_toy_model(), sampler=UniDimSliceSampler(num_slices=4),
        collect_phantom_samples=True, root_allocation_degree=2,
    )
    assert runner.initialise().samples.phantom_samples.log_L.shape[1] == 3
    distributed = DistributedNestedSampler(
        make_toy_model(), coordinator_port=5555,
        sampler=runner.sampler, collect_phantom_samples=True,
    )
    config = distributed._resolve_config(distributed.model, (), None)
    assert config.num_phantom_samples == 3


@pytest.mark.parametrize("field", [
    "root_allocation_degree", "replacement_width", "delta_K",
    "initial_capacity", "max_samples",
])
def test_run_counts_reject_nonintegers(field):
    runner = NestedSampler(make_toy_model(), **{field: 30.9})
    with pytest.raises(TypeError, match=field):
        runner.initialise()


def test_parent_graph_assigns_zero_arrivals_to_sentinel_and_respects_plateaus():
    # The sentinel and real zero-likelihood parents share -inf in storage.
    # Degrees and strictness still determine a compatible allocation of edges.
    state = make_state(
        root_out_degree=3,
        log_likelihoods=(-np.inf, 1.0, -np.inf, 1.0, 2.0, 3.0, 4.0),
        out_degree=(1, 1, 1, 1, 0, 0, 0),
        log_L_constraints=(-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 1.0, 1.0),
        max_samples=9,
    )
    parents, children = state.determine_parent_graph().T
    np.testing.assert_array_equal(parents[[0, 2]], [-1, -1])
    np.testing.assert_array_equal(
        np.bincount(parents + 1, minlength=8), [3, 1, 1, 1, 1, 0, 0, 0],
    )
    actual_contours = np.r_[-np.inf, state.samples.log_likelihoods[:7]][parents + 1]
    np.testing.assert_array_equal(actual_contours, state.samples.log_L_constraints[children])


def test_partial_model_bindings_have_distinct_compilation_identity():
    key = jax.random.PRNGKey(307)
    for wrap in (lambda f: f, jax.jit):
        one = Model(wrap(partial(jax.jit(_offset_prior_model), offset=1.0)))
        two = Model(wrap(partial(jax.jit(_offset_prior_model), offset=2.0)))
        assert one.prior_model != two.prior_model
        u = one.sample_U(key)
        np.testing.assert_allclose(two.log_likelihood(u) - one.log_likelihood(u), 1.0)


def test_model_preserves_python_decorator_likelihood_semantics():
    @wraps(_offset_prior_model)
    def decorated(*args, **kwargs):
        return 2.0 * _offset_prior_model(*args, **kwargs)

    plain = Model(_offset_prior_model)
    u = plain.sample_U(jax.random.PRNGKey(307))
    for fn in (decorated, jax.jit(decorated)):
        model = Model(fn)
        np.testing.assert_allclose(model.log_likelihood(u), 2.0 * plain.log_likelihood(u))


def test_state_json_round_trip_preserves_model_inputs_and_host_arrays():
    runner = NestedSampler(Model(_offset_prior_model), root_allocation_degree=2)
    state = runner.initialise(args=(3.0,))
    restored = type(state).from_json(json.loads(json.dumps(state.to_json())))
    for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        assert type(right) is np.ndarray
        np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(restored.to_result().log_Z_mean, state.to_result().log_Z_mean)


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("leaves", {}), ("treedef", None),
])
def test_json_rejects_malformed_envelope(field, value):
    encoded = TreeField(jnp.ones(2)).to_json()
    encoded[field] = value
    with pytest.raises(ValueError, match="JSON"):
        TreeField.from_json(encoded)


@pytest.mark.parametrize("execution", ["goal", "single", "distributed"])
def test_resume_rejects_incompatible_phantom_width_before_execution(execution):
    runner = NestedSampler(make_toy_model(), root_allocation_degree=2)
    state = runner.initialise()
    if execution == "distributed":
        distributed = DistributedNestedSampler(
            runner.model, coordinator_port=5555, collect_phantom_samples=True,
        )
        with pytest.raises(ValueError, match="Saved phantom count"):
            distributed.resume_until_goal(DistributedState.from_state(state), lambda s: True)
    else:
        runner = dataclasses.replace(runner, collect_phantom_samples=True)
        with pytest.raises(ValueError, match="Saved phantom count"):
            if execution == "goal":
                runner.resume_until_goal(state, lambda s: True)
            else:
                runner.run_single_iteration(state)


def test_capacity_limited_state_resumes_after_explicit_budget_increase():
    runner = NestedSampler(
        make_toy_model(), root_allocation_degree=2, replacement_width=1,
        max_samples=4, initial_capacity=4,
    )
    stopped = runner.run()
    assert int(stopped.termination_reason) == 1
    larger = dataclasses.replace(runner, max_samples=8)
    # A hard stop remains sticky until the caller acknowledges the new budget.
    unchanged = larger.resume_until_goal(stopped, lambda s: False)
    assert int(unchanged.num_samples) == 4
    continuation = dataclasses.replace(
        stopped, termination_reason=jnp.zeros_like(stopped.termination_reason),
    )
    continued = larger.resume_until_goal(continuation, lambda s: False)
    assert int(continued.num_samples) == 8
    np.testing.assert_array_equal(
        continued.samples.log_likelihoods[:4], stopped.samples.log_likelihoods,
    )
