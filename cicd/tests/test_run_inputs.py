"""Run inputs belong to State, including default resolution on resumption."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.distributed_core import DistributedNestedSampler
from jaxns.model import Model
from jaxns.priors import Prior
from jaxns.runtime.client import SupervisorClient
from jaxns.sampling.batching import evaluate_request, sample_request


def _prior_model(observations):
    position = Prior(
        tfp.distributions.Uniform(
            low=jnp.zeros_like(observations), high=jnp.ones_like(observations),
        ),
        name="position",
    ).realise(periodic=True)
    scale = Prior(
        tfp.distributions.Exponential(rate=1.0), name="scale",
    ).parameter()
    return -jnp.sum(jnp.square(position - observations)) / scale


def _unusable_prior_model():
    raise AssertionError("Resumption must use the checkpoint model.")


def _assert_same_tree(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _assert_state_inputs(state, args, params):
    _assert_same_tree(state.args, args)
    _assert_same_tree(state.params, params)
    count = int(state.num_samples)
    samples = jax.tree.map(
        lambda value: value[:count], state.samples.U_samples,
    )
    expected = jax.vmap(
        lambda sample: state.model.log_likelihood(
            sample, args=args, params=params,
        )
    )(samples)
    np.testing.assert_allclose(
        state.samples.log_likelihoods[:count], expected,
        rtol=1e-13, atol=1e-13,
    )


def test_runner_reuse_resolves_dimensions_from_each_runs_inputs():
    model = Model(prior_model=_prior_model)
    runner = NestedSampler(model=model, collect_phantom_samples=True)
    for dimension in (1, 3, 1):
        args = (jnp.full((dimension,), 0.3),)
        params = model.init_params(jax.random.PRNGKey(dimension), args=args)
        state = runner.initialise(
            jax.random.PRNGKey(288), args=args, params=params,
        )
        # JAXCTX also retains the base coordinate for the scale parameter.
        model_dimension = dimension + 1
        assert int(state.num_samples) == 30 * model_dimension
        assert state.samples.phantom_samples.log_L.shape[1] == 5 * model_dimension - 1
        _assert_state_inputs(state, args, params)
        # Reusing or serialising the runner must not freeze the last run's
        # dimension-dependent defaults or retain that run's data.
        assert runner.root_allocation_degree is None
        assert runner.sampler is None
        runner = pickle.loads(pickle.dumps(runner))


@pytest.mark.parametrize(
    "entrypoint", ["run", "run_until_goal", "run_single_iteration"],
)
def test_local_run_entrypoints_forward_inputs(entrypoint):
    model = Model(prior_model=_prior_model)
    args = (jnp.asarray([0.2, 0.7]),)
    params = model.init_params(jax.random.PRNGKey(1), args=args)
    runner = NestedSampler(
        model=model, root_allocation_degree=2, replacement_width=1,
        max_samples=6, initial_capacity=6, depth_condition=DepthCondition(),
    )
    options = {"key": jax.random.PRNGKey(2), "args": args, "params": params}
    if entrypoint == "run":
        state = runner.run(**options)
    elif entrypoint == "run_until_goal":
        state = runner.run_until_goal(
            lambda state: int(state.goal_loop_iter) >= 1, **options,
        )
    else:
        state = runner.run_single_iteration(**options)
    _assert_state_inputs(state, args, params)


@pytest.mark.parametrize("automatic", [False, True])
def test_local_resume_resolves_only_checkpoint_inputs(tmp_path, automatic):
    model = Model(prior_model=_prior_model)
    args = (jnp.asarray([0.2, 0.7]),)
    params = model.init_params(jax.random.PRNGKey(3), args=args)
    settings = {
        "root_allocation_degree": 2,
        "replacement_width": 1,
        "max_samples": 8,
        "initial_capacity": 8,
        "depth_condition": DepthCondition(),
    }
    runner = NestedSampler(model=model, **settings)
    first = runner.run_until_goal(
        lambda state: int(state.goal_loop_iter) >= 1,
        key=jax.random.PRNGKey(4), args=args, params=params,
        checkpoint_dir=tmp_path,
    )

    def final_goal(state):
        return int(state.goal_loop_iter) >= 2

    expected = runner.resume_until_goal(first, final_goal)
    resumed_runner = NestedSampler(
        model=Model(prior_model=_unusable_prior_model), **settings,
    )
    if automatic:
        resumed = resumed_runner.run_until_goal(
            final_goal, checkpoint_dir=tmp_path,
            # These are new-run inputs. A checkpoint must take precedence
            # before even tracing the model to resolve sampler topology.
            args=(jnp.zeros((5,)),), params=None,
        )
    else:
        resumed = resumed_runner.resume_until_goal(first, final_goal)
    _assert_same_tree(resumed, expected)


@pytest.mark.parametrize("entrypoint", ["initialise", "run", "run_until_goal"])
def test_distributed_run_and_resume_use_state_owned_inputs(
        monkeypatch, entrypoint,
):
    sessions = []

    class Client:
        def __init__(self):
            self.session = None
            self.results = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def register(self, session_id, session):
            self.session = session
            sessions.append(session)
            return (2,)

        def release(self, session_id):
            pass

        def capacity(self, session_id, timeout_s):
            return 2

        def evaluate_many(self, session_id, tasks):
            session = self.session
            self.results.extend((task_id, evaluate_request(
                session.model, request,
                args=session.args, params=session.params,
            )) for task_id, request in tasks)

        def submit_many(self, session_id, tasks):
            session = self.session
            self.results.extend((task_id, sample_request(
                session.sampler, request, model=session.model,
                args=session.args, params=session.params,
            )) for task_id, request in tasks)

        def receive_group(self, session_id, timeout_s):
            completed = tuple(self.results)
            self.results.clear()
            return completed

        def acknowledge(self, session_id, task_id):
            pass

    monkeypatch.setattr(SupervisorClient, "from_port", lambda port: Client())
    model = Model(prior_model=_prior_model)
    args = (jnp.asarray([0.2, 0.7]),)
    params = model.init_params(jax.random.PRNGKey(5), args=args)
    runner = DistributedNestedSampler(
        model=model, coordinator_port=5555, root_allocation_degree=2,
        initial_capacity=6, max_samples=6, depth_condition=DepthCondition(),
    )
    options = {"key": jax.random.PRNGKey(6), "args": args, "params": params}
    if entrypoint == "initialise":
        checkpoint = runner.initialise(**options)
    elif entrypoint == "run":
        checkpoint = runner.run(**options)
    else:
        checkpoint = runner.run_until_goal(
            lambda state: int(state.goal_loop_iter) >= 1, **options,
        )
    _assert_state_inputs(checkpoint.state, args, params)
    assert sessions[0].sampler._periodic.count(True) == 2
    runner.model = Model(prior_model=_unusable_prior_model)
    resumed = runner.resume_until_goal(checkpoint, lambda state: True)
    assert resumed is checkpoint
    assert sessions[-1].sampler._periodic == sessions[0].sampler._periodic
    _assert_same_tree(sessions[-1].args, checkpoint.state.args)
    _assert_same_tree(sessions[-1].params, checkpoint.state.params)
