"""Keep phantom inference separate from the frozen allocation comparison."""

from functools import partial

import jax
import jaxctx
import numpy as np
import pytest
from jax import numpy as jnp

import jaxns.model as model_module
from benchmarks.issue_312.allocation_observer import run_with_observer
from benchmarks.issue_312.allocation_phantoms import EvidenceInputs
from benchmarks.issue_312.hessian_sampler import ToyHessianSliceSampler
from benchmarks.issue_312.numpy_allocation_evidence import (
    count_operators,
    evidence_from_probabilities,
    prepare_events,
)
from benchmarks.issue_312.reflection_sampler import ReflectionSampler
from benchmarks.issue_312.run_ridge import prior_model
from jaxns.algorithm.depth import _refresh_likelihood_order
from jaxns.algorithm.race_tree import build_block_state
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.samples import PhantomSamples, Samples
from jaxns.shrinkage.phantom import _evidence_batch_from_probabilities
from jaxns.shrinkage.reference import (
    compute_phantom_count_matrices,
    gamma_weighted_phantom_probabilities_from_draws,
)


@pytest.mark.parametrize("kernel", ["hessian", "reflection"])
def test_allocation_phantoms_preserve_sampling_and_evidence_api(kernel, monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_module, "transform", partial(jaxctx.transform, base_dtype=jnp.float64),
    )
    sampler = (ToyHessianSliceSampler(num_slices=90, scale_to_contour=True)
               if kernel == "hessian" else ReflectionSampler(
                   num_cycles=9, slices_per_cycle=1, num_bounces=10,
                   step_size=0.003, boundary_only=False,
               ))
    results = []
    for retain in (False, True):
        runner = NestedSampler(
            model=Model(prior_model=partial(prior_model, 1e-5, 0.3)),
            root_allocation_degree=12, replacement_width=4, delta_K=12,
            initial_capacity=64, sampler=sampler, collect_phantom_samples=retain,
        )
        state = run_with_observer(
            runner, runner.initialise(jax.random.PRNGKey(31244)),
            lambda before, after, work: int(after.depth_loop_iter) >= 3,
            lambda state: False,
        )
        results.append(_refresh_likelihood_order(state).to_result())
    classic, phantom = results
    for left, right in ((classic.U_samples, phantom.U_samples),
                        (classic.log_L, phantom.log_L), (classic.log_dp, phantom.log_dp),
                        (classic.num_likelihood_evaluations_per_sample,
                         phantom.num_likelihood_evaluations_per_sample)):
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
            np.testing.assert_array_equal(a, b)
    assert phantom.log_L_phantom.shape[1] == (89 if kernel == "hessian" else 8)
    count = int(phantom.total_num_samples)
    np.testing.assert_array_equal(phantom.valid_phantom[:count],
                                  np.isfinite(phantom.log_L_constraints[:count]))
    inputs = EvidenceInputs.from_result(phantom)
    path = tmp_path / "evidence.npz"
    inputs.save(path)
    loaded = EvidenceInputs.load(path)
    for a, b in zip(jax.tree.leaves(inputs), jax.tree.leaves(loaded), strict=True):
        np.testing.assert_array_equal(a, b)
    for condition in (False, True):
        key = jax.random.PRNGKey(31245)
        expected = phantom.sample_evidence(128, key=key, phantom_conditioning=condition)
        actual = loaded.draw(key, phantoms=condition, draws=128)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            np.testing.assert_array_equal(a, b)


def test_numpy_phantom_counts_and_evidence_match_dense_reference():
    # A real race tree with 20 roots and 40 children of its first dead point.
    # Integer likelihoods create both singleton and plateau blocks. Forty
    # independent chain clusters activate the gate; it closes in the tail.
    rng = np.random.default_rng(31246)
    values = jnp.concatenate([jnp.arange(20.0), jnp.arange(1.0, 41.0)])
    constraints = jnp.concatenate([jnp.full(20, -jnp.inf), jnp.zeros(40)])
    cluster_valid = jnp.arange(60) >= 20
    phantoms = jnp.asarray(rng.integers(1, 41, size=(60, 8)), dtype=jnp.float64)
    samples = Samples(
        constraints, values, jnp.zeros((60, 1)), jnp.zeros(60, jnp.int32).at[0].set(40),
        jnp.ones(60, jnp.int32), PhantomSamples(None, cluster_valid, phantoms),
    )
    block = build_block_state(samples, jnp.asarray(20), jnp.asarray(60))
    events = jax.device_get(prepare_events(
        block_state=block, log_L_constraints=constraints, valid_phantom=cluster_valid,
        log_L_phantom=phantoms, sample_mask=jnp.ones(60, bool), C_min=20.0,
    ))
    reference = compute_phantom_count_matrices(
        log_L_blocks=block.log_L_blocks, block_valid_mask=block.valid,
        log_L_constraints=constraints, valid_phantom=cluster_valid,
        log_L_phantom=phantoms, sample_mask=np.ones(60, bool), C_min=20.0,
    )
    np.testing.assert_array_equal(events.gate, reference.phantom_gate_active)
    np.testing.assert_allclose(events.kish, reference.kish_participating_cluster_counts)
    assert np.any(events.gate) and np.any(~events.gate[np.asarray(block.valid)])
    operator_a, operator_b = count_operators(events)
    for _ in range(8):
        weights = rng.exponential(size=60) * np.asarray(cluster_valid)
        weighted_a = np.cumsum(operator_a @ weights)[:-1]
        weighted_b = np.cumsum(operator_b @ weights)[:-1]
        np.testing.assert_allclose(weighted_a, weights @ reference.A_cg, atol=1e-10)
        np.testing.assert_allclose(weighted_b, weights @ reference.B_cg, atol=1e-10)
        gt, eq, lt = rng.gamma(2, size=(3, 60))
        eq *= np.asarray(block.block_size) > 1
        expected = gamma_weighted_phantom_probabilities_from_draws(
            block_state=block, A_cg=reference.A_cg, B_cg=reference.B_cg,
            E_cg=reference.E_cg, race_gamma_gt=gt, race_gamma_eq=eq,
            race_gamma_lt=lt, cluster_weights=weights,
        )
        actual = (gt + weighted_b * events.gate) / (
            gt + eq + lt + weighted_a * events.gate
        )
        valid = np.asarray(block.valid)
        np.testing.assert_allclose(actual[valid], expected.p_gt[valid], rtol=1e-12)
        actual_z, actual_dz = evidence_from_probabilities(
            actual[None, valid], np.asarray(block.log_L_blocks)[valid],
        )
        expected_z, _, expected_dz = _evidence_batch_from_probabilities(
            p_gt=jnp.asarray(actual[None]), block_state=block,
        )
        np.testing.assert_allclose(actual_z, expected_z, atol=1e-12)
        np.testing.assert_allclose(actual_dz, np.asarray(expected_dz)[:, valid], atol=1e-12)
