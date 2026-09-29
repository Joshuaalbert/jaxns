"""Zero likelihood is part of the prior, including a positive-mass plateau."""

import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from cicd.tests.distributed_support import half_prior_model
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.samples import SeedPoint
from jaxns.sampling.batching import sample_request
from jaxns.sampling.protocol import ConstrainedSampleRequest
from jaxns.shrinkage import phantom as jax_phantom
from jaxns.shrinkage import reference as ref_phantom
from jaxns.state import State


def test_zero_likelihood_prior_mass_is_retained_in_evidence():
    runner = NestedSampler(
        model=Model(half_prior_model),
        root_allocation_degree=1024,
        replacement_width=8,
        max_samples=2048,
    )
    state = runner.initialise(jax.random.PRNGKey(42))
    evidence = float(jnp.exp(state.expected_log_Z_mean))
    # Dropping zero-likelihood prior draws instead estimates E[L | L > 0] = 1.
    # The fixed-seed tolerance allows finite root-population error around 1/2.
    np.testing.assert_allclose(evidence, 0.5, atol=0.05, rtol=0.0)
    n = int(state.num_samples)
    assert np.any(np.isneginf(state.samples.log_likelihoods[:n]))
    np.testing.assert_array_equal(
        state.samples.num_likelihood_evaluations[:n], 1,
    )
    result = state.to_result().trim()
    assert np.all(np.isneginf(result.log_dp[np.isneginf(result.log_L)]))
    sampled = result.sample_evidence(64, key=jax.random.PRNGKey(43))
    np.testing.assert_allclose(jnp.exp(sampled.log_Z_mean), 0.5, atol=0.05)


@pytest.mark.parametrize("num_slices", [4, 32])
def test_root_draws_and_zero_contour_chains_have_different_support(num_slices):
    model = Model(half_prior_model)
    width = 32
    sampler = UniDimSliceSampler(
        num_slices=num_slices,
        collect_phantom_samples=True,
        max_phantom_samples=num_slices - 1,
    )
    point = jax.tree.map(
        lambda x: jnp.full_like(x, 0.75),
        model.sample_U(jax.random.PRNGKey(0)),
    )
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(10), width),
        valid=jnp.ones((width,), dtype=bool),
        log_L_constraints=jnp.full((width,), -jnp.inf),
        seed_points=SeedPoint(
            U0=jax.tree.map(
                lambda x: jnp.broadcast_to(x, (width,) + x.shape), point,
            ),
            log_L0=jnp.zeros((width,)),
        ),
        sampler_data=None,
        from_root=jnp.arange(width) < width // 2,
    )
    execute = jax.jit(lambda r: sample_request(sampler, r, model=model))
    result = execute(request)
    roots = slice(0, width // 2)
    chains = slice(width // 2, width)
    assert np.any(np.isneginf(result.log_likelihoods[roots]))
    assert np.any(np.isfinite(result.log_likelihoods[roots]))
    np.testing.assert_array_equal(
        result.num_likelihood_evaluations[roots], 1,
    )
    np.testing.assert_array_equal(result.log_likelihoods[chains], 0.0)
    assert result.phantom_samples.valid_mask.shape == (width,)
    np.testing.assert_array_equal(
        result.phantom_samples.valid_mask[roots], False,
    )
    np.testing.assert_array_equal(
        result.phantom_samples.valid_mask[chains], True,
    )
    np.testing.assert_array_equal(result.phantom_samples.log_L[chains], 0.0)

    # An entirely root batch must skip the constrained sampler at runtime.
    roots_only = dataclasses.replace(
        request, from_root=jnp.ones((width,), bool),
    )
    all_roots = execute(roots_only)
    np.testing.assert_array_equal(all_roots.num_likelihood_evaluations, 1)
    np.testing.assert_array_equal(all_roots.phantom_samples.valid_mask, False)


def test_zero_plateau_survives_allocation_and_checkpoint_round_trip(tmp_path):
    runner = NestedSampler(
        model=Model(half_prior_model),
        root_allocation_degree=128,
        replacement_width=8,
        delta_K=128,
        max_samples=2048,
        sampler=UniDimSliceSampler(
            num_slices=4,
            collect_phantom_samples=True,
            max_phantom_samples=3,
        ),
    )
    state = runner.run_until_goal(
        lambda s: int(s.root_out_degree) >= 256,
        key=jax.random.PRNGKey(42),
    )
    state.ensure_consistency()
    n = int(state.num_samples)
    roots = np.asarray(state.samples.num_likelihood_evaluations[:n]) == 1
    assert int(state.root_out_degree) >= 256
    assert np.sum(roots) == int(state.root_out_degree)
    assert np.any(~roots)
    np.testing.assert_array_equal(
        state.samples.phantom_samples.valid_mask[:n], ~roots,
    )
    assert np.any(np.isneginf(state.samples.log_likelihoods[:n]))
    path = str(tmp_path / "zero-plateau.pkl")
    state.save(path)
    restored = State.load(path)
    restored.ensure_consistency()
    result = restored.to_result().trim()
    assert result.valid_phantom.shape == (n,)
    for conditioning in (False, True):
        evidence = result.sample_evidence(
            128, key=jax.random.PRNGKey(12), phantom_conditioning=conditioning,
        )
        np.testing.assert_allclose(jnp.exp(evidence.log_Z_mean), 0.5, atol=0.1)


def test_zero_contour_phantoms_cannot_inform_zero_plateau_mass():
    # The first cluster is a root draw, the second a chain above actual zero.
    inputs = {
        "log_L_blocks": np.asarray([-np.inf, 0.0, 1.0]),
        "block_valid_mask": np.ones((3,), dtype=bool),
        "log_L_constraints": np.asarray([-np.inf, -np.inf]),
        "valid_phantom": np.asarray([False, True]),
        "sample_mask": np.ones((2,), dtype=bool),
        "log_L_phantom": np.asarray([[-np.inf, -np.inf], [0.0, 1.0]]),
        "C_min": 1,
    }
    reference = ref_phantom.compute_phantom_count_matrices(**inputs)
    compiled = jax_phantom.compute_phantom_count_matrices(**inputs)
    np.testing.assert_array_equal(compiled.A_cg, [[0, 0, 0], [0, 2, 1]])
    np.testing.assert_array_equal(compiled.B_cg, [[0, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(compiled.E_cg, [[0, 0, 0], [0, 1, 1]])
    np.testing.assert_array_equal(compiled.A_cg, reference.A_cg)
    np.testing.assert_array_equal(compiled.B_cg, reference.B_cg)
    np.testing.assert_array_equal(compiled.E_cg, reference.E_cg)
