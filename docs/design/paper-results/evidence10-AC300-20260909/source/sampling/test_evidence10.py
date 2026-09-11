"""Independent checks for the current 10D evidence experiment definitions."""

import dataclasses
import pickle
import subprocess
import sys

import jax
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from benchmarks.paper_reproduction.evidence10_cases import CG_BETA
from benchmarks.paper_reproduction.evidence10_cases import GAUSSIAN_COVARIANCE
from benchmarks.paper_reproduction.evidence10_cases import GAUSSIAN_MEAN
from benchmarks.paper_reproduction.evidence10_cases import build_case
from benchmarks.paper_reproduction.evidence10_cases import cg10_hermite_reference
from benchmarks.paper_reproduction.posterior_cases import build_model
from jaxns.core import NestedSampler
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.depth_condition import DepthCondition


@pytest.mark.parametrize("case", ["g10", "cg10"])
def test_gaussian_adapter_prior_and_inverse_shear(case):
    model, reference = build_case(case)
    model = pickle.loads(pickle.dumps(model))
    assert model.U_ndims() == 10
    for seed in range(5):
        u = model.sample_U(jax.random.PRNGKey(seed))
        x = np.asarray(jax.tree.leaves(model.transform_to_X(u))[0])
        z = x.copy()
        if case == "cg10":
            z[1] -= CG_BETA * ((z[0]-3.)**2-1.)
        expected = multivariate_normal.logpdf(
            z, mean=GAUSSIAN_MEAN, cov=GAUSSIAN_COVARIANCE,
        )
        np.testing.assert_allclose(model.log_likelihood(u), expected, rtol=1e-12)
        np.testing.assert_allclose(
            model.log_prior(u), multivariate_normal.logpdf(x, mean=np.zeros(10)),
            rtol=1e-12,
        )
    assert np.isfinite(reference["log_Z"])


def test_g10_evidence_matches_rank_one_formula():
    _, reference = build_case("g10")
    # I + Sigma = 1.01 I + 0.99 11^T: determinant lemma and Sherman--Morrison.
    diagonal, rank_one, dimension = 1.01, .99, 10
    logdet = (dimension-1)*np.log(diagonal) + np.log(diagonal+dimension*rank_one)
    quadratic = 9. * (1./diagonal - rank_one/(diagonal*(diagonal+dimension*rank_one)))
    expected = -.5*(dimension*np.log(2.*np.pi)+logdet+quadratic)
    np.testing.assert_allclose(reference["log_Z"], expected, atol=1e-12, rtol=0)


def test_cg10_evidence_quadrature_refinement():
    _, reference = build_case("cg10")
    for order in (128, 256):
        np.testing.assert_allclose(cg10_hermite_reference(order), reference["log_Z"],
                                   atol=1e-9, rtol=0)


def test_repository_ss10_is_identical_to_previous_run_model():
    model, reference = build_case("ss10")
    old = build_model("ss10", "repository")
    for seed in range(5):
        key = jax.random.PRNGKey(seed)
        u = model.sample_U(key)
        for new_leaf, old_leaf in zip(jax.tree.leaves(u),
                                      jax.tree.leaves(old.sample_U(key)), strict=True):
            np.testing.assert_array_equal(new_leaf, old_leaf)
        np.testing.assert_array_equal(model.log_likelihood(u), old.log_likelihood(u))
    np.testing.assert_allclose(reference["log_Z"], -24.15593480605331,
                               atol=1e-12, rtol=0)


def test_evidence_runner_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.paper_reproduction.run_evidence10", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--resume-from" in result.stdout
    assert "--goal-log-z-uncert" in result.stdout


def test_full_phantom_checkpoint_preserves_continuation():
    model, _ = build_case("ss10")
    sampler = NestedSampler(
        model=model, root_allocation_degree=8, shell_size=4, delta_K=8,
        sampler=UniDimSliceSampler(model=model, num_slices=5,
                                   collect_phantom_samples=True,
                                   max_phantom_samples=4, no_step_out=True),
        collect_phantom_samples=True, max_phantom_samples=4,
        allocation_target="evidence_improving", unlimited_samples=True,
        depth_condition=DepthCondition(dlogZ=np.log1p(.01)),
    )
    bootstrap = dataclasses.replace(sampler, allocation_target="uniform", delta_K=1)
    state = bootstrap.run_until_goal(lambda s: int(s.goal_loop_iter) >= 1,
                                     key=jax.random.PRNGKey(0))
    restored = pickle.loads(pickle.dumps(state))
    expected = sampler.resume_until_goal(state, lambda s: int(s.goal_loop_iter) >= 3).trim()
    actual = sampler.resume_until_goal(restored, lambda s: int(s.goal_loop_iter) >= 3).trim()
    assert int(actual.num_samples) == int(expected.num_samples)
    for left, right in zip(jax.tree.leaves(actual.samples),
                           jax.tree.leaves(expected.samples), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(actual.random_key, expected.random_key)
    np.testing.assert_array_equal(actual.goal_key, expected.goal_key)
