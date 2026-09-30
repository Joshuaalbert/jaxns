"""Finite-ensemble calibration on exactly sampled analytic prior measures.

These tests isolate inference from mode discovery and MCMC mixing. They do not
certify uncertainty calibration when a production run misses structure.
"""

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.algorithm.race_tree import build_block_state
from jaxns.samples import PhantomSamples, Samples
from jaxns.shrinkage.classic import (
    classic_dirichlet_concentrations,
    dirichlet_probability_means,
    expected_evidence_summary,
    expected_log_posterior_weights,
)
from jaxns.shrinkage.phantom import sample_evidence


def _root_race(x, log_likelihood):
    """A complete tree of independent prior draws, with no child lineages."""
    count = x.shape[0]
    samples = Samples(
        U_samples=x,
        log_likelihoods=log_likelihood,
        log_L_constraints=jnp.full((count,), -jnp.inf),
        out_degree=jnp.zeros(count, dtype=jnp.int32),
        num_likelihood_evaluations=jnp.ones(count, dtype=jnp.int32),
        phantom_samples=PhantomSamples(
            U_samples=None, valid_mask=jnp.zeros(count, dtype=bool),
            log_L=jnp.zeros((count, 0)),
        ),
    )
    return build_block_state(samples, jnp.asarray(count, dtype=jnp.int32))


def test_independent_prior_races_calibrate_evidence_and_converge_in_posterior():
    # Uniform x, L=x^4: Z=1/5 and posterior mean E[x]=5/6.
    # Vary only independent information, keeping the likelihood and inference
    # law fixed. Fixed keys make the statistical acceptance test reproducible.
    repetitions = 256
    keys = jax.random.split(jax.random.PRNGKey(303), repetitions)
    posterior_rmse = []
    for count in (32, 512):
        def assess(key, count=count):
            x = jax.random.uniform(key, (count,), dtype=jnp.float64)
            blocks = _root_race(x, 4 * jnp.log(x))
            concentrations = classic_dirichlet_concentrations(blocks)
            evidence = expected_evidence_summary(blocks, concentrations)
            weights = jnp.exp(expected_log_posterior_weights(blocks, concentrations))
            return evidence.log_Z_mean, evidence.log_Z_uncert, jnp.sum(weights * x)

        log_z, uncertainty, posterior_mean = map(
            np.asarray, jax.jit(jax.vmap(assess))(keys),
        )
        posterior_rmse.append(np.sqrt(np.mean((posterior_mean - 5 / 6) ** 2)))
    standardized = (log_z - np.log(1 / 5)) / uncertainty
    # Five ensemble standard errors allow ordinary Monte Carlo variation,
    # while detecting nonzero bias and an incorrect uncertainty scale.
    assert abs(standardized.mean()) < 5 / np.sqrt(repetitions)
    assert abs(standardized.std(ddof=1) - 1) < 5 / np.sqrt(2 * (repetitions - 1))
    assert abs(posterior_mean.mean() - 5 / 6) < (
        5 * posterior_mean.std(ddof=1) / np.sqrt(repetitions)
    )
    assert posterior_rmse[1] < 0.5 * posterior_rmse[0]


def test_known_plateau_volumes_and_atoms_converge_with_independent_information():
    repetitions = 128
    keys = jax.random.split(jax.random.PRNGKey(304), repetitions)
    errors = []
    reference = np.asarray([0.8, 0.5, 0.0, 0.2, 0.3, 0.5])
    for count in (64, 1024):
        def assess(key, count=count):
            x = jax.random.uniform(key, (count,), dtype=jnp.float64)
            log_l = jnp.where(x < 0.2, 0.0, jnp.where(x < 0.5, 1.0, 2.0))
            blocks = _root_race(x, log_l)
            concentrations = classic_dirichlet_concentrations(blocks)
            p_gt, p_eq, _ = dirichlet_probability_means(concentrations)
            volume = jnp.cumprod(p_gt[:3])
            previous = jnp.concatenate((jnp.ones(1), volume[:-1]))
            return jnp.concatenate((volume, previous * p_eq[:3]))

        inferred = np.asarray(jax.jit(jax.vmap(assess))(keys))
        errors.append(np.sqrt(np.mean((inferred - reference) ** 2, axis=0)))
    assert np.all(errors[1] < 0.5 * errors[0])
    np.testing.assert_allclose(inferred.mean(axis=0), reference, atol=0.01)
    # A continuous singleton is not an atom even beside well-resolved plateaus.
    blocks = _root_race(jnp.arange(5.0), jnp.asarray([0.0, 0.0, 1.0, 2.0, 2.0]))
    concentrations = classic_dirichlet_concentrations(blocks)
    assert float(concentrations.alpha_eq[1]) == 0.0


def test_stationary_phantom_conditioning_preserves_analytic_evidence_and_posterior():
    repetitions = 32
    root_count = 128
    phantom_count = 7
    rng = np.random.default_rng(305)
    classic_estimates = []
    phantom_estimates = []
    posterior_means = []
    # Match the compiled result path so this ensemble does not perform a host
    # scatter for every sample in every independently generated race.
    posterior_weights = jax.jit(expected_log_posterior_weights)
    for repetition in range(repetitions):
        roots = rng.uniform(size=root_count)
        # Exact independent chains: each transition forgets its initial state
        # and draws uniformly above its root parent's x. Retained observations
        # and the final classic child have the same constrained-prior marginal.
        chain = roots[:, None] + (1 - roots[:, None]) * rng.uniform(
            size=(root_count, phantom_count + 1),
        )
        x = np.r_[roots, chain[:, -1]]
        count = x.size
        samples = Samples(
            U_samples=jnp.asarray(x),
            log_likelihoods=jnp.asarray(4 * np.log(x)),
            log_L_constraints=jnp.asarray(np.r_[
                np.full(root_count, -np.inf), 4 * np.log(roots),
            ]),
            out_degree=jnp.asarray(np.r_[
                np.ones(root_count), np.zeros(root_count),
            ], dtype=jnp.int32),
            num_likelihood_evaluations=jnp.ones(count, dtype=jnp.int32),
            phantom_samples=PhantomSamples(
                U_samples=None,
                valid_mask=jnp.asarray(np.r_[
                    np.zeros(root_count, dtype=bool), np.ones(root_count, dtype=bool),
                ]),
                log_L=jnp.asarray(np.r_[
                    np.full((root_count, phantom_count), -np.inf),
                    4 * np.log(chain[:, :-1]),
                ]),
            ),
        )
        blocks = build_block_state(samples, jnp.asarray(root_count, dtype=jnp.int32))
        concentrations = classic_dirichlet_concentrations(blocks)
        expected = expected_evidence_summary(blocks, concentrations)
        classic_estimates.append(float(jnp.exp(expected.log_Z_linear_mean)))
        log_weights = posterior_weights(blocks, concentrations)
        posterior_means.append(float(jnp.sum(jnp.exp(log_weights) * x)))
        # Public inference validates all associations, then exercises the real
        # count matrices, Kish gate, shared cluster weights, and evidence path.
        incoming = jnp.zeros(count, dtype=blocks.incoming_K.dtype).at[
            blocks.block_sample_indices
        ].set(blocks.incoming_K)
        conditioned = sample_evidence(
            key=jax.random.PRNGKey(400 + repetition),
            log_L_constraints=samples.log_L_constraints,
            log_L_classic=samples.log_likelihoods,
            K_classic=incoming,
            valid_phantom=samples.phantom_samples.valid_mask,
            log_L_phantom=samples.phantom_samples.log_L,
            num_samples=jnp.asarray(count),
            num_Z_samples=128,
            block_state=blocks,
            C_min=20,
            batch_size=32,
            diagnostics=False,
        )
        assert np.any(np.asarray(conditioned.phantom_gate_active))
        phantom_estimates.append(float(jnp.mean(jnp.exp(conditioned.log_Z_samples))))
        np.testing.assert_array_equal(
            posterior_weights(blocks, concentrations), log_weights,
        )
    for estimates in (classic_estimates, phantom_estimates):
        estimates = np.asarray(estimates)
        assert abs(estimates.mean() - 1 / 5) < (
            5 * estimates.std(ddof=1) / np.sqrt(repetitions)
        )
    difference = np.asarray(phantom_estimates) - classic_estimates
    assert abs(difference.mean()) < 5 * difference.std(ddof=1) / np.sqrt(repetitions)
    posterior_means = np.asarray(posterior_means)
    assert abs(posterior_means.mean() - 5 / 6) < (
        5 * posterior_means.std(ddof=1) / np.sqrt(repetitions)
    )
