from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.race_tree import BlockState
from jaxns.shrinkage import (
    classic_dirichlet_concentrations,
    sample_gamma_weighted_phantom_probabilities,
)

tfpd = tfp.distributions


class ContinuousCalibrationEnsemble(NamedTuple):
    deterministic_log_z: np.ndarray
    deterministic_log_z_uncert: np.ndarray
    posterior_x_mean: np.ndarray


def _linear_likelihood_model() -> Model:
    def prior_model():
        x = Prior(tfpd.Uniform(0.0, 1.0), name="x").realise()
        likelihood = jnp.maximum(2.0 * x, jnp.finfo(x.dtype).tiny)
        return jnp.log(likelihood)

    return Model(prior_model=prior_model)


def _plateau_likelihood_model(atom_mass: float) -> Model:
    def prior_model():
        x = Prior(tfpd.Uniform(0.0, 1.0), name="x").realise()
        likelihood = jnp.where(x < atom_mass, 1.0, 3.0)
        return jnp.log(likelihood)

    return Model(prior_model=prior_model)


def _assert_ensemble_mean_consistent_with_zero(
        values: np.ndarray,
        *,
        standard_errors: float = 4.0,
) -> None:
    sample_standard_error = np.std(values, ddof=1) / np.sqrt(values.size)
    assert sample_standard_error > 0.0
    assert abs(np.mean(values)) <= standard_errors * sample_standard_error


@pytest.fixture(scope="module")
def continuous_calibration_ensemble() -> ContinuousCalibrationEnsemble:
    """Reuse one deterministic 30-run ensemble across calibration checks."""
    model = _linear_likelihood_model()
    ns = NestedSampler(
        model=model,
        root_allocation_degree=30,
        shell_size=10,
        max_samples=3000,
        collect_phantom_samples=True,
    )
    deterministic_log_z = []
    deterministic_log_z_uncert = []
    posterior_x_mean = []

    for seed in range(30):
        run_key = jax.random.PRNGKey(seed)
        state = ns.run(run_key)
        results = state.to_result()
        jax.block_until_ready((state, results))

        # Compute this scalar directly from the public posterior measure. The
        # dedicated integration invariant test covers the JIT helper itself;
        # avoiding 30 variable-shape compilations keeps calibration fast.
        num_samples = int(results.total_num_samples)
        x = np.asarray(results.X_samples.get_dotted("x"))[:num_samples]
        weights = np.exp(np.asarray(results.log_dp)[:num_samples])
        x_mean = np.sum(weights * x)

        deterministic_log_z.append(float(results.log_Z_mean))
        deterministic_log_z_uncert.append(float(results.log_Z_uncert))
        posterior_x_mean.append(float(x_mean))

    return ContinuousCalibrationEnsemble(
        deterministic_log_z=np.asarray(deterministic_log_z),
        deterministic_log_z_uncert=np.asarray(deterministic_log_z_uncert),
        posterior_x_mean=np.asarray(posterior_x_mean),
    )


def test_known_evidence_normalized_errors_are_centered_and_unit_scale(
        continuous_calibration_ensemble,
):
    """Repeated reported errors must calibrate against the analytic log Z=0."""
    ensemble = continuous_calibration_ensemble
    z_scores = (
        ensemble.deterministic_log_z
        / ensemble.deterministic_log_z_uncert
    )

    # Under calibrated normal errors, these are direct standard-error and
    # chi-square-derived checks. The broad four-sigma bounds avoid a flaky
    # release gate while still rejecting material bias or scale errors.
    assert abs(np.mean(z_scores)) <= 4.0 / np.sqrt(z_scores.size)
    assert 0.6 <= np.std(z_scores, ddof=1) <= 1.4


def test_known_posterior_ensemble_converges_to_analytic_mean(
        continuous_calibration_ensemble,
):
    """Independent nested-sampling information converges to E[x]=2/3."""
    errors = continuous_calibration_ensemble.posterior_x_mean - (2.0 / 3.0)

    _assert_ensemble_mean_consistent_with_zero(errors)
    assert np.sqrt(np.mean(np.square(errors))) < 0.08


def test_known_likelihood_atom_recovers_strict_volume_and_atom_mass():
    """A two-level likelihood recovers its known strict and equality masses."""
    atom_mass = 0.4
    model = _plateau_likelihood_model(atom_mass)
    ns = NestedSampler(
        model=model,
        root_allocation_degree=128,
        shell_size=8,
        max_samples=256,
    )
    results = ns.run(jax.random.PRNGKey(71)).to_result()
    evidence = results.sample_evidence_mc(
        num_samples=2048,
        conditioning="classic",
        key=jax.random.PRNGKey(72),
    )
    jax.block_until_ready(evidence)

    valid = np.isfinite(np.asarray(evidence.log_L_blocks))
    low_block = np.flatnonzero(
        valid & np.isclose(np.asarray(evidence.log_L_blocks), 0.0)
    )
    assert low_block.size == 1
    block_idx = int(low_block[0])
    inferred_strict = np.asarray(evidence.p_gt_samples)[:, block_idx]
    inferred_atom = np.asarray(evidence.p_eq_samples)[:, block_idx]

    # The truth must lie inside the posterior ensemble at a tolerance that
    # includes both shrinkage uncertainty and the finite 512-point race.
    strict_mc_se = np.std(inferred_strict, ddof=1) / np.sqrt(
        inferred_strict.size
    )
    atom_mc_se = np.std(inferred_atom, ddof=1) / np.sqrt(
        inferred_atom.size
    )
    assert abs(np.mean(inferred_strict) - (1.0 - atom_mass)) < (
        4.0 * np.std(inferred_strict, ddof=1) + 4.0 * strict_mc_se
    )
    assert abs(np.mean(inferred_atom) - atom_mass) < (
        4.0 * np.std(inferred_atom, ddof=1) + 4.0 * atom_mc_se
    )

    # A lone continuous observation cannot establish an equality atom. Keep
    # this endpoint in the same calibration contract so an implementation
    # cannot improve the plateau fit by assigning atoms to singleton blocks.
    singleton_block = BlockState(
        log_L_blocks=jnp.asarray([0.0]),
        block_first_idx=jnp.asarray([0], dtype=jnp.int32),
        block_size=jnp.asarray([1], dtype=jnp.int32),
        incoming_K=jnp.asarray([24], dtype=jnp.int32),
        block_out_degree=jnp.asarray([0], dtype=jnp.int32),
        valid=jnp.asarray([True]),
        block_sample_indices=jnp.asarray([[0]], dtype=jnp.int32),
    )
    singleton = classic_dirichlet_concentrations(singleton_block)
    np.testing.assert_array_equal(singleton.alpha_eq, np.asarray([0.0]))


def test_phantom_conditioning_does_not_introduce_systematic_evidence_bias(
):
    """Matched independent phantoms add information without shrinkage bias."""
    true_p_gt = 24.0 / 25.0
    num_clusters = 200
    block_state = BlockState(
        log_L_blocks=jnp.asarray([0.0]),
        block_first_idx=jnp.asarray([0], dtype=jnp.int32),
        block_size=jnp.asarray([1], dtype=jnp.int32),
        incoming_K=jnp.asarray([24], dtype=jnp.int32),
        block_out_degree=jnp.asarray([0], dtype=jnp.int32),
        valid=jnp.asarray([True]),
        block_sample_indices=jnp.asarray([[0]], dtype=jnp.int32),
    )
    A_cg = jnp.ones((num_clusters, 1))
    B_cg = jnp.concatenate([
        jnp.ones((192, 1)),
        jnp.zeros((8, 1)),
    ])
    E_cg = jnp.zeros_like(A_cg)
    num_draws = 8192
    key = jax.random.PRNGKey(91)
    classic = sample_gamma_weighted_phantom_probabilities(
        key=key,
        block_state=block_state,
        A_cg=jnp.zeros_like(A_cg),
        B_cg=jnp.zeros_like(B_cg),
        E_cg=E_cg,
        num_samples=num_draws,
        C_min=20,
    )
    conditioned = sample_gamma_weighted_phantom_probabilities(
        key=key,
        block_state=block_state,
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        num_samples=num_draws,
        C_min=20,
    )

    classic_shell_mass = 1.0 - np.asarray(classic.p_gt_samples)[:, 0]
    conditioned_shell_mass = (
        1.0 - np.asarray(conditioned.p_gt_samples)[:, 0]
    )
    true_shell_mass = 1.0 - true_p_gt
    for samples in (classic_shell_mass, conditioned_shell_mass):
        standard_error = np.std(samples, ddof=1) / np.sqrt(num_draws)
        assert abs(np.mean(samples) - true_shell_mass) <= 5.0 * standard_error
    assert np.std(conditioned_shell_mass) < np.std(classic_shell_mass)
