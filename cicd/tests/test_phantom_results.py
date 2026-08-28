import dataclasses
from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jaxns.results as results_module
from cicd.tests.distributed_support import make_toy_model
from jaxns.algorithm.race_tree import BlockState
from jaxns.constrained_sampler import AbstractSampler
from jaxns.core import NestedSampler
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.results import BlockData, NestedSamplerResults
from jaxns.samples import PhantomSamples
from jaxns.shrinkage.phantom import sample_mc_shrinkage
from jaxns.termination_condition import TerminationCondition


class ResultCase(NamedTuple):
    results: NestedSamplerResults
    expected_map_idx: int
    expected_resample_idx: int


@dataclasses.dataclass(slots=True, frozen=True)
class _HighPhantomLikelihoodSampler(AbstractSampler, PureDataclassPytree):
    """Sampler probe whose phantom likelihood beats every classic draw."""

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [])

    def num_phantom(self) -> int:
        return 1

    def get_sample(
        self,
        key,
        log_L_constraint,
        seed_point,
        args=(),
        params=None,
    ):
        del key, log_L_constraint, seed_point, args, params
        classic_u = jnp.asarray(0.25, dtype=mp_policy.measure_dtype)
        phantom_u = jnp.asarray([0.75], dtype=mp_policy.measure_dtype)
        return (
            classic_u,
            jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
            jnp.asarray(1, dtype=mp_policy.count_dtype),
            PhantomSamples(
                U_samples=phantom_u,
                log_L=jnp.asarray([10.0], dtype=mp_policy.measure_dtype),
                valid_mask=jnp.asarray([True], dtype=mp_policy.bool_dtype),
            ),
        )


_HighPhantomLikelihoodSampler.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class _MixedPhantomValiditySampler(AbstractSampler, PureDataclassPytree):
    """Sampler probe with one invalid phantom slot in every cluster."""

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [])

    def num_phantom(self) -> int:
        return 2

    def get_sample(
        self,
        key,
        log_L_constraint,
        seed_point,
        args=(),
        params=None,
    ):
        del key, log_L_constraint, seed_point, args, params
        classic_u = jnp.asarray(0.25, dtype=mp_policy.measure_dtype)
        phantom_u = jnp.asarray([0.50, 0.75], dtype=mp_policy.measure_dtype)
        return (
            classic_u,
            jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
            jnp.asarray(1, dtype=mp_policy.count_dtype),
            PhantomSamples(
                U_samples=phantom_u,
                log_L=jnp.asarray([1.0, 2.0], dtype=mp_policy.measure_dtype),
                valid_mask=jnp.asarray(
                    [True, False],
                    dtype=mp_policy.bool_dtype,
                ),
            ),
        )


_MixedPhantomValiditySampler.register_pytree()


def _make_result_case() -> ResultCase:
    u_samples = jnp.asarray([0.10, 0.20, 0.30], dtype=mp_policy.measure_dtype)
    log_l = jnp.asarray([0.0, 1.0, 2.0], dtype=mp_policy.measure_dtype)
    log_posterior_density = jnp.asarray(
        [4.0, 1.0, -1.0],
        dtype=mp_policy.measure_dtype,
    )
    map_idx = 0

    results = NestedSamplerResults(
        log_Z_mean=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        log_Z_uncert=jnp.asarray(0.1, dtype=mp_policy.measure_dtype),
        ess=jnp.asarray(2.0, dtype=mp_policy.measure_dtype),
        H_mean=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        total_num_samples=jnp.asarray(3, dtype=mp_policy.count_dtype),
        total_phantom_samples=jnp.asarray(6, dtype=mp_policy.count_dtype),
        total_num_likelihood_evaluations=jnp.asarray(
            12,
            dtype=mp_policy.count_dtype,
        ),
        log_efficiency=jnp.log(
            jnp.asarray(0.25, dtype=mp_policy.measure_dtype),
        ),
        termination_reason=jnp.asarray(0, dtype=mp_policy.count_dtype),
        U_samples=u_samples,
        X_samples=u_samples,
        log_L_constraints=jnp.asarray(
            [-jnp.inf, 0.0, 1.0],
            dtype=mp_policy.measure_dtype,
        ),
        log_L_phantom=jnp.asarray(
            [
                [50.0, 51.0],
                [52.0, 53.0],
                [54.0, 55.0],
            ],
            dtype=mp_policy.measure_dtype,
        ),
        valid_phantom=jnp.asarray(
            [True, True, True],
            dtype=mp_policy.bool_dtype,
        ),
        log_L=log_l,
        log_dp=jnp.asarray(
            [0.0, -1000.0, -1000.0],
            dtype=mp_policy.measure_dtype,
        ),
        log_X_mean=jnp.asarray(
            [0.0, -0.5, -1.0],
            dtype=mp_policy.measure_dtype,
        ),
        log_posterior_density=log_posterior_density,
        num_live_points_per_sample=jnp.asarray(
            [4, 3, 2],
            dtype=mp_policy.count_dtype,
        ),
        num_likelihood_evaluations_per_sample=jnp.asarray(
            [2, 2, 2],
            dtype=mp_policy.count_dtype,
        ),
        log_L_supremum=log_l[-1],
        U_supremum=u_samples[-1],
        X_supremum=u_samples[-1],
        log_L_map=log_l[map_idx],
        U_map=u_samples[map_idx],
        X_map=u_samples[map_idx],
    )
    return ResultCase(
        results=results,
        expected_map_idx=map_idx,
        expected_resample_idx=0,
    )


def test_integrate_semi_positive_values_over_posterior():
    """The semi-positive path must average values, not exponentiate them."""
    results = dataclasses.replace(
        _make_result_case().results,
        X_samples=jnp.asarray(
            [0.0, 1.0, 2.0],
            dtype=mp_policy.measure_dtype,
        ),
        log_dp=jnp.log(
            jnp.asarray(
                [0.25, 0.75, 0.0],
                dtype=mp_policy.measure_dtype,
            )
        ),
    )

    posterior_mean = results.integrate_fn_over_posterior(
        lambda x: x,
        semi_positive=True,
    )

    np.testing.assert_allclose(posterior_mean, 0.75)


def _make_padded_plateau_result(num_phantom: int) -> NestedSamplerResults:
    u_samples = jnp.asarray([0.10, 0.20], dtype=mp_policy.measure_dtype)
    log_l = jnp.asarray([0.0, 0.0], dtype=mp_policy.measure_dtype)
    valid_phantom = jnp.full(
        (2,),
        num_phantom > 0,
        dtype=mp_policy.bool_dtype,
    )
    log_l_phantom = jnp.full(
        (2, num_phantom),
        0.5,
        dtype=mp_policy.measure_dtype,
    )

    return NestedSamplerResults(
        log_Z_mean=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        log_Z_uncert=jnp.asarray(0.1, dtype=mp_policy.measure_dtype),
        ess=jnp.asarray(2.0, dtype=mp_policy.measure_dtype),
        H_mean=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        total_num_samples=jnp.asarray(2, dtype=mp_policy.count_dtype),
        total_phantom_samples=jnp.asarray(
            2 * num_phantom,
            dtype=mp_policy.count_dtype,
        ),
        total_num_likelihood_evaluations=jnp.asarray(
            2 + 2 * num_phantom,
            dtype=mp_policy.count_dtype,
        ),
        log_efficiency=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        termination_reason=jnp.asarray(0, dtype=mp_policy.count_dtype),
        U_samples=u_samples,
        X_samples=u_samples,
        log_L_constraints=jnp.full(
            (2,),
            -jnp.inf,
            dtype=mp_policy.measure_dtype,
        ),
        log_L_phantom=log_l_phantom,
        valid_phantom=valid_phantom,
        log_L=log_l,
        log_dp=jnp.log(
            jnp.asarray([0.5, 0.5], dtype=mp_policy.measure_dtype),
        ),
        log_X_mean=jnp.asarray([-0.5, -1.0], dtype=mp_policy.measure_dtype),
        log_posterior_density=log_l,
        num_live_points_per_sample=jnp.asarray(
            [2, 2],
            dtype=mp_policy.count_dtype,
        ),
        num_likelihood_evaluations_per_sample=jnp.asarray(
            [1 + num_phantom, 1 + num_phantom],
            dtype=mp_policy.count_dtype,
        ),
        log_L_supremum=log_l[0],
        U_supremum=u_samples[0],
        X_supremum=u_samples[0],
        log_L_map=log_l[0],
        U_map=u_samples[0],
        X_map=u_samples[0],
        block_data=BlockData(
            log_L=jnp.asarray(
                [0.0, jnp.inf],
                dtype=mp_policy.measure_dtype,
            ),
            first_idx=jnp.asarray([0, -1], dtype=mp_policy.index_dtype),
            size=jnp.asarray([2, 0], dtype=mp_policy.count_dtype),
            incoming_K=jnp.asarray([2, 0], dtype=mp_policy.count_dtype),
            out_degree=jnp.asarray([0, 0], dtype=mp_policy.count_dtype),
            valid=jnp.asarray([True, False], dtype=mp_policy.bool_dtype),
        ),
    )


def _make_single_block_gamma_public_result() -> NestedSamplerResults:
    num_samples = 20
    u_samples = jnp.linspace(0.05, 0.95, num_samples)
    log_l = jnp.zeros((num_samples,), dtype=mp_policy.measure_dtype)
    log_l_phantom = jnp.asarray(
        np.concatenate(
            [
                np.full((12, 1), 1.0, dtype=float),
                np.full((4, 1), 0.0, dtype=float),
                np.full((4, 1), -1.0, dtype=float),
            ],
            axis=0,
        ),
        dtype=mp_policy.measure_dtype,
    )
    return NestedSamplerResults(
        log_Z_mean=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        log_Z_uncert=jnp.asarray(0.1, dtype=mp_policy.measure_dtype),
        ess=jnp.asarray(10.0, dtype=mp_policy.measure_dtype),
        H_mean=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        total_num_samples=jnp.asarray(num_samples, dtype=mp_policy.count_dtype),
        total_phantom_samples=jnp.asarray(num_samples, dtype=mp_policy.count_dtype),
        total_num_likelihood_evaluations=jnp.asarray(40, dtype=mp_policy.count_dtype),
        log_efficiency=jnp.log(jnp.asarray(0.5, dtype=mp_policy.measure_dtype)),
        termination_reason=jnp.asarray(0, dtype=mp_policy.count_dtype),
        U_samples=u_samples,
        X_samples=u_samples,
        log_L_constraints=jnp.full((num_samples,), -jnp.inf, dtype=mp_policy.measure_dtype),
        log_L_phantom=log_l_phantom,
        valid_phantom=jnp.ones((num_samples,), dtype=mp_policy.bool_dtype),
        log_L=log_l,
        log_dp=jnp.full((num_samples,), -jnp.log(num_samples), dtype=mp_policy.measure_dtype),
        log_X_mean=-jnp.linspace(0.0, 1.0, num_samples),
        log_posterior_density=log_l,
        num_live_points_per_sample=jnp.full((num_samples,), 40, dtype=mp_policy.count_dtype),
        num_likelihood_evaluations_per_sample=jnp.full((num_samples,), 2, dtype=mp_policy.count_dtype),
        log_L_supremum=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        U_supremum=u_samples[0],
        X_supremum=u_samples[0],
        log_L_map=jnp.asarray(0.0, dtype=mp_policy.measure_dtype),
        U_map=u_samples[0],
        X_map=u_samples[0],
        block_data=BlockData(
            log_L=jnp.asarray([0.0], dtype=mp_policy.measure_dtype),
            first_idx=jnp.asarray([0], dtype=mp_policy.index_dtype),
            size=jnp.asarray([num_samples], dtype=mp_policy.count_dtype),
            incoming_K=jnp.asarray([40], dtype=mp_policy.count_dtype),
            out_degree=jnp.asarray([0], dtype=mp_policy.count_dtype),
            valid=jnp.asarray([True], dtype=mp_policy.bool_dtype),
            start=jnp.asarray([0], dtype=mp_policy.index_dtype),
            stop=jnp.asarray([num_samples], dtype=mp_policy.index_dtype),
            sample_indices=jnp.arange(
                num_samples,
                dtype=mp_policy.index_dtype,
            )[None, :],
        ),
    )


def _assert_result_probability_mean(samples, expected: np.ndarray, *, atol: float) -> None:
    got = np.stack(
        [
            np.asarray(samples.p_gt_samples)[:, 0],
            np.asarray(samples.p_eq_samples)[:, 0],
            np.asarray(samples.p_lt_samples)[:, 0],
        ],
        axis=-1,
    )
    np.testing.assert_allclose(np.mean(got, axis=0), expected, atol=atol)


def _run_high_phantom_probe():
    sampler = _HighPhantomLikelihoodSampler()
    ns = NestedSampler(
        model=make_toy_model(),
        sampler=sampler,
        target_num_live_points=2,
        max_samples=3,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=3),
        store_phantom_samples=True,
    )
    return ns.run(jax.random.PRNGKey(11))


def _run_mixed_validity_probe():
    sampler = _MixedPhantomValiditySampler()
    ns = NestedSampler(
        model=make_toy_model(),
        sampler=sampler,
        target_num_live_points=2,
        max_samples=3,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=3),
        store_phantom_samples=True,
    )
    return ns.run(jax.random.PRNGKey(23))


def test_results_expose_public_phantom_conditioning_diagnostics():
    result_case = _make_result_case()

    diagnostics = result_case.results.phantom_conditioning_diagnostics(
        C_min=2,
    )

    for field_name in (
            "kish_participating_cluster_counts",
            "phantom_gate_active",
            "A_g",
            "B_g",
            "E_g",
            "R_g",
    ):
        assert hasattr(diagnostics, field_name), (
            "phantom_conditioning_diagnostics() must expose "
            f"{field_name}."
        )

    np.testing.assert_allclose(
        np.asarray(diagnostics.kish_participating_cluster_counts),
        np.asarray([1.0, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics.phantom_gate_active, dtype=bool),
        np.asarray([False, True, True]),
    )
    np.testing.assert_allclose(np.asarray(diagnostics.A_g), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(np.asarray(diagnostics.B_g), [2.0, 4.0, 6.0])
    np.testing.assert_allclose(np.asarray(diagnostics.E_g), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(diagnostics.R_g), [0.0, 0.0, 0.0])


def test_results_sample_mc_shrinkage_exposes_kish_gate_diagnostics_not_target_rho():
    result_case = _make_result_case()

    evidence_samples = result_case.results.sample_mc_shrinkage(
        num_samples=4,
        key=jax.random.PRNGKey(7),
        C_min=2,
    )

    np.testing.assert_allclose(
        np.asarray(evidence_samples.log_L_blocks),
        np.asarray(result_case.results.log_L),
    )
    for field_name in (
            "kish_participating_cluster_counts",
            "phantom_gate_active",
            "phantom_A",
            "phantom_B",
            "phantom_E",
            "phantom_R",
    ):
        assert hasattr(evidence_samples, field_name), (
            "EvidenceSamples from result.sample_mc_shrinkage() must expose "
            f"{field_name}."
        )
        assert np.asarray(getattr(evidence_samples, field_name)).shape == (
            result_case.results.log_L.shape
        )
    for old_name in ("rho_values", "rho_fit", "rho_samples", "rho_eta_samples"):
        if hasattr(evidence_samples, old_name):
            assert getattr(evidence_samples, old_name) is None


def test_results_sample_mc_shrinkage_uses_gamma_conditioning_when_gate_active():
    results = _make_single_block_gamma_public_result()
    num_draws = 4096

    active = results.sample_mc_shrinkage(
        num_samples=num_draws,
        key=jax.random.PRNGKey(103),
        C_min=20,
    )
    inactive = results.sample_mc_shrinkage(
        num_samples=num_draws,
        key=jax.random.PRNGKey(103),
        C_min=21,
    )

    for samples in (active, inactive):
        for field_name in ("p_gt_samples", "p_eq_samples", "p_lt_samples"):
            assert hasattr(samples, field_name), (
                "NestedSamplerResults.sample_mc_shrinkage() must return "
                f"{field_name} so the public path target can be verified."
            )

    classic_alpha = np.asarray([21.0, 20.5, 0.5], dtype=float)
    active_alpha = np.asarray([33.0, 24.5, 4.5], dtype=float)
    _assert_result_probability_mean(
        inactive,
        classic_alpha / np.sum(classic_alpha),
        atol=0.02,
    )
    _assert_result_probability_mean(
        active,
        active_alpha / np.sum(active_alpha),
        atol=0.02,
    )
    assert abs(
        float(np.mean(np.asarray(active.p_gt_samples)[:, 0]))
        - float(np.mean(np.asarray(inactive.p_gt_samples)[:, 0]))
    ) > 0.03
    np.testing.assert_allclose(
        np.asarray(active.kish_participating_cluster_counts),
        np.asarray([20.0]),
    )
    np.testing.assert_array_equal(
        np.asarray(active.phantom_gate_active, dtype=bool),
        np.asarray([True]),
    )
    np.testing.assert_array_equal(
        np.asarray(inactive.phantom_gate_active, dtype=bool),
        np.asarray([False]),
    )


@pytest.mark.parametrize("batch_size", [1, 4, 10, 64])
def test_sample_evidence_mc_batches_have_exact_requested_shape(batch_size):
    """Cover single, partial-final, exact, and larger-than-request batches."""
    results = _make_single_block_gamma_public_result()

    samples = results.sample_evidence_mc(
        num_samples=10,
        conditioning="phantom",
        key=jax.random.PRNGKey(211),
        batch_size=batch_size,
    )

    assert samples.log_Z_samples.shape == (10,)
    assert samples.H_samples.shape == (10,)
    assert samples.p_gt_samples is None
    assert samples.p_eq_samples is None
    assert samples.p_lt_samples is None
    assert samples.phantom_add_gt_samples is None
    assert np.all(np.isfinite(np.asarray(samples.log_Z_samples)))


def test_sample_evidence_mc_batching_is_reproducible_and_matches_moments():
    """A fixed key is repeatable and batching preserves the sampled law."""
    results = _make_single_block_gamma_public_result()
    key = jax.random.PRNGKey(223)
    num_draws = 4096

    batched = results.sample_evidence_mc(
        num_samples=num_draws,
        conditioning="phantom",
        key=key,
        batch_size=257,
    )
    repeated = results.sample_evidence_mc(
        num_samples=num_draws,
        conditioning="phantom",
        key=key,
        batch_size=257,
    )
    unbatched = results.sample_mc_shrinkage(
        num_samples=num_draws,
        conditioning="phantom",
        key=key,
        diagnostics=False,
    )

    np.testing.assert_array_equal(
        np.asarray(batched.log_Z_samples),
        np.asarray(repeated.log_Z_samples),
    )
    np.testing.assert_allclose(
        float(batched.log_Z_mean),
        float(unbatched.log_Z_mean),
        atol=0.03,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        float(batched.log_Z_uncert),
        float(unbatched.log_Z_uncert),
        atol=0.03,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(batched.p_gt_mean),
        np.asarray(unbatched.p_gt_mean),
        atol=0.02,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.exp(np.asarray(batched.log_dZ_mean)),
        np.exp(np.asarray(unbatched.log_dZ_mean)),
        atol=0.02,
        rtol=0.0,
    )


def test_sample_evidence_mc_one_batch_preserves_fixed_key_draws():
    results = _make_single_block_gamma_public_result()
    key = jax.random.PRNGKey(227)

    unbatched = results.sample_mc_shrinkage(
        num_samples=10,
        conditioning="phantom",
        key=key,
        diagnostics=False,
    )
    larger_batch = results.sample_evidence_mc(
        num_samples=10,
        conditioning="phantom",
        key=key,
        batch_size=100,
    )

    np.testing.assert_array_equal(
        np.asarray(larger_batch.log_Z_samples),
        np.asarray(unbatched.log_Z_samples),
    )
    np.testing.assert_array_equal(
        np.asarray(larger_batch.H_samples),
        np.asarray(unbatched.H_samples),
    )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_sample_evidence_mc_rejects_invalid_batch_sizes(batch_size):
    results = _make_single_block_gamma_public_result()

    with pytest.raises(ValueError, match="batch_size.*positive integer"):
        results.sample_evidence_mc(
            num_samples=10,
            conditioning="phantom",
            key=jax.random.PRNGKey(229),
            batch_size=batch_size,
        )


def test_batched_summary_preserves_block_models_and_kish_gate():
    plateau = _make_single_block_gamma_public_result()
    active = plateau.sample_evidence_mc(
        num_samples=128,
        conditioning="phantom",
        key=jax.random.PRNGKey(233),
        batch_size=31,
        C_min=20,
    )
    inactive = plateau.sample_evidence_mc(
        num_samples=128,
        conditioning="phantom",
        key=jax.random.PRNGKey(233),
        batch_size=31,
        C_min=21,
    )
    singleton = _make_result_case().results.sample_evidence_mc(
        num_samples=32,
        conditioning="phantom",
        key=jax.random.PRNGKey(239),
        batch_size=7,
        C_min=1,
    )
    classic = _make_padded_plateau_result(0).sample_evidence_mc(
        num_samples=9,
        conditioning="classic",
        key=jax.random.PRNGKey(241),
        batch_size=1,
    )

    assert bool(np.asarray(active.phantom_gate_active)[0])
    assert not bool(np.asarray(inactive.phantom_gate_active)[0])
    assert float(np.asarray(active.p_eq_mean)[0]) > 0.0
    np.testing.assert_array_equal(
        np.asarray(singleton.p_eq_mean),
        np.zeros_like(np.asarray(singleton.p_eq_mean)),
    )
    np.testing.assert_array_equal(
        np.asarray(classic.phantom_gate_active),
        np.zeros_like(np.asarray(classic.phantom_gate_active), dtype=bool),
    )


def test_full_diagnostics_support_a_partial_final_batch():
    results = _make_single_block_gamma_public_result()

    samples = results.sample_evidence_mc(
        num_samples=10,
        conditioning="phantom",
        key=jax.random.PRNGKey(251),
        batch_size=4,
        diagnostics=True,
    )

    assert samples.p_gt_samples.shape == (10, 1)
    assert samples.p_eq_samples.shape == (10, 1)
    assert samples.p_lt_samples.shape == (10, 1)
    assert samples.phantom_add_gt_samples.shape == (10, 1)


def test_legacy_sample_evidence_uses_the_bounded_default():
    results = _make_single_block_gamma_public_result()
    key = jax.random.PRNGKey(257)

    default = results.sample_evidence(num_samples=100, key=key)
    explicit = results.sample_evidence(
        num_samples=100,
        batch_size=64,
        key=key,
    )

    np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


def test_results_sample_mc_shrinkage_matches_explicit_block_state_public_call():
    results = _make_single_block_gamma_public_result()
    block_state = results_module._block_state_from_results(results)
    assert block_state is not None

    key = jax.random.PRNGKey(107)
    result_samples = results.sample_mc_shrinkage(
        num_samples=16,
        key=key,
        C_min=20,
    )
    direct_samples = sample_mc_shrinkage(
        key=key,
        log_L_constraints=results.log_L_constraints,
        log_L_classic=results.log_L,
        K_classic=results.num_live_points_per_sample,
        valid_phantom=results.valid_phantom,
        log_L_phantom=results.log_L_phantom,
        num_samples=results.total_num_samples,
        num_Z_samples=16,
        block_state=block_state,
        C_min=20,
    )

    for field in dataclasses.fields(result_samples):
        expected = getattr(direct_samples, field.name)
        got = getattr(result_samples, field.name)
        if expected is None or got is None:
            assert got is expected
            continue
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_results_sample_mc_shrinkage_uses_block_state_helper(monkeypatch):
    results = _make_single_block_gamma_public_result()
    block_state = results_module._block_state_from_results(results)
    assert block_state is not None
    helper_calls = []
    sentinel = object()

    def fake_block_state_helper(*, block_state, **kwargs):
        helper_calls.append((block_state, kwargs))
        return sentinel

    def fail_generic_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "NestedSamplerResults.sample_mc_shrinkage() must use the "
            "block-state helper when result block arrays are present."
        )

    monkeypatch.setattr(
        results_module,
        "_sample_mc_shrinkage_with_block_state",
        fake_block_state_helper,
        raising=False,
    )
    monkeypatch.setattr(
        results_module,
        "_sample_mc_shrinkage",
        fail_generic_path,
    )

    got = results.sample_mc_shrinkage(
        num_samples=4,
        key=jax.random.PRNGKey(109),
        C_min=20,
    )

    assert got is sentinel
    assert len(helper_calls) == 1
    called_block_state, kwargs = helper_calls[0]
    np.testing.assert_allclose(
        np.asarray(called_block_state.log_L_blocks),
        np.asarray(block_state.log_L_blocks),
    )
    np.testing.assert_array_equal(
        np.asarray(called_block_state.block_size),
        np.asarray(block_state.block_size),
    )
    assert kwargs["num_samples"] == 4
    assert kwargs["C_min"] == 20


def test_trim_keeps_log_l_blocks_aligned_to_trimmed_sample_size():
    result_case = _make_result_case()
    results = dataclasses.replace(
        result_case.results,
        total_num_samples=jnp.asarray(2, dtype=mp_policy.count_dtype),
        block_data=BlockData(
            log_L=jnp.asarray(
                [0.0, 1.0, jnp.inf],
                dtype=mp_policy.measure_dtype,
            ),
            first_idx=jnp.asarray([0, 1, -1], dtype=mp_policy.index_dtype),
            size=jnp.asarray([1, 1, 0], dtype=mp_policy.count_dtype),
            incoming_K=jnp.asarray([4, 3, 0], dtype=mp_policy.count_dtype),
            out_degree=jnp.zeros((3,), dtype=mp_policy.count_dtype),
            valid=jnp.asarray([True, True, False], dtype=mp_policy.bool_dtype),
        ),
    )

    trimmed = results.trim()

    assert trimmed.log_L.shape == (2,)
    assert trimmed.block_data is not None
    assert trimmed.block_data.log_L.shape == (2,)
    evidence_samples = trimmed.sample_mc_shrinkage(
        num_samples=4,
        key=jax.random.PRNGKey(13),
    )
    np.testing.assert_allclose(
        np.asarray(trimmed.block_data.log_L),
        np.asarray(evidence_samples.log_L_blocks),
    )


@pytest.mark.parametrize("num_phantom", [0, 1])
def test_plateau_result_with_padded_inf_block_has_finite_h_samples(
        num_phantom,
):
    results = _make_padded_plateau_result(num_phantom)

    evidence_samples = results.sample_mc_shrinkage(
        num_samples=8,
        key=jax.random.PRNGKey(53 + num_phantom),
    )

    log_l_blocks = np.asarray(evidence_samples.log_L_blocks)
    assert np.isposinf(log_l_blocks[-1])
    assert np.all(np.isfinite(np.asarray(evidence_samples.H_samples)))


def test_result_point_estimates_do_not_have_or_use_phantom_coordinates():
    result_case = _make_result_case()
    results = result_case.results
    map_idx = result_case.expected_map_idx

    assert (
        float(jnp.max(results.log_L_phantom))
        > float(results.log_L_supremum)
    )
    for attr_name in (
        "U_phantom",
        "X_phantom",
        "U_phantom_samples",
        "X_phantom_samples",
        "phantom_U_samples",
        "phantom_X_samples",
    ):
        assert not hasattr(results, attr_name)

    np.testing.assert_allclose(results.log_L_supremum, jnp.max(results.log_L))
    np.testing.assert_allclose(
        results.U_supremum,
        results.U_samples[jnp.argmax(results.log_L)],
    )
    np.testing.assert_allclose(
        results.X_supremum,
        results.X_samples[jnp.argmax(results.log_L)],
    )
    np.testing.assert_allclose(results.log_L_map, results.log_L[map_idx])
    np.testing.assert_allclose(results.U_map, results.U_samples[map_idx])
    np.testing.assert_allclose(results.X_map, results.X_samples[map_idx])


def test_resample_keeps_phantom_likelihood_metadata_aligned():
    result_case = _make_result_case()
    results = result_case.results

    resampled = results.resample(
        num_samples=1,
        replace=True,
        key=jax.random.PRNGKey(0),
    )

    expected_idx = result_case.expected_resample_idx
    assert int(resampled.total_num_samples) == 1
    assert resampled.log_L_constraints.shape == (1,)
    assert resampled.log_L_phantom.shape == (
        1,
        results.log_L_phantom.shape[1],
    )
    assert resampled.valid_phantom.shape == (1,)
    # Posterior resampling destroys the race-tree block identities, so block
    # diagnostics must not masquerade as aligned data on the resampled result.
    assert resampled.block_data is None
    assert not hasattr(resampled, "U_phantom_samples")
    assert not hasattr(resampled, "X_phantom_samples")
    np.testing.assert_allclose(
        resampled.log_L,
        results.log_L[expected_idx:expected_idx + 1],
    )
    np.testing.assert_allclose(
        resampled.log_L_constraints,
        results.log_L_constraints[expected_idx:expected_idx + 1],
    )
    np.testing.assert_allclose(
        resampled.log_L_phantom,
        results.log_L_phantom[expected_idx:expected_idx + 1],
    )
    np.testing.assert_array_equal(
        np.asarray(resampled.valid_phantom),
        np.asarray(results.valid_phantom[expected_idx:expected_idx + 1]),
    )
    for field_name in (
        "effective_parent_idx",
        "requested_parent_idx",
        "requested_log_L_constraint",
        "phantom_seed_idx",
    ):
        assert not hasattr(resampled, field_name)


def test_results_sample_mc_shrinkage_rejects_malformed_arrays_before_jit():
    result_case = _make_result_case()
    results = dataclasses.replace(
        result_case.results,
        valid_phantom=jnp.asarray(
            [
                [True, True],
                [True, True],
                [True, True],
            ],
            dtype=mp_policy.bool_dtype,
        ),
    )

    with pytest.raises(
        ValueError,
        match="valid_phantom.*one-dimensional.*per-cluster",
    ):
        results.sample_mc_shrinkage(
            num_samples=4,
            key=jax.random.PRNGKey(31),
        )


def test_results_sample_mc_shrinkage_rejects_strict_contour_equality():
    result_case = _make_result_case()
    results = dataclasses.replace(
        result_case.results,
        log_L_constraints=jnp.asarray(
            [0.0, 0.0, 1.0],
            dtype=mp_policy.measure_dtype,
        ),
    )

    with pytest.raises(ValueError, match="Strict contour.*must be greater"):
        results.sample_mc_shrinkage(
            num_samples=4,
            key=jax.random.PRNGKey(41),
        )


def test_public_sample_mc_shrinkage_rejects_strict_contour_equality():
    with pytest.raises(ValueError, match="Strict contour.*must be greater"):
        sample_mc_shrinkage(
            key=jax.random.PRNGKey(43),
            log_L_constraints=jnp.asarray(
                [0.0, 0.0],
                dtype=mp_policy.measure_dtype,
            ),
            log_L_classic=jnp.asarray(
                [0.0, 1.0],
                dtype=mp_policy.measure_dtype,
            ),
            K_classic=jnp.asarray([1, 1], dtype=mp_policy.count_dtype),
            valid_phantom=jnp.asarray(
                [False, False],
                dtype=mp_policy.bool_dtype,
            ),
            log_L_phantom=jnp.zeros((2, 0), dtype=mp_policy.measure_dtype),
            num_samples=jnp.asarray(2, dtype=mp_policy.count_dtype),
            num_Z_samples=2,
        )


def test_public_sample_mc_shrinkage_rejects_stale_block_likelihoods_before_jit():
    block_state = BlockState(
        log_L_blocks=jnp.asarray(
            [0.0, 2.0],
            dtype=mp_policy.measure_dtype,
        ),
        block_first_idx=jnp.asarray([0, 1], dtype=mp_policy.index_dtype),
        block_size=jnp.asarray([1, 1], dtype=mp_policy.count_dtype),
        incoming_K=jnp.asarray([1, 1], dtype=mp_policy.count_dtype),
        block_out_degree=jnp.asarray([0, 0], dtype=mp_policy.count_dtype),
        valid=jnp.asarray([True, True], dtype=mp_policy.bool_dtype),
        block_sample_indices=jnp.asarray(
            [
                [0],
                [1],
            ],
            dtype=mp_policy.index_dtype,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"block_state\.log_L_blocks.*log_L_classic",
    ):
        sample_mc_shrinkage(
            key=jax.random.PRNGKey(47),
            log_L_constraints=jnp.full(
                (2,),
                -jnp.inf,
                dtype=mp_policy.measure_dtype,
            ),
            log_L_classic=jnp.asarray(
                [0.0, 1.0],
                dtype=mp_policy.measure_dtype,
            ),
            K_classic=jnp.asarray([1, 1], dtype=mp_policy.count_dtype),
            valid_phantom=jnp.asarray(
                [False, False],
                dtype=mp_policy.bool_dtype,
            ),
            log_L_phantom=jnp.zeros((2, 0), dtype=mp_policy.measure_dtype),
            num_samples=jnp.asarray(2, dtype=mp_policy.count_dtype),
            num_Z_samples=2,
            block_state=block_state,
        )


def test_results_sample_mc_shrinkage_rejects_stale_block_likelihoods_before_jit():
    result_case = _make_result_case()
    results = dataclasses.replace(
        result_case.results,
        total_num_samples=jnp.asarray(2, dtype=mp_policy.count_dtype),
        block_data=BlockData(
            log_L=jnp.asarray(
                [0.0, 2.0],
                dtype=mp_policy.measure_dtype,
            ),
            first_idx=jnp.asarray([0, 1], dtype=mp_policy.index_dtype),
            size=jnp.asarray([1, 1], dtype=mp_policy.count_dtype),
            incoming_K=jnp.asarray([1, 1], dtype=mp_policy.count_dtype),
            out_degree=jnp.asarray([0, 0], dtype=mp_policy.count_dtype),
            valid=jnp.asarray([True, True], dtype=mp_policy.bool_dtype),
        ),
        valid_phantom=jnp.asarray(
            [True, True, False],
            dtype=mp_policy.bool_dtype,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"block_state\.log_L_blocks.*log_L_classic",
    ):
        results.sample_mc_shrinkage(
            num_samples=2,
            key=jax.random.PRNGKey(49),
        )


def test_results_cluster_block_fields_in_block_data():
    result_fields = {field.name for field in dataclasses.fields(NestedSamplerResults)}

    assert "block_data" in result_fields
    assert {
        field_name for field_name in result_fields
        if field_name.startswith("block_")
    } == {"block_data"}

    block_data = _make_single_block_gamma_public_result().block_data
    assert block_data is not None
    assert isinstance(block_data, PureDataclassPytree)
    with pytest.raises(dataclasses.FrozenInstanceError):
        block_data.size = jnp.asarray([1], dtype=mp_policy.count_dtype)


def test_nested_sampler_keeps_phantom_likelihood_only_with_legacy_flag():
    state = _run_high_phantom_probe()
    num_samples = int(state.num_samples)

    assert state.samples.phantom_samples.U_samples is None
    assert state.samples.phantom_samples.log_L[:num_samples].shape[-1] == 1
    assert bool(
        jnp.any(state.samples.phantom_samples.valid_mask[:num_samples]),
    )


def test_state_to_result_preserves_per_cluster_phantom_validity():
    state = _run_mixed_validity_probe()
    num_samples = int(state.num_samples)
    per_phantom_validity = np.asarray(
        state.samples.phantom_samples.valid_mask[:num_samples],
        dtype=bool,
    )
    expected_cluster_validity = np.all(per_phantom_validity, axis=-1)

    results = state.to_result().trim()

    np.testing.assert_array_equal(
        np.asarray(results.valid_phantom, dtype=bool),
        expected_cluster_validity,
    )
    assert results.log_L_phantom.shape == per_phantom_validity.shape
    assert int(results.total_phantom_samples) == int(
        np.sum(per_phantom_validity),
    )
    assert not np.any(np.asarray(results.valid_phantom, dtype=bool))


def test_phantom_likelihoods_do_not_drive_supremum_or_map_outputs():
    state = _run_high_phantom_probe()
    num_samples = int(state.num_samples)
    classic_log_l = state.samples.log_likelihoods[:num_samples]
    phantom_log_l = state.samples.phantom_samples.log_L[:num_samples]

    assert float(jnp.max(phantom_log_l)) > float(jnp.max(classic_log_l))
    np.testing.assert_allclose(state.log_L_supremum, jnp.max(classic_log_l))

    results = state.to_result().trim()
    np.testing.assert_allclose(results.log_L_supremum, jnp.max(results.log_L))
    np.testing.assert_allclose(
        results.log_L_map,
        results.log_L[jnp.argmax(results.log_posterior_density)],
    )
    assert results.X_samples.shape[0] == int(results.total_num_samples)
    for attr_name in ("U_phantom_samples", "X_phantom_samples"):
        assert not hasattr(results, attr_name)
