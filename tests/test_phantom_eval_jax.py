from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp

import jaxns.phantom_eval as jax_phantom
import jaxns.phantom_eval_ref as ref_phantom
from jaxns.race_tree import BlockState


class PhantomEvalFixture(NamedTuple):
    log_L_blocks: np.ndarray
    block_valid_mask: np.ndarray
    log_L_constraints: np.ndarray
    valid_phantom: np.ndarray
    sample_mask: np.ndarray
    log_L_phantom: np.ndarray


def _require_callable(module, name: str):
    fn = getattr(module, name, None)
    assert callable(fn), f"{module.__name__}.{name} is required by Ticket 0013."
    return fn


def _fixture() -> PhantomEvalFixture:
    return PhantomEvalFixture(
        log_L_blocks=np.asarray([0.0, 1.0, 2.0, np.inf], dtype=float),
        block_valid_mask=np.asarray([True, True, True, False], dtype=bool),
        log_L_constraints=np.asarray([-np.inf, 0.0, 0.5, -np.inf], dtype=float),
        valid_phantom=np.asarray([True, True, True, False], dtype=bool),
        sample_mask=np.asarray([True, True, True, True], dtype=bool),
        log_L_phantom=np.asarray(
            [
                [0.0, 0.4, 1.0, 3.0],
                [1.0, 1.5, 2.0, 2.5],
                [0.7, 2.0, 2.2, 3.0],
                [9.0, 9.0, 9.0, 9.0],
            ],
            dtype=float,
        ),
    )


def _count_call(module, fixture: PhantomEvalFixture):
    fn = _require_callable(module, "compute_phantom_count_matrices")
    array = jnp.asarray if module is jax_phantom else np.asarray
    return fn(
        log_L_blocks=array(fixture.log_L_blocks),
        block_valid_mask=array(fixture.block_valid_mask),
        log_L_constraints=array(fixture.log_L_constraints),
        valid_phantom=array(fixture.valid_phantom),
        log_L_phantom=array(fixture.log_L_phantom),
        sample_mask=array(fixture.sample_mask),
    )


def _assert_counts_match(jax_counts, ref_counts) -> None:
    for field_name in (
            "A_cg",
            "B_cg",
            "E_cg",
            "R_cg",
            "A_g",
            "B_g",
            "E_g",
            "R_g",
            "kish_participating_cluster_counts",
            "phantom_gate_active",
    ):
        assert hasattr(jax_counts, field_name), f"JAX counts missing {field_name}."
        assert hasattr(ref_counts, field_name), f"Reference counts missing {field_name}."
        np.testing.assert_allclose(
            np.asarray(getattr(jax_counts, field_name)),
            np.asarray(getattr(ref_counts, field_name)),
        )


def _block_state() -> BlockState:
    return BlockState(
        log_L_blocks=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        block_first_idx=jnp.asarray([0, 1], dtype=jnp.int32),
        block_size=jnp.asarray([1, 2], dtype=jnp.int32),
        incoming_K=jnp.asarray([6, 5], dtype=jnp.int32),
        block_out_degree=jnp.asarray([0, 0], dtype=jnp.int32),
        valid=jnp.asarray([True, True]),
        block_sample_indices=jnp.asarray([[0, -1], [1, 2]], dtype=jnp.int32),
    )


def test_jax_and_reference_per_cluster_counts_match_plateau_fixture():
    fixture = _fixture()

    jax_counts = _count_call(jax_phantom, fixture)
    ref_counts = _count_call(ref_phantom, fixture)

    _assert_counts_match(jax_counts, ref_counts)
    np.testing.assert_allclose(
        np.asarray(jax_counts.R_cg),
        np.asarray(jax_counts.A_cg) - np.asarray(jax_counts.B_cg) - np.asarray(jax_counts.E_cg),
    )


def test_jax_count_matrices_are_jittable_and_preserve_R_component():
    fixture = _fixture()
    count_fn = _require_callable(jax_phantom, "compute_phantom_count_matrices")

    @jax.jit
    def _run():
        return count_fn(
            log_L_blocks=jnp.asarray(fixture.log_L_blocks),
            block_valid_mask=jnp.asarray(fixture.block_valid_mask),
            log_L_constraints=jnp.asarray(fixture.log_L_constraints),
            valid_phantom=jnp.asarray(fixture.valid_phantom),
            log_L_phantom=jnp.asarray(fixture.log_L_phantom),
            sample_mask=jnp.asarray(fixture.sample_mask),
        )

    counts = _run()

    np.testing.assert_allclose(
        np.asarray(counts.R_cg),
        np.asarray(counts.A_cg) - np.asarray(counts.B_cg) - np.asarray(counts.E_cg),
    )


def test_jax_and_reference_explicit_gamma_conditioning_match():
    jax_fn = _require_callable(
        jax_phantom,
        "gamma_weighted_phantom_probabilities_from_draws",
    )
    ref_fn = _require_callable(
        ref_phantom,
        "gamma_weighted_phantom_probabilities_from_draws",
    )
    block_state = _block_state()
    A_cg = np.asarray([[4.0, 2.0], [1.0, 3.0]], dtype=float)
    B_cg = np.asarray([[2.0, 1.0], [0.0, 1.0]], dtype=float)
    E_cg = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    race_gt = np.asarray([7.0, 11.0], dtype=float)
    race_eq = np.asarray([2.0, 3.0], dtype=float)
    race_lt = np.asarray([5.0, 13.0], dtype=float)
    cluster_weights = np.asarray([0.25, 2.0], dtype=float)

    jax_draw = jax_fn(
        block_state=block_state,
        A_cg=jnp.asarray(A_cg),
        B_cg=jnp.asarray(B_cg),
        E_cg=jnp.asarray(E_cg),
        race_gamma_gt=jnp.asarray(race_gt),
        race_gamma_eq=jnp.asarray(race_eq),
        race_gamma_lt=jnp.asarray(race_lt),
        cluster_weights=jnp.asarray(cluster_weights),
        C_min=1,
    )
    ref_draw = ref_fn(
        block_state=block_state,
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=cluster_weights,
        C_min=1,
    )

    R_cg = A_cg - B_cg - E_cg
    expected_gt = race_gt + cluster_weights @ B_cg
    expected_eq = race_eq + cluster_weights @ E_cg
    expected_lt = race_lt + cluster_weights @ R_cg
    expected_total = expected_gt + expected_eq + expected_lt
    expected_by_field = {
        "p_gt": expected_gt / expected_total,
        "p_eq": expected_eq / expected_total,
        "p_lt": expected_lt / expected_total,
        "phantom_add_gt": cluster_weights @ B_cg,
        "phantom_add_eq": cluster_weights @ E_cg,
        "phantom_add_lt": cluster_weights @ R_cg,
        "kish_participating_cluster_counts": np.asarray(
            [
                np.square(np.sum(A_cg[:, 0])) / np.sum(np.square(A_cg[:, 0])),
                np.square(np.sum(A_cg[:, 1])) / np.sum(np.square(A_cg[:, 1])),
            ],
            dtype=float,
        ),
        "phantom_gate_active": np.asarray([True, True]),
    }
    for field_name in (
            "p_gt",
            "p_eq",
            "p_lt",
            "phantom_add_gt",
            "phantom_add_eq",
            "phantom_add_lt",
            "kish_participating_cluster_counts",
            "phantom_gate_active",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(jax_draw, field_name)),
            np.asarray(getattr(ref_draw, field_name)),
        )
        np.testing.assert_allclose(
            np.asarray(getattr(jax_draw, field_name)),
            expected_by_field[field_name],
        )


def test_jax_and_reference_sample_mc_shrinkage_emit_new_diagnostics():
    fixture = _fixture()
    log_L_classic = np.asarray([0.0, 1.0, 2.0, 99.0], dtype=float)
    K_classic = np.asarray([6, 5, 4, 1], dtype=np.int32)
    num_samples = np.int32(3)

    jax_out = jax_phantom.sample_mc_shrinkage(
        key=jax.random.PRNGKey(17),
        log_L_constraints=jnp.asarray(fixture.log_L_constraints),
        log_L_classic=jnp.asarray(log_L_classic),
        K_classic=jnp.asarray(K_classic),
        valid_phantom=jnp.asarray(fixture.valid_phantom),
        log_L_phantom=jnp.asarray(fixture.log_L_phantom),
        num_samples=jnp.asarray(num_samples, dtype=jnp.int32),
        num_Z_samples=8,
        C_min=2,
    )
    ref_out = ref_phantom.sample_mc_shrinkage(
        seed=17,
        log_L_constraints=fixture.log_L_constraints,
        log_L_classic=log_L_classic,
        K_classic=K_classic,
        valid_phantom=fixture.valid_phantom,
        log_L_phantom=fixture.log_L_phantom,
        num_samples=num_samples,
        num_Z_samples=8,
        C_min=2,
    )

    for field_name in (
            "kish_participating_cluster_counts",
            "phantom_gate_active",
            "phantom_A",
            "phantom_B",
            "phantom_E",
            "phantom_R",
    ):
        assert hasattr(jax_out, field_name), f"JAX EvidenceSamples missing {field_name}."
        assert hasattr(ref_out, field_name), f"Reference EvidenceSamples missing {field_name}."
        np.testing.assert_allclose(
            np.asarray(getattr(jax_out, field_name)),
            np.asarray(getattr(ref_out, field_name)),
            rtol=1e-6,
            atol=1e-6,
        )

    for old_name in ("rho_values", "rho_fit", "rho_samples", "rho_eta_samples"):
        if hasattr(jax_out, old_name):
            assert getattr(jax_out, old_name) is None

    for out in (jax_out, ref_out):
        np.testing.assert_allclose(
            np.asarray(out.p_gt_mean),
            np.mean(np.asarray(out.p_gt_samples), axis=0),
        )
        np.testing.assert_allclose(
            np.asarray(out.p_eq_mean),
            np.mean(np.asarray(out.p_eq_samples), axis=0),
        )
        np.testing.assert_allclose(
            np.asarray(out.p_lt_mean),
            np.mean(np.asarray(out.p_lt_samples), axis=0),
        )


def test_reused_seed_clusters_share_kish_independence_group():
    fixture = _fixture()
    counts = _count_call(jax_phantom, fixture)
    group_idx = jnp.asarray([0, 0, 2, 3], dtype=jnp.int32)
    output = jax_phantom.sample_mc_shrinkage(
        key=jax.random.PRNGKey(19),
        log_L_constraints=jnp.asarray(fixture.log_L_constraints),
        log_L_classic=jnp.asarray([0.0, 1.0, 2.0, 99.0]),
        K_classic=jnp.asarray([6, 5, 4, 1], dtype=jnp.int32),
        valid_phantom=jnp.asarray(fixture.valid_phantom),
        log_L_phantom=jnp.asarray(fixture.log_L_phantom),
        num_samples=jnp.asarray(3, dtype=jnp.int32),
        num_Z_samples=8,
        C_min=1,
        phantom_group_idx=group_idx,
    )

    A_cg = np.asarray(counts.A_cg[:3, :3])
    grouped_A = np.stack([A_cg[0] + A_cg[1], A_cg[2]])
    expected_kish = np.square(np.sum(grouped_A, axis=0)) / np.sum(
        np.square(grouped_A),
        axis=0,
    )
    np.testing.assert_allclose(
        np.asarray(output.kish_participating_cluster_counts),
        expected_kish,
    )
