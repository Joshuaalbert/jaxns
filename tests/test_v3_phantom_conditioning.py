from itertools import product
from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jaxns.phantom_eval as jax_phantom
import jaxns.phantom_eval_ref as ref_phantom
import jaxns.v3_shrinkage as v3_shrinkage
from jaxns.race_tree import BlockState


class PhantomCountCase(NamedTuple):
    log_L_blocks: np.ndarray
    log_L_constraints: np.ndarray
    log_L_phantom: np.ndarray
    valid_phantom: np.ndarray
    expected_A: np.ndarray
    expected_B: np.ndarray
    expected_E: np.ndarray


class RhoCase(NamedTuple):
    log_L_constraints: np.ndarray
    log_L_classic: np.ndarray
    K_classic: np.ndarray
    valid_phantom: np.ndarray
    log_L_phantom: np.ndarray
    num_samples: np.int32


class DegenerateRhoCase(NamedTuple):
    name: str
    log_L_constraints: np.ndarray
    log_L_classic: np.ndarray
    K_classic: np.ndarray
    valid_phantom: np.ndarray
    log_L_phantom: np.ndarray
    num_samples: np.int32


class MalformedPhantomInputCase(NamedTuple):
    name: str
    overrides: dict
    match: str


class CapturingShrinkageRng:
    def __init__(self):
        self.dirichlet_parameters: list[np.ndarray] = []

    def integers(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        count = 1 if size is None else int(np.prod(size))
        draws = np.arange(count, dtype=np.int64)
        draws = low + (draws % (high - low))
        if size is None:
            return int(draws[0])
        return draws.reshape(size)

    def gamma(self, shape, scale=1.0, size=None):
        alpha = np.asarray(shape, dtype=float)
        self.dirichlet_parameters.append(alpha.copy())
        values = alpha * float(scale)
        if size is not None:
            values = np.broadcast_to(values, size).copy()
        return values


def _manual_boundary_counts(
        log_L_blocks: np.ndarray,
        log_L_constraints: np.ndarray,
        log_L_phantom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_blocks = log_L_blocks.shape[0]
    A = np.zeros((num_blocks,), dtype=float)
    B = np.zeros((num_blocks,), dtype=float)
    E = np.zeros((num_blocks,), dtype=float)
    for constraint, phantom_row in zip(
            log_L_constraints,
            log_L_phantom,
            strict=True,
    ):
        left_constraint = np.searchsorted(
            log_L_blocks,
            constraint,
            side="left",
        )
        start = 0 if np.isneginf(constraint) else left_constraint + 1
        start = min(start, num_blocks)
        for phantom_log_L in phantom_row:
            left = np.searchsorted(log_L_blocks, phantom_log_L, side="left")
            a_hi = min(left + 1, num_blocks)
            b_hi = min(left, num_blocks)
            if a_hi > start:
                A[start:a_hi] += 1.0
            if b_hi > start:
                B[start:b_hi] += 1.0
            if (
                    left < num_blocks
                    and left >= start
                    and log_L_blocks[left] == phantom_log_L
            ):
                E[left] += 1.0
    return A, B, E


def _expected_log_dz_from_dirichlet_mean(
        log_L_blocks: np.ndarray,
        incoming_K: np.ndarray,
        block_size: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
        rho: np.ndarray,
) -> np.ndarray:
    epsilon = np.where(block_size == 1, 1e-6, 0.5)
    alpha_gt = incoming_K - block_size + 1.0 + rho * B
    alpha_eq = block_size + epsilon + rho * E
    alpha_lt = 1.0 - epsilon + rho * (A - B - E)
    shrinkage = alpha_gt / (alpha_gt + alpha_eq + alpha_lt)
    log_X = np.cumsum(np.log(shrinkage))
    log_X_prev = np.concatenate(
        [np.asarray([0.0], dtype=float), log_X[:-1]],
        axis=0,
    )
    log_dX = log_X_prev + np.log1p(-np.exp(log_X - log_X_prev))
    return log_dX + log_L_blocks


def _count_case() -> PhantomCountCase:
    return PhantomCountCase(
        log_L_blocks=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float),
        log_L_constraints=np.asarray(
            [-np.inf, 0.0, 1.0, 2.0, -np.inf],
            dtype=float,
        ),
        log_L_phantom=np.asarray(
            [
                [-0.5, 0.0, 0.2, 1.0],
                [0.5, 1.0, 1.5, 2.0],
                [1.0, 2.5, 2.0, 0.5],
                [2.1, 3.0, 3.5, 4.0],
                [9.0, 9.0, 9.0, 9.0],
            ],
            dtype=float,
        ),
        valid_phantom=np.asarray([True, True, True, True, False]),
        expected_A=np.asarray([4.0, 6.0, 4.0, 5.0]),
        expected_B=np.asarray([2.0, 2.0, 1.0, 2.0]),
        expected_E=np.asarray([1.0, 2.0, 2.0, 1.0]),
    )


def _rho_case() -> RhoCase:
    num_phantom = 6
    return RhoCase(
        log_L_constraints=np.full((5,), -np.inf, dtype=float),
        log_L_classic=np.asarray([0.0, 1.0, 2.0, 3.0, 3.0], dtype=float),
        K_classic=np.asarray([30, 30, 30, 30, 30], dtype=np.int32),
        valid_phantom=np.ones((5,), dtype=bool),
        log_L_phantom=np.asarray(
            [
                [4.0] * num_phantom,
                [1.0] * num_phantom,
                [0.5] * num_phantom,
                [2.0] * num_phantom,
                [3.0] * num_phantom,
            ],
            dtype=float,
        ),
        num_samples=np.int32(5),
    )


def _iid_rho_case() -> RhoCase:
    rng = np.random.default_rng(202503)
    num_clusters = 64
    num_phantom = 16
    log_L_phantom = rng.choice(
        np.asarray([-1.0, 1.0], dtype=float),
        size=(num_clusters, num_phantom),
        p=np.asarray([0.5, 0.5], dtype=float),
    )
    return RhoCase(
        log_L_constraints=np.full((num_clusters,), -np.inf, dtype=float),
        log_L_classic=np.zeros((num_clusters,), dtype=float),
        K_classic=np.full((num_clusters,), 96, dtype=np.int32),
        valid_phantom=np.ones((num_clusters,), dtype=bool),
        log_L_phantom=log_L_phantom,
        num_samples=np.int32(num_clusters),
    )


def _block_state() -> BlockState:
    block_sample_indices = np.asarray(
        [
            [0, -1],
            [1, 2],
            [3, -1],
        ],
        dtype=np.int32,
    )
    return BlockState(
        log_L_blocks=jnp.asarray([0.0, 1.0, 2.0]),
        block_first_idx=jnp.asarray([0, 1, 3], dtype=jnp.int32),
        block_size=jnp.asarray([1, 2, 1], dtype=jnp.int32),
        incoming_K=jnp.asarray([5, 4, 3], dtype=jnp.int32),
        block_out_degree=jnp.zeros((3,), dtype=jnp.int32),
        valid=jnp.ones((3,), dtype=jnp.bool_),
        block_sample_indices=jnp.asarray(block_sample_indices),
    )


def _conditioned_concentrations(block_state, A, B, E, rho):
    try:
        fn = v3_shrinkage.phantom_conditioned_dirichlet_concentrations
    except AttributeError:
        pytest.fail(
            "jaxns.v3_shrinkage.phantom_conditioned_dirichlet_concentrations "
            "is required for Ticket 0003 phantom Dirichlet updates."
        )
    return fn(
        block_state,
        jnp.asarray(A),
        jnp.asarray(B),
        jnp.asarray(E),
        jnp.asarray(rho),
    )


def _fit_low_order_rho_g_curve(
        module,
        array_fn,
        raw_rho_g: np.ndarray,
        race_time: np.ndarray,
        valid_mask: np.ndarray,
) -> np.ndarray:
    try:
        fn = module.fit_low_order_rho_g_curve
    except AttributeError:
        pytest.fail(
            f"{module.__name__}.fit_low_order_rho_g_curve is required so "
            "raw cluster-bootstrap rho_g estimates are smoothed with the "
            "paper's low-order curve against normalized race time."
        )
    return np.asarray(
        fn(
            raw_rho_g=array_fn(raw_rho_g),
            race_time=array_fn(race_time),
            valid_mask=array_fn(valid_mask),
            polynomial_order=2,
            fallback_rho=1.0,
        )
    )


def _estimate_raw_rho_g_from_bootstrap_covariance(
        module,
        array_fn,
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
        bootstrap_covariance: np.ndarray,
) -> np.ndarray:
    try:
        fn = module.estimate_raw_rho_g_from_bootstrap_covariance
    except AttributeError:
        pytest.fail(
            f"{module.__name__}.estimate_raw_rho_g_from_bootstrap_covariance "
            "is required for the paper rho_g estimator "
            "rank(Sigma_g) / trace(Sigma_g^+ Sigma_g^boot)."
        )
    return np.asarray(
        fn(
            A=array_fn(A),
            B=array_fn(B),
            E=array_fn(E),
            bootstrap_covariance=array_fn(bootstrap_covariance),
            fallback_rho=1.0,
        )
    )


def _exact_bootstrap_q_covariance_by_enumeration(
        A_by_cluster: np.ndarray,
        B_by_cluster: np.ndarray,
        E_by_cluster: np.ndarray,
) -> np.ndarray:
    num_clusters, num_blocks = A_by_cluster.shape
    q_by_resample = [[] for _ in range(num_blocks)]
    for bootstrap_indices in product(range(num_clusters), repeat=num_clusters):
        A = np.sum(A_by_cluster[list(bootstrap_indices)], axis=0)
        B = np.sum(B_by_cluster[list(bootstrap_indices)], axis=0)
        E = np.sum(E_by_cluster[list(bootstrap_indices)], axis=0)
        active = A > 0.0
        for block_idx in range(num_blocks):
            if active[block_idx]:
                q_by_resample[block_idx].append(
                    [B[block_idx] / A[block_idx], E[block_idx] / A[block_idx]]
                )

    covariance = np.zeros((num_blocks, 2, 2), dtype=float)
    for block_idx, q_values in enumerate(q_by_resample):
        q = np.asarray(q_values, dtype=float)
        if q.shape[0] > 0:
            centered = q - np.mean(q, axis=0, keepdims=True)
            covariance[block_idx] = centered.T @ centered / float(q.shape[0])
    return covariance


def _centered_count_bootstrap_covariance(
        A_by_cluster: np.ndarray,
        B_by_cluster: np.ndarray,
        E_by_cluster: np.ndarray,
) -> np.ndarray:
    A_total = np.sum(A_by_cluster, axis=0)
    B_total = np.sum(B_by_cluster, axis=0)
    E_total = np.sum(E_by_cluster, axis=0)
    A_safe = np.where(A_total > 0.0, A_total, 1.0)
    q_gt = B_total / A_safe
    q_eq = E_total / A_safe
    centered_gt = B_by_cluster - q_gt[None, :] * A_by_cluster
    centered_eq = E_by_cluster - q_eq[None, :] * A_by_cluster
    scale = A_safe * A_safe
    return np.stack(
        [
            np.stack(
                [
                    np.sum(centered_gt * centered_gt, axis=0) / scale,
                    np.sum(centered_gt * centered_eq, axis=0) / scale,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.sum(centered_gt * centered_eq, axis=0) / scale,
                    np.sum(centered_eq * centered_eq, axis=0) / scale,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def _run_reference(
        case: RhoCase | DegenerateRhoCase,
        num_Z_samples: int = 16,
        rho_grid: np.ndarray | None = None,
):
    return ref_phantom.sample_mc_shrinkage(
        seed=41,
        log_L_constraints=case.log_L_constraints,
        log_L_classic=case.log_L_classic,
        K_classic=case.K_classic,
        valid_phantom=case.valid_phantom,
        log_L_phantom=case.log_L_phantom,
        num_samples=case.num_samples,
        num_Z_samples=num_Z_samples,
        rho_grid=rho_grid,
    )


def _run_jax(
        case: RhoCase | DegenerateRhoCase,
        num_Z_samples: int = 16,
        rho_grid: np.ndarray | None = None,
):
    out = jax_phantom.sample_mc_shrinkage(
        key=jax.random.PRNGKey(41),
        log_L_constraints=jnp.asarray(case.log_L_constraints),
        log_L_classic=jnp.asarray(case.log_L_classic),
        K_classic=jnp.asarray(case.K_classic),
        valid_phantom=jnp.asarray(case.valid_phantom),
        log_L_phantom=jnp.asarray(case.log_L_phantom),
        num_samples=jnp.asarray(case.num_samples, dtype=jnp.int32),
        num_Z_samples=num_Z_samples,
        rho_grid=None if rho_grid is None else jnp.asarray(rho_grid),
        rho_prior="none",
    )
    return jax.tree_util.tree_map(lambda value: np.asarray(value), out)


def _well_formed_sample_kwargs() -> dict:
    return {
        "log_L_constraints": np.asarray([-np.inf, 0.0, 1.0], dtype=float),
        "log_L_classic": np.asarray([0.0, 1.0, 2.0], dtype=float),
        "K_classic": np.asarray([6, 5, 4], dtype=np.int32),
        "valid_phantom": np.asarray([True, True, True], dtype=bool),
        "log_L_phantom": np.asarray(
            [
                [0.5, 1.5],
                [1.5, 2.5],
                [2.5, 3.5],
            ],
            dtype=float,
        ),
        "num_samples": np.int32(3),
        "num_Z_samples": 4,
    }


def _run_reference_kwargs(kwargs: dict):
    return ref_phantom.sample_mc_shrinkage(seed=17, **kwargs)


def _run_jax_kwargs(kwargs: dict):
    return jax_phantom.sample_mc_shrinkage(
        key=jax.random.PRNGKey(17),
        log_L_constraints=jnp.asarray(kwargs["log_L_constraints"]),
        log_L_classic=jnp.asarray(kwargs["log_L_classic"]),
        K_classic=jnp.asarray(kwargs["K_classic"]),
        valid_phantom=jnp.asarray(kwargs["valid_phantom"]),
        log_L_phantom=jnp.asarray(kwargs["log_L_phantom"]),
        num_samples=jnp.asarray(kwargs["num_samples"], dtype=jnp.int32),
        num_Z_samples=kwargs["num_Z_samples"],
    )


def _malformed_input_cases() -> list[MalformedPhantomInputCase]:
    return [
        MalformedPhantomInputCase(
            name="cluster mask length mismatch",
            overrides={"valid_phantom": np.asarray([True, True], dtype=bool)},
            match="valid_phantom|shape|cluster",
        ),
        MalformedPhantomInputCase(
            name="per-phantom mask passed as cluster mask",
            overrides={
                "valid_phantom": np.asarray(
                    [
                        [True, True],
                        [True, False],
                        [True, True],
                    ],
                    dtype=bool,
                ),
            },
            match="valid_phantom|per-cluster|one-dimensional|shape",
        ),
        MalformedPhantomInputCase(
            name="stale phantom association after num_samples",
            overrides={
                "num_samples": np.int32(2),
                "valid_phantom": np.asarray([True, True, True], dtype=bool),
            },
            match="valid_phantom|num_samples|stale|association",
        ),
        MalformedPhantomInputCase(
            name="constraint length mismatch",
            overrides={
                "log_L_constraints": np.asarray([-np.inf, 0.0], dtype=float),
            },
            match="log_L_constraints|shape",
        ),
        MalformedPhantomInputCase(
            name="parent constraint above classic likelihood",
            overrides={
                "log_L_constraints": np.asarray(
                    [-np.inf, 5.0, 1.0],
                    dtype=float,
                ),
            },
            match="log_L_constraints|constraint|parent|classic",
        ),
        MalformedPhantomInputCase(
            name="parent constraint equal to classic likelihood",
            overrides={
                "log_L_constraints": np.asarray(
                    [-np.inf, 1.0, 1.0],
                    dtype=float,
                ),
            },
            match="Strict contour|greater|log_L_constraint",
        ),
        MalformedPhantomInputCase(
            name="phantom likelihood rank mismatch",
            overrides={
                "log_L_phantom": np.asarray([0.5, 1.5, 2.5], dtype=float),
            },
            match="log_L_phantom|two-dimensional|rank|shape",
        ),
        MalformedPhantomInputCase(
            name="phantom likelihood cluster count mismatch",
            overrides={
                "log_L_phantom": np.asarray(
                    [
                        [0.5, 1.5],
                        [1.5, 2.5],
                    ],
                    dtype=float,
                ),
            },
            match="log_L_phantom|shape|cluster",
        ),
    ]


def test_reference_counts_include_equality_and_parent_contour_eligibility():
    case = _count_case()
    valid_idx = np.where(case.valid_phantom)[0]

    A, B, E = ref_phantom._boundary_counts_from_clusters(
        log_L_blocks=case.log_L_blocks,
        log_L_constraints=case.log_L_constraints[valid_idx],
        log_L_phantom=case.log_L_phantom[valid_idx],
    )

    np.testing.assert_allclose(A, case.expected_A)
    np.testing.assert_allclose(B, case.expected_B)
    np.testing.assert_allclose(E, case.expected_E)

    pooled_without_parent_gate_A = np.asarray([4.0, 10.0, 8.0, 5.0])
    assert not np.array_equal(case.expected_A, pooled_without_parent_gate_A)


def test_jax_and_reference_counts_match_parent_eligibility_fixture():
    case = _count_case()
    valid_idx = np.where(case.valid_phantom)[0]
    ref_A, ref_B, ref_E = ref_phantom._boundary_counts_from_clusters(
        log_L_blocks=case.log_L_blocks,
        log_L_constraints=case.log_L_constraints[valid_idx],
        log_L_phantom=case.log_L_phantom[valid_idx],
    )

    start_idx = np.asarray([0, 1, 2, 3, 0], dtype=np.int32)
    cluster_multiplicity = case.valid_phantom.astype(np.float32)
    event_cluster_idx = np.repeat(
        np.arange(case.log_L_constraints.shape[0], dtype=np.int32),
        case.log_L_phantom.shape[1],
    )
    event_log_L = case.log_L_phantom.reshape((-1,))
    left = np.searchsorted(case.log_L_blocks, event_log_L, side="left")
    event_a_hi = np.minimum(left + 1, case.log_L_blocks.shape[0])
    event_b_hi = np.minimum(left, case.log_L_blocks.shape[0])
    event_start = start_idx[event_cluster_idx]
    event_A_active = event_a_hi > event_start
    event_B_active = event_b_hi > event_start
    eq_ok = (left < case.log_L_blocks.shape[0]) & (
        case.log_L_blocks[np.minimum(left, case.log_L_blocks.shape[0] - 1)]
        == event_log_L
    )
    event_eq_idx = np.where(eq_ok, left, 0).astype(np.int32)
    event_eq_active = (
        eq_ok
        & (event_eq_idx >= event_start)
        & case.valid_phantom[event_cluster_idx]
    )
    count_A_start = np.bincount(
        event_cluster_idx,
        weights=event_A_active.astype(np.float32),
        minlength=case.log_L_constraints.shape[0],
    )
    count_B_start = np.bincount(
        event_cluster_idx,
        weights=event_B_active.astype(np.float32),
        minlength=case.log_L_constraints.shape[0],
    )

    got_A, got_B, got_E = jax_phantom._boundary_counts_from_multiplicity(
        cluster_multiplicity=jnp.asarray(cluster_multiplicity),
        start_idx=jnp.asarray(start_idx),
        count_A_start_per_cluster=jnp.asarray(count_A_start),
        count_B_start_per_cluster=jnp.asarray(count_B_start),
        event_cluster_idx=jnp.asarray(event_cluster_idx),
        event_a_hi=jnp.asarray(event_a_hi),
        event_b_hi=jnp.asarray(event_b_hi),
        event_A_active=jnp.asarray(event_A_active),
        event_B_active=jnp.asarray(event_B_active),
        event_eq_idx=jnp.asarray(event_eq_idx),
        event_eq_active=jnp.asarray(event_eq_active),
        num_blocks=case.log_L_blocks.shape[0],
    )

    np.testing.assert_allclose(np.asarray(got_A), ref_A)
    np.testing.assert_allclose(np.asarray(got_B), ref_B)
    np.testing.assert_allclose(np.asarray(got_E), ref_E)


def test_phantom_dirichlet_updates_all_components_and_classic_fallback():
    block_state = _block_state()
    A = np.asarray([6.0, 4.0, 0.0], dtype=np.float32)
    B = np.asarray([2.0, 1.0, 0.0], dtype=np.float32)
    E = np.asarray([2.0, 1.0, 0.0], dtype=np.float32)
    rho = np.asarray([0.25, 0.5, 1.0], dtype=np.float32)

    concentrations = _conditioned_concentrations(block_state, A, B, E, rho)

    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_gt),
        np.asarray([5.5, 3.5, 3.0]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_eq),
        np.asarray([1.500001, 3.0, 1.000001]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_lt),
        np.asarray([1.499999, 1.5, 0.999999]),
        rtol=1e-6,
    )


def test_phantom_dirichlet_conditioning_uses_fitted_rho_curve_not_raw_estimates():
    block_state = _block_state()
    A = np.asarray([8.0, 8.0, 0.0], dtype=np.float32)
    B = np.asarray([4.0, 1.0, 0.0], dtype=np.float32)
    E = np.asarray([2.0, 3.0, 0.0], dtype=np.float32)
    raw_rho = np.asarray([1.0, 0.10, 1.0], dtype=np.float32)
    fitted_rho = np.asarray([0.25, 0.75, 1.0], dtype=np.float32)

    concentrations = _conditioned_concentrations(
        block_state,
        A,
        B,
        E,
        fitted_rho,
    )
    raw_concentrations = _conditioned_concentrations(
        block_state,
        A,
        B,
        E,
        raw_rho,
    )

    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_gt),
        np.asarray([6.0, 3.75, 3.0]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_eq),
        np.asarray([1.500001, 4.75, 1.000001]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(concentrations.alpha_lt),
        np.asarray([1.499999, 3.5, 0.999999]),
        rtol=1e-6,
    )
    assert not np.allclose(
        np.asarray(concentrations.alpha_gt),
        np.asarray(raw_concentrations.alpha_gt),
    )
    assert not np.allclose(
        np.asarray(concentrations.alpha_eq),
        np.asarray(raw_concentrations.alpha_eq),
    )


def test_sample_mc_shrinkage_uses_fitted_rho_for_actual_phantom_shrinkage(
        monkeypatch,
):
    kwargs = _well_formed_sample_kwargs()
    fitted_rho = 0.25
    sampled_rho = 0.90
    capturing_rng = CapturingShrinkageRng()

    monkeypatch.setattr(
        ref_phantom.np.random,
        "default_rng",
        lambda seed: capturing_rng,
    )
    monkeypatch.setattr(
        ref_phantom,
        "fit_low_order_rho_g_curve",
        lambda raw_rho_g, race_time, valid_mask, polynomial_order, fallback_rho: (
            np.where(np.asarray(valid_mask), fitted_rho, np.nan)
        ),
    )
    monkeypatch.setattr(
        ref_phantom,
        "_sample_rho_from_likelihood",
        lambda *args, **kwargs: sampled_rho,
    )

    ref_phantom.sample_mc_shrinkage(seed=17, **kwargs)

    assert capturing_rng.dirichlet_parameters, (
        "sample_mc_shrinkage must feed phantom-conditioned Dirichlet "
        "parameters into the actual shrinkage sampler."
    )
    alpha = capturing_rng.dirichlet_parameters[0]
    log_L_blocks = np.asarray([0.0, 1.0, 2.0], dtype=float)
    valid_idx = np.where(kwargs["valid_phantom"])[0]
    A, B, E = ref_phantom._boundary_counts_from_clusters(
        log_L_blocks=log_L_blocks,
        log_L_constraints=kwargs["log_L_constraints"][valid_idx],
        log_L_phantom=kwargs["log_L_phantom"][valid_idx],
        cluster_indices=np.arange(valid_idx.size, dtype=np.int64),
    )
    block_size = np.ones_like(kwargs["K_classic"], dtype=float)
    epsilon = np.where(block_size == 1, 1e-6, 0.5)
    expected_alpha = np.stack(
        [
            kwargs["K_classic"].astype(float) - block_size + 1.0 + fitted_rho * B,
            block_size + epsilon + fitted_rho * E,
            1.0 - epsilon + fitted_rho * (A - B - E),
        ],
        axis=-1,
    )
    sampled_alpha = np.stack(
        [
            kwargs["K_classic"].astype(float) - block_size + 1.0 + sampled_rho * B,
            block_size + epsilon + sampled_rho * E,
            1.0 - epsilon + sampled_rho * (A - B - E),
        ],
        axis=-1,
    )

    np.testing.assert_allclose(alpha, expected_alpha)
    assert not np.allclose(alpha, sampled_alpha)


def test_public_jax_sample_mc_shrinkage_uses_fitted_per_block_rho_curve(
        monkeypatch,
):
    log_L_constraints = np.asarray([-np.inf, -np.inf, -np.inf], dtype=float)
    log_L_classic = np.asarray([0.0, 1.0, 2.0], dtype=float)
    K_classic = np.asarray([9, 8, 7], dtype=np.int32)
    valid_phantom = np.asarray([True, True, True], dtype=bool)
    log_L_phantom = np.asarray(
        [
            [0.0, 0.5, 2.5],
            [1.0, 1.5, 2.5],
            [2.0, 2.5, 2.5],
        ],
        dtype=float,
    )
    fitted_rho = np.asarray([0.20, 0.55, 0.90], dtype=np.float32)
    sampled_scalar_rho = 0.95
    fit_calls = []

    def fake_low_order_fit(
            *,
            raw_rho_g,
            race_time,
            valid_mask,
            polynomial_order,
            fallback_rho,
    ):
        del race_time
        fit_calls.append((polynomial_order, fallback_rho))
        sentinel = jnp.asarray(fitted_rho, dtype=raw_rho_g.dtype)
        return jnp.where(valid_mask, sentinel, jnp.nan)

    def fake_scalar_rho(*args, **kwargs):
        del args
        rho_grid = kwargs["rho_grid"]
        return jnp.asarray(sampled_scalar_rho, dtype=rho_grid.dtype)

    def deterministic_randint(key, shape, minval, maxval, dtype=jnp.int32):
        del key, minval
        draws = jnp.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        return jnp.mod(draws, maxval).astype(dtype)

    def gamma_mean(key, a, shape=None, dtype=None):
        del key
        value = a
        if shape is not None:
            value = jnp.broadcast_to(value, shape)
        if dtype is not None:
            value = value.astype(dtype)
        return value

    monkeypatch.setattr(
        jax_phantom,
        "fit_low_order_rho_g_curve",
        fake_low_order_fit,
        raising=False,
    )
    monkeypatch.setattr(jax_phantom, "_fit_rho_mle", fake_scalar_rho)
    monkeypatch.setattr(jax_phantom, "_sample_rho_from_likelihood", fake_scalar_rho)
    monkeypatch.setattr(jax_phantom.random, "randint", deterministic_randint)
    monkeypatch.setattr(jax_phantom.random, "gamma", gamma_mean)
    if hasattr(jax_phantom._sample_mc_shrinkage, "clear_cache"):
        jax_phantom._sample_mc_shrinkage.clear_cache()

    try:
        out = jax_phantom.sample_mc_shrinkage(
            key=jax.random.PRNGKey(101),
            log_L_constraints=jnp.asarray(log_L_constraints),
            log_L_classic=jnp.asarray(log_L_classic),
            K_classic=jnp.asarray(K_classic),
            valid_phantom=jnp.asarray(valid_phantom),
            log_L_phantom=jnp.asarray(log_L_phantom),
            num_samples=jnp.asarray(3, dtype=jnp.int32),
            num_Z_samples=1,
            rho_grid=jnp.asarray([0.01, 0.1, 1.0], dtype=jnp.float32),
            rho_prior="none",
        )
    finally:
        if hasattr(jax_phantom._sample_mc_shrinkage, "clear_cache"):
            jax_phantom._sample_mc_shrinkage.clear_cache()

    out = jax.tree_util.tree_map(lambda value: np.asarray(value), out)
    valid_blocks = np.isfinite(out.log_L_blocks)

    assert fit_calls, (
        "Public JAX sample_mc_shrinkage must call the fitted per-block "
        "rho_g curve and use it in the shrinkage sampler."
    )
    np.testing.assert_allclose(out.rho_fit[valid_blocks], fitted_rho)
    assert np.unique(out.rho_fit[valid_blocks]).size > 1

    A, B, E = _manual_boundary_counts(
        out.log_L_blocks[valid_blocks],
        log_L_constraints,
        log_L_phantom,
    )
    expected = _expected_log_dz_from_dirichlet_mean(
        out.log_L_blocks[valid_blocks],
        out.incoming_K[valid_blocks].astype(float),
        out.block_size[valid_blocks].astype(float),
        A,
        B,
        E,
        fitted_rho,
    )
    sampled_scalar_expected = _expected_log_dz_from_dirichlet_mean(
        out.log_L_blocks[valid_blocks],
        out.incoming_K[valid_blocks].astype(float),
        out.block_size[valid_blocks].astype(float),
        A,
        B,
        E,
        np.full_like(fitted_rho, sampled_scalar_rho),
    )

    np.testing.assert_allclose(
        out.log_dZ_mean[valid_blocks],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(expected, sampled_scalar_expected)


def test_phantom_dirichlet_rejects_invalid_count_relations_and_rho():
    block_state = _block_state()
    A = jnp.asarray([2.0, 2.0, 2.0])
    B = jnp.asarray([1.0, 2.0, 1.0])
    E = jnp.asarray([1.0, 1.0, 0.0])
    rho = jnp.asarray([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="B.*E.*A|count|Dirichlet"):
        _conditioned_concentrations(block_state, A, B, E, rho)

    with pytest.raises(ValueError, match="rho|positive|finite|<= 1"):
        _conditioned_concentrations(
            block_state,
            jnp.asarray([2.0, 2.0, 2.0]),
            jnp.asarray([1.0, 1.0, 1.0]),
            jnp.asarray([0.0, 0.0, 0.0]),
            jnp.asarray([1.0, 0.0, np.nan]),
        )

    with pytest.raises(ValueError, match="shape|align"):
        _conditioned_concentrations(
            block_state,
            jnp.asarray([2.0, 2.0]),
            jnp.asarray([1.0, 1.0]),
            jnp.asarray([0.0, 0.0]),
            jnp.asarray([0.5, 0.5]),
        )


@pytest.mark.parametrize(
    ("module", "array_fn"),
    [
        (ref_phantom, np.asarray),
        (jax_phantom, jnp.asarray),
    ],
    ids=["reference", "jax"],
)
def test_raw_rho_g_formula_uses_q_pseudoinverse_rank_trace(
        module,
        array_fn,
):
    A = np.asarray([16.0, 20.0], dtype=float)
    B = np.asarray([8.0, 10.0], dtype=float)
    E = np.asarray([4.0, 10.0], dtype=float)

    # Block 0 has q_g=(1/2, 1/4) and full-rank Sigma_g.
    sigma_full_rank = np.asarray(
        [
            [1.0 / 64.0, -1.0 / 128.0],
            [-1.0 / 128.0, 3.0 / 256.0],
        ],
        dtype=float,
    )
    # Block 1 has q_g=(1/2, 1/2), p_< = 0, and rank-one Sigma_g;
    # this forces the Moore-Penrose path rather than a matrix inverse.
    sigma_rank_one = np.asarray(
        [
            [1.0 / 80.0, -1.0 / 80.0],
            [-1.0 / 80.0, 1.0 / 80.0],
        ],
        dtype=float,
    )
    bootstrap_covariance = np.stack(
        [
            2.0 * sigma_full_rank,
            4.0 * sigma_rank_one,
        ],
        axis=0,
    )

    rho = _estimate_raw_rho_g_from_bootstrap_covariance(
        module,
        array_fn,
        A,
        B,
        E,
        bootstrap_covariance,
    )

    np.testing.assert_allclose(rho, np.asarray([0.5, 0.25]), rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
    ("module", "array_fn"),
    [
        (ref_phantom, np.asarray),
        (jax_phantom, jnp.asarray),
    ],
    ids=["reference", "jax"],
)
def test_bootstrap_covariance_recomputes_q_with_bootstrap_denominators(
        module,
        array_fn,
):
    A_by_cluster = np.asarray(
        [
            [1.0],
            [3.0],
            [2.0],
        ],
        dtype=float,
    )
    B_by_cluster = np.asarray(
        [
            [1.0],
            [0.0],
            [1.0],
        ],
        dtype=float,
    )
    E_by_cluster = np.asarray(
        [
            [0.0],
            [1.0],
            [1.0],
        ],
        dtype=float,
    )

    expected = _exact_bootstrap_q_covariance_by_enumeration(
        A_by_cluster,
        B_by_cluster,
        E_by_cluster,
    )
    centered_count = _centered_count_bootstrap_covariance(
        A_by_cluster,
        B_by_cluster,
        E_by_cluster,
    )
    got = np.asarray(
        module._bootstrap_covariance_from_cluster_counts(
            array_fn(A_by_cluster),
            array_fn(B_by_cluster),
            array_fn(E_by_cluster),
        )
    )

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-8)
    assert not np.allclose(centered_count, expected, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize(
    ("module", "array_fn"),
    [
        (ref_phantom, np.asarray),
        (jax_phantom, jnp.asarray),
    ],
    ids=["reference", "jax"],
)
def test_low_order_rho_fit_uses_normalized_race_time(
        module,
        array_fn,
):
    race_time = np.asarray([2.0, 5.0, 13.0, 20.0, 99.0], dtype=float)
    valid_mask = np.asarray([True, True, True, True, False], dtype=bool)
    normalized_time = race_time[:4] / race_time[3]
    raw_rho = 0.2 + 0.3 * normalized_time + 0.4 * normalized_time ** 2
    raw_rho = np.concatenate([raw_rho, np.asarray([0.05], dtype=float)])

    rho_fit = _fit_low_order_rho_g_curve(
        module,
        array_fn,
        raw_rho,
        race_time,
        valid_mask,
    )

    np.testing.assert_allclose(
        rho_fit[:4],
        raw_rho[:4],
        rtol=0.0,
        atol=1e-8,
    )
    assert np.isnan(rho_fit[4])


@pytest.mark.parametrize(
    "runner",
    [_run_reference_kwargs, _run_jax_kwargs],
    ids=["reference", "jax"],
)
@pytest.mark.parametrize(
    "case",
    _malformed_input_cases(),
    ids=lambda case: case.name,
)
def test_sample_mc_shrinkage_rejects_malformed_phantom_metadata(
        runner,
        case,
):
    kwargs = _well_formed_sample_kwargs()
    kwargs.update(case.overrides)

    with pytest.raises(ValueError, match=case.match):
        runner(kwargs)


def test_sample_mc_shrinkage_exposes_raw_cluster_bootstrap_rho_g_and_fit():
    case = _rho_case()
    ref_out = _run_reference(case)
    jax_out = _run_jax(case)
    valid_blocks = np.isfinite(ref_out.log_L_blocks)

    np.testing.assert_allclose(
        jax_out.rho_values[valid_blocks],
        ref_out.rho_values[valid_blocks],
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.any(ref_out.rho_values[valid_blocks] < 0.25)

    for rho_fit in (
        ref_out.rho_fit[valid_blocks],
        jax_out.rho_fit[valid_blocks],
    ):
        assert rho_fit.shape == ref_out.rho_values[valid_blocks].shape
        assert np.all(np.isfinite(rho_fit))
        assert np.all((rho_fit > 0.0) & (rho_fit <= 1.0))


def test_iid_synthetic_cluster_raw_rho_g_is_near_one():
    case = _iid_rho_case()
    rho_grid = np.asarray([0.01, 0.1, 0.5, 1.0], dtype=float)
    assert int(np.sum(case.valid_phantom)) == 64

    ref_out = _run_reference(case, num_Z_samples=4, rho_grid=rho_grid)
    jax_out = _run_jax(case, num_Z_samples=4, rho_grid=rho_grid)
    valid_blocks = np.isfinite(ref_out.log_L_blocks)

    for rho_values in (
        ref_out.rho_values[valid_blocks],
        ref_out.rho_fit[valid_blocks],
        jax_out.rho_values[valid_blocks],
        jax_out.rho_fit[valid_blocks],
    ):
        np.testing.assert_allclose(rho_values, 1.0, rtol=0.0, atol=0.25)


def _degenerate_cases() -> list[DegenerateRhoCase]:
    return [
        DegenerateRhoCase(
            name="empty valid phantom clusters",
            log_L_constraints=np.asarray([-np.inf, 0.0, 1.0], dtype=float),
            log_L_classic=np.asarray([0.0, 1.0, 2.0], dtype=float),
            K_classic=np.asarray([10, 8, 6], dtype=np.int32),
            valid_phantom=np.asarray([False, False, False], dtype=bool),
            log_L_phantom=np.full((3, 3), 0.5, dtype=float),
            num_samples=np.int32(3),
        ),
        DegenerateRhoCase(
            name="single repeated cluster",
            log_L_constraints=np.asarray([-np.inf, 0.0, 1.0], dtype=float),
            log_L_classic=np.asarray([0.0, 1.0, 2.0], dtype=float),
            K_classic=np.asarray([10, 8, 6], dtype=np.int32),
            valid_phantom=np.asarray([True, False, False], dtype=bool),
            log_L_phantom=np.asarray(
                [
                    [1.0, 1.0, 1.0],
                    [9.0, 9.0, 9.0],
                    [9.0, 9.0, 9.0],
                ],
                dtype=float,
            ),
            num_samples=np.int32(3),
        ),
        DegenerateRhoCase(
            name="rank zero iid covariance",
            log_L_constraints=np.asarray(
                [-np.inf, -np.inf, -np.inf],
                dtype=float,
            ),
            log_L_classic=np.asarray([0.0, 1.0, 2.0], dtype=float),
            K_classic=np.asarray([10, 8, 6], dtype=np.int32),
            valid_phantom=np.asarray([True, True, True], dtype=bool),
            log_L_phantom=np.full((3, 4), 4.0, dtype=float),
            num_samples=np.int32(3),
        ),
        DegenerateRhoCase(
            name="valid parent constraints with no active phantom counts",
            log_L_constraints=np.asarray([-1.0, 0.0, 1.0], dtype=float),
            log_L_classic=np.asarray([0.0, 1.0, 2.0], dtype=float),
            K_classic=np.asarray([10, 8, 6], dtype=np.int32),
            valid_phantom=np.asarray([True, True, True], dtype=bool),
            log_L_phantom=np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                ],
                dtype=float,
            ),
            num_samples=np.int32(3),
        ),
    ]


@pytest.mark.parametrize(
    "case",
    _degenerate_cases(),
    ids=lambda case: case.name,
)
def test_degenerate_rho_paths_are_finite_and_jax_reference_aligned(case):
    ref_out = _run_reference(case, num_Z_samples=8)
    jax_out = _run_jax(case, num_Z_samples=8)

    valid_blocks = np.isfinite(ref_out.log_L_blocks)
    np.testing.assert_allclose(
        jax_out.rho_values[valid_blocks],
        ref_out.rho_values[valid_blocks],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        jax_out.rho_fit[valid_blocks],
        ref_out.rho_fit[valid_blocks],
        rtol=1e-5,
        atol=1e-5,
    )
    for values in (
        ref_out.rho_values[valid_blocks],
        ref_out.rho_fit[valid_blocks],
        jax_out.rho_values[valid_blocks],
        jax_out.rho_fit[valid_blocks],
    ):
        assert np.all(np.isfinite(values))
        assert np.all((values > 0.0) & (values <= 1.0))
