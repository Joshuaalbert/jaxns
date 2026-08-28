import inspect
from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jaxns.shrinkage.phantom as jax_phantom
import jaxns.shrinkage.reference as ref_phantom
from jaxns import shrinkage
from jaxns.algorithm.race_tree import BlockState


class PerClusterCountFixture(NamedTuple):
    log_L_blocks: np.ndarray
    block_valid_mask: np.ndarray
    log_L_constraints: np.ndarray
    valid_phantom: np.ndarray
    sample_mask: np.ndarray
    log_L_phantom: np.ndarray
    A_cg: np.ndarray
    B_cg: np.ndarray
    E_cg: np.ndarray
    R_cg: np.ndarray


class MalformedCountCase(NamedTuple):
    name: str
    overrides: dict
    match: str


def _require_callable(module, name: str):
    fn = getattr(module, name, None)
    assert callable(fn), f"{module.__name__}.{name} is required by Ticket 0013."
    return fn


def _count_fixture() -> PerClusterCountFixture:
    return PerClusterCountFixture(
        log_L_blocks=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float),
        block_valid_mask=np.ones((4,), dtype=bool),
        log_L_constraints=np.asarray(
            [-np.inf, 0.0, 1.5, 3.0],
            dtype=float,
        ),
        valid_phantom=np.asarray([True, True, True, True], dtype=bool),
        sample_mask=np.asarray([True, True, True, True], dtype=bool),
        log_L_phantom=np.asarray(
            [
                [-0.5, 0.0, 0.2, 1.0],
                [0.5, 1.0, 1.5, 2.0],
                [2.0, 2.5, 3.0, 0.5],
                [4.0, 4.0, 4.0, 4.0],
            ],
            dtype=float,
        ),
        A_cg=np.asarray(
            [
                [4.0, 2.0, 0.0, 0.0],
                [0.0, 4.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        B_cg=np.asarray(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        E_cg=np.asarray(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        R_cg=np.asarray(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )


def _block_state(num_blocks: int = 1, *, plateau: bool = False) -> BlockState:
    block_size = 2 if plateau else 1
    sample_indices = np.arange(
        num_blocks * block_size,
        dtype=np.int32,
    ).reshape((num_blocks, block_size))
    return BlockState(
        log_L_blocks=jnp.arange(num_blocks, dtype=jnp.float32),
        block_first_idx=jnp.asarray(sample_indices[:, 0]),
        block_size=jnp.full((num_blocks,), block_size, dtype=jnp.int32),
        incoming_K=jnp.full((num_blocks,), 5, dtype=jnp.int32),
        block_out_degree=jnp.zeros((num_blocks,), dtype=jnp.int32),
        valid=jnp.ones((num_blocks,), dtype=jnp.bool_),
        block_sample_indices=jnp.asarray(sample_indices),
    )


def _as_count_arrays(counts) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for field_name in ("A_cg", "B_cg", "E_cg", "R_cg"):
        assert hasattr(counts, field_name), (
            "compute_phantom_count_matrices(...) must return per-cluster "
            f"{field_name} counts."
        )
    return (
        np.asarray(counts.A_cg, dtype=float),
        np.asarray(counts.B_cg, dtype=float),
        np.asarray(counts.E_cg, dtype=float),
        np.asarray(counts.R_cg, dtype=float),
    )


def _call_count_matrices(module, fixture: PerClusterCountFixture):
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


def _call_deterministic_gamma(
        *,
        block_state: BlockState,
        A_cg: np.ndarray,
        B_cg: np.ndarray,
        E_cg: np.ndarray,
        race_gamma_gt: np.ndarray,
        race_gamma_eq: np.ndarray,
        race_gamma_lt: np.ndarray,
        cluster_weights: np.ndarray,
        C_min: float,
):
    fn = _require_callable(
        shrinkage,
        "gamma_weighted_phantom_probabilities_from_draws",
    )
    return fn(
        block_state=block_state,
        A_cg=jnp.asarray(A_cg),
        B_cg=jnp.asarray(B_cg),
        E_cg=jnp.asarray(E_cg),
        race_gamma_gt=jnp.asarray(race_gamma_gt),
        race_gamma_eq=jnp.asarray(race_gamma_eq),
        race_gamma_lt=jnp.asarray(race_gamma_lt),
        cluster_weights=jnp.asarray(cluster_weights),
        C_min=C_min,
    )


def _p_fields(draw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for field_name in ("p_gt", "p_eq", "p_lt"):
        assert hasattr(draw, field_name), (
            "gamma_weighted_phantom_probabilities_from_draws(...) must return "
            f"{field_name}."
        )
    return (
        np.asarray(draw.p_gt, dtype=float),
        np.asarray(draw.p_eq, dtype=float),
        np.asarray(draw.p_lt, dtype=float),
    )


def _assert_probability_samples_match_draw_normalization(samples) -> None:
    m_gt = np.asarray(samples.race_gamma_gt) + np.asarray(
        samples.phantom_add_gt_samples
    )
    m_eq = np.asarray(samples.race_gamma_eq) + np.asarray(
        samples.phantom_add_eq_samples
    )
    m_lt = np.asarray(samples.race_gamma_lt) + np.asarray(
        samples.phantom_add_lt_samples
    )
    total = m_gt + m_eq + m_lt
    np.testing.assert_allclose(
        np.asarray(samples.p_gt_samples),
        m_gt / total,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(samples.p_eq_samples),
        m_eq / total,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(samples.p_lt_samples),
        m_lt / total,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        (
            np.asarray(samples.p_gt_samples)
            + np.asarray(samples.p_eq_samples)
            + np.asarray(samples.p_lt_samples)
        ),
        np.ones_like(total),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "module",
    [ref_phantom, jax_phantom],
    ids=["reference", "jax"],
)
def test_per_cluster_counts_are_parent_contour_gated_with_equality_and_open_interval(
        module,
):
    fixture = _count_fixture()

    counts = _call_count_matrices(module, fixture)
    A_cg, B_cg, E_cg, R_cg = _as_count_arrays(counts)

    np.testing.assert_allclose(A_cg, fixture.A_cg)
    np.testing.assert_allclose(B_cg, fixture.B_cg)
    np.testing.assert_allclose(E_cg, fixture.E_cg)
    np.testing.assert_allclose(R_cg, fixture.R_cg)
    np.testing.assert_allclose(R_cg, A_cg - B_cg - E_cg)
    assert not np.any(A_cg[3]), (
        "A cluster generated under a contour stricter than every block parent "
        "must not condition those blocks."
    )

    for field_name, expected in (
            ("A_g", fixture.A_cg.sum(axis=0)),
            ("B_g", fixture.B_cg.sum(axis=0)),
            ("E_g", fixture.E_cg.sum(axis=0)),
            ("R_g", fixture.R_cg.sum(axis=0)),
    ):
        assert hasattr(counts, field_name), (
            "count diagnostics must expose aggregate "
            f"{field_name} summaries."
        )
        np.testing.assert_allclose(np.asarray(getattr(counts, field_name)), expected)


def test_count_matrix_validation_rejects_malformed_count_relations() -> None:
    validate_counts = _require_callable(
        shrinkage,
        "validate_phantom_count_matrices",
    )
    block_valid_mask = jnp.asarray([True, True])
    base = {
        "A_cg": jnp.asarray([[2.0, 0.0], [1.0, 1.0]]),
        "B_cg": jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
        "E_cg": jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "block_valid_mask": block_valid_mask,
    }

    validate_counts(**base)

    bad_shape = dict(base, E_cg=jnp.asarray([[1.0], [0.0]]))
    with pytest.raises(ValueError, match="shape|align"):
        validate_counts(**bad_shape)

    nonfinite = dict(base, A_cg=jnp.asarray([[2.0, np.nan], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite|valid"):
        validate_counts(**nonfinite)

    negative = dict(base, B_cg=jnp.asarray([[1.0, 0.0], [-1.0, 0.0]]))
    with pytest.raises(ValueError, match="negative|non-negative"):
        validate_counts(**negative)

    impossible = dict(base, E_cg=jnp.asarray([[2.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="B.*E.*A|count"):
        validate_counts(**impossible)


@pytest.mark.parametrize(
    "case",
    [
        MalformedCountCase(
            name="constraint length mismatch",
            overrides={"log_L_constraints": np.asarray([-np.inf, 0.0])},
            match="log_L_constraints|cluster axis|shape",
        ),
        MalformedCountCase(
            name="per-phantom validity mask",
            overrides={
                "valid_phantom": np.asarray(
                    [
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, True, True, True],
                    ],
                    dtype=bool,
                )
            },
            match="valid_phantom|per-cluster|one-dimensional",
        ),
        MalformedCountCase(
            name="stale valid cluster association",
            overrides={"sample_mask": np.asarray([True, True, False, True])},
            match="stale|sample_mask|num_samples|association",
        ),
        MalformedCountCase(
            name="phantom likelihood cluster mismatch",
            overrides={"log_L_phantom": np.ones((3, 4), dtype=float)},
            match="log_L_phantom|cluster axis|shape",
        ),
    ],
    ids=lambda case: case.name,
)
def test_count_input_validation_rejects_stale_and_malformed_metadata(case) -> None:
    fixture = _count_fixture()
    kwargs = fixture._asdict()
    kwargs.update(case.overrides)
    bad_fixture = PerClusterCountFixture(**kwargs)

    with pytest.raises(ValueError, match=case.match):
        _call_count_matrices(jax_phantom, bad_fixture)


def test_kish_gate_default_C_min_configurable_and_requires_independent_clusters():
    kish_fn = _require_callable(
        shrinkage,
        "compute_kish_participating_cluster_counts",
    )
    gate_fn = _require_callable(shrinkage, "compute_phantom_gate_active")
    signature = inspect.signature(gate_fn)
    assert signature.parameters["C_min"].default == 20

    A_cg = np.zeros((20, 4), dtype=float)
    A_cg[0, 0] = 100.0
    A_cg[:5, 1] = 1.0
    A_cg[:20, 2] = 1.0

    kish = np.asarray(kish_fn(jnp.asarray(A_cg)), dtype=float)
    gate_default = np.asarray(gate_fn(jnp.asarray(A_cg)), dtype=bool)
    gate_configured = np.asarray(gate_fn(jnp.asarray(A_cg), C_min=5), dtype=bool)

    np.testing.assert_allclose(kish, [1.0, 5.0, 20.0, 0.0])
    np.testing.assert_array_equal(
        gate_default,
        np.asarray([False, False, True, False]),
    )
    np.testing.assert_array_equal(
        gate_configured,
        np.asarray([False, True, True, False]),
    )


def test_deterministic_gamma_helper_reuses_cluster_weights_across_blocks_and_components():
    block_state = _block_state(num_blocks=2, plateau=True)
    A_cg = np.asarray(
        [
            [3.0, 2.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    B_cg = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    E_cg = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    race_gt = np.asarray([10.0, 20.0], dtype=float)
    race_eq = np.asarray([2.0, 3.0], dtype=float)
    race_lt = np.asarray([4.0, 5.0], dtype=float)
    cluster_weights = np.asarray([0.5, 2.0], dtype=float)

    draw = _call_deterministic_gamma(
        block_state=block_state,
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=cluster_weights,
        C_min=1.0,
    )
    p_gt, p_eq, p_lt = _p_fields(draw)

    R_cg = A_cg - B_cg - E_cg
    expected_gt = race_gt + cluster_weights @ B_cg
    expected_eq = race_eq + cluster_weights @ E_cg
    expected_lt = race_lt + cluster_weights @ R_cg
    expected_total = expected_gt + expected_eq + expected_lt

    np.testing.assert_allclose(p_gt, expected_gt / expected_total)
    np.testing.assert_allclose(p_eq, expected_eq / expected_total)
    np.testing.assert_allclose(p_lt, expected_lt / expected_total)

    for field_name, expected in (
            ("phantom_add_gt", cluster_weights @ B_cg),
            ("phantom_add_eq", cluster_weights @ E_cg),
            ("phantom_add_lt", cluster_weights @ R_cg),
    ):
        assert hasattr(draw, field_name), (
            "deterministic gamma helper must expose weighted phantom "
            f"{field_name} additions for diagnostics."
        )
        np.testing.assert_allclose(np.asarray(getattr(draw, field_name)), expected)


def test_no_phantom_and_gate_inactive_blocks_reduce_to_classic_race_gammas():
    block_state = _block_state(num_blocks=2, plateau=True)
    A_cg = np.asarray([[0.0, 100.0]], dtype=float)
    B_cg = np.asarray([[0.0, 40.0]], dtype=float)
    E_cg = np.asarray([[0.0, 20.0]], dtype=float)
    race_gt = np.asarray([3.0, 5.0], dtype=float)
    race_eq = np.asarray([2.0, 7.0], dtype=float)
    race_lt = np.asarray([1.0, 11.0], dtype=float)

    draw = _call_deterministic_gamma(
        block_state=block_state,
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=np.asarray([1000.0], dtype=float),
        C_min=20.0,
    )
    p_gt, p_eq, p_lt = _p_fields(draw)
    total = race_gt + race_eq + race_lt

    np.testing.assert_allclose(p_gt, race_gt / total)
    np.testing.assert_allclose(p_eq, race_eq / total)
    np.testing.assert_allclose(p_lt, race_lt / total)
    np.testing.assert_array_equal(
        np.asarray(draw.phantom_gate_active, dtype=bool),
        np.asarray([False, False]),
    )


def test_plateau_equality_mass_and_R_component_are_separate_gamma_targets():
    block_state = _block_state(num_blocks=1, plateau=True)
    race_gt = np.asarray([2.0], dtype=float)
    race_eq = np.asarray([1.0], dtype=float)
    race_lt = np.asarray([1.0], dtype=float)
    weights = np.asarray([3.0], dtype=float)

    equality_draw = _call_deterministic_gamma(
        block_state=block_state,
        A_cg=np.asarray([[2.0]], dtype=float),
        B_cg=np.asarray([[0.0]], dtype=float),
        E_cg=np.asarray([[2.0]], dtype=float),
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=weights,
        C_min=1.0,
    )
    open_interval_draw = _call_deterministic_gamma(
        block_state=block_state,
        A_cg=np.asarray([[2.0]], dtype=float),
        B_cg=np.asarray([[0.0]], dtype=float),
        E_cg=np.asarray([[0.0]], dtype=float),
        race_gamma_gt=race_gt,
        race_gamma_eq=race_eq,
        race_gamma_lt=race_lt,
        cluster_weights=weights,
        C_min=1.0,
    )

    _, equality_p_eq, equality_p_lt = _p_fields(equality_draw)
    _, open_p_eq, open_p_lt = _p_fields(open_interval_draw)
    assert equality_p_eq[0] > open_p_eq[0]
    assert open_p_lt[0] > equality_p_lt[0]
    np.testing.assert_allclose(
        np.asarray(equality_draw.phantom_add_lt),
        np.asarray([0.0]),
    )
    np.testing.assert_allclose(
        np.asarray(open_interval_draw.phantom_add_lt),
        np.asarray([6.0]),
    )


@pytest.mark.parametrize("num_samples", [1, 7])
def test_jax_gamma_weighted_samples_are_raw_per_draw_normalization(
        num_samples: int,
) -> None:
    sample_fn = _require_callable(
        shrinkage,
        "sample_gamma_weighted_phantom_probabilities",
    )
    block_state = _block_state(num_blocks=2)
    A_cg = np.ones((20, 2), dtype=float)
    B_cg = np.asarray(
        [
            (1.0 if idx < 8 else 0.0, 1.0 if idx < 5 else 0.0)
            for idx in range(20)
        ],
        dtype=float,
    )
    E_cg = np.asarray(
        [
            (1.0 if 8 <= idx < 12 else 0.0, 1.0 if 5 <= idx < 10 else 0.0)
            for idx in range(20)
        ],
        dtype=float,
    )

    samples = sample_fn(
        key=jax.random.PRNGKey(101),
        block_state=block_state,
        A_cg=jnp.asarray(A_cg),
        B_cg=jnp.asarray(B_cg),
        E_cg=jnp.asarray(E_cg),
        num_samples=num_samples,
        C_min=20,
    )

    _assert_probability_samples_match_draw_normalization(samples)


@pytest.mark.parametrize("num_samples", [1, 7])
def test_reference_gamma_weighted_samples_are_raw_per_draw_normalization(
        num_samples: int,
) -> None:
    sample_fn = _require_callable(
        ref_phantom,
        "sample_gamma_weighted_phantom_probabilities",
    )
    block_state = _block_state(num_blocks=2)
    A_cg = np.ones((20, 2), dtype=float)
    B_cg = np.asarray(
        [
            (1.0 if idx < 8 else 0.0, 1.0 if idx < 5 else 0.0)
            for idx in range(20)
        ],
        dtype=float,
    )
    E_cg = np.asarray(
        [
            (1.0 if 8 <= idx < 12 else 0.0, 1.0 if 5 <= idx < 10 else 0.0)
            for idx in range(20)
        ],
        dtype=float,
    )

    samples = sample_fn(
        rng=np.random.default_rng(101),
        block_state=block_state,
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        num_samples=num_samples,
        C_min=20,
    )

    _assert_probability_samples_match_draw_normalization(samples)


def test_singleton_block_stays_two_class_and_correlated_cluster_inflates_variance():
    sample_fn = _require_callable(
        shrinkage,
        "sample_gamma_weighted_phantom_probabilities",
    )
    block_state = _block_state(num_blocks=1)
    num_samples = 8192
    singleton_A = np.ones((50, 1), dtype=float)
    singleton_B = np.concatenate(
        [np.ones((20, 1)), np.zeros((30, 1))],
        axis=0,
    )
    singleton_E = np.concatenate(
        [np.zeros((20, 1)), np.ones((10, 1)), np.zeros((20, 1))],
        axis=0,
    )

    singleton = sample_fn(
        key=jax.random.PRNGKey(7),
        block_state=block_state,
        A_cg=jnp.asarray(singleton_A),
        B_cg=jnp.asarray(singleton_B),
        E_cg=jnp.asarray(singleton_E),
        num_samples=num_samples,
        C_min=20,
    )

    # The classic singleton race contributes (K_g, 1)=(5, 1). Phantom
    # equality observations remain in the A-B complement because no atom is
    # present, producing the exact two-class Dirichlet (25, 31).
    alpha = np.asarray([25.0, 0.0, 31.0], dtype=float)
    alpha0 = np.sum(alpha)
    expected_mean = alpha / alpha0
    expected_var = alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1.0))
    singleton_p = np.stack(
        [
            np.asarray(singleton.p_gt_samples)[:, 0],
            np.asarray(singleton.p_eq_samples)[:, 0],
            np.asarray(singleton.p_lt_samples)[:, 0],
        ],
        axis=-1,
    )
    np.testing.assert_allclose(np.mean(singleton_p, axis=0), expected_mean, atol=0.015)
    np.testing.assert_allclose(np.var(singleton_p, axis=0), expected_var, atol=0.001)

    correlated = sample_fn(
        key=jax.random.PRNGKey(8),
        block_state=block_state,
        A_cg=jnp.asarray([[50.0]]),
        B_cg=jnp.asarray([[20.0]]),
        E_cg=jnp.asarray([[10.0]]),
        num_samples=num_samples,
        C_min=1,
    )
    for field_name, expected_mean in (
            ("phantom_add_gt_samples", 20.0),
            ("phantom_add_eq_samples", 0.0),
            ("phantom_add_lt_samples", 30.0),
    ):
        assert hasattr(correlated, field_name), (
            "sample_gamma_weighted_phantom_probabilities(...) must expose "
            f"{field_name} so correlated-cluster mean preservation can be "
            "checked directly."
        )
        np.testing.assert_allclose(
            np.mean(np.asarray(getattr(correlated, field_name))[:, 0]),
            expected_mean,
            rtol=0.0,
            atol=0.5,
        )
    correlated_p = np.stack(
        [
            np.asarray(correlated.p_gt_samples)[:, 0],
            np.asarray(correlated.p_eq_samples)[:, 0],
            np.asarray(correlated.p_lt_samples)[:, 0],
        ],
        axis=-1,
    )
    correlated_p_gt = correlated_p[:, 0]
    assert np.var(correlated_p_gt) > 1.25 * np.var(singleton_p[:, 0])


def test_gamma_race_components_and_cluster_weights_are_independent_draws():
    draw_fn = _require_callable(
        shrinkage,
        "sample_gamma_weighted_phantom_draws",
    )
    draws = draw_fn(
        key=jax.random.PRNGKey(1234),
        block_state=_block_state(num_blocks=1),
        num_clusters=3,
        num_samples=4096,
    )
    for field_name in (
            "race_gamma_gt",
            "race_gamma_eq",
            "race_gamma_lt",
            "cluster_weights",
    ):
        assert hasattr(draws, field_name), (
            "sample_gamma_weighted_phantom_draws(...) must expose raw "
            f"{field_name} draws for independence diagnostics."
        )

    series = np.stack(
        [
            np.asarray(draws.race_gamma_gt)[:, 0],
            np.asarray(draws.race_gamma_lt)[:, 0],
            np.asarray(draws.cluster_weights)[:, 0],
            np.asarray(draws.cluster_weights)[:, 1],
            np.asarray(draws.cluster_weights)[:, 2],
        ],
        axis=0,
    )
    corr = np.corrcoef(series)
    off_diag = corr[np.triu_indices_from(corr, k=1)]
    assert np.max(np.abs(off_diag)) < 0.06
