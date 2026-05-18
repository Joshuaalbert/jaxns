"""Deterministic fixture definitions for v3 validation harness tests."""

from __future__ import annotations

import math
from typing import NamedTuple


class AnalyticEvidenceFixture(NamedTuple):
    name: str
    likelihood_levels: tuple[float, ...]
    equality_masses: tuple[float, ...]
    strict_survival_after_block: tuple[float, ...]
    logZ_ref: float


class PlateauEqualityRecoveryFixture(NamedTuple):
    name: str
    likelihood_level: float
    equality_mass_ref: float
    block_size: int
    incoming_K: int
    epsilon: float
    phantom_A: int
    phantom_B: int
    phantom_E: int
    phantom_R: int
    A_cg: tuple[tuple[float, ...], ...]
    B_cg: tuple[tuple[float, ...], ...]
    E_cg: tuple[tuple[float, ...], ...]
    R_cg: tuple[tuple[float, ...], ...]
    C_min: int
    kish_participating_cluster_count: float
    phantom_gate_active: bool
    no_phantom_equality_mean: float
    phantom_equality_mean: float
    logZ_ref: float


class GammaWeightedPhantomConditioningFixture(NamedTuple):
    name: str
    log_L_blocks: tuple[float, ...]
    block_size: tuple[int, ...]
    incoming_K: tuple[int, ...]
    block_sample_indices: tuple[tuple[int, ...], ...]
    A_cg: tuple[tuple[float, ...], ...]
    B_cg: tuple[tuple[float, ...], ...]
    E_cg: tuple[tuple[float, ...], ...]
    R_cg: tuple[tuple[float, ...], ...]
    race_gamma_gt: tuple[float, ...]
    race_gamma_eq: tuple[float, ...]
    race_gamma_lt: tuple[float, ...]
    cluster_weights: tuple[float, ...]
    C_min: int
    kish_participating_cluster_counts: tuple[float, ...]
    phantom_gate_active: tuple[bool, ...]
    expected_p_gt: tuple[float, ...]
    expected_p_eq: tuple[float, ...]
    expected_p_lt: tuple[float, ...]


class RaceTreeAccountingFixture(NamedTuple):
    name: str
    root_out_degree: int
    sample_indices: tuple[int, ...]
    log_likelihoods: tuple[float, ...]
    log_L_constraints: tuple[float, ...]
    out_degree: tuple[int, ...]
    expected_log_L_blocks: tuple[float, ...]
    expected_block_size: tuple[int, ...]
    expected_incoming_K: tuple[int, ...]
    expected_block_out_degree: tuple[int, ...]
    expected_block_sample_indices: tuple[tuple[int, ...], ...]


class PosteriorWeightingFixture(NamedTuple):
    name: str
    log_L_blocks: tuple[float, ...]
    block_size: tuple[int, ...]
    incoming_K: tuple[int, ...]
    block_sample_indices: tuple[tuple[int, ...], ...]
    alpha_gt: tuple[float, ...]
    alpha_eq: tuple[float, ...]
    alpha_lt: tuple[float, ...]
    expected_sample_weights: tuple[float, ...]


def _strict_survival_after(
        equality_masses: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        sum(equality_masses[idx + 1:])
        for idx in range(len(equality_masses))
    )


def _log_evidence(
        likelihood_levels: tuple[float, ...],
        equality_masses: tuple[float, ...],
) -> float:
    return math.log(sum(
        likelihood * mass
        for likelihood, mass in zip(
            likelihood_levels,
            equality_masses,
            strict=True,
        )
    ))


def _classic_equality_mean(
        *,
        incoming_K: int,
        block_size: int,
        epsilon: float,
) -> float:
    alpha_gt = float(incoming_K - block_size + 1)
    alpha_eq = float(block_size + epsilon)
    alpha_lt = float(1.0 - epsilon)
    return alpha_eq / (alpha_gt + alpha_eq + alpha_lt)


def _gamma_weighted_equality_target_mean(
        *,
        incoming_K: int,
        block_size: int,
        epsilon: float,
        B_cg: tuple[tuple[float, ...], ...],
        E_cg: tuple[tuple[float, ...], ...],
        R_cg: tuple[tuple[float, ...], ...],
        gate_active: bool,
) -> float:
    alpha_gt = float(incoming_K - block_size + 1)
    alpha_eq = float(block_size + epsilon)
    alpha_lt = float(1.0 - epsilon)
    gate = 1.0 if gate_active else 0.0
    add_gt = gate * sum(row[0] for row in B_cg)
    add_eq = gate * sum(row[0] for row in E_cg)
    add_lt = gate * sum(row[0] for row in R_cg)
    return (alpha_eq + add_eq) / (
        alpha_gt + add_gt + alpha_eq + add_eq + alpha_lt + add_lt
    )


def analytic_evidence_fixtures() -> tuple[AnalyticEvidenceFixture, ...]:
    """Small exact evidence fixtures with known survival curves."""
    three_atom_levels = (1.0, 3.0, 7.0)
    three_atom_masses = (0.4, 0.4, 0.2)
    plateau_levels = (1.0, 5.0, 10.0)
    plateau_masses = (0.5, 0.3, 0.2)
    return (
        AnalyticEvidenceFixture(
            name="three_atom_step",
            likelihood_levels=three_atom_levels,
            equality_masses=three_atom_masses,
            strict_survival_after_block=_strict_survival_after(
                three_atom_masses,
            ),
            logZ_ref=_log_evidence(three_atom_levels, three_atom_masses),
        ),
        AnalyticEvidenceFixture(
            name="plateau_step",
            likelihood_levels=plateau_levels,
            equality_masses=plateau_masses,
            strict_survival_after_block=_strict_survival_after(
                plateau_masses,
            ),
            logZ_ref=_log_evidence(plateau_levels, plateau_masses),
        ),
    )


def plateau_equality_recovery_fixture() -> PlateauEqualityRecoveryFixture:
    """Plateau atom fixture where phantom equality counts recover atom mass."""
    incoming_K = 12
    block_size = 3
    epsilon = 0.5
    A_cg = tuple((1.0,) for _ in range(100))
    B_cg = tuple((1.0 if idx < 20 else 0.0,) for idx in range(100))
    E_cg = tuple((1.0 if 20 <= idx < 50 else 0.0,) for idx in range(100))
    R_cg = tuple(
        (A_cg[idx][0] - B_cg[idx][0] - E_cg[idx][0],)
        for idx in range(100)
    )
    phantom_A = int(sum(row[0] for row in A_cg))
    phantom_B = int(sum(row[0] for row in B_cg))
    phantom_E = int(sum(row[0] for row in E_cg))
    phantom_R = int(sum(row[0] for row in R_cg))
    C_min = 20
    kish = float(phantom_A * phantom_A / phantom_A)
    gate_active = kish >= C_min
    equality_mass_ref = 0.3
    analytic = analytic_evidence_fixtures()[1]
    return PlateauEqualityRecoveryFixture(
        name="plateau_step_equality_recovery",
        likelihood_level=5.0,
        equality_mass_ref=equality_mass_ref,
        block_size=block_size,
        incoming_K=incoming_K,
        epsilon=epsilon,
        phantom_A=phantom_A,
        phantom_B=phantom_B,
        phantom_E=phantom_E,
        phantom_R=phantom_R,
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        R_cg=R_cg,
        C_min=C_min,
        kish_participating_cluster_count=kish,
        phantom_gate_active=gate_active,
        no_phantom_equality_mean=_classic_equality_mean(
            incoming_K=incoming_K,
            block_size=block_size,
            epsilon=epsilon,
        ),
        phantom_equality_mean=_gamma_weighted_equality_target_mean(
            incoming_K=incoming_K,
            block_size=block_size,
            epsilon=epsilon,
            B_cg=B_cg,
            E_cg=E_cg,
            R_cg=R_cg,
            gate_active=gate_active,
        ),
        logZ_ref=analytic.logZ_ref,
    )


def gamma_weighted_phantom_conditioning_fixture(
) -> GammaWeightedPhantomConditioningFixture:
    """Explicit gamma-weighted phantom fixture with shared cluster weights."""
    A_cg = tuple((1.0, 1.0) for _ in range(20))
    B_cg = tuple(
        (
            1.0 if idx < 8 else 0.0,
            1.0 if idx < 5 else 0.0,
        )
        for idx in range(20)
    )
    E_cg = tuple(
        (
            1.0 if 8 <= idx < 12 else 0.0,
            1.0 if 5 <= idx < 10 else 0.0,
        )
        for idx in range(20)
    )
    R_cg = tuple(
        (
            A_cg[idx][0] - B_cg[idx][0] - E_cg[idx][0],
            A_cg[idx][1] - B_cg[idx][1] - E_cg[idx][1],
        )
        for idx in range(20)
    )
    race_gamma_gt = (10.0, 12.0)
    race_gamma_eq = (2.0, 3.0)
    race_gamma_lt = (4.0, 5.0)
    add_gt = (8.0, 5.0)
    add_eq = (4.0, 5.0)
    add_lt = (8.0, 10.0)
    total = (
        race_gamma_gt[0] + add_gt[0] + race_gamma_eq[0] + add_eq[0]
        + race_gamma_lt[0] + add_lt[0],
        race_gamma_gt[1] + add_gt[1] + race_gamma_eq[1] + add_eq[1]
        + race_gamma_lt[1] + add_lt[1],
    )
    return GammaWeightedPhantomConditioningFixture(
        name="gamma_weighted_two_block_conditioning",
        log_L_blocks=(0.0, 1.0),
        block_size=(1, 1),
        incoming_K=(10, 12),
        block_sample_indices=((0,), (1,)),
        A_cg=A_cg,
        B_cg=B_cg,
        E_cg=E_cg,
        R_cg=R_cg,
        race_gamma_gt=race_gamma_gt,
        race_gamma_eq=race_gamma_eq,
        race_gamma_lt=race_gamma_lt,
        cluster_weights=tuple(1.0 for _ in range(20)),
        C_min=20,
        kish_participating_cluster_counts=(20.0, 20.0),
        phantom_gate_active=(True, True),
        expected_p_gt=(
            (race_gamma_gt[0] + add_gt[0]) / total[0],
            (race_gamma_gt[1] + add_gt[1]) / total[1],
        ),
        expected_p_eq=(
            (race_gamma_eq[0] + add_eq[0]) / total[0],
            (race_gamma_eq[1] + add_eq[1]) / total[1],
        ),
        expected_p_lt=(
            (race_gamma_lt[0] + add_lt[0]) / total[0],
            (race_gamma_lt[1] + add_lt[1]) / total[1],
        ),
    )


def race_tree_accounting_fixture() -> RaceTreeAccountingFixture:
    """Race-tree fixture with plateaus and hand-computable K_g recurrence."""
    return RaceTreeAccountingFixture(
        name="plateau_race_tree_accounting",
        root_out_degree=3,
        sample_indices=(100, 101, 102, 103, 104, 105),
        log_likelihoods=(2.0, 1.0, 2.0, 4.0, 3.0, 3.0),
        log_L_constraints=(1.0, -math.inf, -math.inf, 3.0, 2.0, 1.0),
        out_degree=(1, 2, 0, 0, 0, 0),
        expected_log_L_blocks=(1.0, 2.0, 3.0, 4.0),
        expected_block_size=(1, 2, 2, 1),
        expected_incoming_K=(3, 4, 3, 1),
        expected_block_out_degree=(2, 1, 0, 0),
        expected_block_sample_indices=(
            (101, -1),
            (100, 102),
            (104, 105),
            (103, -1),
        ),
    )


def posterior_weighting_fixtures() -> tuple[PosteriorWeightingFixture, ...]:
    """Posterior weighting fixtures for plateau and non-plateau blocks."""
    return (
        PosteriorWeightingFixture(
            name="plateau_equality_atom_split",
            log_L_blocks=(math.log(5.0), math.log(7.0)),
            block_size=(2, 1),
            incoming_K=(4, 2),
            block_sample_indices=((0, 2), (1, -1)),
            alpha_gt=(2.0, 1.0),
            alpha_eq=(1.0, 0.5),
            alpha_lt=(1.0, 0.5),
            expected_sample_weights=(
                0.20833333333333334,
                0.5833333333333334,
                0.20833333333333334,
            ),
        ),
        PosteriorWeightingFixture(
            name="non_plateau_strict_mass",
            log_L_blocks=(math.log(2.0), math.log(3.0)),
            block_size=(1, 1),
            incoming_K=(5, 4),
            block_sample_indices=((0,), (1,)),
            alpha_gt=(4.0, 1.0),
            alpha_eq=(0.5, 0.5),
            alpha_lt=(0.5, 0.5),
            expected_sample_weights=(0.25, 0.75),
        ),
    )
