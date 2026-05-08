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


class DirichletConcentrationFixture(NamedTuple):
    alpha_gt: float
    alpha_eq: float
    alpha_lt: float


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
    rho_g: float
    no_phantom_equality_mean: float
    phantom_equality_mean: float
    logZ_ref: float


class PhantomCountEffectFixture(NamedTuple):
    name: str
    log_L_blocks: tuple[float, ...]
    block_size: tuple[int, ...]
    incoming_K: tuple[int, ...]
    phantom_A: tuple[int, ...]
    phantom_B: tuple[int, ...]
    phantom_E: tuple[int, ...]
    rho_g: tuple[float, ...]
    classic_concentrations: DirichletConcentrationFixture
    conditioned_concentrations: DirichletConcentrationFixture


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


def _classic_concentrations(
        *,
        incoming_K: int,
        block_size: int,
        epsilon: float,
) -> DirichletConcentrationFixture:
    return DirichletConcentrationFixture(
        alpha_gt=float(incoming_K - block_size + 1),
        alpha_eq=float(block_size + epsilon),
        alpha_lt=float(1.0 - epsilon),
    )


def _conditioned_concentrations(
        *,
        incoming_K: int,
        block_size: int,
        epsilon: float,
        phantom_A: int,
        phantom_B: int,
        phantom_E: int,
        rho_g: float,
) -> DirichletConcentrationFixture:
    if phantom_B + phantom_E > phantom_A:
        raise ValueError(
            "Phantom count fixture must satisfy B_g + E_g <= A_g."
        )
    classic = _classic_concentrations(
        incoming_K=incoming_K,
        block_size=block_size,
        epsilon=epsilon,
    )
    return DirichletConcentrationFixture(
        alpha_gt=classic.alpha_gt + rho_g * phantom_B,
        alpha_eq=classic.alpha_eq + rho_g * phantom_E,
        alpha_lt=classic.alpha_lt + rho_g * (
            phantom_A - phantom_B - phantom_E
        ),
    )


def _mean_eq(concentrations: DirichletConcentrationFixture) -> float:
    total = (
        concentrations.alpha_gt
        + concentrations.alpha_eq
        + concentrations.alpha_lt
    )
    return concentrations.alpha_eq / total


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
    phantom_A = 100
    phantom_B = 20
    phantom_E = 30
    rho_g = 1.0
    equality_mass_ref = 0.3
    classic = _classic_concentrations(
        incoming_K=incoming_K,
        block_size=block_size,
        epsilon=epsilon,
    )
    conditioned = _conditioned_concentrations(
        incoming_K=incoming_K,
        block_size=block_size,
        epsilon=epsilon,
        phantom_A=phantom_A,
        phantom_B=phantom_B,
        phantom_E=phantom_E,
        rho_g=rho_g,
    )
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
        rho_g=rho_g,
        no_phantom_equality_mean=_mean_eq(classic),
        phantom_equality_mean=_mean_eq(conditioned),
        logZ_ref=analytic.logZ_ref,
    )


def phantom_count_effect_fixture() -> PhantomCountEffectFixture:
    """One-block fixture proving phantoms update all Dirichlet components."""
    incoming_K = 6
    block_size = 2
    epsilon = 0.5
    phantom_A = 20
    phantom_B = 7
    phantom_E = 5
    rho_g = 0.4
    return PhantomCountEffectFixture(
        name="single_block_count_effect",
        log_L_blocks=(math.log(4.0),),
        block_size=(block_size,),
        incoming_K=(incoming_K,),
        phantom_A=(phantom_A,),
        phantom_B=(phantom_B,),
        phantom_E=(phantom_E,),
        rho_g=(rho_g,),
        classic_concentrations=_classic_concentrations(
            incoming_K=incoming_K,
            block_size=block_size,
            epsilon=epsilon,
        ),
        conditioned_concentrations=_conditioned_concentrations(
            incoming_K=incoming_K,
            block_size=block_size,
            epsilon=epsilon,
            phantom_A=phantom_A,
            phantom_B=phantom_B,
            phantom_E=phantom_E,
            rho_g=rho_g,
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
