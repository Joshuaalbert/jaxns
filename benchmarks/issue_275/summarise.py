"""Render the issue 275 benchmark artifacts as a review report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def _groups(name: str) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in _load(name)["records"]:
        grouped[record["label"]].append(record)
    return dict(grouped)


def _row(label: str, records: list[dict]) -> str:
    errors = np.asarray([record["log_Z_error"] for record in records])
    evaluations = np.asarray([
        record["likelihood_evaluations"] for record in records
    ])
    warm = np.asarray([
        record["wall_s"] for record in records if not record["cold"]
    ])
    cold = next(
        record["wall_s"] for record in records if record["cold"]
    )
    moment = [
        record["circular_moment_error"]
        for record in records
        if "circular_moment_error" in record
    ]
    moment_text = (
        "--"
        if not moment
        else f"{np.sqrt(np.mean(np.square(moment))):.4f}"
    )
    return (
        f"| {label} | {len(records)} | {np.mean(errors):+.4f} | "
        f"{np.sqrt(np.mean(np.square(errors))):.4f} | {moment_text} | "
        f"{np.median(evaluations):.0f} | {cold:.3f} | "
        f"{np.median(warm):.3f} |"
    )


def _table(groups: dict[str, list[dict]]) -> str:
    lines = [
        (
            "| Problem/method | n | log Z bias | log Z RMS | circular moment "
            "RMS | median likelihood evals | cold wall s | median warm wall "
            "s |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_row(label, records) for label, records in groups.items())
    return "\n".join(lines)


def _paired_ratio(
    hard: list[dict],
    chart: list[dict],
    field: str,
    statistic: str,
    include_cold: bool = True,
) -> tuple[float, float, float]:
    """Return chart/hard ratio and a paired seed-bootstrap interval."""
    hard = sorted(hard, key=lambda record: record["seed"])
    chart = sorted(chart, key=lambda record: record["seed"])
    if not include_cold:
        # Seed zero includes compilation and is deliberately excluded from the
        # steady-state wall-time comparison on both matched configurations.
        hard = [record for record in hard if not record["cold"]]
        chart = [record for record in chart if not record["cold"]]
    hard_seeds = [record["seed"] for record in hard]
    chart_seeds = [record["seed"] for record in chart]
    if hard_seeds != chart_seeds:
        raise ValueError("Paired comparisons require identical seed sets.")
    hard_values = np.asarray([record[field] for record in hard])
    chart_values = np.asarray([record[field] for record in chart])

    def calculate(values: np.ndarray) -> float:
        if statistic == "rms":
            return float(np.sqrt(np.mean(np.square(values))))
        if statistic == "median":
            return float(np.median(values))
        raise ValueError(f"Unknown paired statistic {statistic!r}.")

    ratio = calculate(chart_values) / calculate(hard_values)
    # Resampling paired seed indices preserves the common NS random-key
    # control, so the interval measures the implementation difference rather
    # than variation introduced by comparing unrelated ensembles.
    random = np.random.default_rng(275)
    indices = random.integers(
        low=0,
        high=len(hard_values),
        size=(10_000, len(hard_values)),
    )
    # [B, N] paired bootstrap samples. Vectorising this reduction keeps report
    # generation cheap even though every displayed interval uses 10,000 draws.
    hard_samples = hard_values[indices]
    chart_samples = chart_values[indices]
    if statistic == "rms":
        hard_statistics = np.sqrt(np.mean(np.square(hard_samples), axis=1))
        chart_statistics = np.sqrt(
            np.mean(np.square(chart_samples), axis=1)
        )
    else:
        hard_statistics = np.median(hard_samples, axis=1)
        chart_statistics = np.median(chart_samples, axis=1)
    ratios = chart_statistics / hard_statistics
    lower, upper = np.quantile(ratios, [0.025, 0.975])
    return ratio, float(lower), float(upper)


def _ratio_text(interval: tuple[float, float, float]) -> str:
    ratio, lower, upper = interval
    return f"{ratio:.2f} [{lower:.2f}, {upper:.2f}]"


def _paired_bias_difference(
    hard: list[dict],
    chart: list[dict],
) -> tuple[float, float, float]:
    """Return chart-minus-hard bias and a paired seed-bootstrap interval."""
    hard = sorted(hard, key=lambda record: record["seed"])
    chart = sorted(chart, key=lambda record: record["seed"])
    hard_seeds = [record["seed"] for record in hard]
    chart_seeds = [record["seed"] for record in chart]
    if hard_seeds != chart_seeds:
        raise ValueError("Paired comparisons require identical seed sets.")
    differences = np.asarray([
        chart_record["log_Z_error"] - hard_record["log_Z_error"]
        for hard_record, chart_record in zip(hard, chart, strict=True)
    ])
    random = np.random.default_rng(275)
    indices = random.integers(
        low=0,
        high=len(differences),
        size=(10_000, len(differences)),
    )
    estimates = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(np.mean(differences)), float(lower), float(upper)


def _difference_text(interval: tuple[float, float, float]) -> str:
    difference, lower, upper = interval
    return f"{difference:+.3f} [{lower:+.3f}, {upper:+.3f}]"


def _geometry_comparison(
    groups: dict[str, list[dict]],
    problem: str,
    placements: tuple[str, ...],
    root_degrees: tuple[int, ...],
) -> str:
    lines = [
        (
            "| Placement | root degree | log Z bias difference | log Z RMS "
            "ratio | moment RMS ratio | likelihood-eval ratio | warm-time "
            "ratio |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for placement in placements:
        for root_degree in root_degrees:
            hard = groups[
                f"{problem}_hard_{placement}K{root_degree}"
            ]
            chart = groups[
                f"{problem}_random_chart_{placement}K{root_degree}"
            ]
            placement_name = placement.removesuffix("_") or "same model"
            lines.append(
                f"| {placement_name} | {root_degree} | "
                f"{_difference_text(_paired_bias_difference(hard, chart))} | "
                f"{_ratio_text(_paired_ratio(hard, chart, 'log_Z_error', 'rms'))} | "
                f"{_ratio_text(_paired_ratio(hard, chart, 'circular_moment_error', 'rms'))} | "
                f"{_ratio_text(_paired_ratio(hard, chart, 'likelihood_evaluations', 'median'))} | "
                f"{_ratio_text(_paired_ratio(hard, chart, 'wall_s', 'median', include_cold=False))} |"
            )
    return "\n".join(lines)


def main() -> None:
    develop = _load("all_false_develop.json")["records"]
    periodic = _load("all_false_periodic.json")["records"]
    paired_fields = (
        "log_Z",
        "log_Z_uncert",
        "log_Z_error",
        "likelihood_evaluations",
        "classic_samples",
        "phantom_samples",
    )
    paired_equal = all(
        left[field] == right[field]
        for left, right in zip(develop, periodic, strict=True)
        for field in paired_fields
    )
    hot_develop = _load("hotpath_develop.json")
    hot_periodic = _load("hotpath_periodic.json")
    hot_enabled = _load("hotpath_enabled.json")
    ridge = _groups("ridge_sweep.json")
    jones_geometry = _groups("jones_geometry.json")

    report = f"""# Issue 275: periodic random-chart evidence

## Method

All accuracy summaries use 30 independent nested-sampling seeds per row. The
Jones evidence uses 256 final MC shrinkage draws per run; the analytic circular
problems use the deterministic race expectation. Cold wall time is the first
end-to-end run after `jax.clear_caches()`. Warm wall time is the median of the
remaining 29 runs and excludes final MC evidence sampling. Likelihood calls are
logical scientific evaluations reported by JAXNS.

Measurements ran on the JAX 0.10.0 CPU backend with one Intel i7-8750H
(6 cores/12 threads). JAXNS enabled x64 for evidence arithmetic; JAXCTX's
homogeneous U coordinates used their normal float32 base dtype. The direct
hotpath shape was 8 concurrent chains, 32 slice transitions, and one U
coordinate. Compiler memory values are XLA `memory_analysis()` reports, not
process RSS.

The Jones reference is independent of JAXNS. Circular constant phase is
integrated through its modified-Bessel form, unknown HalfNormal measurement
noise is integrated in log scale, and DTEC/clock use tensor Gauss-Legendre
quadrature. Orders 32, 64, 128, 256, and 512 all gave
`log Z = 34.803948945405`; noise quadrature orders 256 and 512 agreed at the
displayed precision.

## Ordinary all-false path versus develop

Paired scientific outputs and evaluation counts identical: **{paired_equal}**.

{_table({"develop": develop, "all-false random-chart branch": periodic})}

The constrained-sampler StableHLO SHA-256 is
`{hot_develop['stablehlo_sha256']}` on develop and
`{hot_periodic['stablehlo_sha256']}` on the all-false branch. Both have
{hot_develop['stablehlo_lines']} lines, zero remainder operations,
{hot_develop['temporary_bytes']} temporary bytes, and identical argument/output
memory. Direct compile time was {hot_develop['compile_s']:.3f} s versus
{hot_periodic['compile_s']:.3f} s; median execution was
{1000 * hot_develop['median_execute_s']:.3f} ms versus
{1000 * hot_periodic['median_execute_s']:.3f} ms.

With periodic charts enabled, the same hotpath used
{hot_enabled['remainder_operations']} remainder operations,
{hot_enabled['temporary_bytes']} temporary bytes, compiled in
{hot_enabled['compile_s']:.3f} s, and executed in a median
{1000 * hot_enabled['median_execute_s']:.3f} ms.
This deliberately exposes the modulo cost on an unrealistically cheap toy
likelihood. In the end-to-end default-allocation controls below, random charts
add 3--12% to warm wall time because likelihood work dominates.

## Known circular posterior at three chart placements

{_table(_groups('von_mises.json'))}

## Sharp seam stress: hard cube versus random chart

This uses concentration 100 and 120 root lineages to make split modes at
the canonical seam consequential.

{_table(_groups('seam.json'))}

Moving the physical mode from the seam to the interior changes the hard-cube
median likelihood count by 41% and its circular-moment RMS by 56%. Random
charts reduce those placement effects to 2% for both metrics, while their warm
end-to-end wall times agree within 1%. The log-evidence biases remain small
relative to the per-run RMS errors for every placement.

## Marginal geometry control: coupled periodic ridge

The initial seam stress only showed placement invariance. This stronger
control holds the physical target, stopping condition, slice count, lineage
count, and random keys fixed while changing only whether the two angular
coordinates have hard cube boundaries or independent random periodic charts.
The target is a coupled toroidal ridge; at the canonical seam its posterior is
split among four cube corners. Root lineage counts 4, 8, 16, and 32 deliberately
test under-resolved mode recovery, while 60 is the normal JAXNS default for
this two-dimensional model. The interior placement is a negative control.

{_table(ridge)}

The following comparisons use a 10,000-resample paired seed bootstrap;
brackets are 95% intervals. Bias is chart minus hard, while the other columns
are chart/hard ratios for which values below one favour random charts.

{_geometry_comparison(
        ridge,
        problem='ridge',
        placements=('seam_', 'interior_'),
        root_degrees=(4, 8, 16, 32, 60),
    )}

## Marginal geometry control: Jones calibration

This is the full Jones model with the same data, priors, stopping condition,
slice count, random keys, and MC evidence calculation in each pair. Only the
constant phase coordinate changes from hard bounded to periodic. Root degree
120 is the normal four-dimensional default; lower values test whether topology
helps when the run begins with less mode coverage.

{_table(jones_geometry)}

{_geometry_comparison(
        jones_geometry,
        problem='jones',
        placements=('',),
        root_degrees=(16, 32, 64, 120),
    )}

## Interpretation

The marginal value is specific rather than universal. On the coupled ridge at
the seam, random charts reduce circular posterior-moment RMS by 57--70% at
every tested lineage allocation. At the normal 60-root allocation the ratio is
0.43 [0.32, 0.58], whereas the interior negative control is 1.20
[0.93, 1.56]. This is strong evidence that charts repair seam-fragmented
posterior exploration, rather than generally improving an easy interior
target.

There is no evidence that charts change log-evidence bias: every paired bias
difference interval contains zero. Nor is there a consistent evidence-RMS
improvement. The normal ridge setting has a higher RMS point estimate, 1.38
[1.00, 1.85], while the normal Jones setting is statistically unresolved at
0.92 [0.70, 1.18]. The feature should therefore not be presented as an
evidence-accuracy optimisation.

On Jones, the normal 120-root setting has unchanged likelihood work, 1.00
[0.98, 1.02], 3% more warm wall time, and a smaller periodic-phase moment RMS,
0.81 [0.66, 0.99]. Together with the ridge result, this supports keeping random
charts as the topology-correct sampler for explicitly periodic coordinates.
The cost is isolated: when no coordinate is periodic, scientific results,
evaluation counts, StableHLO, and compiler memory are identical to develop.

## Full Jones calibration problem

Twenty channels span 700--702.6 MHz. The inferred variables are DTEC in mTECU,
clock in ns, periodic constant phase, and unknown measurement uncertainty; the
truth noise is 0.1.

{_table(_groups('jones.json'))}
"""
    (ROOT / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
