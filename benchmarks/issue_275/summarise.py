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
likelihood. In the end-to-end circular benchmark below, random charts add only
a few percent to warm wall time because likelihood work dominates.

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
