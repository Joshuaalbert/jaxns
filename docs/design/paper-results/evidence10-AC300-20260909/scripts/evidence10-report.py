"""Audit completed cohorts and write paired evidence and recovery summaries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--completed-only", action="store_true")
args = parser.parse_args()
out = args.root
report = out / "report"
report.mkdir(exist_ok=True)
sources = json.loads((out / "SOURCES.json").read_text())
analysis_sources = (
    json.loads((out / "ANALYSIS_SOURCES.json").read_text())
    if (out / "ANALYSIS_SOURCES.json").exists()
    else {}
)
retries = json.loads((out / "RETRIES.json").read_text()) if (out / "RETRIES.json").exists() else []
if not args.completed_only:
    assert json.loads((out / "FINISHED.json").read_text())["complete"]
indices = np.random.default_rng(20260909).integers(0, 30, (10000, 30))
summary = {}
audit = {
    "source": sources,
    "cells": 0,
    "record_sha256": {},
    "bootstrap_seed": 20260909,
    "bootstrap_resamples": 10000,
    "state_storage": {},
    "continuation_checks": [],
}
tables = []
for stage in ("0.05", "0.02", "0.01", "0.005"):
    cases = ("g10", "cg10", "ss10") if stage == "0.05" else ("ss10",)
    for case in cases:
        cells = [out / stage / case / f"seed-{seed:02d}" for seed in range(30)]
        if not all((cell / "ANALYSIS.json").exists() for cell in cells):
            continue
        analyses, cores, coverage = [], [], []
        for seed, cell in enumerate(cells):
            core = json.loads((cell / "CORE.json").read_text())
            a = json.loads((cell / "ANALYSIS.json").read_text())
            m = json.loads((cell / "MANIFEST.json").read_text())
            assert core["complete"] and core["classic_log_Z_uncert"] < float(stage)
            assert core["seed"] == a["seed"] == m["seed"] == seed
            assert core["case"] == a["case"] == m["case"] == case
            assert m["source_commit"] == a["source_commit"] == sources["commit"]
            if "analysis_source_commit" in a:
                analysis_source = analysis_sources[a["analysis_source_commit"]]
                assert analysis_source["src_tree"] == sources["src_tree"]
                assert a["analysis_source_sha256"] == analysis_source["files"]
            assert m["case_revision"] == a["case_revision"] == "evidence10-AC300-20260909"
            assert m["goal_classic_log_Z_uncert"] == float(stage)
            assert m["root_degree"] == m["delta_K"] == 300
            assert m["shell_size"] == m["num_slices"] == 100
            assert m["retained_phantoms"] == 99 and m["all_phantom_seeds"]
            assert m["allocation_target"] == "evidence_improving"
            assert m["allocation_weights"] is None and m["direction"] == "isotropic"
            assert m["bootstrap"] == {"allocation_target": "uniform", "delta_K": 1, "goals": 1}
            assert m["checkpoint_preserves_capacity"]
            assert len(m["affinity"]) == len(a["affinity"]) == 1
            assert a["mc_draws"] == 2048 and a["prefix_sizes"] == list(range(0, 91, 10))
            for name, digest in m["source_sha256"].items():
                assert sources["files"][name] == digest
            posterior = np.load(cell / "classic_posterior.npz")
            x = posterior["x"]  # [N,D]
            w = np.exp(posterior["log_dp"] - logsumexp(posterior["log_dp"]))
            assert x.shape == (core["classic_samples"], 10)
            np.testing.assert_allclose(1.0 / np.sum(w**2), core["classic_kish_ess"], rtol=1e-10)
            np.testing.assert_allclose(a["classic_kish_ess"], core["classic_kish_ess"], rtol=1e-10)
            if case == "ss10":
                means = np.zeros((2, 10))
                means[0, :2], means[1, :2] = 6.0, 2.5
                log_l = norm.logpdf(
                    x[:, None, :], means[None, :, :], np.sqrt([0.08, 0.8])[None, :, None]
                ).sum(axis=-1)
                mass = w @ np.exp(log_l[:, 0] - logsumexp(log_l, axis=1))
                np.testing.assert_allclose(mass, a["classic_spike_mass"], atol=1e-12)
            draws = np.load(cell / "evidence_draws.npz")["log_Z"]  # [draw,prefix]
            assert draws.shape == (2048, 10) and np.isfinite(draws).all()
            np.testing.assert_allclose(draws.mean(axis=0), a["log_Z_mean"], rtol=1e-12)
            np.testing.assert_allclose(draws.std(axis=0, ddof=1), a["log_Z_uncert"], rtol=1e-12)
            low, high = np.quantile(draws, [0.025, 0.975], axis=0)
            truth = a["reference"]["log_Z"]
            coverage.append((low <= truth) & (truth <= high))
            for name in (
                "CORE.json",
                "ANALYSIS.json",
                "MANIFEST.json",
                "classic_posterior.npz",
                "evidence_draws.npz",
                "progress.jsonl",
            ):
                path = cell / name
                audit["record_sha256"][str(path.relative_to(out))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            state = cell / "state.pkl"
            checkpoint = cell / "checkpoint.pkl"
            archive = (
                json.loads((cell / "STATE_ARCHIVE.json").read_text())
                if (cell / "STATE_ARCHIVE.json").exists()
                else None
            )
            assert state.exists() and checkpoint.exists()
            shared_file = checkpoint.samefile(state)
            if archive:
                assert archive["verified_after_copy"]
                assert state.stat().st_size == checkpoint.stat().st_size == archive["bytes"]
            if not args.completed_only or archive is None:
                assert shared_file
            audit["state_storage"][str(state.relative_to(out))] = {
                "bytes": state.stat().st_size,
                "capacity": core["state_capacity"],
                "checkpoint_shares_file": shared_file,
                "archive": archive,
            }
            if m["resume_from"]:
                previous = json.loads((Path(m["resume_from"]) / "CORE.json").read_text())
                assert (
                    core["goal_progress"][: len(previous["goal_progress"])]
                    == previous["goal_progress"]
                )
                assert core["iteration"] > previous["iteration"]
                assert core["classic_samples"] > previous["classic_samples"]
                assert core["likelihood_evaluations"] > previous["likelihood_evaluations"]
                audit["continuation_checks"].append(
                    {"stage": stage, "seed": seed, "retained_goal_prefix": True}
                )
            cores.append(core)
            analyses.append(a)
            audit["cells"] += 1
        truth = analyses[0]["reference"]["log_Z"]
        errors = np.asarray([a["log_Z_mean"] for a in analyses]) - truth  # [seed,prefix]
        boot_rmse = np.sqrt(np.mean(errors[indices] ** 2, axis=1))  # [bootstrap,prefix]
        boot_delta = boot_rmse[:, -1] - boot_rmse[:, 0]
        calls = np.asarray([c["likelihood_evaluations"] for c in cores])
        goals = np.asarray([c["iteration"] for c in cores])
        r = dict(
            stage=stage,
            case=case,
            seeds=list(range(30)),
            truth_log_Z=truth,
            path=str(out / stage / case),
            errors=errors.tolist(),
            rmse=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            rmse_se=boot_rmse.std(axis=0, ddof=1).tolist(),
            rmse_ci95=np.quantile(boot_rmse, [0.025, 0.975], axis=0).T.tolist(),
            bias=errors.mean(axis=0).tolist(),
            bias_sd=errors.std(axis=0, ddof=1).tolist(),
            mean_reported_sd=np.mean([a["log_Z_uncert"] for a in analyses], axis=0).tolist(),
            coverage95=np.mean(coverage, axis=0).tolist(),
            coverage_by_seed=np.asarray(coverage).tolist(),
            mean_gate_fraction=np.mean([a["gate_fraction"] for a in analyses], axis=0).tolist(),
            paired_ph90_minus_classic_rmse_ci95=np.quantile(boot_delta, [0.025, 0.975]).tolist(),
            paired_ph90_minus_classic_rmse_se=float(boot_delta.std(ddof=1)),
            calls=calls.tolist(),
            mean_calls=float(calls.mean()),
            sd_calls=float(calls.std(ddof=1)),
            goals=goals.tolist(),
            mean_goals=float(goals.mean()),
            sd_goals=float(goals.std(ddof=1)),
            mean_core_seconds=float(np.mean([c["run_seconds"] for c in cores])),
            mean_stage_seconds=float(np.mean([c["stage_run_seconds"] for c in cores])),
            mean_analysis_seconds=float(np.mean([a["analysis_seconds"] for a in analyses])),
            peak_core_rss_gib=max(c["peak_rss_bytes"] for c in cores) / 1024**3,
            peak_analysis_rss_gib=max(a["peak_rss_bytes"] for a in analyses) / 1024**3,
            mean_classic_samples=float(np.mean([c["classic_samples"] for c in cores])),
            mean_kish_ess=float(np.mean([c["classic_kish_ess"] for c in cores])),
            goal_sigmas=[c["classic_log_Z_uncert"] for c in cores],
            analysis_source_commits=sorted(
                {a.get("analysis_source_commit", a["source_commit"]) for a in analyses}
            ),
            tied_blocks_by_seed=[a.get("num_tied_blocks", 0) for a in analyses],
            largest_blocks_by_seed=[a.get("largest_block", 1) for a in analyses],
        )
        if case == "ss10":
            masses = np.asarray([a["classic_spike_mass"] for a in analyses])
            mass_error = masses - 0.5000077444258325
            boot_mass = np.sqrt(np.mean(mass_error[indices] ** 2, axis=1))
            r["posterior"] = dict(
                truth=0.5000077444258325,
                mass_by_seed=masses.tolist(),
                mean=float(masses.mean()),
                min=float(masses.min()),
                max=float(masses.max()),
                rmse=float(np.sqrt(np.mean(mass_error**2))),
                rmse_se=float(boot_mass.std(ddof=1)),
                rmse_ci95=np.quantile(boot_mass, [0.025, 0.975]).tolist(),
                max_mass_error=float(np.abs(mass_error).max()),
                recovered=bool(
                    np.sqrt(np.mean(mass_error**2)) <= 0.05 and np.abs(mass_error).max() <= 0.15
                ),
                seeds_below_10_percent=int(np.sum(masses < 0.1)),
                seeds_within_5_percentage_points=int(np.sum(np.abs(mass_error) <= 0.05)),
            )
        summary.setdefault(stage, {})[case] = r
        label = f"{case}_{stage.replace('.', 'p')}"
        best = int(np.argmin(r["rmse"]))
        rows = []
        for p in range(10):
            values = [
                f"${p}D$" + (" (classic)" if p == 0 else ""),
                f"{r['bias'][p]:+.4f} $\\pm$ {r['bias_sd'][p]:.4f}",
                f"{r['rmse'][p]:.4f} $\\pm$ {r['rmse_se'][p]:.4f}",
                f"{r['mean_reported_sd'][p]:.4f}",
                f"{100 * r['coverage95'][p]:.0f}\\%",
            ]
            if p == best:
                values[2] = r"\textbf{" + values[2] + "}"
            rows.append(" & ".join(values) + r" \\")
        table = (
            "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
            + f"\\caption{{{case.upper()} A+C300, classic uncertainty target {stage}, 30 trees. "
            + "Bias errors are across-tree SDs; RMSE errors are bootstrap SEs. "
            + "SD is the mean reported shrinkage uncertainty, and coverage is the fraction "
            + "of 95\\% shrinkage intervals containing the reference log evidence. "
            + f"Mean likelihood calls: $({r['mean_calls'] / 1e6:.3f}"
            + f"\\pm{r['sd_calls'] / 1e6:.3f})\\times10^6$; "
            + f"mean goals: ${r['mean_goals']:.2f}$. Bold marks the lowest point RMSE.}}\n"
            + f"\\label{{tab:phantom_evidence_{label}}}\n"
            + "\\begin{tabular}{lrrrr}\n\\toprule\n"
            + r"Prefix & Bias $\pm$ SD & RMSE $\pm$ SE & Mean SD & 95\% coverage \\"
            + "\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
        )
        (report / f"{label}_table.tex").write_text(table)
        tables.append(table)

active = {}
max_active = 0
completed_phases = 0
for line in (out / "dispatch.log").read_text().splitlines(keepends=True):
    if not line.startswith("{") or not line.endswith("\n"):
        continue
    event = json.loads(line)
    if event["event"] == "start":
        assert event["cpu"] not in active
        active[event["cpu"]] = event["pid"]
        max_active = max(max_active, len(active))
        assert max_active <= 60
    elif event["event"] == "finish":
        assert active.pop(event["cpu"]) == event["pid"]
        if event["success"]:
            completed_phases += 1
        else:
            matching = [
                retry
                for retry in retries
                if retry["stage"] == event["stage"]
                and retry["job"] == event["job"]
                and retry["exit_code"] == event["exit_code"]
            ]
            assert len(matching) == 1
            audit.setdefault("failed_analysis_attempts", []).append(event)
if not args.completed_only:
    assert not active and completed_phases == 2 * audit["cells"]
    assert all(retry["resolved"] for retry in retries)
audit["analysis_sources"] = analysis_sources
audit["retries"] = retries
audit.update(max_active=max_active, completed_phases=completed_phases)
previous_path = Path("/largedata/albert/jaxns-posterior10-20260908/ss10/seed-00/CORE.json")
new_path = out / "0.05/ss10/seed-00/CORE.json"
if new_path.exists():
    previous = json.loads(previous_path.read_text())
    new = json.loads(new_path.read_text())
    keys = (
        "iteration",
        "classic_samples",
        "likelihood_evaluations",
        "classic_log_Z",
        "classic_log_Z_uncert",
        "classic_kish_ess",
    )
    for key in keys:
        np.testing.assert_allclose(previous[key], new[key], rtol=1e-12, atol=0)
    audit["ss10_seed0_matches_previous_baseline"] = True
    new_analysis = out / "0.05/ss10/seed-00/ANALYSIS.json"
    if new_analysis.exists():
        previous_a = json.loads(previous_path.with_name("ANALYSIS.json").read_text())
        new_a = json.loads(new_analysis.read_text())
        for key in ("log_Z_mean", "log_Z_uncert"):
            np.testing.assert_array_equal(previous_a[key], new_a[key])
        np.testing.assert_array_equal(
            previous_a["classic_spike_responsibility_mass"], new_a["classic_spike_mass"]
        )
        audit["ss10_seed0_all_prefixes_and_mass_match_previous"] = True

(report / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
(report / "AUDIT.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
(report / "ALL_TABLES.tex").write_text("\n".join(tables))
lines = [
    "# A+C300: 30-seed evidence and SS10 recovery",
    "",
    (
        "| Problem | Stop | Classic RMSE | Ph90 RMSE | Mean reported SD, "
        "classic / Ph90 | Calls (M) | Goals |"
    ),
    "|---|---:|---:|---:|---:|---:|---:|",
]
for stage, cases in summary.items():
    for case, r in cases.items():
        lines.append(
            f"| {case.upper()} | {stage} | {r['rmse'][0]:.4f} ± {r['rmse_se'][0]:.4f} | "
            f"{r['rmse'][-1]:.4f} ± {r['rmse_se'][-1]:.4f} | "
            f"{r['mean_reported_sd'][0]:.4f} / {r['mean_reported_sd'][-1]:.4f} | "
            f"{r['mean_calls'] / 1e6:.3f} | {r['mean_goals']:.2f} |"
        )
lines.extend(
    [
        "",
        "| SS10 stop | Mean spike mass | Spike-mass RMSE | Range | Recovered |",
        "|---:|---:|---:|---:|---|",
    ]
)
for stage, cases in summary.items():
    if "ss10" not in cases:
        continue
    p = cases["ss10"]["posterior"]
    lines.append(
        f"| {stage} | {p['mean']:.5f} | {p['rmse']:.5f} ± {p['rmse_se']:.5f} | "
        f"{p['min']:.5f}–{p['max']:.5f} | {p['recovered']} |"
    )
lines.extend(
    [
        "",
        "The true spike fraction is 0.5000077444. RMSE uncertainties are whole-seed bootstrap SEs.",
        (
            "Each prefix uses the same classic posterior. Calls and goals for "
            "resumed stages are cumulative."
        ),
        "The 95% interval for Ph90 RMSE minus classic RMSE is paired across trees:",
        "",
    ]
)
for stage, cases in summary.items():
    for case, r in cases.items():
        low, high = r["paired_ph90_minus_classic_rmse_ci95"]
        lines.append(f"- {case.upper()} at {stage}: [{low:+.5f}, {high:+.5f}].")
(report / "COMPARISON.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
