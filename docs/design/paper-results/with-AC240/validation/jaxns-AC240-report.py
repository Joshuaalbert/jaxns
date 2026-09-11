"""Audit A+C and append it to the frozen R/A/C/Cprime comparison."""

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path("/largedata/albert/jaxns-AC240-20260908")
REPORT = OUT / "report"
REPORT.mkdir(exist_ok=True)
BASELINE = Path(
    "/largedata/albert/jaxns-ss8-css8-20260908/report/summary.json"
)
summary = json.loads(BASELINE.read_text())
baseline_bytes = {v: json.dumps(r, sort_keys=True) for v, r in summary.items()}
sources = json.loads((OUT / "SOURCES.json").read_text())["AC240"]
CASES = {
    "basic_mvn": "G8",
    "weak_curved_mvn8": "CG8",
    "spike_slab": "SS8",
    "curved_spike_slab8": "CSS8",
}
indices = np.random.default_rng(20260908).integers(0, 30, (10000, 30))
assert json.loads((OUT / "FINISHED.json").read_text())["cells"] == 120
summary["AC240"] = {}
audit = dict(
    new_cells=0,
    source_commit=sources["commit"],
    source_tree=sources["src_tree"],
    baseline_summary_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
    bootstrap_seed=20260908,
    bootstrap_resamples=10000,
    first_goal_A_checks=[],
)
resources = {}
for case in CASES:
    root = OUT / "AC240"
    analyses, cores, hashes = [], [], {}
    for seed in range(30):
        cell = root / case / f"seed-{seed:02d}"
        core = json.loads((cell / "CORE.json").read_text())
        a = json.loads((cell / "ANALYSIS.json").read_text())
        manifest = json.loads((cell / "MANIFEST.json").read_text())
        assert core["seed"] == a["seed"] == manifest["seed"] == seed
        assert core["case"] == a["case"] == manifest["case"] == case
        assert core["expected_log_Z_uncert"] < 0.05
        assert (
            len(manifest["affinity"]) == 1
            and manifest["direction"] == "isotropic"
        )
        assert manifest["root_degree"] == 240 and manifest["shell_size"] == 80
        assert (
            manifest["retained_phantoms"] == 79
            and manifest["evidence_prefix_max"] == 72
        )
        assert (
            manifest["phantom_seeding"]
            and manifest["bootstrap_phantom_seeding"]
        )
        assert (
            manifest["allocation_target"] == "evidence_improving"
            and manifest["delta_K"] == 240
        )
        assert (
            manifest["bootstrap_target"] == "uniform"
            and manifest["bootstrap_delta_K"] == 1
        )
        assert manifest["case_revision"] == "ss8-css8-20260908"
        assert manifest["source_commit"] == sources["commit"]
        for name in ("cases.py", "references.py", "run.py", "prefix_sweep.py"):
            assert manifest["protocol_sha256"][name] == sources["files"][name]
        assert (cell / "state.pkl").is_file()
        assert a["mc_draws"] == 2048
        aroot = Path(summary["A240"][case]["path"])
        first = json.loads(
            (aroot / case / f"seed-{seed:02d}" / "CORE.json").read_text()
        )["goal_progress"][0]
        np.testing.assert_allclose(
            core["goal_progress"][0]["samples"],
            first["samples"],
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            core["goal_progress"][0]["uncertainty"],
            first["uncertainty"],
            rtol=1e-12,
        )
        for key in (
            "expected_log_Z_mean",
            "likelihood_evaluations",
            "root_out_degree",
        ):
            if key in first:
                np.testing.assert_allclose(
                    core["goal_progress"][0][key], first[key], rtol=1e-12
                )
        audit["first_goal_A_checks"].append(
            dict(case=case, seed=seed, matched=True)
        )
        if case in ("spike_slab", "curved_spike_slab8"):
            with np.load(cell / "classic_posterior.npz") as data:
                weights = np.exp(data["log_dp"])
                np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)
                np.testing.assert_allclose(
                    weights[data["is_spike"]].sum(), a["mode_mass"], atol=1e-12
                )
        with np.load(cell / "evidence_draws.npz") as data:
            draws = data["log_Z"]
            assert draws.shape == (2048, 10) and np.isfinite(draws).all()
            np.testing.assert_allclose(
                draws.mean(axis=0), a["log_Z_mean"], rtol=1e-12
            )
            np.testing.assert_allclose(
                draws.std(axis=0, ddof=1), a["log_Z_uncert"], rtol=1e-12
            )
        for name in (
            "CORE.json",
            "ANALYSIS.json",
            "MANIFEST.json",
            "evidence_draws.npz",
        ):
            hashes[str(cell / name)] = hashlib.sha256(
                (cell / name).read_bytes()
            ).hexdigest()
        analyses.append(a)
        cores.append(core)
        audit["new_cells"] += 1
    truth = summary["R240"][case]["truth_log_Z"]
    np.testing.assert_allclose(
        [a["truth_log_Z"] for a in analyses], truth, atol=1e-10
    )
    errors = np.array([a["log_Z_mean"] for a in analyses]) - truth
    calls = np.array([c["likelihood_evaluations"] for c in cores])
    goals = np.array([len(c["goal_progress"]) for c in cores])
    boot = np.sqrt(np.mean(errors[indices] ** 2, axis=1))
    r = dict(
        path=str(root),
        source_commits=[sources["commit"]],
        truth_log_Z=truth,
        errors=errors.tolist(),
        rmse=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        rmse_se=boot.std(axis=0, ddof=1).tolist(),
        bias=errors.mean(axis=0).tolist(),
        bias_sd=errors.std(axis=0, ddof=1).tolist(),
        calls=calls.tolist(),
        mean_calls=float(calls.mean()),
        sd_calls=float(calls.std(ddof=1)),
        goals=goals.tolist(),
        mean_goals=float(goals.mean()),
        median_core_seconds=float(
            np.median([c["run_seconds"] for c in cores])
        ),
        goal_sigma=[c["expected_log_Z_uncert"] for c in cores],
        record_sha256=hashes,
    )
    if case in ("spike_slab", "curved_spike_slab8"):
        mass = np.array([a["mode_mass"] for a in analyses])
        truth = analyses[0]["mode_mass_truth"]
        e = mass - truth
        np.testing.assert_allclose(
            truth, summary["R240"][case]["posterior"]["truth"], atol=1e-12
        )
        r["posterior"] = dict(
            mass_by_seed=mass.tolist(),
            truth=truth,
            mean=float(mass.mean()),
            rmse=float(np.sqrt(np.mean(e**2))),
            rmse_se=float(
                np.sqrt(np.mean(e[indices] ** 2, axis=1)).std(ddof=1)
            ),
            min=float(mass.min()),
            max=float(mass.max()),
        )
    summary["AC240"][case] = r

# Verify saved baseline summaries and their raw input files remain unchanged.
for variant, original in baseline_bytes.items():
    assert json.dumps(summary[variant], sort_keys=True) == original
    for r in summary[variant].values():
        for path, digest in r["record_sha256"].items():
            assert (
                hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            )
# Gather timing and memory for all policies from original core records.
for variant, cases in summary.items():
    resources[variant] = {}
    for case, r in cases.items():
        root = Path(r["path"])
        cores = [
            json.loads(
                (root / case / f"seed-{seed:02d}" / "CORE.json").read_text()
            )
            for seed in range(30)
        ]
        analyses = [
            json.loads(
                (
                    root / case / f"seed-{seed:02d}" / "ANALYSIS.json"
                ).read_text()
            )
            for seed in range(30)
        ]
        resources[variant][case] = dict(
            core_seconds=[c["run_seconds"] for c in cores],
            core_rss_bytes=[c["peak_rss_bytes"] for c in cores],
            analysis_seconds=[a["analysis_seconds"] for a in analyses],
            analysis_rss_bytes=[a["peak_rss_bytes"] for a in analyses],
            classic_samples=[c["classic_samples"] for c in cores],
            state_file_bytes=[
                (root / case / f"seed-{seed:02d}" / "state.pkl").stat().st_size
                for seed in range(30)
            ],
        )
active = {}
completed = 0
max_active = 0
for line in (OUT / "dispatch.log").read_text().splitlines():
    if not line.startswith("{"):
        continue
    row = json.loads(line)
    if row["event"] == "start":
        assert row["cpu"] not in active
        active[row["cpu"]] = row["pid"]
        max_active = max(max_active, len(active))
        assert max_active <= 60
    elif row["event"] == "finish":
        assert (
            active.pop(row["cpu"]) == row["pid"]
            and row["success"]
            and row["exit_code"] == 0
        )
        completed += 1
assert not active and completed == 240
audit.update(
    completed_phases=completed,
    max_active_workers=max_active,
    baseline_summaries_and_records_unchanged=True,
)
for name, value in [
    ("summary.json", summary),
    ("resources.json", resources),
    ("AUDIT.json", audit),
]:
    (REPORT / name).write_text(json.dumps(value, indent=2) + "\n")

lines = [
    "# R/A/C/C′/A+C at 240 roots",
    "",
    "Thirty seeds per problem and variant. A+C combines all-phantom seeds with C’s evidence-improving allocation after a first uniform goal identical to A’s. R/A/C/C′ retain their previous measurements. SS8 uses covariance multiplier 0.43 and correlation 0.031; CSS8 is its component-centered β=0.4 shear. All runs use CPU float64, isotropic directions, width 80, 80 slice transitions and classic expected log-evidence uncertainty<0.05.",
    "",
    "Evidence analysis uses 2048 paired shrinkage draws and prefixes 0, 8, ..., 72. A and A+C retain all 79 phantoms for seed selection. Mode RMSE is measured once per classic posterior and does not depend on phantom prefix. The paper tables remain R240.",
    "",
    "| Problem | Variant | Classic RMSE | Ph72 RMSE | Mode RMSE | Mean calls (M) | Mean goals |",
    "|---|---|---:|---:|---:|---:|---:|",
]


def label(v):
    return {"Cprime240": "C′240", "AC240": "A+C240"}.get(v, v)


for case, problem in CASES.items():
    for v, r in ((v, cases[case]) for v, cases in summary.items()):
        mode = f"{r['posterior']['rmse']:.4f}" if "posterior" in r else "—"
        lines.append(
            f"| {problem} | {label(v)} | {r['rmse'][0]:.4f} ± {r['rmse_se'][0]:.4f} | {r['rmse'][9]:.4f} ± {r['rmse_se'][9]:.4f} | {mode} | {r['mean_calls'] / 1e6:.3f} | {r['mean_goals']:.2f} |"
        )
lines += [
    "",
    "Evidence errors are log-Z RMSE ± bootstrap SE (10,000 seed resamples). Mode reference is the component evidence fraction; hard component assignment can introduce a small overlap discrepancy.",
    "",
    "## A+C policy contrasts",
    "",
    "Differences are A+C minus the reference; negative RMSE differences favor A+C. Intervals are paired-seed bootstrap 95% intervals, unadjusted for multiple comparisons. An interval crossing zero is unresolved, not evidence of equivalence. Calls and core-time ratios compare means. Core times include compilation and different cohort host loads.",
    "",
    "| Problem | Reference | Classic RMSE difference [95% CI] | Ph72 RMSE difference [95% CI] | Mode RMSE difference [95% CI] | Calls ratio [95% CI] |",
    "|---|---|---|---|---|---|",
]
contrasts = {}
for case, problem in CASES.items():
    r = summary["AC240"][case]
    e = np.array(r["errors"])
    boot = np.sqrt(np.mean(e[indices] ** 2, axis=1))
    for other in ("R240", "A240", "C240", "Cprime240"):
        b = summary[other][case]
        be = np.array(b["errors"])
        d = np.array(r["rmse"]) - b["rmse"]
        ci = np.quantile(
            boot - np.sqrt(np.mean(be[indices] ** 2, axis=1)),
            [0.025, 0.975],
            axis=0,
        )
        calls = np.array(r["calls"])
        bcalls = np.array(b["calls"])
        ratio = calls.mean() / bcalls.mean()
        rci = np.quantile(
            calls[indices].mean(axis=1) / bcalls[indices].mean(axis=1),
            [0.025, 0.975],
        )
        record = dict(
            rmse_difference=d.tolist(),
            rmse_difference_ci95=ci.tolist(),
            calls_ratio=float(ratio),
            calls_ratio_ci95=rci.tolist(),
        )
        mode = "—"
        if "posterior" in r:
            me = (
                np.array(r["posterior"]["mass_by_seed"])
                - r["posterior"]["truth"]
            )
            bm = (
                np.array(b["posterior"]["mass_by_seed"])
                - b["posterior"]["truth"]
            )
            md = r["posterior"]["rmse"] - b["posterior"]["rmse"]
            mci = np.quantile(
                np.sqrt(np.mean(me[indices] ** 2, axis=1))
                - np.sqrt(np.mean(bm[indices] ** 2, axis=1)),
                [0.025, 0.975],
            )
            record.update(mode_rmse_difference=md, mode_rmse_ci95=mci.tolist())
            mode = f"{md:+.4f} [{mci[0]:+.4f}, {mci[1]:+.4f}]"
        contrasts[f"AC240-{other}/{case}"] = record
        entries = [
            f"{d[i]:+.4f} [{ci[0, i]:+.4f}, {ci[1, i]:+.4f}]" for i in (0, 9)
        ]
        lines.append(
            f"| {problem} | {label(other)} | "
            + " | ".join(entries)
            + f" | {mode} | {ratio:.3f} [{rci[0]:.3f}, {rci[1]:.3f}] |"
        )
lines += [
    "",
    "## Runtime and memory",
    "",
    "Times include compilation. RSS is peak resident process memory; saved-state size is the serialized trimmed State. Cohorts ran under different host loads, so likelihood counts are the more controlled measure of work.",
    "",
    "| Problem | Variant | Median core (s) | Median analysis (s) | Median / max core RSS (GiB) | Median classics | Median state (MiB) |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for case, problem in CASES.items():
    for v in summary:
        r = resources[v][case]
        rss = np.array(r["core_rss_bytes"]) / 2**30
        lines.append(
            f"| {problem} | {label(v)} | {np.median(r['core_seconds']):.1f} | {np.median(r['analysis_seconds']):.1f} | {np.median(rss):.3f} / {rss.max():.3f} | {np.median(r['classic_samples']):.0f} | {np.median(r['state_file_bytes']) / 2**20:.1f} |"
        )
(REPORT / "COMPARISON.md").write_text("\n".join(lines) + "\n")
(REPORT / "contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
print("\n".join(lines[:29]))
