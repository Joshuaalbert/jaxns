"""Audit installed evidence draws and paper tables without loading sampler states."""

import hashlib
import json
from pathlib import Path

import numpy as np

report = Path(__file__).resolve().parent.parent
paper = report.parents[1] / "paper.tex"
summary = json.loads((report / "SUMMARY.json").read_text())
audit = json.loads((report / "AUDIT.json").read_text())
contrasts = json.loads((report / "SS10_CONTRASTS.json").read_text())
snapshot = json.loads((report / "SNAPSHOT.json").read_text())
assert snapshot["cells"] == audit["cells"] == 120
assert snapshot["ladder_status"] == "cancelled"
assert not snapshot["full_ladder_complete"]
assert list(summary) == ["0.05", "0.02"]
assert list(summary["0.05"]) == ["g10", "cg10", "ss10"]
assert list(summary["0.02"]) == ["ss10"]
assert contrasts["summary_sha256"] == hashlib.sha256(
    (report / "SUMMARY.json").read_bytes()
).hexdigest()
indices = np.random.default_rng(20260909).integers(0, 30, (10000, 30))
files_verified = 0
cohorts = []
for stage, cases in summary.items():
    for case, row in cases.items():
        assert row["seeds"] == list(range(30))
        errors, deviations, coverages, calls, goals, masses = [], [], [], [], [], []
        for seed in range(30):
            relative = Path(stage) / case / f"seed-{seed:02d}"
            cell = report / "records" / relative
            for name in (
                "MANIFEST.json", "CORE.json", "ANALYSIS.json",
                "evidence_draws.npz", "progress.jsonl",
            ):
                digest = hashlib.sha256((cell / name).read_bytes()).hexdigest()
                assert digest == audit["record_sha256"][str(relative / name)]
                files_verified += 1
            core = json.loads((cell / "CORE.json").read_text())
            analysis = json.loads((cell / "ANALYSIS.json").read_text())
            assert core["complete"] and core["classic_log_Z_uncert"] < float(stage)
            assert core["seed"] == analysis["seed"] == seed
            assert analysis["prefix_sizes"] == list(range(0, 100, 10))
            with np.load(cell / "evidence_draws.npz") as saved:
                draws = saved["log_Z"]  # [2048,prefix]
            assert draws.shape == (2048, 10) and np.isfinite(draws).all()
            truth = analysis["reference"]["log_Z"]
            np.testing.assert_allclose(draws.mean(axis=0), analysis["log_Z_mean"])
            errors.append(draws.mean(axis=0) - truth)
            deviations.append(draws.std(axis=0, ddof=1))
            low, high = np.quantile(draws, [0.025, 0.975], axis=0)
            coverages.append((low <= truth) & (truth <= high))
            calls.append(core["likelihood_evaluations"])
            goals.append(core["iteration"])
            if case == "ss10":
                masses.append(analysis["classic_spike_mass"])
            if stage == "0.02":
                previous = report / "records/0.05/ss10" / f"seed-{seed:02d}/CORE.json"
                earlier = json.loads(previous.read_text())
                assert core["goal_progress"][:len(earlier["goal_progress"])] == (
                    earlier["goal_progress"]
                )
                assert core["likelihood_evaluations"] > earlier["likelihood_evaluations"]
        errors = np.asarray(errors)  # [30,prefix]
        boot_errors = errors[indices]  # [10000,30,prefix]
        boot_rmse = np.sqrt(np.mean(boot_errors ** 2, axis=1))
        paired_rmse = boot_rmse[:, -1] - boot_rmse[:, 0]
        expected = dict(
            errors=errors,
            bias=errors.mean(axis=0),
            bias_sd=errors.std(axis=0, ddof=1),
            rmse=np.sqrt(np.mean(errors ** 2, axis=0)),
            rmse_se=boot_rmse.std(axis=0, ddof=1),
            mean_reported_sd=np.mean(deviations, axis=0),
            coverage95=np.mean(coverages, axis=0),
            coverage_by_seed=coverages,
            calls=calls,
            goals=goals,
            mean_calls=np.mean(calls),
            mean_goals=np.mean(goals),
            paired_ph90_minus_classic_rmse_ci95=np.quantile(paired_rmse, [0.025, 0.975]),
        )
        for field, value in expected.items():
            np.testing.assert_allclose(value, row[field], rtol=1e-10, atol=1e-12)
        if case == "ss10":
            mass_error = np.asarray(masses) - row["posterior"]["truth"]
            np.testing.assert_allclose(masses, row["posterior"]["mass_by_seed"])
            np.testing.assert_allclose(
                np.sqrt(np.mean(mass_error ** 2)), row["posterior"]["rmse"],
            )
            assert not row["posterior"]["recovered"]
            assert np.sqrt(np.mean(mass_error ** 2)) > 0.05
            boot_bias = boot_errors.mean(axis=1)  # [10000,prefix]
            delta = np.abs(boot_bias[:, -1]) - np.abs(boot_bias[:, 0])
            np.testing.assert_allclose(
                np.quantile(delta, [0.025, 0.975]),
                contrasts["stages"][stage]["absolute_bias_ci95"],
                rtol=1e-10, atol=1e-12,
            )
        table = report / f"{case}_{stage.replace('.', 'p')}_table.tex"
        assert table.read_text().strip() in paper.read_text()
        cohorts.append(f"{case}/{stage}: 30 seeds")
assert (report / "SS10_BIAS_TABLE.tex").read_text().strip() in paper.read_text()
assert "in progress" not in paper.read_text()
result = dict(
    cells=120, files_verified=files_verified, cohorts=cohorts,
    evidence_statistics_from_saved_draws=True, paired_bias_intervals_verified=True,
    matched_continuations_verified=True, full_prefix_tables_match=True,
    classic_posterior_mass_source="Previously audited ANALYSIS.json; no new state reduction",
    summary_sha256=contrasts["summary_sha256"],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(report / "SNAPSHOT_VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
