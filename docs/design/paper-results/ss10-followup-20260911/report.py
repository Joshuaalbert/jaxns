"""Rebuild the completed SS10 follow-up statistics and figure from saved records.

Run from any directory with conda run -n jaxns_py python <path>/report.py.
This reads compact records only; it never loads states or launches sampling.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

root = Path(__file__).resolve().parent
previous = root.parent / "evidence10-AC300-20260909"
truth = 0.5000077444258325
true_log_z = -24.15593480605331
indices = np.random.default_rng(20260909).integers(0, 30, (10000, 30))
paired_indices = np.random.default_rng(292).integers(0, 30, (20000, 30))
summary = {"stages": {}, "posterior_completed": []}
stage_records = {}
for stage in ("0.05", "0.02", "0.015"):
    cores, masses = [], []
    for seed in range(30):
        if stage == "0.015":
            cell = root / "records" / stage / f"seed-{seed:02d}"
        else:
            cell = previous / "records" / stage / "ss10" / f"seed-{seed:02d}"
        core = json.loads((cell / "CORE.json").read_text())
        assert core["seed"] == seed and core["complete"]
        assert core["classic_log_Z_uncert"] < float(stage)
        if stage == "0.015":
            mass = core["accuracy"]["spike_mass"]
            assert core["accuracy"]["log_Z"] == core["classic_log_Z"]
        else:
            mass = json.loads((cell / "ANALYSIS.json").read_text())["classic_spike_mass"]
        cores.append(core)
        masses.append(mass)
    mass = np.asarray(masses)  # [30], classic-weighted component responsibilities
    error = np.asarray([c["classic_log_Z"] - true_log_z for c in cores])
    sd = np.asarray([c["classic_log_Z_uncert"] for c in cores])
    mass_error = mass - truth
    bootstrap_rmse = np.sqrt(np.mean(mass_error[indices] ** 2, axis=1))
    stage_records[stage] = dict(cores=cores, mass=mass, error=error)
    summary["stages"][stage] = dict(
        seeds=list(range(30)), mean_spike_mass=float(mass.mean()),
        spike_mass_rmse=float(np.sqrt(np.mean(mass_error ** 2))),
        spike_mass_rmse_se=float(bootstrap_rmse.std(ddof=1)),
        spike_mass_range=[float(mass.min()), float(mass.max())],
        max_abs_spike_mass_error=float(np.max(np.abs(mass_error))),
        recovered=bool(np.sqrt(np.mean(mass_error ** 2)) <= .05
                       and np.max(np.abs(mass_error)) <= .15),
        classic_expected_log_z_bias=float(error.mean()),
        classic_expected_log_z_rmse=float(np.sqrt(np.mean(error ** 2))),
        mean_expected_sd=float(sd.mean()),
        nominal_normal_95_coverage=int(np.sum(np.abs(error) <= 1.96 * sd)),
        mean_cumulative_calls=float(np.mean([c["likelihood_evaluations"] for c in cores])),
        mean_cumulative_goals=float(np.mean([c["iteration"] for c in cores])),
    )
old, new = stage_records["0.02"], stage_records["0.015"]
for seed, (before, after) in enumerate(zip(old["cores"], new["cores"], strict=True)):
    assert after["likelihood_evaluations"] - before["likelihood_evaluations"] == (
        after["added_likelihood_evaluations"]
    )
    assert after["iteration"] - before["iteration"] == after["added_goals"]
summary["paired_rmse_change_ci95"] = {}
for metric, earlier, later in (
    ("log_Z", old["error"], new["error"]),
    ("spike_mass", old["mass"] - truth, new["mass"] - truth),
):
    delta = (np.sqrt(np.mean(later[paired_indices] ** 2, axis=1))
             - np.sqrt(np.mean(earlier[paired_indices] ** 2, axis=1)))
    summary["paired_rmse_change_ci95"][metric] = np.quantile(delta, [.025, .975]).tolist()

stopped = json.loads((root / "POSTERIOR_STOPPED.json").read_text())
for seed in sorted(stopped["completed"]):
    core = json.loads((root / "records/posterior-ess2" / f"seed-{seed:02d}/CORE.json").read_text())
    before = new["cores"][seed]
    assert core["complete"] and core["allocation_target"] == "posterior_improving"
    assert core["initial_ess"] == before["accuracy"]["kish_ess"]
    assert core["ess_target"] == 2 * core["initial_ess"]
    assert core["accuracy"]["kish_ess"] >= core["ess_target"]
    summary["posterior_completed"].append(dict(
        seed=seed, initial_ess=core["initial_ess"], ess_ratio=core["ess_ratio"],
        before_spike_mass=before["accuracy"]["spike_mass"],
        after_spike_mass=core["accuracy"]["spike_mass"],
        added_goals=core["added_goals"],
        added_calls=core["added_likelihood_evaluations"],
    ))
assert len(summary["posterior_completed"]) == 7
assert len(stopped["queued_cancelled"]) == 18 and len(stopped["previously_active"]) == 5
summary["posterior_status"] = "Cancelled: 7 completed, 5 interrupted, 18 queued cancelled"
summary["interpretation"] = (
    "Classic expected evidence at all three stops; no new phantom-prefix reductions at 0.015. "
    "Posterior results are a completion-selected subset, not a 30-seed cohort estimate."
)
summary["bootstrap"] = dict(se_resamples=10000, se_seed=20260909,
                            paired_resamples=20000, paired_seed=292)
(root / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         r"\caption{SS10 classic expected-evidence diagnostics for the 30 matched continuations. "
         r"These use the deterministic classic expectation and its reported SD, rather than "
         r"Monte Carlo shrinkage means. Coverage uses the normal approximation "
         r"$\widehat{\log Z}\pm1.96\widehat\sigma^{\rm exp}_{\log Z}$; "
         r"it is distinct from central Monte Carlo interval coverage in the prefix tables. "
         r"Calls and goals are cumulative cohort means.}",
         r"\label{tab:ss10_classic_followup}", r"\begin{tabular}{lrrrrrr}",
         r"\toprule", r"Stop & Bias & RMSE & Mean SD & 95\% coverage & Calls ($10^6$) & Goals\\",
         r"\midrule"]
for stage in ("0.02", "0.015"):
    row = summary["stages"][stage]
    lines.append(
        f"${stage}$ & {row['classic_expected_log_z_bias']:.4f} & "
        f"{row['classic_expected_log_z_rmse']:.4f} & {row['mean_expected_sd']:.4f} & "
        f"{row['nominal_normal_95_coverage']}/30 & "
        f"{row['mean_cumulative_calls'] / 1e6:.3f} & {row['mean_cumulative_goals']:.2f} " + r"\\"
    )
lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
(root / "CLASSIC_TABLE.tex").write_text("\n".join(lines) + "\n")

plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
for seed in range(30):
    axes[0].plot(range(3), [stage_records[s]["mass"][seed] for s in stage_records],
                 color="#3478ad", alpha=.35, marker="o", markersize=3, linewidth=.8)
axes[0].set_xticks(range(3), ["0.05", "0.02", "0.015"])
axes[0].set_xlabel("Classic uncertainty target (30 matched seeds)")
axes[0].set_title("Evidence allocation: gradual recovery")
for row in summary["posterior_completed"]:
    axes[1].plot([1, row["ess_ratio"]], [row["before_spike_mass"], row["after_spike_mass"]],
                 marker="o", linewidth=1, label=f"Seed {row['seed']}")
axes[1].set_xticks([1, 2], ["1×", "2×"])
axes[1].set_xlabel("Classic Kish ESS / initial 0.015 ESS")
axes[1].set_title("Posterior allocation: 7 completed before cancellation")
axes[1].legend(fontsize=8, ncol=2, loc="lower right", frameon=False)
for ax in axes:
    ax.axhline(truth, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Classic posterior spike mass")
    ax.set_ylim(0, .9)
    ax.grid(alpha=.15)
for suffix in ("pdf", "png"):
    fig.savefig(root / f"ss10_followup.{suffix}", dpi=170)
plt.close(fig)

# Detect accidental changes to the installed raw evidence on later rebuilds.
paths = sorted((root / "records").rglob("*")) + [root / "POSTERIOR_STOPPED.json"]
digests = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in paths if p.is_file()}
manifest = root / "RECORD_SHA256.json"
if manifest.exists():
    assert json.loads(manifest.read_text()) == digests
else:
    manifest.write_text(json.dumps(digests, indent=2) + "\n")
print(json.dumps(summary, indent=2))
