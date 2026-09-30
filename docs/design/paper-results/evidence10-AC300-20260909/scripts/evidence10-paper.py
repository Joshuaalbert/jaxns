"""Replace retired paper experiments with the completed A+C300 cohorts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/largedata/albert/jaxns-evidence10-AC300-20260909")
REPORT = ROOT / "report"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--completed-only", action="store_true")
parser.add_argument("--report-dir", type=Path, default=REPORT)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
REPORT = args.report_dir
cancelled = (REPORT / "CANCELLED.json").exists() or (ROOT / "CANCELLED.json").exists()
summary = json.loads((REPORT / "SUMMARY.json").read_text())
finished = (
    json.loads((ROOT / "FINISHED.json").read_text())
    if (ROOT / "FINISHED.json").exists()
    else {"complete": False, "stages": list(summary)}
)
assert finished["complete"] or args.completed_only
baseline = summary["0.05"]
assert all(case in baseline for case in ("g10", "cg10", "ss10"))
stages = [stage for stage in ("0.05", "0.02", "0.01", "0.005") if stage in summary]
assert all(summary[stage]["ss10"]["seeds"] == list(range(30)) for stage in stages)
last = stages[-1]
old = Path(__file__).with_name("evidence10-template.tex").read_text()
problems = Path(__file__).with_name("evidence10-problems.tex").read_text()
tables = "\n".join(
    (REPORT / f"{case}_0p05_table.tex").read_text() for case in ("g10", "cg10", "ss10")
)
for stage in stages[1:]:
    tables += "\n" + (REPORT / f"ss10_{stage.replace('.', 'p')}_table.tex").read_text()
results = ["\\clearpage\n\\subsection{Results and limitations}\n"]
for case in ("g10", "cg10", "ss10"):
    r = baseline[case]
    change = 100 * (r["rmse"][-1] / r["rmse"][0] - 1.0)
    low, high = r["paired_ph90_minus_classic_rmse_ci95"]
    results.append(
        f"For {case.upper()}, the pre-specified $9D=90$ phantom prefix changes log-evidence RMSE "
        f"from ${r['rmse'][0]:.4f}$ to ${r['rmse'][-1]:.4f}$, "
        f"a ${abs(change):.1f}\\%$ {'reduction' if change < 0 else 'increase'}.\n"
        f"The paired bootstrap $95\\%$ interval for phantom minus classic RMSE is "
        f"$[{low:+.4f},{high:+.4f}]$.\n"
    )
    if high < 0:
        results.append("This paired comparison resolves lower phantom-conditioned RMSE.\n")
    elif low > 0:
        results.append("This paired comparison resolves higher phantom-conditioned RMSE.\n")
    else:
        results.append("The 30-seed comparison does not resolve an RMSE difference.\n")
g, cg = baseline["g10"], baseline["cg10"]
results.append(
    "Point accuracy and interval calibration require separate assessment. "
    f"At Ph90, empirical $95\\%$ interval coverage is "
    f"${100 * g['coverage95'][-1]:.0f}\\%$ for G10 and "
    f"${100 * cg['coverage95'][-1]:.0f}\\%$ for CG10, compared with "
    f"${100 * g['coverage95'][0]:.0f}\\%$ and ${100 * cg['coverage95'][0]:.0f}\\%$ "
    "for classic shrinkage. These are finite-cohort coverage measurements.\n"
)
results.append(r"""
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{images/evidence_accuracy.pdf}
    \caption{Evidence accuracy on the same 30 A+C300 trees per problem at classic uncertainty
below $0.05$. Points show empirical RMSE with bootstrap SE; dashed lines show mean reported
shrinkage SD. Adding phantom conditioning uses no further likelihood evaluations and leaves the
classic posterior unchanged.}
    \label{fig:evidence_accuracy}
\end{figure*}

\subsection{Spike recovery and conditional certainty}
""")
p = baseline["ss10"]["posterior"]
r = baseline["ss10"]
results.append(
    f"At the $0.05$ stop, SS10 has mean classic spike mass ${p['mean']:.5f}$ "
    f"against the reference $0.5000077444$, with mass RMSE ${p['rmse']:.4f}$ "
    f"and seed range $[{p['min']:.5f},{p['max']:.5f}]$.\n"
    f"There are ${p['seeds_below_10_percent']}$ of $30$ seeds with less than $10\\%$ spike mass.\n"
    f"The mean reported log-evidence SD changes from ${r['mean_reported_sd'][0]:.4f}$ "
    f"to ${r['mean_reported_sd'][-1]:.4f}$ under Ph90, while central $95\\%$ "
    f"interval coverage changes from ${100 * r['coverage95'][0]:.0f}\\%$ "
    f"to ${100 * r['coverage95'][-1]:.0f}\\%$.\n"
)
results.append(r"""
The uncertainty ladder separates posterior recovery from precision conditional on the
represented tree.
Every stage uses the same seed identities and retains all preceding sampling work.
Table~\ref{tab:ss10_recovery} reports classic posterior recovery independently of phantom
prefix; Table~\ref{tab:ss10_ladder_evidence} compares the evidence estimators and their cost.

\begin{table*}[t]
\centering
\caption{SS10 classic posterior recovery across the matched uncertainty ladder, 30 seeds per
stage. The reference spike fraction is $0.5000077444$. RMSE errors are bootstrap SEs. Recovery
requires RMSE at most $0.05$ and no absolute seed error above $0.15$; the rule was fixed before
the tighter runs.}
\label{tab:ss10_recovery}
\begin{tabular}{lrrrr}
\toprule
Stop & Mean spike mass & Spike-mass RMSE $\pm$ SE & Seed range & Recovered\\
\midrule
""")
for stage in stages:
    p = summary[stage]["ss10"]["posterior"]
    results.append(
        f"${stage}$ & {p['mean']:.5f} & {p['rmse']:.4f} $\\pm$ {p['rmse_se']:.4f} "
        f"& [{p['min']:.4f}, {p['max']:.4f}] & {'Yes' if p['recovered'] else 'No'} " + r"\\" + "\n"
    )
results.append(r"""\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\scriptsize
\caption{SS10 paired evidence accuracy and cumulative effort across the uncertainty ladder. RMSE
errors are bootstrap SEs; reported SDs are means over 30 trees. Likelihood calls and goals
include the work from all preceding stages. Classic and Ph90 reductions use the same completed
tree at each stop.}
\label{tab:ss10_ladder_evidence}
\begin{tabular}{lrrrrrr}
\toprule
Stop & Classic RMSE $\pm$ SE & Ph90 RMSE $\pm$ SE & Classic SD & Ph90 SD & Calls ($10^6$) & Goals\\
\midrule
""")
for stage in stages:
    r = summary[stage]["ss10"]
    results.append(
        f"${stage}$ & {r['rmse'][0]:.4f} $\\pm$ {r['rmse_se'][0]:.4f} "
        f"& {r['rmse'][-1]:.4f} $\\pm$ {r['rmse_se'][-1]:.4f} "
        f"& {r['mean_reported_sd'][0]:.4f} & {r['mean_reported_sd'][-1]:.4f} "
        f"& {r['mean_calls'] / 1e6:.3f} & {r['mean_goals']:.2f} " + r"\\" + "\n"
    )
results.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
final = summary[last]["ss10"]
p = final["posterior"]
if cancelled:
    results.append(
        f"The tightest completed stop is ${last}$; neither completed stage meets "
        "the spike-recovery criterion. Further sampling was cancelled because "
        "of its computational cost. Partial $0.01$ checkpoints and the unstarted "
        "$0.005$ stage are excluded.\n"
    )
elif not finished["complete"]:
    results.append(
        f"The tightest completed stop reported here is ${last}$. "
        "The remaining uncertainty-ladder runs are in progress; their results "
        "are not included in these tables.\n"
    )
elif p["recovered"]:
    results.append(
        f"The pre-specified spike-recovery criterion is first met at uncertainty ${last}$.\n"
    )
else:
    results.append(
        f"The spike-recovery criterion is not met even at the final uncertainty ${last}$.\n"
        "Tighter reported uncertainty therefore does not solve the "
        "component-mass error in this experiment.\n"
    )
low, high = final["paired_ph90_minus_classic_rmse_ci95"]
results.append(
    f"At the tightest reported stage, Ph90 minus classic evidence RMSE is "
    f"${final['rmse'][-1] - final['rmse'][0]:+.4f}$, with paired bootstrap "
    f"$95\\%$ interval $[{low:+.4f},{high:+.4f}]$.\n"
)
if high < 0:
    results.append("The paired comparison resolves lower phantom-conditioned RMSE at this stage.\n")
elif low > 0:
    results.append(
        "The paired comparison resolves higher phantom-conditioned RMSE at this stage.\n"
    )
else:
    results.append(
        "This finite-seed comparison does not resolve a difference in RMSE at this stage.\n"
    )
if "0.02" in summary:
    earlier = baseline["ss10"]
    tighter = summary["0.02"]["ss10"]
    results.append(
        f"Tightening the classic target from $0.05$ to $0.02$ uses "
        f"${tighter['mean_calls'] / earlier['mean_calls']:.2f}$ times the cumulative "
        f"mean likelihood calls. Mean spike mass increases from "
        f"${earlier['posterior']['mean']:.3f}$ to ${tighter['posterior']['mean']:.3f}$, "
        f"and spike-mass RMSE falls from ${earlier['posterior']['rmse']:.3f}$ to "
        f"${tighter['posterior']['rmse']:.3f}$. Nevertheless, "
        f"${tighter['posterior']['seeds_below_10_percent']}$ of $30$ seeds still "
        "assign less than $10\\%$ mass to the spike, whose reference mass is "
        "$50.0008\\%$. More work improves the represented component masses "
        "without resolving the geometry sufficiently for accurate inference.\n"
    )

# Whole-tree pairing is retained for bias, just as in the audited RMSE contrast.
indices = np.random.default_rng(20260909).integers(0, 30, (10000, 30))
contrasts = {}
bias_table = [r"""
\begin{table*}[t]
\centering
\scriptsize
\caption{SS10 bias and interval calibration on the same 30 trees at each stop.
Bias is mean log-evidence error. The absolute-bias contrast is Ph90 minus classic;
brackets give its paired bootstrap $95\%$ interval. Coverage counts how many
central $95\%$ shrinkage intervals contain the reference evidence.}
\label{tab:ss10_bias_coverage}
\begin{tabular}{lrrrrr}
\toprule
Stop & Classic bias & Ph90 bias & $\Delta|\mathrm{bias}|$ [95\% interval]
& Classic coverage & Ph90 coverage\\
\midrule
"""]
for stage in stages:
    r = summary[stage]["ss10"]
    errors = np.asarray(r["errors"])  # [30,prefix]
    assert errors.shape == (30, 10)
    bias = errors.mean(axis=0)  # [prefix]
    np.testing.assert_allclose(bias, r["bias"], rtol=0, atol=1e-12)
    boot_bias = errors[indices].mean(axis=1)  # [bootstrap,prefix]
    signed_delta = boot_bias[:, -1] - boot_bias[:, 0]
    absolute_delta = np.abs(boot_bias[:, -1]) - np.abs(boot_bias[:, 0])
    low, high = np.quantile(absolute_delta, [0.025, 0.975])
    absolute_point = float(abs(bias[-1]) - abs(bias[0]))
    coverage = np.rint(30 * np.asarray(r["coverage95"])).astype(int)
    contrasts[stage] = dict(
        signed_bias_ph90_minus_classic=float(bias[-1] - bias[0]),
        signed_bias_ci95=np.quantile(signed_delta, [0.025, 0.975]).tolist(),
        absolute_bias_ph90_minus_classic=absolute_point,
        absolute_bias_ci95=[float(low), float(high)],
        rmse_ph90_minus_classic=r["rmse"][-1] - r["rmse"][0],
        rmse_ci95=r["paired_ph90_minus_classic_rmse_ci95"],
        classic_coverage_count=int(coverage[0]),
        ph90_coverage_count=int(coverage[-1]),
    )
    bias_table.append(
        f"${stage}$ & {bias[0]:+.4f} & {bias[-1]:+.4f} "
        f"& {absolute_point:+.4f} [{low:+.4f}, {high:+.4f}] "
        f"& {coverage[0]}/30 & {coverage[-1]}/30 " + r"\\" + "\n"
    )
bias_table.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
bias_table = "\n".join(bias_table)
(REPORT / "SS10_BIAS_TABLE.tex").write_text(bias_table)
(REPORT / "SS10_CONTRASTS.json").write_text(json.dumps(dict(
    summary_sha256=hashlib.sha256((REPORT / "SUMMARY.json").read_bytes()).hexdigest(),
    bootstrap_seed=20260909,
    bootstrap_resamples=10000,
    definition="Ph90 minus classic; absolute bias is absolute mean error, not MAE",
    stages=contrasts,
), indent=2, allow_nan=False) + "\n")
results.append(bias_table)
results.append(r"""
At both completed stops, the classic and Ph90 estimates share a large negative
bias (Table~\ref{tab:ss10_bias_coverage}). The absolute cohort bias is slightly
smaller with Ph90, but both paired intervals include zero, as do the RMSE
contrasts. These cohorts show no resolved additional bias or RMSE from phantom
conditioning; they do not establish equivalence or a general no-harm guarantee.

The limitation is clearest in interval calibration. At $0.02$, the mean
reported SD falls from $0.0199$ to $0.0059$ with Ph90 while empirical RMSE
remains above $0.52$. Classic intervals cover the reference in only $2/30$
trees and Ph90 intervals in $0/30$, compared with $3/30$ and $1/30$ at $0.05$.
Thus the unresolved component-mass error already affects classic inference;
conditioning on more phantom information narrows uncertainty within that
misrepresented tree without correcting the exploration error. Interpreting
this conditional precision as total accuracy gives a false sense of certainty.
Representative constrained-prior exploration and recovery of component masses
are prerequisites for interpreting the gain, and interval calibration still
requires independent checks even when point accuracy improves.
""")
results.append(r"""
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{images/ss10_recovery_ladder.pdf}
    \caption{SS10 uncertainty ladder. Left: classic spike mass for each of the 30 matched seeds,
with the exact mass dashed. Middle: empirical log-evidence RMSE and mean reported shrinkage SD
for classic and Ph90 conditioning. Right: cumulative mean likelihood calls. Connecting lines
identify continued trees rather than independent runs.}
    \label{fig:ss10_recovery_ladder}
\end{figure*}

Seed $0$ at the baseline stop reproduces the earlier exploratory run: $18$ goals,
$16.574\times10^6$ likelihood calls, and $0.2654\%$ classic spike mass against $50.0008\%$.
Spike-dominated classic and retained phantom samples were present during its first goal, so this
error is not equivalent to never visiting the spike.
The classic shrinkage mean is $-24.7449$ with SD $0.0494$; Ph90 gives $-24.8583$ with SD
$0.0131$, close to the slab-only log evidence $-24.8491$.
Figure~\ref{fig:ss10_corner} shows that baseline posterior.

\begin{figure*}[p]
    \centering
    \includegraphics[width=\textwidth]{images/ss10_corner.pdf}
    \caption{SS10, seed $0$, A+C300 at classic expected uncertainty below $0.05$. Blue panels
use classic posterior weights; orange diagonal curves are exact finite-box mixture marginals.
The recovered spike mass is $0.2654\%$ against $50.0008\%$. Pair panels use weighted histograms
with 68\% and 95\% contour levels after 0.6-bin smoothing. Phantom states enter seed selection
but are not posterior samples.}
    \label{fig:ss10_corner}
\end{figure*}

These tests assess evidence conditioning on completed trees, not a guarantee of stationary
exploration or correct component allocation.
Phantom seeding and evidence-improving allocation are fixed across the paired reductions, so the
prefix comparison does not isolate their separate effects.
Evidence-only phantom conditioning cannot repair a missing mode in the classic posterior.

""")
start = old.index("\\subsection{Known-evidence problems}")
end = old.index("\\section{Posterior recovery}")
paper = old[:start] + problems + tables + "\n".join(results) + old[end:]
start = paper.index("\\subsection{Repository spike--slab (SS10)}")
end = paper.index("\\section{Conclusions}", start)
paper = paper[:start] + paper[end:]
paper = paper.replace(
    (
        "We next examine three ten-dimensional posterior targets with ESS "
        "stopping, followed by a supplementary spike--slab test with "
        "evidence-uncertainty stopping, using the earlier sampler "
        "comparison's repository definitions."
    ),
    (
        "We next examine three ten-dimensional posterior targets with ESS "
        "stopping, using the earlier sampler comparison's repository "
        "definitions."
    ),
)
paper = paper.replace(
    (
        "These are exploratory single-seed experiments with different seed"
        " pools, allocation rules, and stopping criteria from the R240 "
        "evidence tables. The following mixed-allocation protocol applies "
        "to Eggbox, Rosenbrock, and Rastrigin; the SS10 protocol is "
        "specified separately in Section~\\ref{sec:ss10}."
    ),
    (
        "These exploratory single-seed experiments share all-phantom seed "
        "eligibility with the evidence tables, but use different "
        "allocation weights and stopping criteria. The following "
        "mixed-allocation protocol applies to Eggbox, Rosenbrock, and "
        "Rastrigin."
    ),
)
paper = paper.replace(
    (
        "The R240 baseline uses classic seed points only.\nThe "
        "posterior-recovery intervention in "
        "Section~\\ref{sec:posterior_recovery} additionally retains every "
        "phantom unit-prior coordinate and extends the existing "
        "seed-selection population with those point identities."
    ),
    (
        "The experiments in Sections~\\ref{sec:experiments} "
        "and~\\ref{sec:posterior_recovery} retain every phantom unit-prior "
        "coordinate and extend the existing seed-selection population with"
        " those point identities (intervention A)."
    ),
)
changes = [100 * (1 - baseline[c]["rmse"][-1] / baseline[c]["rmse"][0]) for c in ("g10", "cg10")]
if min(changes) > 0:
    claim = (
        "On G10 and CG10, conditioning on 90 phantoms reduces point-estimate "
        f"evidence RMSE by {changes[0]:.1f}\\% and {changes[1]:.1f}\\%, "
        "respectively, without additional likelihood evaluations."
    )
else:
    claim = (
        "Thirty-seed G10, CG10, and SS10 experiments compare empirical "
        "evidence error with the conditional uncertainty reported by "
        "phantom shrinkage."
    )
resolved = [
    case.upper() for case in ("g10", "cg10")
    if baseline[case]["paired_ph90_minus_classic_rmse_ci95"][1] < 0
]
if resolved == ["G10"]:
    claim += " Only the G10 reduction is resolved by the paired 30-seed comparison."
paper = paper.replace(
    (
        "On the four R240 benchmark problems, the leading 72 phantoms "
        "reduce point-estimate evidence RMSE by 17\\% to 52\\% without "
        "additional likelihood evaluations."
    ),
    claim
    + (
        "\nOn SS10, continuing 30 matched trees from classic uncertainty $0.05$ "
        "to $0.02$ improves spike-mass recovery but does not meet the recovery "
        "criterion. Phantom conditioning shows no resolved additional evidence "
        "bias or RMSE, while its narrower conditional intervals have worse "
        "coverage of the reference evidence."
    ),
)
start = paper.index("Without additional likelihood effort,")
end = paper.index("Phantom conditioning is therefore", start)
conclusion = claim + "\n"
if cancelled:
    conclusion += (
        "The completed SS10 $0.05$ and $0.02$ cohorts both fail the spike-recovery "
        "criterion, despite improved component-mass estimates with more work. "
        "Further stages were cancelled because of computational cost. Neither "
        "cohort resolves additional evidence bias or RMSE from phantom "
        "conditioning, but narrower intervals worsen empirical coverage. "
        "The geometry error already affects classic inference and is not "
        "repaired by conditioning on phantoms from the same represented tree.\n"
    )
elif not finished["complete"]:
    conclusion += (
        f"The tightest completed SS10 stop reported here is ${last}$. "
        "Further matched uncertainty-ladder runs are in progress; the "
        "current tables contain only completed 30-seed cohorts.\n"
    )
elif p["recovered"]:
    conclusion += (
        "The matched SS10 uncertainty ladder reaches the stated posterior "
        f"spike-recovery criterion at ${last}$; the paired evidence comparison "
        "assesses phantom benefit separately.\n"
    )
else:
    conclusion += (
        "The SS10 uncertainty ladder does not meet the stated spike-recovery "
        f"criterion by ${last}$, despite every run reaching its requested "
        "classic uncertainty.\n"
    )
conclusion += (
    "Classic posterior mass is shared by all phantom-prefix "
    "reductions. The ESS-target experiments likewise reach their "
    "requested values while retaining mode-weight and tail-recovery "
    "errors. These operational stopping diagnostics therefore require "
    "independent posterior and evidence checks.\n"
)
paper = paper[:start] + conclusion + paper[end:]
assert not any(retired in paper for retired in ("G8", "CG8", "SS8", "CSS8", "R240"))
output = args.output or REPORT / "paper.tex"
output.write_text(paper)
print("Wrote", output)
