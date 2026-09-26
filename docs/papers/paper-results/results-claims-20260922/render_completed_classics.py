"""Validate and render completed classic calculations from dorrie's records.

The input contains all thirty sampling records per cohort. Its evidence
summaries match a lognormal distribution to the first two linear-evidence
moments; they must not be labelled as Monte Carlo phantom-prefix reductions.
Bootstrap statistics were computed on dorrie using shared whole-seed indices.
"""

import json
from pathlib import Path

import numpy as np


def main() -> None:
    """Check cohort completeness, paired continuations, and reported metrics."""
    root = Path(__file__).resolve().parent
    data = json.loads((root / "INTERIM_CORE_WORKSTATION.json").read_text())
    names = [
        ("g10_e", "G10", "Evidence"),
        ("g10_uniform", "G10", "Uniform"),
        ("cg10_e", "CG10", "Evidence"),
        ("cg10_p_followup", "CG10", "Posterior continuation"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        (
            r"\caption{Completed classic calculations for the $(3,-3)$ Gaussian "
            r"models, $30$ seeds per row. Evidence RMSE and mean reported SD use "
            r"the moment-matched summary in Eq.~\ref{eq:classic_moment_summary}; "
            r"RMSE uncertainties are paired-bootstrap SEs. Calls and Kish ESS "
            r"are mean $\pm$ across-run SD. Calls include discovery and, for "
            r"posterior continuation, the preceding evidence run.}"
        ),
        r"\label{tab:completed_classic_cohorts}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Case & Allocation & RMSE & Mean SD & Calls ($10^6$) "
            r"& Kish ESS ($10^3$)\\"
        ),
        r"\midrule",
    ]
    for arm, case, allocation in names:
        cohort = data["arms"][arm]
        rows = cohort["records"]
        summary = cohort["summary"]
        if [r["seed"] for r in rows] != list(range(30)):
            raise ValueError(f"Incomplete seed cohort: {arm}")
        if summary["n"] != 30:
            raise ValueError(f"Incorrect summary sample count: {arm}")
        for row in rows:
            if row["reference"]["mean"] != [3., -3.] + [0.] * 8:
                raise ValueError(f"Different Gaussian model in {arm}")
        calls = np.array([r["likelihood_evaluations"] for r in rows])
        ess = np.array([r["classic_kish_ess"] for r in rows])
        errors = np.array([
            r["classic_log_Z"] - r["reference"]["log_Z"] for r in rows
        ])
        np.testing.assert_allclose(
            [calls.mean(), calls.std(ddof=1), ess.mean(), ess.std(ddof=1),
             np.sqrt(np.mean(errors**2))],
            [summary["calls_mean"], summary["calls_sd"], summary["ess_mean"],
             summary["ess_sd"], summary["classic_expected_rmse"]],
            rtol=1e-12,
        )
        if arm != "cg10_p_followup" and not all(
            r["classic_log_Z_uncert"] < .05 for r in rows
        ):
            raise ValueError(f"Evidence stopping goal was not met: {arm}")
        lines.append(
            f"{case} & {allocation} & "
            f"${summary['classic_expected_rmse']:.4f}"
            rf"\pm{summary['classic_expected_rmse_bootstrap_se']:.4f}$ & "
            f"{summary['mean_expected_sd']:.5f} & "
            f"${summary['calls_mean'] / 1e6:.3f}"
            rf"\pm{summary['calls_sd'] / 1e6:.3f}$ & "
            f"${summary['ess_mean'] / 1e3:.3f}"
            rf"\pm{summary['ess_sd'] / 1e3:.3f}$\\"
        )

    base = data["arms"]["cg10_e"]["records"]
    followup = data["arms"]["cg10_p_followup"]["records"]
    for before, after in zip(base, followup, strict=True):
        if after["followup"]["state_sha256"] != before["state_sha256"]:
            raise ValueError("Posterior continuation has a different parent state.")
        if after["classic_kish_ess"] < 2 * before["classic_kish_ess"]:
            raise ValueError("A posterior continuation missed its fixed ESS goal.")
    expected_saving = 100 * (
        1 - data["arms"]["g10_e"]["summary"]["calls_mean"]
        / data["arms"]["g10_uniform"]["summary"]["calls_mean"]
    )
    np.testing.assert_allclose(expected_saving, data["allocation_saving_percent"])
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (root / "COMPLETED_CLASSIC_TABLE.tex").write_text("\n".join(lines) + "\n")
    print("Validated 120 records and 30 paired ESS continuations; table rendered.")


if __name__ == "__main__":
    main()
