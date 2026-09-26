"""Independently recompute paper summary metrics from the 30 seed rows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument(
        "--cohort", choices=("classic_seeds", "uniform"),
        default="classic_seeds",
        help="Explicit cohort key in the archived summary.",
    )
    parser.add_argument(
        "--reference-check", type=Path,
        help="Independent reference calculation for a changed model mean.",
    )
    args = parser.parse_args()
    references = {
        "mean": [2, -3] + [0] * 8,
        "g10_conjugate_log_Z": -16.819723286090568,
        "cg10_direct_quad_log_Z": -17.89275622710132,
    }
    if args.reference_check is not None:
        references = json.loads(args.reference_check.read_text())
    summary = json.loads(args.summary.read_text())
    assert summary["bootstrap_resamples"] == 100000
    assert summary["bootstrap_seed"] == 20260911

    indices = np.random.default_rng(20260911).integers(0, 30, (100000, 30))
    # Count seed multiplicities instead of indexing replicate tensors. This
    # independently evaluates the same paired whole-tree bootstrap protocol.
    counts = np.stack(
        [(indices == seed).sum(axis=1) for seed in range(30)], axis=1,
    )
    for case, cohorts in summary["problems"].items():
        data = cohorts[args.cohort]
        errors = np.asarray(data["per_seed_errors"])
        deviations = np.asarray(data["per_seed_reported_sd"])
        assert errors.shape == deviations.shape == (30, 10)
        assert np.isfinite(errors).all() and np.isfinite(deviations).all()
        assert (deviations > 0).all()
        boot_bias = counts @ errors / 30
        boot_rmse = np.sqrt(counts @ errors**2 / 30)
        computed = {
            "bias": errors.mean(axis=0),
            "error_sd": errors.std(axis=0, ddof=1),
            "rmse": np.sqrt(np.mean(errors**2, axis=0)),
            "mean_reported_sd": deviations.mean(axis=0),
            "rmse_bootstrap_se": boot_rmse.std(axis=0, ddof=1),
            "mean_mc_error": deviations.mean(axis=0) / np.sqrt(2048),
            "paired_rmse_minus_classic_ci95": np.quantile(
                boot_rmse - boot_rmse[:, :1], [0.025, 0.975], axis=0,
            ).T,
            "paired_absolute_bias_minus_classic_ci95": np.quantile(
                abs(boot_bias) - abs(boot_bias[:, :1]),
                [0.025, 0.975], axis=0,
            ).T,
            "mean_likelihood_evaluations": np.mean(data["calls"]),
            "sd_likelihood_evaluations": np.std(data["calls"], ddof=1),
            "mean_goals": np.mean(data["goals"]),
        }
        for name, value in computed.items():
            np.testing.assert_allclose(
                value, data[name], rtol=0, atol=1e-10,
                err_msg=f"{case}: {name}",
            )
        if args.records is not None:
            reference = {
                "g10": references["g10_conjugate_log_Z"],
                "cg10": references["cg10_direct_quad_log_Z"],
            }[case]
            covered = []
            for seed in range(30):
                cell = args.records / case / f"seed-{seed:02d}"
                core = json.loads((cell / "CORE.json").read_text())
                analysis = json.loads((cell / "ANALYSIS.json").read_text())
                assert core["complete"] and analysis["complete"]
                assert core["case"] == analysis["case"] == case
                assert core["seed"] == analysis["seed"] == seed
                assert core["phantom_U_is_None"]
                assert core["isotropic_without_fitted_GMM"]
                assert analysis[
                    "prefixes_share_exact_classic_tree_and_posterior"
                ]
                assert analysis["prefix_sizes"] == list(range(0, 91, 10))
                assert core["state_sha256"] == analysis["state_sha256"]
                assert (
                    core["classic_posterior_sha256"]
                    == analysis["classic_posterior_sha256"]
                )
                assert (
                    core["likelihood_evaluations"]
                    == analysis["likelihood_evaluations"]
                    == data["calls"][seed]
                )
                assert core["iteration"] == data["goals"][seed]
                assert analysis["protocol_sha256"] == summary["protocol"]
                np.testing.assert_array_equal(
                    analysis["reference"]["mean"], references["mean"],
                )
                np.testing.assert_allclose(
                    analysis["reference"]["log_Z"], reference,
                    rtol=0, atol=1e-11,
                )
                progress_lines = (
                    (cell / "progress.jsonl").read_text().splitlines()
                )
                progress = [json.loads(line) for line in progress_lines]
                assert progress == core["goal_progress"]
                assert all(
                    item["classic_log_Z_uncert"] >= 0.05
                    for item in progress[:-1]
                )
                assert progress[-1]["classic_log_Z_uncert"] < 0.05
                draw_path = cell / "evidence_draws.npz"
                assert (
                    hashlib.sha256(draw_path.read_bytes()).hexdigest()
                    == analysis["evidence_draws_sha256"]
                )
                with np.load(draw_path) as stored:
                    draws = stored["log_Z"]
                assert draws.shape == (2048, 10)
                assert np.isfinite(draws).all()
                np.testing.assert_allclose(
                    draws.mean(axis=0) - reference, errors[seed],
                    rtol=0, atol=1e-11,
                )
                np.testing.assert_allclose(
                    draws.std(axis=0, ddof=1), deviations[seed],
                    rtol=0, atol=1e-12,
                )
                low, high = np.quantile(draws, [0.025, 0.975], axis=0)
                covered.append((low <= reference) & (reference <= high))
            np.testing.assert_allclose(
                np.mean(covered, axis=0), data["coverage95"],
                rtol=0, atol=1e-12,
            )
            print(f"{case}: raw draws, coverage and stopping checks agree")
        print(f"{case}: all 30 seeds, 10 prefixes and paired metrics agree")


if __name__ == "__main__":
    main()
