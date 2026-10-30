"""Let the historical volume-reduction criterion terminate without a K=10 cap."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "gmm_fit")
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import matplotlib
    import numpy as np
    from gmm_split_reference import (
        DensityFit,
        draw_mixture,
        log_density,
        marginal,
        split_fit,
    )
    from scipy.special import erf

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = options.directory
    records = []
    for dataset in ("oracle", "saved-0", "saved-1", "saved-2"):
        original = json.loads((directory / f"{dataset}.json").read_text())
        with np.load(directory / f"{dataset}.npz") as archive:
            train, test = archive["train"], archive["test"]
        seed = original["input"]["seed"]
        # Each leaf needs at least D+1 members. This cap cannot be reached;
        # termination must come from the volume, support, or rank criterion.
        unreachable_cap = len(train) // (train.shape[1] + 1) + 1
        fitted, trace = split_fit(train, unreachable_cap, seed)
        count = len(fitted.masses)
        assert count < unreachable_cap
        samples = draw_mixture(fitted, 65536, seed + 100)
        residual = np.abs(samples[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * samples[:, 1]))
        rejection_reasons = {}
        for event in trace:
            if not event["accepted"]:
                rejection_reasons[event["reason"]] = rejection_reasons.get(event["reason"], 0) + 1
        record = {
            "dataset": dataset, "components": count, "attempts": len(trace),
            "rejection_reasons": rejection_reasons,
            "train_mean_log_density": float(log_density(train, fitted).mean()),
            "test_mean_log_density": float(log_density(test, fitted).mean()),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "fraction_residual_within_3sigma": float(np.mean(residual < 3e-5)),
            "largest_mass": float(fitted.masses.max()),
            "source_data_checksum": original["data_checksum"], "split_trace": trace,
        }
        records.append(record)
        np.savez_compressed(directory / f"{dataset}-adaptive-split.npz",
                            centres=fitted.centres, covariances=fitted.covariances,
                            masses=fitted.masses)
        print(json.dumps({key: value for key, value in record.items() if key != "split_trace"}),
              flush=True)

        comparisons = [(f"Split until stop: K={count}", fitted, "#33a02c")]
        for name, label, colour in (("split", "Split: K=10", "#aaaa00"),
                                    ("current-500", "EM 500: K=10", "#1f78b4")):
            item = next(row for row in original["models"] if row["name"] == name)
            comparisons.append((label, DensityFit(np.asarray(item["centres"]),
                                                  np.asarray(item["covariances"]),
                                                  np.asarray(item["masses"])), colour))
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for axis, dimension in zip(axes.flat, (0, 1, 5, 16, 17)):
            grid = np.linspace(-0.05, 1.05, 1500)
            axis.hist(test[:, dimension], bins=70, density=True, color="gray", alpha=0.25)
            for label, fit, colour in comparisons:
                axis.plot(grid, np.exp(log_density(grid[:, None], marginal(fit, [dimension]))),
                          label=label, color=colour)
            axis.set_title(f"x{dimension}")
            axis.set_ylabel("Marginal density")
            axis.set_xlim(-0.05, 1.05)
        axis = axes.flat[-1]
        thresholds = np.geomspace(1e-7, 1, 500)
        for label, fit, colour in comparisons:
            points = draw_mixture(fit, 65536, seed + 100)
            residual = np.sort(np.abs(
                points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1]),
            ))
            axis.plot(thresholds, np.searchsorted(residual, thresholds) / len(residual),
                      label=label, color=colour)
        axis.plot(thresholds, erf(thresholds / (1e-5 * np.sqrt(2))), "k--", label="Known posterior")
        axis.set_xscale("log")
        axis.set_title("Residual from the curve")
        axis.set_xlabel("Absolute residual")
        axis.set_ylabel("Cumulative probability")
        axis.legend(fontsize=8)
        figure.suptitle(f"{dataset}: removing the ten-component split cap")
        figure.savefig(directory / f"{dataset}-adaptive-split.png", dpi=160)
        figure.savefig(directory / f"{dataset}-adaptive-split.pdf")
        plt.close(figure)
    (directory / "adaptive-split.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
