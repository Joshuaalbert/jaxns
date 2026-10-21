"""Plot analytic marginals and joint geometry of the ridge density fits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from gmm_split_reference import DensityFit, draw_mixture, log_density, marginal
from scipy.special import erf
from scipy.stats import norm

STYLES = {
    "current-10": ("EM: 10 iterations", "#d95f02", "-"),
    "current-100": ("EM: 100 iterations", "#7570b3", "--"),
    "current-500": ("EM: 500 iterations", "#1f78b4", "-"),
    "split": ("K-means splits", "#33a02c", "-"),
    "split-em-500": ("Splits + 500 EM", "#e7298a", "-"),
    "original-weighted-10": ("Original weighted fit: 10 EM", "#555555", ":"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "gmm_fit")
    options = parser.parse_args()
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = options.directory
    all_rows = []
    convergence, convergence_axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for dataset, convergence_axis in zip(
            ("oracle", "saved-0", "saved-1", "saved-2"), convergence_axes.flat,
    ):
        result = json.loads((directory / f"{dataset}.json").read_text())
        data_path = directory / f"{dataset}.npz"
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != result["data_checksum"]:
            raise ValueError(f"Data checksum mismatch for {dataset}")
        if result["em_iterations"] != 500 or not result["numpy_em_reference_passed"]:
            raise ValueError(f"Incomplete fit sweep for {dataset}")
        with np.load(data_path) as archive:
            points = archive["test"].astype(np.float64)
        fits = {
            row["name"]: DensityFit(np.asarray(row["centres"]),
                                    np.asarray(row["covariances"]), np.asarray(row["masses"]))
            for row in result["models"]
        }
        for row in result["models"]:
            all_rows.append({"dataset": dataset, **{
                key: value for key, value in row.items()
                if key not in ("centres", "covariances", "masses", "mean")
            }})
        title = ("Independent samples from the full posterior" if dataset == "oracle"
                 else f"Resamples of the saved collapsed posterior: seed {dataset[-1]}")

        def draw_marginal(axis, dimension, population, models):
            low, high = ((-0.05, 1.05) if dimension in (0, 1, 16, 17) else (0.2, 0.8))
            grid = np.linspace(low, high, 1200)
            # Histogram is the empirical reference. Density curves are exact
            # Gaussian marginals, not KDEs of Monte Carlo mixture samples.
            axis.hist(population[:, dimension], bins=70, density=True, color="#777777",
                      alpha=0.25, label="Input population")
            for name, (label, colour, linestyle) in STYLES.items():
                if name in models:
                    density = np.exp(log_density(
                        grid[:, None], marginal(models[name], [dimension]),
                    ))
                    axis.plot(grid, density, color=colour, ls=linestyle, lw=1.5, label=label)
            if dimension in (1, 16, 17):
                axis.plot(grid, ((grid >= 0) & (grid <= 1)).astype(float),
                          color="black", ls="--", lw=1, label="Known posterior")
            elif 2 <= dimension <= 15:
                axis.plot(grid, norm.pdf(grid, 0.5, 0.05), color="black", ls="--", lw=1,
                          label="Known posterior")
            axis.set_xlim(low, high)
            axis.set_title(f"x{dimension}")
            axis.set_ylabel("Marginal density")
            axis.grid(alpha=0.15)

        figure, axes = plt.subplots(2, 3, figsize=(15, 8))
        for axis, dimension in zip(axes.flat, (0, 1, 5, 16, 17)):
            draw_marginal(axis, dimension, points, fits)
        residual_axis = axes.flat[-1]
        thresholds = np.geomspace(1e-7, 1, 500)
        input_residual = np.sort(np.abs(
            points[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * points[:, 1]),
        ))
        residual_axis.plot(thresholds, np.searchsorted(input_residual, thresholds) / len(points),
                           color="#777777", lw=2, label="Input population")
        for name, (label, colour, linestyle) in STYLES.items():
            if name not in fits:
                continue
            samples = draw_mixture(fits[name], 65536, result["input"]["seed"] + 100)
            residual = np.sort(np.abs(
                samples[:, 0] - 0.5 - 0.3 * np.sin(2 * np.pi * samples[:, 1]),
            ))
            residual_axis.plot(thresholds, np.searchsorted(residual, thresholds) / len(residual),
                               color=colour, ls=linestyle, label=label)
        residual_axis.plot(thresholds, erf(thresholds / (1e-5 * np.sqrt(2))), "k--", lw=1,
                           label="Known posterior")
        residual_axis.set_xscale("log")
        residual_axis.set_xlabel("Absolute residual from sine ridge")
        residual_axis.set_ylabel("Cumulative probability")
        residual_axis.set_title("Joint geometry: residual width")
        residual_axis.grid(alpha=0.15)
        handles, labels = axes.flat[1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
                      ncol=4, fontsize=9)
        figure.suptitle(title + " — ten components", fontsize=14)
        figure.subplots_adjust(top=0.81, bottom=0.08, hspace=0.30, wspace=0.25)
        figure.savefig(directory / f"{dataset}-marginals.png", dpi=160)
        figure.savefig(directory / f"{dataset}-marginals.pdf")
        plt.close(figure)

        figure, axes = plt.subplots(3, 6, figsize=(19, 10))
        for dimension, axis in enumerate(axes.flat):
            draw_marginal(axis, dimension, points, fits)
            axis.set_ylabel("Density")
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
                      ncol=4, fontsize=9)
        figure.suptitle(title + " — all 18 marginals", fontsize=14)
        figure.subplots_adjust(top=0.84, hspace=0.35, wspace=0.35)
        figure.savefig(directory / f"{dataset}-all-marginals.png", dpi=150)
        figure.savefig(directory / f"{dataset}-all-marginals.pdf")
        plt.close(figure)

        # Project the full 18-D fits analytically onto the curved pair. Using
        # a marginal, rather than a slice through other coordinates, avoids
        # hiding mass in the fourteen Gaussian and two inert dimensions.
        if dataset == "oracle":
            xlim, ylim = (-0.03, 1.03), (0.15, 0.85)
        else:
            bounds = np.quantile(points[:, [1, 0]], [0.001, 0.999], axis=0)
            padding = 0.2 * (bounds[1] - bounds[0])
            xlim = (bounds[0, 0] - padding[0], bounds[1, 0] + padding[0])
            ylim = (bounds[0, 1] - padding[1], bounds[1, 1] + padding[1])
        x = np.linspace(*xlim, 500)
        y = np.linspace(*ylim, 400)
        xx, yy = np.meshgrid(x, y)
        grid = np.stack((xx.ravel(), yy.ravel()), axis=1)
        joint_names = [name for name in STYLES if name != "original-weighted-10"]
        log_images = [log_density(grid, marginal(fits[name], [1, 0])).reshape(xx.shape)
                      / np.log(10) for name in joint_names]
        vmax = max(float(np.max(values)) for values in log_images)
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for axis, name, values in zip(axes.flat, joint_names, log_images):
            shown = axis.imshow(values, origin="lower", extent=(*xlim, *ylim),
                                aspect="auto", cmap="magma", vmin=vmax - 5, vmax=vmax)
            axis.plot(x, 0.5 + 0.3 * np.sin(2 * np.pi * x), color="cyan", lw=0.8)
            axis.scatter(fits[name].centres[:, 1], fits[name].centres[:, 0],
                         marker="x", s=18, color="white")
            axis.set_title(STYLES[name][0])
        axis = axes.flat[-1]
        axis.scatter(points[::4, 1], points[::4, 0], s=2, alpha=0.2, color="black")
        axis.plot(x, 0.5 + 0.3 * np.sin(2 * np.pi * x), color="#00aabb", lw=0.8)
        axis.set_title("Input population (every fourth draw)")
        for axis in axes.flat:
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            axis.set_xlabel("x1: ridge coordinate")
            axis.set_ylabel("x0: stiff coordinate")
        figure.colorbar(shown, ax=list(axes.flat), label="log10 joint marginal density", shrink=0.8)
        figure.suptitle(title + "\nCyan: known ridge; crosses: component means", fontsize=13)
        figure.savefig(directory / f"{dataset}-joint.png", dpi=160)
        figure.savefig(directory / f"{dataset}-joint.pdf")
        plt.close(figure)

        for name, colour in (("current", "#1f78b4"), ("split-em", "#e7298a")):
            trace = result["em_traces"][name]
            iterations = [row["iteration"] for row in trace]
            convergence_axis.plot(iterations, [row["test_mean_log_density"] for row in trace],
                                  "o-", color=colour, label=f"{name}: held out")
            convergence_axis.plot(iterations, [row["train_mean_log_density"] for row in trace],
                                  "--", color=colour, label=f"{name}: training")
        convergence_axis.set_xscale("log")
        convergence_axis.set_title(dataset)
        convergence_axis.set_xlabel("EM iterations")
        convergence_axis.set_ylabel("Mean log density (higher is better)")
        convergence_axis.grid(alpha=0.15)
        convergence_axis.legend(fontsize=8)
    convergence.suptitle("EM convergence: ten components, frozen training populations")
    convergence.savefig(directory / "convergence.png", dpi=160)
    convergence.savefig(directory / "convergence.pdf")
    plt.close(convergence)
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote four marginal/joint comparisons, convergence plots "
          f"and {len(all_rows)} summary rows.")


if __name__ == "__main__":
    main()
