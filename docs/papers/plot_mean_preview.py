"""Preview the proposed Gaussian mean without changing frozen experiments.

Run with ``conda run -n jaxns_py python docs/papers/plot_mean_preview.py``.
The density and styling follow the archived evidence10-plot.py figure source;
outputs have separate names so the manuscript retains its current results.
"""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

plt.switch_backend("Agg")


def _plot_panel(
        axes: Axes,
        case: Literal["G10", "CG10", "SS10"],
        mean: tuple[float, float],
        title: str,
) -> QuadContourSet:
    """Draw a 2D likelihood analogue and its prior boundary."""
    # Equal-width domains preserve the spatial scale across all three cases.
    # Gaussian panels extend downward to display the proposed negative mean;
    # old and new means use identical axes in the comparison figure.
    limits = (-4, 8) if case == "SS10" else (-6, 6)
    grid = np.linspace(*limits, 700)
    horizontal, vertical = np.meshgrid(grid, grid)
    points = np.stack([horizontal, vertical], axis=-1)
    if case == "SS10":
        log_likelihood = logsumexp(
            np.stack([
                multivariate_normal.logpdf(
                    points, mean=[6, 6], cov=0.08 * np.eye(2),
                ),
                multivariate_normal.logpdf(
                    points, mean=[2.5, 2.5], cov=0.8 * np.eye(2),
                ),
            ]),
            axis=0,
        )
        axes.add_patch(patches.Rectangle(
            (-4, -4), 12, 12, fill=False, edgecolor="#bf3c32",
            linestyle="--", linewidth=2, zorder=5, clip_on=False,
        ))
    else:
        latent = points.copy()
        if case == "CG10":
            # The shear stays centered on the latent component. Changing
            # mu_G also changes its x_1 center, preserving the zero expected
            # displacement of x_2 under the latent Gaussian.
            latent[..., 1] -= 0.4 * (
                (latent[..., 0] - mean[0]) ** 2 - 1.0
            )
        log_likelihood = multivariate_normal.logpdf(
            latent, mean=mean, cov=[[1, 0.99], [0.99, 1]],
        )
        axes.add_patch(patches.Circle(
            (0, 0), 1, fill=False, color="#bf3c32", linestyle="--",
            linewidth=1.6, zorder=5,
        ))

    relative = log_likelihood - log_likelihood.max()
    artist = axes.contourf(
        horizontal, vertical, relative, levels=np.linspace(-16, 0, 33),
        cmap="viridis", extend="min",
    )
    axes.contour(
        horizontal, vertical, relative, levels=[-12, -8, -4, -1],
        colors="white", linewidths=0.5, alpha=0.75,
    )
    axes.set(
        title=title, xlabel="$x_1$", ylabel="$x_2$",
        xlim=limits, ylim=limits, aspect="equal",
    )
    return artist


def main() -> None:
    """Save the proposed paper figure and compare all three mean choices."""
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument(
        "--mean22", action="store_true",
        help="Preview (2,-2) and preserve the earlier (2,-3) figure files.",
    )
    choices.add_argument(
        "--mean33", action="store_true",
        help="Draw the paper's (3,-3) models and preserve earlier figures.",
    )
    choices.add_argument(
        "--mean32", action="store_true",
        help="Draw the paper's (3,-2) models and preserve earlier figures.",
    )
    args = parser.parse_args()
    mean = (2.0, -2.0) if args.mean22 else (2.0, -3.0)
    second = 2 if args.mean22 else 3
    mean_label = rf"$\mu_{{\rm G}}=2e_1-{second}e_2$"
    figure_name = f"evidence_problems_mu_2e1_minus{second}e2"
    if args.mean33:
        mean = (3.0, -3.0)
        mean_label = r"$\mu_{\rm G}=3e_1-3e_2$"
        figure_name = "evidence_problems_mu_3e1_minus3e2"
    if args.mean32:
        mean = (3.0, -2.0)
        mean_label = r"$\mu_{\rm G}=3e_1-2e_2$"
        figure_name = "evidence_problems_mu_3e1_minus2e2"
    output = Path(__file__).resolve().parent / "images"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })

    figure, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.1), constrained_layout=True,
    )
    for axis, case in zip(axes, ("G10", "CG10", "SS10"), strict=True):
        title = f"{case} analogue"
        if case != "SS10":
            title += "\n" + mean_label
        artist = _plot_panel(axis, case, mean, title)
    figure.colorbar(
        artist, ax=axes, shrink=0.75, label="Relative log likelihood",
    )
    figure.savefig(output / f"{figure_name}.pdf")
    figure.savefig(output / f"{figure_name}.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        3, 2, figsize=(9.5, 12.5), constrained_layout=True,
    )
    comparisons = (
        ((3.0, 0.0), r"Current: $\mu_{\rm G}=3e_1$"),
        ((0.0, -3.0), r"Previous: $\mu_{\rm G}=-3e_2$"),
        ((2.0, -3.0), r"Proposed: $\mu_{\rm G}=2e_1-3e_2$"),
    )
    comparison_name = "gaussian_mean_comparison"
    if args.mean22:
        comparisons = (
            ((3.0, 0.0), r"Original: $\mu_{\rm G}=3e_1$"),
            ((2.0, -3.0), r"Previous: $\mu_{\rm G}=2e_1-3e_2$"),
            ((2.0, -2.0), r"New: $\mu_{\rm G}=2e_1-2e_2$"),
        )
        comparison_name = "gaussian_mean22_comparison"
    if args.mean33:
        comparisons = (
            ((2.0, -3.0), r"Previous: $\mu_{\rm G}=2e_1-3e_2$"),
            ((2.0, -2.0), r"Previous: $\mu_{\rm G}=2e_1-2e_2$"),
            ((3.0, -3.0), r"New: $\mu_{\rm G}=3e_1-3e_2$"),
        )
        comparison_name = "gaussian_mean33_comparison"
    if args.mean32:
        comparisons = (
            ((2.0, -3.0), r"Previous: $\mu_{\rm G}=2e_1-3e_2$"),
            ((3.0, -3.0), r"Previous: $\mu_{\rm G}=3e_1-3e_2$"),
            ((3.0, -2.0), r"New: $\mu_{\rm G}=3e_1-2e_2$"),
        )
        comparison_name = "gaussian_mean32_comparison"
    for row, (mean, label) in enumerate(comparisons):
        for column, case in enumerate(("G10", "CG10")):
            artist = _plot_panel(
                axes[row, column], case, mean, f"{case}\n{label}",
            )
    figure.colorbar(
        artist, ax=axes.ravel().tolist(), shrink=0.75,
        label="Relative log likelihood",
    )
    figure.savefig(output / f"{comparison_name}.pdf")
    figure.savefig(output / f"{comparison_name}.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
