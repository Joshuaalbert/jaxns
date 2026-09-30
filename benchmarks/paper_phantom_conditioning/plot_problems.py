"""Plot the known-evidence two-dimensional paper benchmarks."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.special import logsumexp

from benchmarks.paper_phantom_conditioning.cases import (
    CG8_BETA,
    CG8_LIKELIHOOD_COVARIANCE,
    CG8_LIKELIHOOD_MEAN,
    CG8_PRIOR_COVARIANCE,
    CG8_PRIOR_MEAN,
    CSS8_COMPONENT_COVARIANCES,
    CSS8_COMPONENT_MEANS,
    CSS8_PRIOR_COVARIANCE,
    CSS8_PRIOR_MEAN,
    G8_LIKELIHOOD_COVARIANCE,
    G8_LIKELIHOOD_MEAN,
    G8_PRIOR_COVARIANCE,
    G8_PRIOR_MEAN,
    SS8_COMPONENT_COVARIANCES,
    SS8_COMPONENT_MEANS,
    SS8_PRIOR_COVARIANCE,
    SS8_PRIOR_MEAN,
)


def _normal_log_density(points, mean, covariance):
    """Evaluate a normal log density at points with shape ``[..., 2]``."""
    displacement = points - mean
    precision = np.linalg.inv(covariance)
    quadratic = np.einsum(
        "...i,ij,...j->...",
        displacement,
        precision,
        displacement,
    )
    _, log_determinant = np.linalg.slogdet(covariance)
    return -np.log(2.0 * np.pi) - 0.5 * log_determinant - 0.5 * quadratic


def _mixture_log_density(points, means, covariances):
    """Evaluate a sum of two normal densities."""
    components = np.stack([
        _normal_log_density(points, mean, covariance)
        for mean, covariance in zip(means, covariances)
    ])
    return logsumexp(components, axis=0)


def _inverse_curve(points, beta, center, variance):
    """Map curved problem coordinates back to the Gaussian coordinates."""
    gaussian = points.copy()
    gaussian[..., 1] -= beta * (
        (gaussian[..., 0] - center) ** 2 - variance
    )
    return gaussian


def _plot_panel(
        axes,
        horizontal,
        vertical,
        log_likelihood,
        log_prior,
        title,
        horizontal_label,
        vertical_label,
):
    """Draw one likelihood field and its one-sigma prior contour."""
    relative = log_likelihood - np.max(log_likelihood)
    # Use one shared likelihood scale so ridge thickness and curvature remain
    # visually comparable across panels without hiding either mixture mode.
    levels = np.linspace(-12.0, 0.0, 25)
    filled = axes.contourf(
        horizontal,
        vertical,
        np.maximum(relative, levels[0]),
        levels=levels,
        cmap="viridis",
        norm=Normalize(levels[0], levels[-1]),
        extend="min",
    )
    axes.contour(
        horizontal,
        vertical,
        relative,
        levels=[-8.0, -4.0, -2.0, -0.5],
        colors="white",
        linewidths=0.6,
        alpha=0.8,
    )

    # A one-sigma contour is one Mahalanobis unit from the prior mean, hence a
    # log-density drop of 1 / 2.  This shows the prior scale directly rather
    # than using a dimension-dependent enclosed-probability convention.
    axes.contour(
        horizontal,
        vertical,
        log_prior,
        levels=[np.max(log_prior) - 0.5],
        colors="#d62728",
        linestyles="--",
        linewidths=1.4,
    )
    axes.set_title(title, fontsize=10)
    axes.set_xlabel(horizontal_label)
    axes.set_ylabel(vertical_label)
    axes.set_aspect("equal", adjustable="box")
    return filled


def main():
    """Create the paper's 2x2 known-evidence problem figure."""
    figure, axes_grid = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.2),
        constrained_layout=True,
    )
    axes = axes_grid.ravel()

    horizontal, vertical = np.meshgrid(
        np.linspace(-3.5, 6.5, 420),
        np.linspace(-4.5, 4.5, 420),
    )
    points = np.stack([horizontal, vertical], axis=-1)
    filled = _plot_panel(
        axes[0],
        horizontal,
        vertical,
        _normal_log_density(
            points,
            np.asarray(G8_LIKELIHOOD_MEAN[:2]),
            np.asarray(G8_LIKELIHOOD_COVARIANCE[:2, :2]),
        ),
        _normal_log_density(
            points,
            np.asarray(G8_PRIOR_MEAN[:2]),
            np.asarray(G8_PRIOR_COVARIANCE[:2, :2]),
        ),
        "G8: shifted correlated likelihood",
        "$x_1$",
        "$x_2$",
    )

    horizontal, vertical = np.meshgrid(
        np.linspace(-3.5, 8.0, 420),
        np.linspace(-4.0, 9.0, 420),
    )
    points = np.stack([horizontal, vertical], axis=-1)
    gaussian = _inverse_curve(
        points,
        CG8_BETA,
        CG8_LIKELIHOOD_MEAN[0],
        CG8_LIKELIHOOD_COVARIANCE[0, 0],
    )
    _plot_panel(
        axes[1],
        horizontal,
        vertical,
        _normal_log_density(
            gaussian,
            np.asarray(CG8_LIKELIHOOD_MEAN[:2]),
            np.asarray(CG8_LIKELIHOOD_COVARIANCE[:2, :2]),
        ),
        _normal_log_density(
            points,
            np.asarray(CG8_PRIOR_MEAN[:2]),
            np.asarray(CG8_PRIOR_COVARIANCE[:2, :2]),
        ),
        "CG8: shifted curved correlated likelihood",
        "$x_1$",
        "$x_2$",
    )

    coordinates = np.asarray([0, 1])
    horizontal, vertical = np.meshgrid(
        np.linspace(-7.0, 7.0, 420),
        np.linspace(-6.0, 6.0, 420),
    )
    points = np.stack([horizontal, vertical], axis=-1)
    means = np.asarray(SS8_COMPONENT_MEANS)[:, coordinates]
    covariances = np.asarray(SS8_COMPONENT_COVARIANCES)[
        :, coordinates[:, None], coordinates
    ]
    _plot_panel(
        axes[2],
        horizontal,
        vertical,
        _mixture_log_density(
            points,
            means,
            covariances,
        ),
        _normal_log_density(
            points,
            np.asarray(SS8_PRIOR_MEAN[:2]),
            np.asarray(SS8_PRIOR_COVARIANCE)[
                coordinates[:, None], coordinates
            ],
        ),
        "SS8: spike--slab mixture",
        "$x_1$",
        "$x_2$",
    )

    horizontal, vertical = np.meshgrid(
        np.linspace(-7.0, 7.0, 420),
        np.linspace(-6.0, 6.0, 420),
    )
    points = np.stack([horizontal, vertical], axis=-1)
    _plot_panel(
        axes[3],
        horizontal,
        vertical,
        _mixture_log_density(
            points,
            np.asarray(CSS8_COMPONENT_MEANS[:, :2]),
            np.asarray(CSS8_COMPONENT_COVARIANCES[:, :2, :2]),
        ),
        _normal_log_density(
            points,
            np.asarray(CSS8_PRIOR_MEAN[:2]),
            np.asarray(CSS8_PRIOR_COVARIANCE[:2, :2]),
        ),
        "CSS8: correlated spike--slab mixture",
        "$x_1$",
        "$x_2$",
    )

    colorbar = figure.colorbar(filled, ax=axes.tolist(), shrink=0.85, pad=0.02)
    colorbar.set_label(r"Relative log likelihood, $\log L-\max\log L$")
    output = Path(__file__).resolve().parents[2] / "docs/papers/images"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "evidence_problems.pdf", bbox_inches="tight")
    figure.savefig(output / "evidence_problems.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
