"""NumPy density-fit experiment using the retired ellipsoid split criterion.

The historical owner is src/jaxns/sampling/ellipsoid.py at d700ead. Only its
enclosing-volume acceptance rule and breadth-first traversal are retained.
Two-means proposes partitions; Gaussian moments, not enclosing radii, define
the output density. No nested-sampling transition uses this experiment.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans


@dataclass(frozen=True, slots=True)
class DensityFit:
    centres: np.ndarray
    covariances: np.ndarray
    masses: np.ndarray


def cluster_moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return empirical moments and the volume of their enclosing ellipsoid."""
    points = np.asarray(points, dtype=np.float64)
    dimension = points.shape[1]
    centre = points.mean(axis=0)
    delta = points - centre
    covariance = delta.T @ delta / len(points)
    factor = np.linalg.cholesky(covariance)
    whitened = solve_triangular(factor, delta.T, lower=True)
    radius_squared = np.max(np.sum(whitened ** 2, axis=0))
    log_volume = (
        0.5 * dimension * np.log(np.pi) - gammaln(0.5 * dimension + 1)
        + np.log(np.diag(factor)).sum()
        + 0.5 * dimension * np.log(radius_squared)
    )
    return centre, covariance, float(log_volume)


def split_fit(
        points: np.ndarray,
        max_components: int,
        seed: int,
        regularisation: float = 1e-6,
) -> tuple[DensityFit, list[dict]]:
    """Recursively accept two-means children only if their hull volume falls.

    A posterior cloud has no single known constrained-prior volume. The old
    volume lower bound is therefore absent: this isolates V0 + V1 < Vparent.
    Each accepted child needs D+1 members and nonsingular empirical covariance.
    K-means uses raw U coordinates, without whitening or handpicked dimensions.
    """
    points = np.asarray(points, dtype=np.float64)
    dimension = points.shape[1]
    clusters = [np.arange(len(points))]
    depths = [0]
    done = [False]
    trace = []
    while len(clusters) < max_components and not all(done):
        parent = min((i for i in range(len(clusters)) if not done[i]),
                     key=lambda i: (depths[i], i))
        indices = clusters[parent]
        event = {"parent": parent, "depth": depths[parent], "count": len(indices),
                 "accepted": False}
        if len(indices) < 2 * (dimension + 1):
            event["reason"] = "insufficient_members"
            done[parent] = True
            trace.append(event)
            continue
        labels = KMeans(
            n_clusters=2, n_init=10, max_iter=100, tol=1e-6,
            random_state=seed + len(trace),
        ).fit_predict(points[indices])
        children = [indices[labels == label] for label in (0, 1)]
        event["child_counts"] = [len(child) for child in children]
        if min(event["child_counts"]) < dimension + 1:
            event["reason"] = "insufficient_child_members"
            done[parent] = True
            trace.append(event)
            continue
        try:
            parent_log_volume = cluster_moments(points[indices])[2]
            child_log_volumes = [cluster_moments(points[child])[2] for child in children]
        except np.linalg.LinAlgError:
            event["reason"] = "singular_covariance"
            done[parent] = True
            trace.append(event)
            continue
        log_ratio = float(np.logaddexp(*child_log_volumes) - parent_log_volume)
        event.update(parent_log_volume=parent_log_volume,
                     child_log_volumes=child_log_volumes, log_volume_ratio=log_ratio)
        if log_ratio < 0:
            clusters[parent] = children[0]
            clusters.append(children[1])
            depths[parent] += 1
            depths.append(depths[parent])
            done.append(False)
            event["accepted"] = True
            event["reason"] = "volume_reduced"
        else:
            done[parent] = True
            event["reason"] = "volume_not_reduced"
        trace.append(event)
    centres, covariances, masses = [], [], []
    for indices in clusters:
        centre, covariance, _ = cluster_moments(points[indices])
        # Enclosing radii belong only to the acceptance criterion. Inflating
        # the Gaussian covariance by them would no longer fit the density.
        scale = max(np.trace(covariance) / dimension, np.finfo(float).eps)
        centres.append(centre)
        covariances.append(covariance + regularisation * scale * np.eye(dimension))
        masses.append(len(indices) / len(points))
    return DensityFit(np.asarray(centres), np.asarray(covariances), np.asarray(masses)), trace


def log_density(points: np.ndarray, fit: DensityFit) -> np.ndarray:
    """Evaluate the normalized Gaussian density in the supplied dimensions."""
    dimension = points.shape[1]
    terms = []
    for centre, covariance, mass in zip(fit.centres, fit.covariances, fit.masses):
        factor = np.linalg.cholesky(covariance)
        scaled = solve_triangular(factor, (points - centre).T, lower=True)
        terms.append(np.log(mass) - 0.5 * (
            dimension * np.log(2 * np.pi)
            + 2 * np.log(np.diag(factor)).sum()
            + np.sum(scaled ** 2, axis=0)
        ))
    return logsumexp(np.asarray(terms), axis=0)


def marginal(fit: DensityFit, dimensions: list[int]) -> DensityFit:
    """Marginalize analytically by selecting the covariance submatrix."""
    return DensityFit(fit.centres[:, dimensions],
                      fit.covariances[:, dimensions][:, :, dimensions], fit.masses)


def draw_mixture(fit: DensityFit, count: int, seed: int) -> np.ndarray:
    """Draw untruncated density samples; clipping would hide poor support fit."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(fit.masses), size=count, p=fit.masses / fit.masses.sum())
    points = np.empty((count, fit.centres.shape[1]))
    for component in range(len(fit.masses)):
        mask = labels == component
        points[mask] = rng.multivariate_normal(
            fit.centres[component], fit.covariances[component], size=mask.sum(),
        )
    return points
