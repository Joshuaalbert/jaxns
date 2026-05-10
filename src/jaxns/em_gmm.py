import dataclasses
import math
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap, lax
from jax._src.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal

from jaxns.mixed_precision import mp_policy


def initialize_params(key, data, n_components: int):
    """
    Initialize the parameters of a Gaussian Mixture Model.

    Args:
        key: the random key
        data: [n, d] array of data
        n_components: number of components

    Returns:
        means: [num_clusters, d] array of means
    """
    n, d = data.shape

    # Initialize means by selecting random data points
    assign_idx = random.choice(key, n, shape=(n_components,), replace=False)
    means = data[assign_idx]

    # Initialize covariances as the empirical covariance of the data
    # cov = jnp.cov(data, rowvar=False)
    cov = jnp.diag(jnp.var(data, axis=0))
    covariances = jnp.repeat(cov[None, ...], n_components, axis=0)

    # Initialize mixture weights uniformly
    log_weights = jnp.full((n_components,), -jnp.log(n_components), mp_policy.measure_dtype)

    return means, covariances, log_weights


def e_step(data, means, covariances, log_weights, mask):
    """
    Compute the responsibilities of each Gaussian for each data point.

    Args:
        data: [n, d] array of data
        means: [num_clusters, d] array of means
        covariances: [num_clusters, d, d] array of covariances
        log_weights: [num_clusters] array of log weights
        mask: [n] boolean array indicating which data points to use

    Returns:
        log_responsibilities: [num_clusters, n] array of log responsibilities
    """
    n, d = data.shape
    # Compute the probabilities of each data point belonging to each Gaussian
    logpdf = vmap(lambda m, c: multivariate_normal.logpdf(data, m, c))(means, covariances)  # num_clusters, num_data
    if mask is not None:
        logpdf = jnp.where(mask[None, :], logpdf, mp_policy.cast_to_measure(-jnp.inf))
    logpdf_weighted = logpdf + log_weights[:, None]
    # Normalize probabilities
    log_responsibilities = logpdf_weighted - logsumexp(logpdf_weighted, axis=0)
    return log_responsibilities


def m_step(data, log_responsibilities):
    """
    Update the parameters of the Gaussian Mixture Model.

    Args:
        data: [n, d] array of data
        log_responsibilities: [num_clusters, n] array of log responsibilities

    Returns:
        means: [num_clusters, d] array of means
    """
    n_components, num_data = log_responsibilities.shape
    _, d = data.shape

    # Update means, covariances, and weights
    log_weights = logsumexp(log_responsibilities, axis=1) - jnp.log(num_data)  # num_components

    # num_components, num_data X num_data, D -> num_components, D
    weighted_responsibilities = jnp.exp(log_responsibilities - log_weights[:, None] - jnp.log(num_data))
    means = jnp.matmul(weighted_responsibilities, data)

    centered_data = data[None, :, :] - means[:, None, :]  # num_components, num_data, D

    covariances = jnp.einsum("cn,cnd,cne->cde", weighted_responsibilities, centered_data, centered_data)
    covariances = covariances + 1e-4 * jnp.eye(d)
    return (
        mp_policy.cast_to_measure(means), mp_policy.cast_to_measure(covariances), mp_policy.cast_to_measure(log_weights)
    )


# No invariance under jit...
def em_gmm(key, data, n_components, mask: Union[jax.Array, None] = None, n_iters=10, tol=1e-6):
    """
    Fit a Gaussian Mixture Model to the data using the Expectation-Maximization algorithm.

    Args:
        key: the random key
        data: [n, d] array of data
        n_components: number of components
        mask: [n] boolean array indicating which data points to use
        n_iters: maximum number of iterations
        tol: convergence tolerance

    Returns:
        cluster_id: [n] array of cluster assignments
        params: tuple of (means, covariances, log_weights)
        total_iters: total number of iterations use
    """
    means, covariances, log_weights = initialize_params(key, data, n_components)
    params = (means, covariances, log_weights)

    def body(state):
        _, i, params = state
        log_responsibilities = e_step(data, *params, mask=mask)
        new_params = m_step(data, log_responsibilities)
        done = False
        for param, new_param in zip(params, new_params):
            done = done | (jnp.all(jnp.abs(jnp.asarray(param) - jnp.asarray(new_param)) < tol)) | (i >= n_iters)

        return done, i + 1, new_params

    def cond(state):
        done, _, params = state
        return jnp.bitwise_not(done)

    _, total_iters, params = lax.while_loop(
        cond,
        body,
        (jnp.asarray(False, jnp.bool_), jnp.asarray(0, jnp.int32), params)
    )

    cluster_id = jnp.argmax(e_step(data, *params, mask=mask), axis=0)
    return cluster_id, params, total_iters


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionGMMKernel:
    component_means: Any
    component_radii: Any
    component_rotations: Any
    component_probabilities: Any


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionFittingDiagnostics:
    active_kernel_mode: str = "isotropic"
    active_kernel_version: int = 0
    attempted_update_count: int = 0
    successful_update_count: int = 0
    fallback_active: bool = False
    fallback_reason: str | None = None
    fitting_path: str | None = None
    D_dim: int = 0
    N_pos: int = 0
    N_eff: float = 0.0
    unique_finite_row_count: int = 0
    excluded_nonfinite_rows: int = 0
    excluded_nonpositive_weight_rows: int = 0
    excluded_phantom_rows: int = 0
    phantom_coordinate_policy: str = "discarded"
    weight_sum: float = 0.0
    M_D: int = 20000
    resampling_seed: int | None = None
    original_N_pos: int = 0
    original_N_eff: float = 0.0
    resampled_row_count: int = 0
    requested_K_D: int = 0
    final_K_D: int = 0
    component_means: Any = dataclasses.field(default_factory=lambda: np.empty((0, 0)))
    component_radii: Any = dataclasses.field(default_factory=lambda: np.empty((0, 0)))
    component_integrated_volumes: Any = dataclasses.field(default_factory=lambda: np.empty((0,)))
    component_probabilities: Any = dataclasses.field(default_factory=lambda: np.empty((0,)))
    dropped_component_reasons: tuple[str, ...] = ()
    dropped_nonfinite: int = 0
    dropped_nonpositive_volume: int = 0
    dropped_low_responsibility_neff: int = 0
    em_iteration_count: int = 0
    em_converged: bool = False
    sigma2_floor: float = 0.0
    num_shrinkage_draws: int = 0
    num_plateau_samples: int = 0
    num_plateau_blocks: int = 0
    shell_epoch: int = 0
    allocation_target: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class PosteriorFittingWeightsResult:
    weights: np.ndarray
    diagnostics: DirectionFittingDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionFittingDataset:
    rows: np.ndarray
    weights: np.ndarray
    diagnostics: DirectionFittingDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionComponentCount:
    requested_K_D: int
    final_K_D: int
    fallback_active: bool


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionCovarianceComponents:
    component_means: np.ndarray
    component_radii: np.ndarray
    component_rotations: np.ndarray
    responsibility_effective_sizes: np.ndarray
    diagnostics: DirectionFittingDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionComponentProbabilities:
    component_probabilities: np.ndarray
    component_integrated_volumes: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionFilterResult:
    component_means: np.ndarray
    component_radii: np.ndarray
    component_rotations: np.ndarray
    component_probabilities: np.ndarray
    component_integrated_volumes: np.ndarray
    kernel: Any
    diagnostics: DirectionFittingDiagnostics


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionKernelFitResult:
    kernel: Any
    diagnostics: DirectionFittingDiagnostics
    coordinator: Any = None


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionKernelDispatchRequest:
    chain_id: str
    worker_id: str | None = None
    shell_epoch: int = 0
    allocation_target: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionKernelFitRequest:
    shell_epoch: int
    allocation_target: str
    samples_U: Any
    posterior_weights: Any
    key: Any
    n_components: int | None = None
    max_fit_rows: int = 20000
    resampling_seed: int | None = None
    hard_adaptation_errors: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionAdaptationContext:
    component_means: Any = None
    component_radii: Any = None
    component_rotations: Any = None
    component_probabilities: Any = None
    component_integrated_volumes: Any = None
    kernel_version: int = 0
    allocation_target: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionKernelDispatchSnapshot:
    chain_id: str
    worker_id: str | None
    shell_epoch: int
    allocation_target: str | None
    kernel: Any
    kernel_version: int
    diagnostics: DirectionFittingDiagnostics

    def direction_adaptation_context(self) -> DirectionAdaptationContext:
        if not _kernel_has_components(self.kernel):
            return DirectionAdaptationContext(
                kernel_version=self.kernel_version,
                allocation_target=self.allocation_target,
            )
        return DirectionAdaptationContext(
            component_means=np.asarray(self.kernel.component_means).copy(),
            component_radii=np.asarray(self.kernel.component_radii).copy(),
            component_rotations=np.asarray(self.kernel.component_rotations).copy(),
            component_probabilities=np.asarray(
                self.kernel.component_probabilities
            ).copy(),
            component_integrated_volumes=_kernel_integrated_volumes(
                self.kernel
            ),
            kernel_version=self.kernel_version,
            allocation_target=self.allocation_target,
        )


def _kernel_has_components(kernel: Any) -> bool:
    return all(
        hasattr(kernel, name)
        for name in (
            "component_means",
            "component_radii",
            "component_rotations",
            "component_probabilities",
        )
    )


def _kernel_integrated_volumes(kernel: Any) -> np.ndarray | None:
    if not _kernel_has_components(kernel):
        return None
    radii = np.asarray(kernel.component_radii, dtype=float)
    if radii.size == 0:
        return np.empty((0,), dtype=float)
    return np.prod(radii, axis=1)


def _as_fitting_matrix(samples_U: Any) -> np.ndarray:
    if hasattr(samples_U, "tree"):
        samples_U = samples_U.tree
    leaves = jax.tree.leaves(samples_U)
    if not leaves:
        array = np.asarray(samples_U, dtype=float)
        if array.ndim == 1:
            return array[:, None]
        return array.reshape((array.shape[0], -1))
    first = np.asarray(leaves[0])
    if first.ndim == 0:
        return np.asarray(samples_U, dtype=float).reshape((1, 1))
    num_samples = first.shape[0]
    flat_leaves = [
        np.asarray(leaf, dtype=float).reshape((num_samples, -1))
        for leaf in leaves
    ]
    return np.concatenate(flat_leaves, axis=1)


def _normalize_positive_weights(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("posterior weights must have positive finite sum.")
    return weights / total


def _effective_sample_size(weights: np.ndarray) -> float:
    weight_square_sum = float(np.sum(np.square(weights)))
    if weight_square_sum <= 0.0 or not np.isfinite(weight_square_sum):
        return 0.0
    return 1.0 / weight_square_sum


def posterior_fitting_weights_from_shrinkage_draws(
        *,
        block_ids: Any,
        block_likelihoods: Any,
        x_prev_draws: Any,
        p_gt_draws: Any,
        p_eq_draws: Any,
) -> PosteriorFittingWeightsResult:
    block_ids = np.asarray(block_ids, dtype=int)
    block_likelihoods = np.asarray(block_likelihoods, dtype=float)
    x_prev_draws = np.asarray(x_prev_draws, dtype=float)
    p_gt_draws = np.asarray(p_gt_draws, dtype=float)
    p_eq_draws = np.asarray(p_eq_draws, dtype=float)

    if x_prev_draws.ndim == 1:
        x_prev_draws = x_prev_draws[None, :]
    if p_gt_draws.ndim == 1:
        p_gt_draws = p_gt_draws[None, :]
    if p_eq_draws.ndim == 1:
        p_eq_draws = p_eq_draws[None, :]

    unique_blocks, block_counts = np.unique(block_ids, return_counts=True)
    count_by_block = dict(zip(unique_blocks.tolist(), block_counts.tolist()))
    per_draw_weights = np.zeros(
        (x_prev_draws.shape[0], block_ids.shape[0]),
        dtype=float,
    )
    plateau_blocks = 0
    plateau_samples = 0
    for sample_index, block_id in enumerate(block_ids):
        block_count = count_by_block[int(block_id)]
        likelihood = block_likelihoods[int(block_id)]
        if block_count > 1:
            draw_masses = (
                likelihood
                * x_prev_draws[:, int(block_id)]
                * p_eq_draws[:, int(block_id)]
                / float(block_count)
            )
            plateau_samples += 1
        else:
            draw_masses = (
                likelihood
                * x_prev_draws[:, int(block_id)]
                * (1.0 - p_gt_draws[:, int(block_id)])
            )
        per_draw_weights[:, sample_index] = draw_masses
    for block_id in unique_blocks:
        if count_by_block[int(block_id)] > 1:
            plateau_blocks += 1

    draw_totals = np.sum(per_draw_weights, axis=1)
    if (
            not np.all(np.isfinite(draw_totals))
            or np.any(draw_totals <= 0.0)
    ):
        raise ValueError("each shrinkage draw must have positive finite mass.")
    weights = np.mean(per_draw_weights / draw_totals[:, None], axis=0)
    weights = _normalize_positive_weights(weights)
    diagnostics = DirectionFittingDiagnostics(
        num_shrinkage_draws=int(x_prev_draws.shape[0]),
        num_plateau_samples=int(plateau_samples),
        num_plateau_blocks=int(plateau_blocks),
        weight_sum=float(np.sum(weights)),
        N_pos=int(np.sum(weights > 0.0)),
        N_eff=_effective_sample_size(weights),
    )
    return PosteriorFittingWeightsResult(weights=weights, diagnostics=diagnostics)


def systematic_resample_indices(
        *,
        weights: Any,
        num_samples: int,
        offset: float | None = None,
) -> np.ndarray:
    weights = _normalize_positive_weights(np.asarray(weights, dtype=float))
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if offset is None:
        offset = 0.0
    positions = float(offset) + np.arange(num_samples, dtype=float) / num_samples
    positions = np.minimum(positions, np.nextafter(1.0, 0.0))
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def build_direction_fitting_dataset(
        *,
        samples_U: Any,
        posterior_weights: Any,
        valid_mask: Any | None = None,
        retained_phantom_log_likelihoods: Any | None = None,
        retained_phantom_valid_mask: Any | None = None,
        max_fit_rows: int = 20000,
        resampling_seed: int | None = None,
) -> DirectionFittingDataset:
    del retained_phantom_log_likelihoods
    del retained_phantom_valid_mask
    rows = _as_fitting_matrix(samples_U)
    weights = np.asarray(posterior_weights, dtype=float).reshape((-1,))
    if rows.shape[0] != weights.shape[0]:
        raise ValueError("samples_U and posterior_weights length mismatch.")
    if max_fit_rows < 1:
        raise ValueError("max_fit_rows must be positive.")

    if valid_mask is None:
        valid_mask_array = np.ones(rows.shape[0], dtype=bool)
    else:
        valid_mask_array = np.asarray(valid_mask, dtype=bool).reshape((-1,))
        if valid_mask_array.shape[0] != rows.shape[0]:
            raise ValueError("valid_mask length mismatch.")

    finite_mask = np.all(np.isfinite(rows), axis=1)
    positive_weight_mask = np.isfinite(weights) & (weights > 0.0)
    keep_mask = valid_mask_array & finite_mask & positive_weight_mask
    filtered_rows = rows[keep_mask]
    if filtered_rows.shape[0] == 0:
        filtered_weights = np.empty((0,), dtype=float)
    else:
        filtered_weights = _normalize_positive_weights(weights[keep_mask])
    original_n_pos = int(filtered_rows.shape[0])
    original_n_eff = _effective_sample_size(filtered_weights)
    fitting_path = "weighted_em"
    resampled_row_count = 0

    if original_n_pos > max_fit_rows:
        rng = np.random.default_rng(resampling_seed)
        offset = float(rng.uniform(0.0, 1.0 / max_fit_rows))
        indices = systematic_resample_indices(
            weights=filtered_weights,
            num_samples=max_fit_rows,
            offset=offset,
        )
        filtered_rows = filtered_rows[indices]
        filtered_weights = np.full(max_fit_rows, 1.0 / max_fit_rows, dtype=float)
        fitting_path = "bounded_resampling"
        resampled_row_count = int(max_fit_rows)
    fitting_unique_finite_row_count = int(
        np.unique(filtered_rows, axis=0).shape[0]
        if filtered_rows.size
        else 0
    )

    diagnostics = DirectionFittingDiagnostics(
        fitting_path=fitting_path,
        D_dim=int(rows.shape[1]) if rows.ndim == 2 else 0,
        N_pos=int(filtered_rows.shape[0]),
        N_eff=_effective_sample_size(filtered_weights),
        unique_finite_row_count=fitting_unique_finite_row_count,
        excluded_nonfinite_rows=int(np.sum(valid_mask_array & ~finite_mask)),
        excluded_nonpositive_weight_rows=int(
            np.sum(valid_mask_array & finite_mask & ~positive_weight_mask)
        ),
        excluded_phantom_rows=0,
        phantom_coordinate_policy="discarded",
        weight_sum=float(np.sum(filtered_weights)),
        M_D=int(max_fit_rows),
        resampling_seed=resampling_seed,
        original_N_pos=original_n_pos,
        original_N_eff=original_n_eff,
        resampled_row_count=resampled_row_count,
    )
    return DirectionFittingDataset(
        rows=filtered_rows,
        weights=filtered_weights,
        diagnostics=diagnostics,
    )


def choose_direction_component_count(
        *,
        n_eff: float,
        d_dim: int,
        unique_row_count: int,
) -> DirectionComponentCount:
    requested = min(8, max(1, int(math.floor(n_eff / (2 * (d_dim + 1))))))
    final = min(requested, int(unique_row_count))
    return DirectionComponentCount(
        requested_K_D=int(requested),
        final_K_D=int(final),
        fallback_active=final < 1,
    )


def _weighted_global_covariance(
        rows: np.ndarray,
        weights: np.ndarray,
) -> np.ndarray:
    mean = weights @ rows
    centered = rows - mean
    return np.einsum("n,nd,ne->de", weights, centered, centered)


def direction_covariance_components(
        *,
        rows: Any,
        posterior_weights: Any,
        responsibilities: Any,
) -> DirectionCovarianceComponents:
    rows = np.asarray(rows, dtype=float)
    weights = _normalize_positive_weights(
        np.asarray(posterior_weights, dtype=float).reshape((-1,))
    )
    responsibilities = np.asarray(responsibilities, dtype=float)
    if responsibilities.shape[0] != rows.shape[0] and (
            responsibilities.shape[1] == rows.shape[0]
    ):
        responsibilities = responsibilities.T
    if responsibilities.shape[0] != rows.shape[0]:
        raise ValueError("responsibilities must have one row per fitting row.")

    n_components = responsibilities.shape[1]
    d_dim = rows.shape[1]
    global_covariance = _weighted_global_covariance(rows, weights)
    trace = float(np.trace(global_covariance))
    sigma2_floor = max(1e-12, 1e-6 * trace / d_dim)

    means = np.zeros((n_components, d_dim), dtype=float)
    radii = np.zeros((n_components, d_dim), dtype=float)
    rotations = np.zeros((n_components, d_dim, d_dim), dtype=float)
    responsibility_effective_sizes = np.zeros(n_components, dtype=float)
    for component_index in range(n_components):
        component_weights = responsibilities[:, component_index] * weights
        component_weight_sum = float(np.sum(component_weights))
        if component_weight_sum <= 0.0 or not np.isfinite(component_weight_sum):
            means[component_index] = np.nan
            radii[component_index] = np.nan
            rotations[component_index] = np.nan
            continue
        normalized_component_weights = component_weights / component_weight_sum
        means[component_index] = normalized_component_weights @ rows
        centered = rows - means[component_index]
        covariance = np.einsum(
            "n,nd,ne->de",
            normalized_component_weights,
            centered,
            centered,
        )
        covariance = covariance + sigma2_floor * np.eye(d_dim)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        radii[component_index] = np.sqrt(
            np.maximum(eigenvalues, sigma2_floor)
        )
        rotations[component_index] = eigenvectors
        responsibility_effective_sizes[component_index] = (
            component_weight_sum ** 2 / float(np.sum(component_weights ** 2))
        )

    diagnostics = DirectionFittingDiagnostics(
        D_dim=int(d_dim),
        N_pos=int(rows.shape[0]),
        N_eff=_effective_sample_size(weights),
        sigma2_floor=float(sigma2_floor),
        component_means=means,
        component_radii=radii,
    )
    return DirectionCovarianceComponents(
        component_means=means,
        component_radii=radii,
        component_rotations=rotations,
        responsibility_effective_sizes=responsibility_effective_sizes,
        diagnostics=diagnostics,
    )


def direction_component_probabilities_from_radii(
        *,
        component_radii: Any,
) -> DirectionComponentProbabilities:
    radii = np.asarray(component_radii, dtype=float)
    volumes = np.prod(radii, axis=1)
    valid_volumes = np.where(
        np.isfinite(volumes) & (volumes > 0.0),
        volumes,
        0.0,
    )
    total_volume = float(np.sum(valid_volumes))
    if total_volume <= 0.0 or not np.isfinite(total_volume):
        raise ValueError(
            "component volumes must have positive finite normalizing volume."
        )
    probabilities = valid_volumes / total_volume
    return DirectionComponentProbabilities(
        component_probabilities=probabilities,
        component_integrated_volumes=valid_volumes,
    )


def filter_direction_components(
        *,
        component_means: Any,
        component_radii: Any,
        component_rotations: Any,
        responsibility_effective_sizes: Any,
        d_dim: int,
        previous_kernel: Any = None,
) -> DirectionFilterResult:
    means = np.asarray(component_means, dtype=float)
    radii = np.asarray(component_radii, dtype=float)
    rotations = np.asarray(component_rotations, dtype=float)
    neff = np.asarray(responsibility_effective_sizes, dtype=float)
    volumes = np.prod(radii, axis=1)

    nonfinite_mask = (
        ~np.all(np.isfinite(means), axis=1)
        | ~np.all(np.isfinite(radii), axis=1)
        | ~np.all(np.isfinite(rotations), axis=(1, 2))
    )
    nonpositive_volume_mask = ~(np.isfinite(volumes) & (volumes > 0.0))
    low_neff_mask = (~np.isfinite(neff)) | (neff < (d_dim + 1))
    valid_mask = ~(nonfinite_mask | nonpositive_volume_mask | low_neff_mask)

    dropped_reasons = []
    if int(np.sum(nonfinite_mask)):
        dropped_reasons.append("nonfinite_parameters")
    if int(np.sum(nonpositive_volume_mask & ~nonfinite_mask)):
        dropped_reasons.append("nonpositive_volume")
    if int(np.sum(low_neff_mask & ~nonfinite_mask & ~nonpositive_volume_mask)):
        dropped_reasons.append("low_responsibility_neff")

    if not np.any(valid_mask):
        diagnostics = DirectionFittingDiagnostics(
            fallback_active=True,
            fallback_reason="no_valid_components",
            dropped_component_reasons=tuple(dropped_reasons),
            dropped_nonfinite=int(np.sum(nonfinite_mask)),
            dropped_nonpositive_volume=int(
                np.sum(nonpositive_volume_mask & ~nonfinite_mask)
            ),
            dropped_low_responsibility_neff=int(
                np.sum(low_neff_mask & ~nonfinite_mask & ~nonpositive_volume_mask)
            ),
            requested_K_D=int(means.shape[0]),
            final_K_D=0,
        )
        return DirectionFilterResult(
            component_means=np.empty((0, d_dim), dtype=float),
            component_radii=np.empty((0, d_dim), dtype=float),
            component_rotations=np.empty((0, d_dim, d_dim), dtype=float),
            component_probabilities=np.empty((0,), dtype=float),
            component_integrated_volumes=np.empty((0,), dtype=float),
            kernel=previous_kernel if previous_kernel is not None else "isotropic",
            diagnostics=diagnostics,
        )

    filtered_means = means[valid_mask]
    filtered_radii = radii[valid_mask]
    filtered_rotations = rotations[valid_mask]
    probability_result = direction_component_probabilities_from_radii(
        component_radii=filtered_radii
    )
    kernel = DirectionGMMKernel(
        component_means=jnp.asarray(filtered_means),
        component_radii=jnp.asarray(filtered_radii),
        component_rotations=jnp.asarray(filtered_rotations),
        component_probabilities=jnp.asarray(
            probability_result.component_probabilities
        ),
    )
    diagnostics = DirectionFittingDiagnostics(
        active_kernel_mode="gmm",
        fallback_active=False,
        requested_K_D=int(means.shape[0]),
        final_K_D=int(filtered_means.shape[0]),
        component_means=filtered_means,
        component_radii=filtered_radii,
        component_integrated_volumes=probability_result.component_integrated_volumes,
        component_probabilities=probability_result.component_probabilities,
        dropped_component_reasons=tuple(dropped_reasons),
        dropped_nonfinite=int(np.sum(nonfinite_mask)),
        dropped_nonpositive_volume=int(
            np.sum(nonpositive_volume_mask & ~nonfinite_mask)
        ),
        dropped_low_responsibility_neff=int(
            np.sum(low_neff_mask & ~nonfinite_mask & ~nonpositive_volume_mask)
        ),
    )
    return DirectionFilterResult(
        component_means=filtered_means,
        component_radii=filtered_radii,
        component_rotations=filtered_rotations,
        component_probabilities=probability_result.component_probabilities,
        component_integrated_volumes=probability_result.component_integrated_volumes,
        kernel=kernel,
        diagnostics=diagnostics,
    )


def _initial_means_by_weighted_projection(
        key: Any,
        rows: np.ndarray,
        weights: np.ndarray,
        n_components: int,
) -> np.ndarray:
    del key
    if n_components == 1:
        return (weights @ rows)[None, :]
    covariance = _weighted_global_covariance(rows, weights)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        projections = rows @ direction
    except np.linalg.LinAlgError:
        projections = rows[:, 0]
    order = np.argsort(projections)
    sorted_rows = rows[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    quantiles = (np.arange(n_components, dtype=float) + 0.5) / n_components
    indices = np.searchsorted(cumulative, quantiles, side="left")
    indices = np.clip(indices, 0, rows.shape[0] - 1)
    return sorted_rows[indices]


def _weighted_em_direction_backend(
        *,
        rows: np.ndarray,
        posterior_weights: np.ndarray,
        n_components: int,
        key: Any,
        max_iters: int = 100,
        tol: float = 1e-6,
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=float)
    weights = _normalize_positive_weights(np.asarray(posterior_weights, dtype=float))
    n_rows, d_dim = rows.shape
    means = _initial_means_by_weighted_projection(
        key=key,
        rows=rows,
        weights=weights,
        n_components=n_components,
    )
    global_covariance = _weighted_global_covariance(rows, weights)
    sigma2_floor = max(1e-12, 1e-6 * float(np.trace(global_covariance)) / d_dim)
    covariances = np.repeat(
        (global_covariance + sigma2_floor * np.eye(d_dim))[None, :, :],
        n_components,
        axis=0,
    )
    mixture_weights = np.full(n_components, 1.0 / n_components, dtype=float)
    responsibilities = np.full(
        (n_rows, n_components),
        1.0 / n_components,
        dtype=float,
    )
    converged = False

    for iteration in range(1, max_iters + 1):
        log_prob = np.empty((n_rows, n_components), dtype=float)
        for component_index in range(n_components):
            covariance = covariances[component_index]
            try:
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0.0:
                    raise np.linalg.LinAlgError("non-positive covariance")
                inverse = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                covariance = covariance + sigma2_floor * np.eye(d_dim)
                sign, logdet = np.linalg.slogdet(covariance)
                inverse = np.linalg.inv(covariance)
            centered = rows - means[component_index]
            mahalanobis = np.einsum("nd,de,ne->n", centered, inverse, centered)
            log_prob[:, component_index] = (
                math.log(max(mixture_weights[component_index], 1e-300))
                - 0.5 * (d_dim * math.log(2.0 * math.pi) + logdet + mahalanobis)
            )
        row_log_norm = _numpy_logsumexp(log_prob, axis=1)
        new_responsibilities = np.exp(log_prob - row_log_norm[:, None])
        effective_component_weights = weights[:, None] * new_responsibilities
        component_masses = np.sum(effective_component_weights, axis=0)
        component_masses = np.maximum(component_masses, 1e-300)
        new_mixture_weights = component_masses / np.sum(component_masses)
        new_means = effective_component_weights.T @ rows
        new_means = new_means / component_masses[:, None]
        new_covariances = np.empty_like(covariances)
        for component_index in range(n_components):
            centered = rows - new_means[component_index]
            component_weights = (
                effective_component_weights[:, component_index]
                / component_masses[component_index]
            )
            new_covariances[component_index] = np.einsum(
                "n,nd,ne->de",
                component_weights,
                centered,
                centered,
            ) + sigma2_floor * np.eye(d_dim)
        max_delta = max(
            float(np.max(np.abs(new_means - means))),
            float(np.max(np.abs(new_mixture_weights - mixture_weights))),
        )
        means = new_means
        covariances = new_covariances
        mixture_weights = new_mixture_weights
        responsibilities = new_responsibilities
        if max_delta < tol:
            converged = True
            break

    return {
        "component_means": means,
        "responsibilities": responsibilities,
        "em_iteration_count": iteration,
        "em_converged": converged,
    }


def _numpy_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return (
        np.squeeze(max_values, axis=axis)
        + np.log(np.sum(np.exp(values - max_values), axis=axis))
    )


def _fallback_fit_result(
        *,
        reason: str,
        previous_kernel: Any = None,
        diagnostics: DirectionFittingDiagnostics | None = None,
        hard_adaptation_errors: bool = False,
) -> DirectionKernelFitResult:
    if hard_adaptation_errors:
        raise RuntimeError(f"direction kernel adaptation failed: {reason}")
    if diagnostics is None:
        diagnostics = DirectionFittingDiagnostics()
    diagnostics = dataclasses.replace(
        diagnostics,
        active_kernel_mode=(
            "gmm" if previous_kernel is not None and _kernel_has_components(
                previous_kernel
            ) else "isotropic"
        ),
        fallback_active=True,
        fallback_reason=reason,
    )
    return DirectionKernelFitResult(
        kernel=previous_kernel if previous_kernel is not None else "isotropic",
        diagnostics=diagnostics,
    )


def fit_posterior_weighted_direction_gmm(
        *,
        key: Any,
        rows: Any,
        posterior_weights: Any,
        n_components: int | None = None,
        max_fit_rows: int = 20000,
        resampling_seed: int | None = None,
        previous_kernel: Any = None,
        hard_adaptation_errors: bool = False,
        em_backend: Any | None = None,
) -> DirectionKernelFitResult:
    raw_rows = _as_fitting_matrix(rows)
    raw_weights = np.asarray(posterior_weights, dtype=float).reshape((-1,))
    base_diagnostics = DirectionFittingDiagnostics(
        D_dim=int(raw_rows.shape[1]) if raw_rows.ndim == 2 else 0,
        M_D=int(max_fit_rows),
        resampling_seed=resampling_seed,
    )
    if raw_rows.shape[0] != raw_weights.shape[0]:
        raise ValueError("rows and posterior_weights length mismatch.")
    try:
        dataset = build_direction_fitting_dataset(
            samples_U=raw_rows,
            posterior_weights=raw_weights,
            max_fit_rows=max_fit_rows,
            resampling_seed=resampling_seed,
        )
    except ValueError as error:
        return _fallback_fit_result(
            reason=str(error),
            previous_kernel=previous_kernel,
            diagnostics=base_diagnostics,
            hard_adaptation_errors=hard_adaptation_errors,
        )
    diagnostics = dataset.diagnostics
    if np.any(raw_weights < 0.0):
        return _fallback_fit_result(
            reason="negative_posterior_weights",
            previous_kernel=previous_kernel,
            diagnostics=diagnostics,
            hard_adaptation_errors=hard_adaptation_errors,
        )
    if diagnostics.original_N_pos == 0:
        if diagnostics.excluded_nonfinite_rows and np.any(raw_weights > 0.0):
            return _fallback_fit_result(
                reason="nonfinite_coordinates",
                previous_kernel=previous_kernel,
                diagnostics=diagnostics,
                hard_adaptation_errors=hard_adaptation_errors,
            )
        return _fallback_fit_result(
            reason="no_positive_posterior_weights",
            previous_kernel=previous_kernel,
            diagnostics=diagnostics,
            hard_adaptation_errors=hard_adaptation_errors,
        )

    d_dim = diagnostics.D_dim
    original_nonfinite_rows = diagnostics.excluded_nonfinite_rows
    if diagnostics.N_pos < d_dim + 1:
        reason = (
            "nonfinite_coordinates"
            if original_nonfinite_rows
            else "insufficient_N_pos"
        )
        return _fallback_fit_result(
            reason=reason,
            previous_kernel=previous_kernel,
            diagnostics=diagnostics,
            hard_adaptation_errors=hard_adaptation_errors,
        )
    min_effective_size = max(20.0, float(2 * (d_dim + 1)))
    if diagnostics.original_N_eff < min_effective_size:
        return _fallback_fit_result(
            reason="insufficient_N_eff",
            previous_kernel=previous_kernel,
            diagnostics=diagnostics,
            hard_adaptation_errors=hard_adaptation_errors,
        )
    if diagnostics.unique_finite_row_count < d_dim + 1:
        return _fallback_fit_result(
            reason="insufficient_unique_rows",
            previous_kernel=previous_kernel,
            diagnostics=diagnostics,
            hard_adaptation_errors=hard_adaptation_errors,
        )

    if n_components is None:
        count = choose_direction_component_count(
            n_eff=diagnostics.original_N_eff,
            d_dim=d_dim,
            unique_row_count=diagnostics.unique_finite_row_count,
        )
        requested_k = count.requested_K_D
        final_k = count.final_K_D
    else:
        requested_k = int(n_components)
        final_k = min(requested_k, diagnostics.unique_finite_row_count)
    if final_k < 1:
        return _fallback_fit_result(
            reason="insufficient_unique_rows",
            previous_kernel=previous_kernel,
            diagnostics=dataclasses.replace(
                diagnostics,
                requested_K_D=requested_k,
                final_K_D=final_k,
            ),
            hard_adaptation_errors=hard_adaptation_errors,
        )

    backend = em_backend or _weighted_em_direction_backend
    try:
        backend_result = backend(
            rows=dataset.rows,
            posterior_weights=dataset.weights,
            n_components=final_k,
            key=key,
        )
        responsibilities = np.asarray(backend_result["responsibilities"], dtype=float)
        covariance_components = direction_covariance_components(
            rows=dataset.rows,
            posterior_weights=dataset.weights,
            responsibilities=responsibilities,
        )
        responsibility_effective_sizes = (
            covariance_components.responsibility_effective_sizes
        )
        filter_result = filter_direction_components(
            component_means=backend_result.get(
                "component_means",
                covariance_components.component_means,
            ),
            component_radii=covariance_components.component_radii,
            component_rotations=covariance_components.component_rotations,
            responsibility_effective_sizes=responsibility_effective_sizes,
            d_dim=d_dim,
            previous_kernel=previous_kernel,
        )
    except Exception:
        return _fallback_fit_result(
            reason=(
                "nonfinite_covariance"
                if original_nonfinite_rows
                else "covariance_failure"
            ),
            previous_kernel=previous_kernel,
            diagnostics=diagnostics,
            hard_adaptation_errors=hard_adaptation_errors,
        )

    if filter_result.diagnostics.fallback_active:
        return DirectionKernelFitResult(
            kernel=filter_result.kernel,
            diagnostics=dataclasses.replace(
                diagnostics,
                fallback_active=True,
                fallback_reason=filter_result.diagnostics.fallback_reason,
                N_pos=diagnostics.original_N_pos,
                N_eff=diagnostics.original_N_eff,
                dropped_component_reasons=(
                    filter_result.diagnostics.dropped_component_reasons
                ),
                dropped_nonfinite=filter_result.diagnostics.dropped_nonfinite,
                dropped_nonpositive_volume=(
                    filter_result.diagnostics.dropped_nonpositive_volume
                ),
                dropped_low_responsibility_neff=(
                    filter_result.diagnostics.dropped_low_responsibility_neff
                ),
                requested_K_D=requested_k,
                final_K_D=0,
                em_iteration_count=int(
                    backend_result.get("em_iteration_count", 0)
                ),
                em_converged=bool(backend_result.get("em_converged", False)),
                sigma2_floor=covariance_components.diagnostics.sigma2_floor,
            ),
        )

    final_diagnostics = dataclasses.replace(
        diagnostics,
        active_kernel_mode="gmm",
        active_kernel_version=1,
        fallback_active=False,
        fallback_reason=None,
        N_pos=diagnostics.original_N_pos,
        N_eff=diagnostics.original_N_eff,
        requested_K_D=requested_k,
        final_K_D=filter_result.diagnostics.final_K_D,
        component_means=filter_result.component_means,
        component_radii=filter_result.component_radii,
        component_integrated_volumes=filter_result.component_integrated_volumes,
        component_probabilities=filter_result.component_probabilities,
        dropped_component_reasons=filter_result.diagnostics.dropped_component_reasons,
        dropped_nonfinite=filter_result.diagnostics.dropped_nonfinite,
        dropped_nonpositive_volume=(
            filter_result.diagnostics.dropped_nonpositive_volume
        ),
        dropped_low_responsibility_neff=(
            filter_result.diagnostics.dropped_low_responsibility_neff
        ),
        em_iteration_count=int(backend_result.get("em_iteration_count", 0)),
        em_converged=bool(backend_result.get("em_converged", False)),
        sigma2_floor=covariance_components.diagnostics.sigma2_floor,
    )
    return DirectionKernelFitResult(
        kernel=filter_result.kernel,
        diagnostics=final_diagnostics,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionKernelAdaptationState:
    active_kernel: Any = "isotropic"
    active_kernel_mode: str = "isotropic"
    active_kernel_version: int = 0
    update_every_shells: int = 5
    attempted_update_count: int = 0
    successful_update_count: int = 0
    distinct_shell_count: int = 0
    last_successful_shell_count: int = 0
    last_log_likelihood: float | None = None
    fallback_reason: str | None = None

    @classmethod
    def initial(
            cls,
            *,
            active_kernel: Any = "isotropic",
            active_kernel_version: int = 0,
            update_every_shells: int = 5,
    ) -> "DirectionKernelAdaptationState":
        mode = "gmm" if _kernel_has_components(active_kernel) else "isotropic"
        return cls(
            active_kernel=active_kernel,
            active_kernel_mode=mode,
            active_kernel_version=int(active_kernel_version),
            update_every_shells=int(update_every_shells),
        )

    def dispatch_snapshot(
            self,
            *,
            chain_id: str,
            worker_id: str | None = None,
            shell_epoch: int | None = None,
            allocation_target: str | None = None,
    ) -> DirectionKernelDispatchSnapshot:
        resolved_shell_epoch = (
            self.distinct_shell_count if shell_epoch is None else int(shell_epoch)
        )
        diagnostics = DirectionFittingDiagnostics(
            active_kernel_mode=self.active_kernel_mode,
            active_kernel_version=self.active_kernel_version,
            attempted_update_count=self.attempted_update_count,
            successful_update_count=self.successful_update_count,
            fallback_reason=self.fallback_reason,
            shell_epoch=resolved_shell_epoch,
            allocation_target=allocation_target,
        )
        return DirectionKernelDispatchSnapshot(
            chain_id=chain_id,
            worker_id=worker_id,
            shell_epoch=resolved_shell_epoch,
            allocation_target=allocation_target,
            kernel=self.active_kernel,
            kernel_version=self.active_kernel_version,
            diagnostics=diagnostics,
        )

    def observe_completed_shell(
            self,
            *,
            log_likelihood: float,
            fit_dataset_ready: bool,
            fit_success: bool,
    ) -> "DirectionKernelAdaptationState":
        is_distinct = (
            self.last_log_likelihood is None
            or float(log_likelihood) != float(self.last_log_likelihood)
        )
        distinct_shell_count = self.distinct_shell_count + int(is_distinct)
        state = dataclasses.replace(
            self,
            distinct_shell_count=distinct_shell_count,
            last_log_likelihood=float(log_likelihood),
        )
        if not is_distinct or not fit_dataset_ready:
            return state
        shells_since_success = (
            distinct_shell_count - state.last_successful_shell_count
        )
        eligible = (
            state.successful_update_count == 0
            or shells_since_success >= state.update_every_shells
        )
        if not eligible:
            return state
        attempted_update_count = state.attempted_update_count + 1
        if fit_success:
            return dataclasses.replace(
                state,
                active_kernel_mode=(
                    "gmm" if _kernel_has_components(state.active_kernel)
                    else "isotropic"
                ),
                active_kernel_version=state.active_kernel_version + 1,
                attempted_update_count=attempted_update_count,
                successful_update_count=state.successful_update_count + 1,
                last_successful_shell_count=distinct_shell_count,
                fallback_reason=None,
            )
        return dataclasses.replace(
            state,
            attempted_update_count=attempted_update_count,
            fallback_reason="fit_failed",
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DirectionKernelAdaptationCoordinator:
    active_kernel: Any = "isotropic"
    active_kernel_mode: str = "isotropic"
    active_kernel_version: int = 0
    update_every_shells: int = 5
    attempted_update_count: int = 0
    successful_update_count: int = 0
    fallback_reason: str | None = None

    @classmethod
    def initial(
            cls,
            *,
            active_kernel: Any = "isotropic",
            active_kernel_version: int = 0,
            update_every_shells: int = 5,
    ) -> "DirectionKernelAdaptationCoordinator":
        return cls(
            active_kernel=active_kernel,
            active_kernel_mode=(
                "gmm" if _kernel_has_components(active_kernel) else "isotropic"
            ),
            active_kernel_version=int(active_kernel_version),
            update_every_shells=int(update_every_shells),
        )

    def prepare_dispatch_snapshot(
            self,
            request: DirectionKernelDispatchRequest,
    ) -> DirectionKernelDispatchSnapshot:
        diagnostics = DirectionFittingDiagnostics(
            active_kernel_mode=self.active_kernel_mode,
            active_kernel_version=self.active_kernel_version,
            attempted_update_count=self.attempted_update_count,
            successful_update_count=self.successful_update_count,
            fallback_reason=self.fallback_reason,
            shell_epoch=int(request.shell_epoch),
            allocation_target=request.allocation_target,
        )
        return DirectionKernelDispatchSnapshot(
            chain_id=request.chain_id,
            worker_id=request.worker_id,
            shell_epoch=int(request.shell_epoch),
            allocation_target=request.allocation_target,
            kernel=self.active_kernel,
            kernel_version=self.active_kernel_version,
            diagnostics=diagnostics,
        )

    def replace_active_kernel(
            self,
            *,
            kernel: Any,
            shell_epoch: int,
            update_reason: str,
    ) -> "DirectionKernelAdaptationCoordinator":
        del shell_epoch
        del update_reason
        return dataclasses.replace(
            self,
            active_kernel=kernel,
            active_kernel_mode="gmm" if _kernel_has_components(kernel) else "isotropic",
            active_kernel_version=self.active_kernel_version + 1,
            fallback_reason=None,
        )

    def request_direction_kernel_fit(
            self,
            request: DirectionKernelFitRequest,
    ) -> DirectionKernelFitResult:
        attempted_update_count = self.attempted_update_count + 1
        fit_result = fit_posterior_weighted_direction_gmm(
            key=request.key,
            rows=request.samples_U,
            posterior_weights=request.posterior_weights,
            n_components=request.n_components,
            max_fit_rows=request.max_fit_rows,
            resampling_seed=request.resampling_seed,
            previous_kernel=self.active_kernel,
            hard_adaptation_errors=request.hard_adaptation_errors,
        )
        if fit_result.diagnostics.fallback_active:
            updated = dataclasses.replace(
                self,
                attempted_update_count=attempted_update_count,
                fallback_reason=fit_result.diagnostics.fallback_reason,
            )
            diagnostics = dataclasses.replace(
                fit_result.diagnostics,
                attempted_update_count=attempted_update_count,
                successful_update_count=self.successful_update_count,
                active_kernel_version=self.active_kernel_version,
                shell_epoch=int(request.shell_epoch),
                allocation_target=request.allocation_target,
            )
            return DirectionKernelFitResult(
                kernel=fit_result.kernel,
                diagnostics=diagnostics,
                coordinator=updated,
            )

        updated = dataclasses.replace(
            self,
            active_kernel=fit_result.kernel,
            active_kernel_mode="gmm",
            active_kernel_version=self.active_kernel_version + 1,
            attempted_update_count=attempted_update_count,
            successful_update_count=self.successful_update_count + 1,
            fallback_reason=None,
        )
        diagnostics = dataclasses.replace(
            fit_result.diagnostics,
            attempted_update_count=attempted_update_count,
            successful_update_count=updated.successful_update_count,
            active_kernel_version=updated.active_kernel_version,
            shell_epoch=int(request.shell_epoch),
            allocation_target=request.allocation_target,
        )
        return DirectionKernelFitResult(
            kernel=fit_result.kernel,
            diagnostics=diagnostics,
            coordinator=updated,
        )
