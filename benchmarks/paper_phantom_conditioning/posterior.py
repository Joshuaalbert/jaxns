"""Classic-posterior diagnostics for the paper experiment."""

import numpy as np


def classic_mode_mass(
        log_dp: np.ndarray,
        first_mode: np.ndarray | None,
) -> float | None:
    """Sum classic expected posterior weights inside the first mode.

    The completed race tree owns this diagnostic. It is intentionally
    independent of all phantom-prefix evidence calculations.
    """
    if first_mode is None:
        return None
    log_weights = np.asarray(log_dp)
    membership = np.asarray(first_mode, dtype=bool)
    if log_weights.ndim != 1 or membership.shape != log_weights.shape:
        raise ValueError("Mode membership must match the classic samples.")
    return float(np.sum(np.exp(log_weights[membership])))
