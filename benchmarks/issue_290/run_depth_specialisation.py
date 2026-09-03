"""Prove unused allocation policy must not specialise compiled depth."""

import json
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from benchmarks.issue_290.run_scheduler import make_state
from cicd.tests.test_core import DeterministicSampler
from jaxns.algorithm.depth import _run_depth, _start_schedule_round
from jaxns.depth_condition import DepthCondition
from jaxns.mixed_precision import mp_policy


@partial(
    jax.jit,
    inline=True,
    static_argnames=(
        "shell_size",
        "allocation_target",
        "root_degree",
        "delta_K",
        "max_samples",
    ),
)
def legacy_specialisation(
        state,
        sampler,
        depth_cond,
        *,
        shell_size,
        allocation_target,
        root_degree,
        delta_K,
        max_samples,
):
    """Reference the former cache key while leaving its dead values unused."""
    del shell_size, allocation_target, root_degree, delta_K
    return _run_depth(
        state,
        sampler,
        depth_cond,
        max_samples=max_samples,
    )


def main() -> None:
    """Compare executable-cache cardinality for two dead policy values."""
    width = 8
    capacity = 32
    condition = DepthCondition()
    state = make_state(
        16,
        width,
        2,
        include_allocation_iteration=True,
    )
    state = _start_schedule_round(
        state,
        condition,
        shell_size=width,
        allocation_target="uniform",
        root_degree=width,
        delta_K=1,
    )
    sampler = DeterministicSampler()

    _run_depth.clear_cache()
    started = time.perf_counter()
    current_first = _run_depth(
        state,
        sampler,
        condition,
        max_samples=capacity,
    )
    jax.block_until_ready(current_first)
    current_first_seconds = time.perf_counter() - started
    current_cache_first = _run_depth._cache_size()
    started = time.perf_counter()
    current_second = _run_depth(
        state,
        sampler,
        condition,
        max_samples=capacity,
    )
    jax.block_until_ready(current_second)
    current_second_seconds = time.perf_counter() - started
    current_cache_second = _run_depth._cache_size()

    legacy_specialisation.clear_cache()
    started = time.perf_counter()
    legacy_first = legacy_specialisation(
        state,
        sampler,
        condition,
        shell_size=width,
        allocation_target="uniform",
        root_degree=width,
        delta_K=1,
        max_samples=capacity,
    )
    jax.block_until_ready(legacy_first)
    legacy_first_seconds = time.perf_counter() - started
    legacy_cache_first = legacy_specialisation._cache_size()
    started = time.perf_counter()
    legacy_second = legacy_specialisation(
        state,
        sampler,
        condition,
        shell_size=4 * width,
        allocation_target="posterior_improving",
        root_degree=2 * width,
        delta_K=width,
        max_samples=capacity,
    )
    jax.block_until_ready(legacy_second)
    legacy_second_seconds = time.perf_counter() - started
    legacy_cache_second = legacy_specialisation._cache_size()

    current_hlo = _run_depth.lower(
        state,
        sampler,
        condition,
        max_samples=capacity,
    ).as_text()
    legacy_hlo_first = legacy_specialisation.lower(
        state,
        sampler,
        condition,
        shell_size=width,
        allocation_target="uniform",
        root_degree=width,
        delta_K=1,
        max_samples=capacity,
    ).as_text()
    legacy_hlo_second = legacy_specialisation.lower(
        state,
        sampler,
        condition,
        shell_size=4 * width,
        allocation_target="posterior_improving",
        root_degree=2 * width,
        delta_K=width,
        max_samples=capacity,
    ).as_text()
    print(json.dumps({
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "x64": bool(jax.config.x64_enabled),
        "measure_dtype": str(jnp.dtype(mp_policy.measure_dtype)),
        "current_cache_after_first": current_cache_first,
        "current_cache_after_second": current_cache_second,
        "legacy_cache_after_first": legacy_cache_first,
        "legacy_cache_after_second": legacy_cache_second,
        "current_first_seconds": current_first_seconds,
        "current_second_seconds": current_second_seconds,
        "legacy_first_seconds": legacy_first_seconds,
        "legacy_second_seconds": legacy_second_seconds,
        "current_hlo_bytes": len(current_hlo.encode()),
        "legacy_first_hlo_bytes": len(legacy_hlo_first.encode()),
        "legacy_second_hlo_bytes": len(legacy_hlo_second.encode()),
    }))


if __name__ == "__main__":
    main()
