"""Measure plain and retained-disabled isotropic direction execution."""

from __future__ import annotations

import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# A benchmark should measure the checkout containing this script, independent
# of an editable installation for another worktree.
sys.path.insert(0, str(REPO_ROOT / "src"))

from jaxns.constrained_sampler import UniDimSliceSampler, sample_request
from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.samples import SeedPoint
from jaxns.sampling.continuation import _initialise_slice_chains
from jaxns.sampling.ellipsoid import empty_sampler_data
from jaxns.sampling.protocol import ConstrainedSampleRequest


@dataclasses.dataclass(slots=True, frozen=True)
class QuadraticModel(PureDataclassPytree):
    """Small traceable likelihood for deterministic direction comparisons."""

    centre: jax.Array  # [D]

    def log_likelihood(
            self,
            U,
            args=(),
            params=None,
            *,
            allow_nan=True,
    ):
        del args, params, allow_nan
        return -jnp.sum(jnp.square(U - self.centre))


QuadraticModel.register_pytree()


def make_request(
        model: QuadraticModel,
        width: int,
) -> ConstrainedSampleRequest:
    """Construct one deterministic valid constrained-sampling request."""
    seeds = jnp.linspace(
        0.35,
        0.65,
        width * 2,
        dtype=mp_policy.measure_dtype,
    ).reshape((width, 2))  # [S, D]
    log_likelihoods = jax.vmap(model.log_likelihood)(seeds)  # [S]
    return ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(290), width),  # [S, 2]
        valid=jnp.ones((width,), dtype=mp_policy.bool_dtype),  # [S]
        log_L_constraints=jnp.full(
            (width,),
            -0.25,
            dtype=mp_policy.measure_dtype,
        ),  # [S]
        seed_points=SeedPoint(
            U0=seeds,
            log_L0=log_likelihoods,
        ),
        sampler_data=None,
    )


def retain_disabled_fit(
        request: ConstrainedSampleRequest,
) -> ConstrainedSampleRequest:
    """Attach valid anisotropic geometry while selecting isotropic execution."""
    data = empty_sampler_data(num_components=1, dimension=2)
    data = dataclasses.replace(
        data,
        radii=jnp.asarray([[8.0, 1.0]], mp_policy.measure_dtype),  # [K, D]
        rotations=jnp.eye(2, dtype=mp_policy.measure_dtype)[None],  # [K, D, D]
        log_volumes=jnp.zeros((1,), mp_policy.measure_dtype),  # [K]
        log_L_at_mean=jnp.ones((1,), mp_policy.measure_dtype),  # [K]
        valid=jnp.ones((1,), dtype=mp_policy.bool_dtype),  # [K]
        enabled=jnp.asarray(False, mp_policy.bool_dtype),
        iso_prob=jnp.asarray(0.0, mp_policy.measure_dtype),
    )
    return dataclasses.replace(request, sampler_data=data)


def lower_and_compile(function, request):
    """Lower and compile separately, returning timing and optimized HLO."""
    jitted = jax.jit(function)
    started = time.perf_counter()
    lowered = jitted.lower(request)
    lower_seconds = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - started
    hlo = compiled.as_text()
    return compiled, {
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "optimized_hlo_bytes": len(hlo.encode()),
        "optimized_hlo_dot_count": hlo.count(" dot("),
        "optimized_hlo_conditional_count": hlo.count("conditional("),
    }


def assert_tree_close(expected, actual) -> None:
    """Require identical discrete values and at most reduction roundoff."""
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected),
        jax.tree.leaves(actual),
        strict=True,
    ):
        expected_array = np.asarray(expected_leaf)
        actual_array = np.asarray(actual_leaf)
        if np.issubdtype(expected_array.dtype, np.inexact):
            tolerance = 32 * np.finfo(expected_array.dtype).eps
            np.testing.assert_allclose(
                actual_array,
                expected_array,
                rtol=tolerance,
                atol=tolerance,
            )
        else:
            np.testing.assert_array_equal(actual_array, expected_array)


def verify_outputs(
        scenario: str,
        plain_output,
        retained_output,
        *,
        num_slices: int,
) -> None:
    """Verify the disabled branch preserves the plain isotropic key stream."""
    if scenario == "continuation_initializer":
        assert_tree_close(plain_output, retained_output)
        return

    direction_counts = retained_output.num_directions  # [S]
    np.testing.assert_array_equal(
        direction_counts,
        np.full(direction_counts.shape, num_slices),
    )
    np.testing.assert_array_equal(
        retained_output.num_isotropic,
        direction_counts,
    )
    retained_without_diagnostics = dataclasses.replace(
        retained_output,
        num_directions=plain_output.num_directions,
        num_isotropic=plain_output.num_isotropic,
    )
    assert_tree_close(plain_output, retained_without_diagnostics)


def measure_scenario(
        scenario: str,
        function,
        plain_request: ConstrainedSampleRequest,
        retained_request: ConstrainedSampleRequest,
        *,
        num_slices: int,
        warmups: int,
        repetitions: int,
) -> dict[str, object]:
    """Compile variants, warm them, then alternate synchronized executions."""
    plain_compiled, plain_record = lower_and_compile(
        function,
        plain_request,
    )
    retained_compiled, retained_record = lower_and_compile(
        function,
        retained_request,
    )

    plain_output = None
    retained_output = None
    for warmup in range(warmups):
        if warmup % 2 == 0:
            plain_output = plain_compiled(plain_request)
            jax.block_until_ready(plain_output)
            retained_output = retained_compiled(retained_request)
            jax.block_until_ready(retained_output)
        else:
            retained_output = retained_compiled(retained_request)
            jax.block_until_ready(retained_output)
            plain_output = plain_compiled(plain_request)
            jax.block_until_ready(plain_output)
    verify_outputs(
        scenario,
        plain_output,
        retained_output,
        num_slices=num_slices,
    )

    plain_times = []
    retained_times = []
    for repetition in range(repetitions):
        if repetition % 2 == 0:
            started = time.perf_counter()
            plain_output = plain_compiled(plain_request)
            jax.block_until_ready(plain_output)
            plain_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            retained_output = retained_compiled(retained_request)
            jax.block_until_ready(retained_output)
            retained_times.append(time.perf_counter() - started)
        else:
            started = time.perf_counter()
            retained_output = retained_compiled(retained_request)
            jax.block_until_ready(retained_output)
            retained_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            plain_output = plain_compiled(plain_request)
            jax.block_until_ready(plain_output)
            plain_times.append(time.perf_counter() - started)

    plain_median = statistics.median(plain_times)
    retained_median = statistics.median(retained_times)
    plain_range = [min(plain_times), max(plain_times)]
    retained_range = [min(retained_times), max(retained_times)]
    plain_record.update({
        "runtime_median_seconds": plain_median,
        "runtime_range_seconds": plain_range,
        "runtime_min_seconds": plain_range[0],
        "runtime_max_seconds": plain_range[1],
    })
    retained_record.update({
        "runtime_median_seconds": retained_median,
        "runtime_range_seconds": retained_range,
        "runtime_min_seconds": retained_range[0],
        "runtime_max_seconds": retained_range[1],
    })
    return {
        "scenario": scenario,
        "num_chains": plain_request.log_L_constraints.shape[0],
        "num_slices": num_slices,
        "plain": plain_record,
        "retained_disabled": retained_record,
        "retained_over_plain_runtime": retained_median / plain_median,
        "retained_over_plain_hlo_bytes": (
            retained_record["optimized_hlo_bytes"]
            / plain_record["optimized_hlo_bytes"]
        ),
    }


def main() -> None:
    """Print environment metadata and two direction-toggle records as JSON."""
    repetitions = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    warmups = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if repetitions < 1:
        raise ValueError("At least one measured repetition is required.")
    if warmups < 2:
        raise ValueError("At least two synchronized warmups are required.")

    model = QuadraticModel(
        centre=jnp.asarray([0.45, 0.55], mp_policy.measure_dtype)
    )
    continuation_sampler = UniDimSliceSampler(
        model=model,
        num_slices=32,
        collect_phantom_samples=True,
        max_phantom_samples=2,
    )
    continuation_plain = make_request(model, width=8)
    continuation_retained = retain_disabled_fit(continuation_plain)
    continuation_record = measure_scenario(
        "continuation_initializer",
        lambda request: _initialise_slice_chains(
            continuation_sampler,
            request,
        ),
        continuation_plain,
        continuation_retained,
        num_slices=continuation_sampler.num_slices,
        warmups=warmups,
        repetitions=repetitions,
    )

    reference_sampler = UniDimSliceSampler(
        model=model,
        num_slices=8,
        collect_phantom_samples=True,
        max_phantom_samples=2,
    )
    reference_plain = make_request(model, width=2)
    reference_retained = retain_disabled_fit(reference_plain)
    reference_record = measure_scenario(
        "complete_reference",
        lambda request: sample_request(reference_sampler, request),
        reference_plain,
        reference_retained,
        num_slices=reference_sampler.num_slices,
        warmups=warmups,
        repetitions=repetitions,
    )

    output = {
        "metadata": {
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.x64_enabled),
            "measure_dtype": str(jnp.dtype(mp_policy.measure_dtype)),
            "warmups_per_variant": warmups,
            "measured_repetitions_per_variant": repetitions,
        },
        "records": [continuation_record, reference_record],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
