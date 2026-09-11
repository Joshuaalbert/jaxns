"""Measure one selector variant on the same saved A tree, on one pinned core."""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import pickle
import platform
import resource
import time
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

import jaxns
from benchmarks.phantom_seeding.block_eager import block_eager_stationary_seeds
from benchmarks.phantom_seeding.reference import reference_stationary_seeds
from jaxns.algorithm import depth
from jaxns.depth_condition import DepthCondition
from jaxns.sampling.phantom_index import build_phantom_seed_index
from jaxns.sampling.phantom_seeds import gather_seed_points

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument(
    "--variant", choices=("reference", "lazy", "blocks", "combined"), required=True
)
parser.add_argument("--block-size", type=int, default=256)
parser.add_argument("--shell-size", type=int, default=100)
parser.add_argument("--queries", choices=("mixed", "recent", "root"), default="mixed")
parser.add_argument("--repeats", type=int, default=11)
parser.add_argument(
    "--phantom-slots", type=int, help="Selector-only prefix projection."
)
parser.add_argument(
    "--coordinate-repeats",
    type=int,
    default=1,
    help="Repeat U coordinates to measure gather/storage scaling only.",
)
parser.add_argument("--capacity-factor", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
assert Path(jaxns.__file__).resolve().parents[2] == root
assert len(os.sched_getaffinity(0)) == 1 and jax.config.jax_enable_x64
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(args.output)
source_hashes = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((root / "src/jaxns").rglob("*.py"))
}
benchmark_hashes = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(Path(__file__).parent.glob("*.py"))
}
with args.state.open("rb") as stream:
    state = pickle.load(stream)
assert state.phantom_seed_index is None
assert args.coordinate_repeats >= 1 and args.capacity_factor >= 1
# These synthetic views exercise the selector only, never the model/sampler.
# Both implementations receive the identical changed population and geometry.
if args.phantom_slots is not None:
    phantoms = state.samples.phantom_samples
    assert 1 <= args.phantom_slots <= phantoms.log_L.shape[1]
    likelihood = phantoms.log_L[:, : args.phantom_slots]
    valid = phantoms.valid_mask[:, : args.phantom_slots]
    state = dataclasses.replace(
        state,
        samples=dataclasses.replace(
            state.samples,
            phantom_samples=dataclasses.replace(
                phantoms,
                log_L=likelihood,
                valid_mask=valid,
                U_samples=jax.tree.map(
                    lambda u: u[:, : args.phantom_slots], phantoms.U_samples
                ),
                seed_log_L_sorted=jnp.sort(jnp.where(valid, likelihood, -jnp.inf)),
            ),
        ),
    )
if args.coordinate_repeats > 1:
    state = dataclasses.replace(
        state,
        samples=dataclasses.replace(
            state.samples,
            U_samples=jax.tree.map(
                lambda u: jnp.concatenate([u] * args.coordinate_repeats, axis=-1),
                state.samples.U_samples,
            ),
            phantom_samples=dataclasses.replace(
                state.samples.phantom_samples,
                U_samples=jax.tree.map(
                    lambda u: jnp.concatenate([u] * args.coordinate_repeats, axis=-1),
                    state.samples.phantom_samples.U_samples,
                ),
            ),
        ),
    )
if args.capacity_factor > 1:
    state = state.resize(state.samples.log_likelihoods.size * args.capacity_factor)
jax.block_until_ready(state)
accepted = int(state.num_samples)
capacity = state.samples.log_likelihoods.size
load_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
births = np.asarray(state.samples.log_L_constraints)[:accepted]
if args.queries == "root":
    constraints = np.full(args.shell_size, -np.inf)
elif args.queries == "recent":
    constraints = births[slice(-args.shell_size, None)]
else:
    finite = births[np.isfinite(births)]
    constraints = np.quantile(finite, np.linspace(0.0, 1.0, args.shell_size))
constraints = jnp.asarray(constraints)
setup_started = time.perf_counter()
index_build_seconds = 0.0
if args.variant in ("blocks", "combined"):
    started = time.perf_counter()
    index = jax.block_until_ready(
        build_phantom_seed_index(
            state.samples,
            state.num_samples,
            block_size=args.block_size,
        )
    )
    index_build_seconds = time.perf_counter() - started
    state = dataclasses.replace(
        state,
        phantom_seed_index=index,
        samples=dataclasses.replace(
            state.samples,
            phantom_samples=dataclasses.replace(
                state.samples.phantom_samples,
                seed_log_L_sorted=None,
            ),
        ),
    )
    del index
    gc.collect()


@jax.jit
def freeze(state):
    blocks, plan, relevant, tail = depth._build_depth_view(
        state,
        DepthCondition(dlogZ=jnp.log1p(0.001)),
        allocation_target="uniform",
        root_degree=300,
        delta_K=1,
    )
    return depth._new_thread_schedule(
        state,
        blocks,
        plan,
        relevant,
        shell_size=args.shell_size,
        tail_K=tail,
    )


schedule = jax.block_until_ready(freeze(state))
selector = {
    "reference": reference_stationary_seeds,
    "lazy": depth._sample_stationary_seeds,
    "blocks": block_eager_stationary_seeds,
    "combined": depth._sample_stationary_seeds,
}[args.variant]


@jax.jit
def select(state, schedule, key, constraints):
    size = constraints.size
    identities = selector(
        key,
        state,
        schedule,
        constraints,
        jnp.isneginf(constraints),
        jnp.ones(size, dtype=bool),
        jnp.full(size, -1),
        jnp.full(size, -jnp.inf),
        jnp.zeros(size, dtype=bool),
    )
    return identities, gather_seed_points(state.samples, identities)


setup_seconds = time.perf_counter() - setup_started
started = time.perf_counter()
lowered = select.lower(state, schedule, jax.random.PRNGKey(0), constraints)
lower_seconds = time.perf_counter() - started
started = time.perf_counter()
compiled = lowered.compile()
compile_seconds = time.perf_counter() - started
memory = compiled.memory_analysis()
args.output.with_suffix(".hlo.txt").write_text(compiled.as_text())
for _ in range(2):
    jax.block_until_ready(
        compiled(state, schedule, jax.random.PRNGKey(99), constraints)
    )
# Linux resets this process's high-water RSS only; separate warm execution
# from the often larger pickle-loading peak. Keep total setup peaks too.
setup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
Path("/proc/self/clear_refs").write_text("5\n")
seconds, identities, coordinates, likelihoods = [], [], [], []
for repeat in range(args.repeats):
    key = jax.random.PRNGKey(100 + repeat)
    started = time.perf_counter()
    selected, points = jax.block_until_ready(
        compiled(state, schedule, key, constraints)
    )
    seconds.append(time.perf_counter() - started)
    identities.append(np.asarray(selected))
    coordinates.append(
        np.concatenate(
            [
                np.asarray(leaf).reshape((args.shell_size, -1))
                for leaf in jax.tree.leaves(points.U0)
            ],
            axis=1,
        )
    )
    likelihoods.append(np.asarray(points.log_L0))
np.savez_compressed(
    args.output.with_suffix(".npz"),
    identities=identities,
    U=coordinates,
    log_L=likelihoods,
    constraints=np.asarray(constraints),
)
rss_status = {}
for line in Path("/proc/self/status").read_text().splitlines():
    if line.startswith(("VmRSS:", "VmHWM:")):
        name, value, unit = line.split()
        assert unit == "kB"
        rss_status[name.rstrip(":")] = int(value) * 1024
record = {
    "variant": args.variant,
    "block_size": args.block_size,
    "queries": args.queries,
    "synthetic_view": {
        "phantom_slots": args.phantom_slots,
        "coordinate_repeats": args.coordinate_repeats,
        "capacity_factor": args.capacity_factor,
    },
    "shell_size": args.shell_size,
    "accepted": accepted,
    "capacity": capacity,
    "phantom_slots": state.samples.phantom_samples.log_L.shape[1],
    "dimension": coordinates[0].shape[1],
    "state_path": str(args.state),
    "source_root": str(root),
    "affinity": sorted(os.sched_getaffinity(0)),
    "hostname": platform.node(),
    "machine": platform.machine(),
    "jax": jax.__version__,
    "backend": jax.default_backend(),
    "x64": jax.config.jax_enable_x64,
    "seconds": seconds,
    "median_seconds": float(np.median(seconds)),
    "quartiles_seconds": np.quantile(seconds, [0.25, 0.75]).tolist(),
    "setup_seconds": setup_seconds,
    "index_build_including_compile_seconds": index_build_seconds,
    "lower_seconds": lower_seconds,
    "compile_seconds": compile_seconds,
    "compiler_argument_bytes": memory.argument_size_in_bytes,
    "compiler_output_bytes": memory.output_size_in_bytes,
    "compiler_temporary_bytes": memory.temp_size_in_bytes,
    "compiler_alias_bytes": memory.alias_size_in_bytes,
    "state_array_bytes": sum(leaf.nbytes for leaf in jax.tree.leaves(state)),
    "load_peak_rss_bytes": load_peak,
    "setup_peak_rss_bytes": setup_peak,
    "warm_peak_rss_bytes": rss_status["VmHWM"],
    "warm_final_rss_bytes": rss_status["VmRSS"],
    "total_peak_rss_bytes": max(
        setup_peak,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    ),
    "benchmark_sha256": benchmark_hashes,
    "source_sha256": source_hashes,
}
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(
    json.dumps({key: value for key, value in record.items() if key != "source_sha256"})
)
