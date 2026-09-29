"""Paired evidence benchmark for the phantom-conditioning paper."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import jax
import numpy as np
from jax import numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
# An explicit PYTHONPATH may select an experimental implementation worktree
# while the benchmark definitions remain in this checkout.
sys.path.append(str(REPO_ROOT))

import jaxns
from benchmarks.paper_phantom_conditioning.cases import (
    CSS8_COMPONENT_COVARIANCES,
    CSS8_COMPONENT_MEANS,
    CSS8_PRIOR_COVARIANCE,
    CSS8_PRIOR_MEAN,
    PAPER_CASES,
    SS8_COMPONENT_COVARIANCES,
    SS8_COMPONENT_MEANS,
    SS8_PRIOR_COVARIANCE,
    SS8_PRIOR_MEAN,
)
from benchmarks.paper_phantom_conditioning.posterior import classic_mode_mass
from benchmarks.paper_phantom_conditioning.prefix_sweep import (
    sample_phantom_prefix_sweep_reference,
)
from jaxns.checkpoint import MANIFEST_NAME, CheckpointManager
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler
from jaxns.depth_condition import DepthCondition
from jaxns.results import _incoming_lineages_per_sample

PAPER_CASE_NAMES = (
    "basic_mvn",
    "weak_curved_mvn8",
    "spike_slab",
    "correlated_spike_slab8",
)
DIRECTIONS = ("gmm", "isotropic")
GOAL_LOG_Z_UNCERT = 0.05
FIRST_GMM_LOG_Z_UNCERT = 0.2
SECOND_GMM_LOG_Z_UNCERT = 0.1
DEPTH_DLOG_Z = 1e-3
SLICE_TRANSITIONS_PER_DIMENSION = 10
MAX_PHANTOM_MULTIPLIER = SLICE_TRANSITIONS_PER_DIMENSION - 1
PAPER_SEEDS = 30
MC_DRAWS = 2048
PROTOCOL = "phantom_prefix_sweep_10d_state_mc2048_staged_gmm_shift3"
EVIDENCE_CAPACITY_QUANTUM = 8192
RUN_METADATA_NAME = "RUN.json"


def _normal_log_density(x, mean, covariance):
    """Evaluate a full-covariance Gaussian density in NumPy."""
    displacement = x - mean
    _, log_determinant = np.linalg.slogdet(covariance)
    precision = np.linalg.inv(covariance)
    quadratic = np.einsum(
        "...i,ij,...j->...",
        displacement,
        precision,
        displacement,
    )
    return (
        -0.5 * mean.size * np.log(2.0 * np.pi)
        - 0.5 * log_determinant
        - 0.5 * quadratic
    )


def _mode_membership(
        case_name: str,
        results,
) -> tuple[np.ndarray | None, float | None]:
    """Assign classic samples to a mixture mode and return its true mass."""
    if case_name not in ("spike_slab", "correlated_spike_slab8"):
        return None, None
    samples = np.asarray(jax.tree.leaves(results.X_samples)[0])  # [N, D]
    if case_name == "spike_slab":
        means = np.asarray(SS8_COMPONENT_MEANS)
        covariances = np.asarray(SS8_COMPONENT_COVARIANCES)
        prior_mean = np.asarray(SS8_PRIOR_MEAN)
        prior_covariance = np.asarray(SS8_PRIOR_COVARIANCE)
    else:
        means = np.asarray(CSS8_COMPONENT_MEANS)
        covariances = np.asarray(CSS8_COMPONENT_COVARIANCES)
        prior_mean = np.asarray(CSS8_PRIOR_MEAN)
        prior_covariance = np.asarray(CSS8_PRIOR_COVARIANCE)

    component_log_density = np.stack([
        _normal_log_density(samples, mean, covariance)
        for mean, covariance in zip(
            means,
            covariances,
            strict=True,
        )
    ], axis=1)  # [N, 2]
    first_mode = np.argmax(component_log_density, axis=1) == 0  # [N]
    component_log_evidence = np.asarray([
        _normal_log_density(
            mean,
            prior_mean,
            prior_covariance + covariance,
        )
        for mean, covariance in zip(
            means,
            covariances,
            strict=True,
        )
    ])
    truth = float(np.exp(
        component_log_evidence[0]
        - np.logaddexp.reduce(component_log_evidence)
    ))
    return first_mode, truth


def _environment() -> dict:
    """Record enough context to interpret performance evidence later."""
    return {
        "jaxns_distribution_version": importlib.metadata.version("jaxns"),
        "jaxns_module": os.path.realpath(jaxns.__file__),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.jax_enable_x64),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _sha256(path: Path) -> str:
    """Hash one persisted artifact without materialising it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_bytes(path: Path, payload: bytes) -> None:
    """Durably replace a small metadata file after all bytes are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _publish_json(path: Path, payload: dict) -> None:
    """Publish one canonical JSON object atomically."""
    _publish_bytes(
        path,
        (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
    )


def _append_record(path: Path, record: dict) -> None:
    """Append by atomically replacing the compact JSONL manifest."""
    prior = b"" if not path.exists() else path.read_bytes()
    if prior and not prior.endswith(b"\n"):
        raise ValueError(f"{path} has an incomplete final JSON record.")
    row = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    _publish_bytes(path, prior + row)


def _git_commit() -> str:
    """Identify the exact JAXNS source used for every saved State."""
    # The paper runner intentionally may import a PR worktree through
    # PYTHONPATH while keeping its uncommitted protocol beside paper.tex.
    # Resolve provenance from that imported package, not from this script.
    source_root = Path(jaxns.__file__).resolve().parents[2]
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _checkpoint_metadata(
        checkpoint_dir: Path,
        state_root: Path,
) -> dict:
    """Verify and describe the exact State committed by CheckpointManager."""
    manifest_path = checkpoint_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_path = checkpoint_dir / manifest["state_file"]
    checksum = _sha256(state_path)
    if checksum != manifest["checksum"]:
        raise RuntimeError(f"Checksum mismatch for {state_path}.")
    return {
        "archive_root": str(state_root.resolve()),
        "directory": str(checkpoint_dir.relative_to(state_root)),
        "trimmed": True,
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_schema_version": manifest["schema_version"],
        "generation": manifest["generation"],
        "state_file": manifest["state_file"],
        "state_sha256": checksum,
        "state_bytes": state_path.stat().st_size,
    }


def _completed_records(
        path: Path,
        case: str,
        direction: str,
        phantom_seeding: bool,
) -> dict[int, dict]:
    """Resume a long cell only after complete state-backed JSON rows."""
    if not path.exists():
        return {}
    completed = set()
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Incomplete JSON in {path} at line {line_number}."
            ) from error
        if record["case"] != case:
            raise ValueError(f"Output {path} contains another problem.")
        if record["protocol"] != PROTOCOL:
            raise ValueError(f"Output {path} contains another protocol.")
        if record["direction"] != direction:
            raise ValueError(f"Output {path} contains another direction.")
        if bool(record.get("phantom_seeding", False)) != phantom_seeding:
            raise ValueError(f"Output {path} contains another seed policy.")
        seed = int(record["seed"])
        if seed in completed:
            raise ValueError(f"Output {path} repeats seed {seed}.")
        completed.add(seed)
        records[seed] = record
    return records


def _parse_seeds(value: str) -> list[int]:
    """Parse either ``start:stop`` or a comma-separated seed list."""
    if ":" in value:
        start, stop = value.split(":", maxsplit=1)
        return list(range(int(start), int(stop)))
    return [int(seed) for seed in value.split(",")]


def _bucket_results(results):
    """Slice valid sampler padding to a reusable evidence-capacity bucket."""
    num_samples = int(results.total_num_samples)
    capacity = (
        (num_samples + EVIDENCE_CAPACITY_QUANTUM - 1)
        // EVIDENCE_CAPACITY_QUANTUM
        * EVIDENCE_CAPACITY_QUANTUM
    )
    storage_capacity = results.log_L.shape[0]
    capacity = min(capacity, storage_capacity)

    def slice_sample_axis(value):
        if (
            value is None
            or value.ndim == 0
            or value.shape[0] != storage_capacity
        ):
            return value
        return value[:capacity, ...]

    return jax.tree.map(
        slice_sample_axis,
        results,
        is_leaf=lambda value: value is None,
    ), capacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=PAPER_CASE_NAMES, required=True)
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument(
        "--phantom-seeding",
        choices=("off", "on"),
        default="off",
        help="Whether retained phantoms may seed later constrained chains.",
    )
    parser.add_argument("--seeds", default="0:30")
    parser.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    parser.add_argument(
        "--mc-batch-size",
        type=int,
        default=1,
        help="Array batch per MC task; parallel analysis requires one.",
    )
    parser.add_argument(
        "--mc-workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Host threads sharing one immutable phantom-prefix plan.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "core", "analysis"),
        default="all",
        help="Run both phases, persist only sampler states, or analyse states.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(
            "/data/jaxns-paper-phantom-conditioning/"
            "states-staged-gmm-prior6-scale05"
        ),
    )
    args = parser.parse_args()

    if args.mc_draws < 2:
        raise ValueError("mc-draws must provide at least two draws per state.")
    if args.mc_batch_size < 1:
        raise ValueError("mc-batch-size must be positive.")
    if args.mc_workers < 1:
        raise ValueError("mc-workers must be positive.")
    if args.mc_workers > 1 and args.mc_batch_size != 1:
        raise ValueError(
            "mc-workers greater than one requires mc-batch-size=1."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_records(
        args.output,
        args.case,
        args.direction,
        args.phantom_seeding == "on",
    )

    model, truth = PAPER_CASES[args.case].build()
    dimension = int(model.U_ndims())
    root_degree = 30 * dimension
    replacement_width = 10 * dimension
    num_slices = SLICE_TRANSITIONS_PER_DIMENSION * dimension
    retained_phantoms = MAX_PHANTOM_MULTIPLIER * dimension
    sampler = UniDimSliceSampler(
        num_slices=num_slices,
        collect_phantom_samples=True,
        # Retain the maximal 9D prefix once. Each evidence calculation below
        # masks this to kD states, so all k share the same classic race tree.
        # The sampler always excludes the final classic replacement.
        max_phantom_samples=retained_phantoms,
    )
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=root_degree,
        replacement_width=replacement_width,
        collect_phantom_samples=True,
        sampler=sampler,
        allocation_target="uniform",
        # The 0.05 goal is intentionally demanding. Do not replace it with a
        # storage ceiling that can silently stop the harder cases first.
        unlimited_samples=True,
        # Classic-only runs must also work with develop, which has no
        # phantom-seeding option. Only the explicitly selected experimental
        # arm forwards that argument to its separate implementation branch.
        **({"phantom_seeding": True} if args.phantom_seeding == "on" else {}),
    )
    depth_condition = DepthCondition(
        dlogZ=jnp.log1p(jnp.asarray(DEPTH_DLOG_Z, dtype=jnp.float64)),
    )
    environment = _environment()
    goal_progress = []
    run_started = None
    goal_stage = "final"
    goal_target = GOAL_LOG_Z_UNCERT

    def goal_condition(state) -> bool:
        """Use the cheap classic expectation only at Python goal boundaries."""
        if int(state.goal_loop_iter) == 0:
            return False
        # The goal consumes only scalar race-tree moments. Constructing the
        # full Results object here would validate, transform, and weight every
        # stored sample again at each outer iteration.
        expected_uncertainty = float(state.expected_log_Z_uncert)
        print(
            f"{args.direction} {args.case} seed {seed}: "
            f"{goal_stage}, "
            f"goal {int(state.goal_loop_iter)}, "
            f"{int(state.num_samples):,} classic, "
            f"expected uncertainty {expected_uncertainty:.6f}",
            flush=True,
        )
        if run_started is not None:
            goal_progress.append({
                "goal_loop_iteration": int(state.goal_loop_iter),
                "allocation_loop_iteration": int(
                    state.allocation_loop_iter
                ),
                "classic_samples": int(state.num_samples),
                "likelihood_evaluations": int(
                    state.total_num_likelihood_evaluations
                ),
                "expected_log_Z_uncert": expected_uncertainty,
                "elapsed_seconds": time.perf_counter() - run_started,
                "stage": goal_stage,
                "target_log_Z_uncert": goal_target,
            })
        return expected_uncertainty < goal_target

    for seed in _parse_seeds(args.seeds):
        if seed < 0 or seed >= PAPER_SEEDS:
            raise ValueError("Paper seeds must be in the canonical range 0--29.")
        checkpoint_dir = (
            args.state_root
            / args.direction
            / f"phantom-seeding-{args.phantom_seeding}"
            / args.case
            / f"seed-{seed:02d}"
        )
        if seed in completed:
            with CheckpointManager(checkpoint_dir) as manager:
                state = manager.load()
            if state is None:
                raise RuntimeError(
                    f"{args.case} seed {seed} has a row but no State."
                )
            checkpoint = _checkpoint_metadata(
                checkpoint_dir,
                args.state_root,
            )
            if completed[seed]["state_checkpoint"] != checkpoint:
                raise RuntimeError(
                    f"{args.case} seed {seed} row does not match its State."
                )
            print(
                f"{args.direction} {args.case} seed {seed}: "
                "already complete and verified",
                flush=True,
            )
            continue
        key = jax.random.PRNGKey(seed)
        metadata_path = checkpoint_dir / RUN_METADATA_NAME
        with CheckpointManager(checkpoint_dir) as manager:
            state = manager.load()
            if state is None:
                if args.phase == "analysis":
                    raise RuntimeError(
                        f"{checkpoint_dir} has no State to analyse."
                    )
                goal_progress = []
                run_started = time.perf_counter()
                fit_seconds = []
                if args.direction == "isotropic":
                    goal_stage = "isotropic_to_0.05"
                    goal_target = GOAL_LOG_Z_UNCERT
                    state = nested_sampler.run_until_goal(
                        goal_condition,
                        depth_cond=depth_condition,
                        key=key,
                    )
                else:
                    # Staging is explicit user policy. The single-mode cases
                    # use one component; the two-component spike--slab cases
                    # ask for two rather than letting execution infer modes.
                    num_components = (
                        2
                        if args.case in (
                            "spike_slab",
                            "correlated_spike_slab8",
                        )
                        else 1
                    )
                    goal_stage = "isotropic_to_0.2"
                    goal_target = FIRST_GMM_LOG_Z_UNCERT
                    state = nested_sampler.run_until_goal(
                        goal_condition,
                        depth_cond=depth_condition,
                        key=key,
                    )
                    fit_started = time.perf_counter()
                    state = state.fit_gmm_directions(
                        num_components=num_components,
                        iso_prob=1e-2,
                    )
                    jax.block_until_ready(state)
                    fit_seconds.append(time.perf_counter() - fit_started)

                    goal_stage = "gmm_to_0.1"
                    goal_target = SECOND_GMM_LOG_Z_UNCERT
                    state = nested_sampler.resume_until_goal(
                        state,
                        goal_condition,
                        depth_cond=depth_condition,
                    )
                    fit_started = time.perf_counter()
                    state = state.fit_gmm_directions()
                    jax.block_until_ready(state)
                    fit_seconds.append(time.perf_counter() - fit_started)

                    goal_stage = "gmm_to_0.05"
                    goal_target = GOAL_LOG_Z_UNCERT
                    state = nested_sampler.resume_until_goal(
                        state,
                        goal_condition,
                        depth_cond=depth_condition,
                    )
                jax.block_until_ready(state)
                run_seconds = time.perf_counter() - run_started
                run_started = None
                # Only invalid capacity padding is removed. The archived
                # object remains a complete State and can grow if resumed.
                state = state.trim()
                jax.block_until_ready(state)
                manager.save(state)
                # Analyse the deserialised checkpoint, not an unsaved object.
                loaded = manager.load()
                if loaded is None:
                    raise RuntimeError(
                        f"{args.case} seed {seed} State did not reload."
                    )
                state = loaded
                run_metadata = {
                    "protocol": PROTOCOL,
                    "case": args.case,
                    "direction": args.direction,
                    "phantom_seeding": args.phantom_seeding == "on",
                    "seed": seed,
                    "run_seconds": run_seconds,
                    "fit_seconds": fit_seconds,
                    "goal_progress": goal_progress,
                    "source_commit": _git_commit(),
                }
                _publish_json(metadata_path, run_metadata)
            else:
                if not metadata_path.is_file():
                    raise RuntimeError(
                        f"{checkpoint_dir} has a State without RUN.json; "
                        "preserve it for diagnosis rather than inventing a "
                        "core runtime."
                    )
                run_metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                expected_metadata = {
                    "protocol": PROTOCOL,
                    "case": args.case,
                    "direction": args.direction,
                    "phantom_seeding": args.phantom_seeding == "on",
                    "seed": seed,
                }
                if any(
                    run_metadata[field] != value
                    for field, value in expected_metadata.items()
                ):
                    raise RuntimeError(
                        f"{metadata_path} describes another run."
                    )
                run_seconds = float(run_metadata["run_seconds"])
                fit_seconds = list(run_metadata["fit_seconds"])

        checkpoint = _checkpoint_metadata(
            checkpoint_dir,
            args.state_root,
        )
        achieved_uncertainty = float(state.expected_log_Z_uncert)
        if not achieved_uncertainty < GOAL_LOG_Z_UNCERT:
            raise RuntimeError(
                f"{args.case} seed {seed} stopped at expected logZ "
                f"uncertainty {achieved_uncertainty}."
            )
        if args.phase == "core":
            print(
                f"{args.direction} {args.case} seed {seed}: "
                f"core complete in {run_seconds:.1f} s",
                flush=True,
            )
            continue

        # Quantised padding reuses evidence compilations across nearby sample
        # counts without processing the much larger sampler-storage capacity.
        # total_num_samples and block validity still exclude every padded row.
        num_samples = int(state.num_samples)
        evidence_capacity = (
            (num_samples + EVIDENCE_CAPACITY_QUANTUM - 1)
            // EVIDENCE_CAPACITY_QUANTUM
            * EVIDENCE_CAPACITY_QUANTUM
        )
        # Archive only valid rows, then restore deterministic padding for the
        # analysis kernel so nearby seeds share compiled executable shapes.
        analysis_state = state.resize(evidence_capacity)
        sampler_results = analysis_state.to_result()
        trimmed_results = sampler_results.trim()
        results, evidence_capacity = _bucket_results(sampler_results)
        valid_blocks = np.asarray(results.block_data.valid)
        valid_sizes = np.asarray(results.block_data.size)[valid_blocks]
        if not np.all(valid_sizes == 1):
            raise RuntimeError(
                "The paper prefix sweep supports only continuous problems."
            )
        if results.log_L_phantom.shape[1] != retained_phantoms:
            raise RuntimeError("The sampler did not retain the full 9D prefix.")

        # All ten columns share classic race gammas and cluster weights. This
        # changes only their Monte Carlo coupling, while making every phantom
        # event contribute once rather than once for every longer prefix.
        evidence_key = jax.random.fold_in(key, 1)
        mc_draws = args.mc_draws
        start = time.perf_counter()
        first_mode, mode_mass_truth = _mode_membership(
            args.case,
            trimmed_results,
        )
        mode_mass = classic_mode_mass(
            np.asarray(trimmed_results.log_dp),  # [N]
            first_mode,
        )

        log_Z_samples, phantom_gates = (
            sample_phantom_prefix_sweep_reference(
                key=evidence_key,
                log_L_constraints=results.log_L_constraints,
                K_classic=_incoming_lineages_per_sample(results),
                valid_phantom=results.valid_phantom,
                log_L_phantom=results.log_L_phantom,
                num_samples=results.total_num_samples,
                block_state=results.block_data.to_block_state(),
                dimension=dimension,
                num_draws=mc_draws,
                batch_size=args.mc_batch_size,
                num_workers=args.mc_workers,
                num_groups=MAX_PHANTOM_MULTIPLIER,
            )
        )
        sweep_seconds = time.perf_counter() - start
        log_Z_samples = log_Z_samples[:mc_draws]  # [M_s, 10]
        evidence = {}
        for multiplier in range(MAX_PHANTOM_MULTIPLIER + 1):
            samples = log_Z_samples[:, multiplier]
            gate_active_fraction = (
                0.0
                if multiplier == 0
                else float(np.mean(
                    phantom_gates[multiplier - 1, valid_blocks]
                ))
            )
            evidence[str(multiplier)] = {
                "log_Z_mean": float(np.mean(samples)),
                "log_Z_uncert": float(np.std(samples, ddof=1)),
                "gate_active_fraction": gate_active_fraction,
            }
        print(f"{args.case} seed {seed}: evidence sweep done", flush=True)

        valid_clusters = int(np.sum(np.asarray(results.valid_phantom)))
        record = {
            "protocol": PROTOCOL,
            "case": args.case,
            "direction": args.direction,
            "phantom_seeding": args.phantom_seeding == "on",
            "seed": seed,
            "truth_log_Z": float(truth),
            "dimension": dimension,
            "root_degree": root_degree,
            "replacement_width": replacement_width,
            "num_slices": num_slices,
            "maximum_retained_phantoms_per_chain": retained_phantoms,
            "phantom_multipliers": list(
                range(1, MAX_PHANTOM_MULTIPLIER + 1)
            ),
            "goal_log_Z_uncert": GOAL_LOG_Z_UNCERT,
            "first_gmm_log_Z_uncert": (
                FIRST_GMM_LOG_Z_UNCERT
                if args.direction == "gmm"
                else None
            ),
            "second_gmm_log_Z_uncert": (
                SECOND_GMM_LOG_Z_UNCERT
                if args.direction == "gmm"
                else None
            ),
            "gmm_num_components": (
                state.sampler_data.centres.shape[0]
                if args.direction == "gmm"
                else None
            ),
            "fit_seconds": fit_seconds,
            "achieved_goal_log_Z_uncert": achieved_uncertainty,
            "depth_dlog_Z": DEPTH_DLOG_Z,
            "mc_draws": mc_draws,
            "mc_batch_size": args.mc_batch_size,
            "mc_workers": args.mc_workers,
            "evidence_sweep_seconds": sweep_seconds,
            "run_seconds": run_seconds,
            "goal_loop_iterations": int(state.goal_loop_iter),
            "classic_samples": int(results.total_num_samples),
            "evidence_capacity": evidence_capacity,
            "maximum_phantom_samples": int(
                results.total_phantom_samples
            ),
            "valid_phantom_clusters": valid_clusters,
            "likelihood_evaluations": int(
                results.total_num_likelihood_evaluations
            ),
            "mode_mass": mode_mass,
            "mode_mass_truth": mode_mass_truth,
            "state_checkpoint": checkpoint,
            "provenance": {
                "source_commit": _git_commit(),
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": _sha256(Path(__file__).resolve()),
                "case_module_path": str(
                    (Path(__file__).parent / "cases.py").resolve()
                ),
                "case_module_sha256": _sha256(
                    Path(__file__).parent / "cases.py"
                ),
                "prefix_reducer_path": str(
                    (Path(__file__).parent / "prefix_sweep.py").resolve()
                ),
                "prefix_reducer_sha256": _sha256(
                    Path(__file__).parent / "prefix_sweep.py"
                ),
            },
            "evidence": evidence,
            "environment": environment,
        }
        _append_record(args.output, record)
        completed[seed] = record
        print(
            f"{args.direction} {args.case} seed {seed}: "
            f"{int(results.total_num_likelihood_evaluations):,} evals, "
            f"{run_seconds:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
