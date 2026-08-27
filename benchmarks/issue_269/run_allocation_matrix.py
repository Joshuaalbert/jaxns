"""Measure whether finer allocation cadence improves multimodal survival."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

import jaxns
from benchmarks.issue_246.run_standard import (
    MODE_PROBLEMS,
    _mode_mass,
    _normal_log_density,
)
from cicd.tests.test_ns_standard_problems import (
    STANDARD_PROBLEM_CASES_BY_NAME,
)
from jaxns.constrained_sampler import EllipsoidalDirection, UniDimSliceSampler
from jaxns.core import NestedSampler


@dataclass(frozen=True, slots=True)
class AllocationSchedule:
    """One intentional allocation-cadence comparison row."""

    name: str
    root_degree: int
    delta_k: int
    shell_size: int


SCHEDULES = (
    AllocationSchedule("baseline", 240, 80, 80),
    AllocationSchedule("smaller_root_coarse", 160, 80, 80),
    AllocationSchedule("smaller_root_medium_fixed", 160, 40, 80),
    AllocationSchedule("smaller_root_fine_fixed", 160, 20, 80),
    AllocationSchedule("smaller_root_medium_matched", 160, 40, 40),
    AllocationSchedule("smaller_root_fine_matched", 160, 20, 20),
    AllocationSchedule("aggressive", 120, 20, 20),
)
SCHEDULES_BY_NAME = {schedule.name: schedule for schedule in SCHEDULES}


def _environment(repository_root: Path) -> dict[str, object]:
    """Record enough source identity to reject an installed-package run."""
    source_root = (repository_root / "src").resolve()
    nested_sampler_source = Path(
        inspect.getsourcefile(NestedSampler) or ""
    ).resolve()
    try:
        nested_sampler_source.relative_to(source_root)
    except ValueError as error:
        raise RuntimeError(
            "NestedSampler did not import from this worktree: "
            f"{nested_sampler_source} is outside {source_root}."
        ) from error
    return {
        "jaxns_distribution_version": importlib.metadata.version("jaxns"),
        "jaxns_module": os.path.realpath(jaxns.__file__),
        "nested_sampler_source": str(nested_sampler_source),
        "git_commit": _git_commit(repository_root),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.x64_enabled),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "matplotlib_backend": os.environ.get("MPLBACKEND"),
    }


def _git_commit(repository_root: Path) -> str:
    """Return the exact commit without adding a runtime dependency on GitPython."""
    head = (repository_root / ".git").read_text(encoding="utf-8").strip()
    if not head.startswith("gitdir: "):
        raise RuntimeError("This benchmark expects to run from a Git worktree.")
    git_directory = Path(head.removeprefix("gitdir: "))
    commit = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    if commit.startswith("ref: "):
        ref_path = git_directory / commit.removeprefix("ref: ")
        if ref_path.exists():
            commit = ref_path.read_text(encoding="utf-8").strip()
        else:
            common_directory = (
                git_directory
                / (git_directory / "commondir").read_text(
                    encoding="utf-8"
                ).strip()
            ).resolve()
            commit = (
                common_directory / commit.removeprefix("ref: ")
            ).read_text(encoding="utf-8").strip()
    return commit


def _assign_components(case_name: str, results) -> np.ndarray:
    """Assign append-order samples to the analytic mixture components."""
    config = MODE_PROBLEMS[case_name]
    samples = np.asarray(jax.tree.leaves(results.X_samples)[0])
    samples = samples[:int(results.total_num_samples)]
    component_log_density = np.stack([
        np.log(mass) + _normal_log_density(samples, mean, variance)
        for mass, mean, variance in zip(
            config["masses"],
            config["means"],
            config["variances"],
            strict=True,
        )
    ], axis=1)
    return np.argmax(component_log_density, axis=1)


def _append_boundary(
        boundaries: list[dict[str, int]],
        state,
) -> None:
    """Capture one small host-side snapshot at a Python goal boundary."""
    snapshot = {
        "goal_loop_iter": int(state.goal_loop_iter),
        "depth_loop_iter": int(state.depth_loop_iter),
        "num_samples": int(state.num_samples),
        "root_out_degree": int(state.root_out_degree),
    }
    if not boundaries or snapshot != boundaries[-1]:
        boundaries.append(snapshot)


def _epoch_diagnostics(
        state,
        results,
        boundaries: list[dict[str, int]],
        assignments: np.ndarray,
        num_components: int,
) -> list[dict[str, object]]:
    """Describe sample and newly allocated root coverage per goal epoch."""
    constraints = np.asarray(results.log_L_constraints)
    log_likelihoods = np.asarray(results.log_L)
    epochs = []
    previous_samples = 0
    previous_root_degree = 0
    for boundary in boundaries:
        stop = boundary["num_samples"]
        epoch_assignments = assignments[previous_samples:stop]
        epoch_constraints = constraints[previous_samples:stop]
        epoch_log_likelihoods = log_likelihoods[previous_samples:stop]
        root_mask = np.isneginf(epoch_constraints)
        component_counts = []
        root_component_counts = []
        max_log_likelihood = []
        for component in range(num_components):
            component_mask = epoch_assignments == component
            component_counts.append(int(np.sum(component_mask)))
            root_component_counts.append(int(np.sum(
                component_mask & root_mask
            )))
            if np.any(component_mask):
                max_log_likelihood.append(float(np.max(
                    epoch_log_likelihoods[component_mask]
                )))
            else:
                max_log_likelihood.append(None)
        epochs.append({
            **boundary,
            "samples_added": stop - previous_samples,
            "roots_added": (
                boundary["root_out_degree"] - previous_root_degree
            ),
            "component_counts": component_counts,
            "root_component_counts": root_component_counts,
            "max_log_likelihood": max_log_likelihood,
        })
        previous_samples = stop
        previous_root_degree = boundary["root_out_degree"]
    if previous_samples != int(state.num_samples):
        raise AssertionError("The final goal boundary omitted valid samples.")
    return epochs


def _build_sampler(
        model,
        collect_phantoms: bool,
) -> UniDimSliceSampler:
    """Build the fixed issue-269 constrained-sampling law."""
    dimension = int(model.U_ndims())
    num_slices = 5 * dimension
    retained_phantoms = dimension if collect_phantoms else 0
    direction = EllipsoidalDirection(
        num_components=4,
        min_effective_samples=None,
        num_iterations=10,
        population_size=1024,
        prob_isotropic=1e-2,
    )
    return UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        collect_phantom_samples=collect_phantoms,
        phantom_burn_in=num_slices - 1 - retained_phantoms,
        direction=direction,
    )


def _measure(
        nested_sampler: NestedSampler,
        case_name: str,
        truth: float,
        seed: int,
        mc_draws: int,
        collect_phantoms: bool,
        goal_log_z_uncert: float,
) -> dict[str, object]:
    """Run one seed and retain final and allocation-epoch diagnostics."""
    key = jax.random.PRNGKey(seed)
    boundaries: list[dict[str, int]] = []

    def goal_condition(state) -> bool:
        # The ordinary run goal is normally satisfied after allocation epoch
        # zero. A separate uncertainty goal deliberately asks later epochs to
        # raise the lineage target in delta_K increments, which is the issue
        # 269 mechanism under test. It does not alter a depth transition.
        _append_boundary(boundaries, state)
        if int(state.goal_loop_iter) == 0:
            return False
        register = state.compute_termination_register()
        return float(register.log_Z_uncert) <= goal_log_z_uncert

    started = time.perf_counter()
    state = nested_sampler.run_until_goal(goal_condition, key=key)
    jax.block_until_ready(state)
    run_s = time.perf_counter() - started
    # A compiled terminal return short-circuits the next callback, so retain
    # the final state explicitly rather than silently losing its last batch.
    _append_boundary(boundaries, state)

    started = time.perf_counter()
    results = state.to_result().trim()
    jax.block_until_ready(results)
    result_s = time.perf_counter() - started

    assignments = _assign_components(case_name, results)
    mode_mass, mode_mass_truth = _mode_mass(case_name, results)
    epochs = _epoch_diagnostics(
        state,
        results,
        boundaries,
        assignments,
        len(MODE_PROBLEMS[case_name]["masses"]),
    )
    evidence = None
    mc_s = 0.0
    if mc_draws > 0:
        conditioning = "phantom" if collect_phantoms else "classic"
        started = time.perf_counter()
        evidence = results.sample_evidence_mc(
            num_samples=mc_draws,
            conditioning=conditioning,
            key=jax.random.fold_in(key, 1),
        )
        jax.block_until_ready(evidence)
        mc_s = time.perf_counter() - started

    sampler_data = state.sampler_data
    goal_register_log_z_uncert = float(
        state.compute_termination_register().log_Z_uncert
    )
    likelihood_evaluations = int(
        results.total_num_likelihood_evaluations
    )
    ess = float(results.ess)
    return {
        "seed": seed,
        "run_s": run_s,
        "result_s": result_s,
        "mc_s": mc_s,
        "log_Z_mean": float(results.log_Z_mean),
        "log_Z_error": float(results.log_Z_mean - truth),
        "log_Z_uncert": float(results.log_Z_uncert),
        "z_score": float(
            (results.log_Z_mean - truth) / results.log_Z_uncert
        ),
        "mc_conditioning": (
            None if evidence is None
            else "phantom" if collect_phantoms else "classic"
        ),
        "mc_log_Z_mean": (
            None if evidence is None else float(evidence.log_Z_mean)
        ),
        "mc_log_Z_error": (
            None if evidence is None
            else float(evidence.log_Z_mean - truth)
        ),
        "mc_log_Z_uncert": (
            None if evidence is None else float(evidence.log_Z_uncert)
        ),
        "mc_z_score": (
            None if evidence is None
            else float(
                (evidence.log_Z_mean - truth) / evidence.log_Z_uncert
            )
        ),
        "samples": int(results.total_num_samples),
        "likelihood_evaluations": likelihood_evaluations,
        "ess": ess,
        "likelihood_evaluations_per_ess": likelihood_evaluations / ess,
        "retained_phantoms": int(results.total_phantom_samples),
        "mode_mass": mode_mass,
        "mode_mass_truth": mode_mass_truth,
        "mode_mass_error": mode_mass - mode_mass_truth,
        "goal_loop_iterations": int(state.goal_loop_iter),
        "depth_loop_iterations": int(state.depth_loop_iter),
        "final_root_out_degree": int(state.root_out_degree),
        "termination_reason": int(state.termination_reason),
        "goal_log_Z_uncert": goal_log_z_uncert,
        "goal_register_log_Z_uncert": goal_register_log_z_uncert,
        "goal_reached": goal_register_log_z_uncert <= goal_log_z_uncert,
        "final_allocation_target": (
            int(nested_sampler.root_allocation_degree)
            + max(int(state.goal_loop_iter) - 1, 0)
            * int(nested_sampler.delta_K)
        ),
        "fit_updates": int(sampler_data.num_updates),
        "directions": int(sampler_data.num_directions),
        "isotropic_directions": int(sampler_data.num_isotropic),
        "epochs": epochs,
    }


def _parse_schedules(value: str) -> list[AllocationSchedule]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - SCHEDULES_BY_NAME.keys())
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown schedules: {', '.join(unknown)}"
        )
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("Schedule names must be unique.")
    return [SCHEDULES_BY_NAME[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="spike_slab")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--schedules",
        type=_parse_schedules,
        default=list(SCHEDULES),
        help="Comma-separated schedule names (default: all).",
    )
    parser.add_argument("--phantoms", action="store_true")
    parser.add_argument("--mc-draws", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=24_000)
    parser.add_argument("--goal-log-z-uncert", type=float, default=0.18)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive.")
    if args.mc_draws < 0:
        parser.error("--mc-draws cannot be negative.")
    if args.goal_log_z_uncert <= 0.0:
        parser.error("--goal-log-z-uncert must be positive.")

    repository_root = Path(__file__).resolve().parents[2]
    environment = _environment(repository_root)
    model, truth = STANDARD_PROBLEM_CASES_BY_NAME[args.case].build_case()
    dimension = int(model.U_ndims())
    if args.case not in MODE_PROBLEMS:
        parser.error(f"Case {args.case!r} has no analytic mode definition.")

    direction = EllipsoidalDirection(
        num_components=4,
        min_effective_samples=None,
        num_iterations=10,
        population_size=1024,
        prob_isotropic=1e-2,
    )
    resolved_min_effective_samples = (
        4 * direction.num_components * (dimension + 1)
    )
    scientific_config = {
        "case": args.case,
        "truth_log_Z": float(truth),
        "dimension": dimension,
        "seeds": args.seeds,
        "collect_phantoms": args.phantoms,
        "mc_draws": args.mc_draws,
        "max_samples": args.max_samples,
        "num_slices": 5 * dimension,
        "allocation_target": "uniform",
        "dlogZ": float(np.log1p(1e-3)),
        "goal_log_Z_uncert": args.goal_log_z_uncert,
        "ellipsoidal_direction": {
            "num_components": direction.num_components,
            "min_effective_samples": direction.min_effective_samples,
            "resolved_min_effective_samples": (
                resolved_min_effective_samples
            ),
            "num_iterations": direction.num_iterations,
            "population_size": direction.population_size,
            "prob_isotropic": direction.prob_isotropic,
        },
        "schedules": [
            {
                "name": schedule.name,
                "root_degree": schedule.root_degree,
                "delta_k": schedule.delta_k,
                "shell_size": schedule.shell_size,
            }
            for schedule in args.schedules
        ],
    }
    output = {
        "schema_version": 1,
        "environment": environment,
        "scientific_config": scientific_config,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        if checkpoint.get("scientific_config") != scientific_config:
            raise ValueError(
                "Cannot resume: the scientific configuration differs."
            )
        output["records"].extend(checkpoint.get("records", []))
    elif args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output}"
        )

    def checkpoint() -> None:
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    checkpoint()
    for schedule in args.schedules:
        completed = {
            int(record["seed"])
            for record in output["records"]
            if record["schedule"] == schedule.name
        }
        if len(completed) == args.seeds:
            continue
        # Each row has a distinct static allocation program.  Warm it once,
        # exclude compilation, and release the preceding row's executable so
        # process memory does not grow across the six-shape matrix.
        jax.clear_caches()
        sampler = _build_sampler(model, args.phantoms)
        nested_sampler = NestedSampler(
            model=model,
            root_allocation_degree=schedule.root_degree,
            shell_size=schedule.shell_size,
            delta_K=schedule.delta_k,
            max_samples=args.max_samples,
            collect_phantom_samples=args.phantoms,
            sampler=sampler,
        )
        warm = _measure(
            nested_sampler,
            args.case,
            float(truth),
            -1,
            0,
            args.phantoms,
            args.goal_log_z_uncert,
        )
        for seed in range(args.seeds):
            if seed in completed:
                continue
            record = _measure(
                nested_sampler,
                args.case,
                float(truth),
                seed,
                args.mc_draws,
                args.phantoms,
                args.goal_log_z_uncert,
            )
            record.update({
                "schedule": schedule.name,
                "root_degree": schedule.root_degree,
                "delta_k": schedule.delta_k,
                "shell_size": schedule.shell_size,
                "warm_run_s": warm["run_s"],
                "warm_result_s": warm["result_s"],
            })
            output["records"].append(record)
            checkpoint()
            print(
                json.dumps({
                    "schedule": schedule.name,
                    "seed": seed,
                    "mode_mass_error": record["mode_mass_error"],
                    "log_Z_error": record["log_Z_error"],
                    "run_s": record["run_s"],
                    "goal_loop_iterations": record[
                        "goal_loop_iterations"
                    ],
                }, sort_keys=True),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
