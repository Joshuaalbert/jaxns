"""Observe the exact Dynesty control without changing its random stream."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})

    import dynesty
    import numpy as np
    from dynesty import internal_samplers
    from dynesty.sampler import Sampler

    if dynesty.__version__ != "3.0.0":
        raise ValueError("This instrumentation is specific to Dynesty 3.0.0.")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    original_queue = Sampler._fill_queue
    original_bound = Sampler.update_bound_if_needed
    original_sample = internal_samplers.RSliceSampler.sample
    original_slice = internal_samplers.generic_slice_step
    context = {}
    config = {}
    bounds = []
    chains = []
    steps = []

    def observe_queue(self, loglstar):
        context["iteration"] = self.it
        context["nells"] = 0 if self.unit_cube_sampling else self.bound.nells
        config.update(
            bound_update_interval=self.bound_update_interval,
            first_bound_update_ncall=self.first_bound_update_ncall,
            first_bound_update_eff=self.first_bound_update_eff,
            bound_enlarge=self.bound_enlarge,
            bound_bootstrap=self.bound_bootstrap,
            queue_size=self.queue_size,
            live_points=self.nlive,
        )
        return original_queue(self, loglstar)

    def observe_bound(self, loglstar, ncall=None, force=False):
        previous = self.nbound
        result = original_bound(self, loglstar, ncall=ncall, force=force)
        if self.nbound != previous:
            eigenvalues = np.linalg.eigvalsh(self.bound.covs)
            bounds.append({
                "iteration": self.it,
                "ncall": self.ncall if ncall is None else ncall,
                "log_L_constraint": float(loglstar),
                "forced": force,
                "nells": self.bound.nells,
                "condition_numbers": (eigenvalues[:, -1] / eigenvalues[:, 0]).tolist(),
                "live_std": np.std(self.live_u, axis=0).tolist(),
                "ellipsoid_centres": self.bound.ctrs.tolist(),
            })
        return result

    def observe_slice(u, direction, nonperiodic, loglstar, loglikelihood,
                      prior_transform, doubling, evaluation_history, rstate):
        evaluations_before = len(evaluation_history)
        result = original_slice(
            u, direction, nonperiodic, loglstar, loglikelihood,
            prior_transform, doubling, evaluation_history, rstate,
        )
        _, _, _, calls, expand, contract, warning = result
        # In doubling mode n_expand is 1+2+...+2**(k-1), not k calls.
        expansion_tests = int(expand).bit_length() if doubling else expand
        steps.append((
            calls, len(evaluation_history) - evaluations_before,
            expansion_tests, contract, calls - 2 - expansion_tests - contract,
            int(doubling), int(warning),
            float(np.linalg.norm(direction)),
            float(np.linalg.norm(result[0] - u)),
        ))
        return result

    def observe_sample(args):
        steps.clear()
        result = original_sample(args)
        step_array = np.asarray(steps)
        if len(steps) != 90 or int(step_array[:, 0].sum()) != result.ncalls:
            raise ValueError("Unexpected slice count or instrumented call accounting.")
        chains.append({
            **context,
            "log_L_constraint": float(args.loglstar),
            "scale": float(args.scale),
            "native_calls": result.ncalls,
            "actual_calls": len(result.evaluation_history),
            "expansion_tests": int(step_array[:, 2].sum()),
            "contraction_tests": int(step_array[:, 3].sum()),
            "doubling_acceptance_tests": int(step_array[:, 4].sum()),
            "doubling_steps": int(step_array[:, 5].sum()),
            "expansion_warnings": int(step_array[:, 6].sum()),
            "median_initial_direction_length": float(np.median(step_array[:, 7])),
            "median_step_length": float(np.median(step_array[:, 8])),
            "chain_displacement": (result.u - args.u).tolist(),
        })
        return result

    # Observers call the pinned implementation exactly once, return its exact
    # objects, and neither draw random values nor call the likelihood. Verify
    # this promise against every archived sample and weight after each replay.
    Sampler._fill_queue = observe_queue
    Sampler.update_bound_if_needed = observe_bound
    internal_samplers.RSliceSampler.sample = staticmethod(observe_sample)
    internal_samplers.generic_slice_step = observe_slice

    for seed in options.seeds:
        context.clear()
        config.clear()
        bounds.clear()
        chains.clear()
        output = options.output_dir / f"seed-{seed}.jsonl"
        sys.argv = [
            str(source_dir / "run_dynesty_ridge.py"),
            "--seed", str(seed), "--cpu", str(options.cpu), "--output", str(output),
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
        reference = source_dir / "raw" / f"dynesty-rslice90-150-{seed}.npz"
        with np.load(output.with_suffix(".npz")) as actual, np.load(reference) as expected:
            for name in expected.files:
                np.testing.assert_array_equal(actual[name], expected[name])
        trace = {
            "seed": seed, "config": config, "bounds": bounds, "chains": chains,
            "matches_archived_samples_and_weights": True,
        }
        trace_path = options.output_dir / f"seed-{seed}.trace.json"
        trace_path.write_text(json.dumps(trace) + "\n")
        print(json.dumps({"seed": seed, "trace": str(trace_path), "matched": True}), flush=True)


if __name__ == "__main__":
    main()
