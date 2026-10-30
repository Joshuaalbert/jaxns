"""Apply existing classic and phantom evidence inference to frozen allocation snapshots."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.allocation_phantoms import EvidenceInputs
    from benchmarks.issue_312.numpy_allocation_evidence import draw_numpy_evidence

    source_folder = Path(__file__).resolve().parent
    source_paths = {name: source_folder / name for name in (
        "condition_allocation_phantoms.py", "allocation_phantoms.py",
        "numpy_allocation_evidence.py",
    )}
    for name in ("results.py", "shrinkage/phantom.py", "shrinkage/classic.py"):
        source_paths[f"src/jaxns/{name}"] = source_folder.parents[1] / "src/jaxns" / name
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in source_paths.items()}
    # Separate deterministic MC keys from the constrained-sampling stream.
    # The draw count, key derivation, gate and batch size are fixed for all cells.
    for line in options.records.read_text().splitlines():
        row = json.loads(line)
        if options.seeds is not None and row["seed"] not in options.seeds:
            continue
        archive = Path(row["phantom_archive"])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["phantom_archive_sha256"]
        inputs = EvidenceInputs.load(archive)
        key = jax.random.PRNGKey(312044)
        for value in (row["seed"], row["budget"],
                      ("hessian", "reflection").index(row["kernel"]),
                      ("uniform", "evidence_improving", "posterior_improving")
                      .index(row["allocation"])):
            key = jax.random.fold_in(key, value)
        record = {**row, "mc_draws": 4096, "mc_batch_size": 64, "C_min": 20,
                  "mc_key": np.asarray(key).tolist(), "mc_source_sha256": hashes,
                  "mc_backend": "numpy_PCG64_scipy_sparse_production_event_plan",
                  "numpy_version": np.__version__}
        samples = {}
        for mode, use_phantoms in (("classic_mc", False), ("phantom", True)):
            started = time.monotonic()
            entropy = np.asarray(jax.random.fold_in(key, int(use_phantoms))).tolist()
            ensemble = draw_numpy_evidence(
                inputs, np.random.SeedSequence(entropy), phantoms=use_phantoms,
            )
            values = np.asarray(ensemble.log_Z_samples)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Nonfinite evidence ensemble: {archive}, {mode}")
            samples[f"{mode}_log_Z_samples"] = values
            record.update({
                f"{mode}_log_Z": float(values.mean()),
                f"{mode}_log_Z_uncert": float(values.std()),
                f"{mode}_mcse": float(values.std(ddof=1) / np.sqrt(len(values))),
                f"{mode}_seconds": time.monotonic() - started,
            })
            if use_phantoms:
                valid_blocks = np.asarray(inputs.block_state.valid)
                gate = np.asarray(ensemble.phantom_gate_active)
                contributions = np.asarray(ensemble.log_dZ_mean)
                weights = np.exp(contributions - np.max(contributions))
                weights /= weights.sum()
                record["phantom_gate_blocks"] = int(gate[valid_blocks].sum())
                record["blocks"] = int(valid_blocks.sum())
                record["phantom_gate_evidence_fraction"] = float(weights[gate].sum())
                samples["phantom_gate_active"] = gate
                samples["kish_participating_cluster_counts"] = np.asarray(
                    ensemble.kish_participating_cluster_counts,
                )
            print(f"{row['kernel']} {row['allocation']} seed={row['seed']} "
                  f"budget={row['budget']} {mode}={values.mean():.6f} +/- {values.std():.6f} "
                  f"seconds={record[f'{mode}_seconds']:.1f}", flush=True)
        draws_path = archive.with_name(f"{archive.stem}-evidence.npz")
        np.savez_compressed(draws_path, **samples)
        record["draws_archive"] = str(draws_path)
        record["draws_archive_sha256"] = hashlib.sha256(draws_path.read_bytes()).hexdigest()
        with options.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
