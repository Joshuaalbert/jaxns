"""Check exact replay against the archived pre-NN GP chain implementation."""

import argparse
import gzip
import hashlib
import json
import os
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    os.sched_setaffinity(0, {options.cpu})
    import jax
    import numpy as np
    from jax import numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from benchmarks.issue_312.gp_geometry import GP
    from benchmarks.issue_312.gp_hessian_sampler import GPHessianSliceSampler

    folder = Path(__file__).resolve().parent
    source = gzip.decompress((folder / "raw/gp_hessian_sampler.py.before_nn.gz").read_bytes())
    assert hashlib.sha256(source).hexdigest() == (
        "f62a9fcb82242e1890097eadb96ad173874e42c75c407c6282d807c8694f801b"
    )
    # This fixed, committed source is our own trusted reference implementation,
    # not an arbitrary user-provided archive. Load it under a separate module
    # name so its registered pytree types cannot replace the current classes.
    reference_module = types.ModuleType("issue312_previous_gp_sampler")
    sys.modules[reference_module.__name__] = reference_module
    exec(  # noqa: S102 - execute only the checksum-verified reference above.
        compile(source, "archived_gp_hessian_sampler.py", "exec"), reference_module.__dict__,
    )
    with np.load(folder / "gp_audit/seed-0/fit-020.npz") as saved:
        gp = GP(*(jnp.asarray(saved[name]) for name in GP._fields))
    peak = -np.log(1e-5 * np.sqrt(2 * np.pi)) - 14 * np.log(0.05 * np.sqrt(2 * np.pi))
    previous = reference_module.GPHessianSliceSampler(gp, float(peak))
    current = GPHessianSliceSampler(gp, float(peak))
    rng = np.random.default_rng(31221)
    points = rng.uniform(size=(64, 18))
    ball = rng.normal(size=(64, 15))
    ball /= np.linalg.norm(ball, axis=1, keepdims=True)
    ball *= (np.sqrt(30) * rng.uniform(size=64) ** (1 / 15))[:, None]
    points[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * points[:, 1]) + 1e-5 * ball[:, 0]
    points[:, 2:16] = 0.5 + 0.05 * ball[:, 1:]
    keys = jax.random.split(jax.random.PRNGKey(31222), len(points))

    def likelihood(point):
        residual = point[0] - 0.5 - 0.3 * jnp.sin(2 * jnp.pi * point[1])
        return peak - 0.5 * ((residual / 1e-5) ** 2 + jnp.sum(((point[2:16] - 0.5) / 0.05) ** 2))

    def run(sampler):
        @jax.jit
        @jax.vmap
        def chain(key, point):
            return sampler.run_chain(
                key, point, likelihood(point), jnp.asarray(peak - 15), likelihood,
            )
        return jax.block_until_ready(chain(keys, jnp.asarray(points)))

    expected, observed = run(previous), run(current)
    reference_leaves, current_leaves = map(jax.tree.leaves, (expected, observed))
    assert len(reference_leaves) == len(current_leaves)
    for old, new in zip(reference_leaves, current_leaves):
        np.testing.assert_array_equal(old, new)
    record = {
        "chains": len(points), "slices": 90, "array_leaves": len(current_leaves),
        "bit_identical": True, "includes_intermediate_states": True,
        "original_sampler_sha256": hashlib.sha256(source).hexdigest(),
        "source_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                          for name in ("gp_hessian_sampler.py", "hessian_chain.py")},
    }
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
