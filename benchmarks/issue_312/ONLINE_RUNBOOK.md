# Online reflection experiment entrypoints

Read REFLECTION_ONLINE_PROTOCOL.md first. The protocol, source freeze and
coordinator releases own the allowed phases/seeds. These scripts do not grant
release. CPU70 is reserved for independent review.

Run from this worktree with `PYTHONPATH=.:src`, `JAX_PLATFORMS=cpu`,
`CUDA_VISIBLE_DEVICES=''`, `JAX_ENABLE_X64=1`, and OMP/OPENBLAS/MKL thread counts1.
Use `taskset -c CPU conda run --no-capture-output -n jaxns_py python -m ...`;
each numerical entrypoint also pins itself to `--cpu` before importing JAX.

The initial isolated preflight is `smoke_reflection_online`. It records its own
source inventory and has no evaluation mode. The shared released runner is
`run_reflection_online`, extending the existing fixed-work harness. It refuses
training/evaluation without the exact coordinator release and source hashes.

After source freeze, the shared smoke command is:

```text
benchmarks.issue_312.run_reflection_online --cpu 2 --arm online \
  --allocation uniform --seed 3122920 --phase smoke --output OUT/smoke/uniform
```

Use CPU3, allocation `evidence_improving` and seed3122921 for the other smoke.
Both take first complete batch crossings of30k,60k,300k calls. The longer allowed
preflight demonstrates real parameter changes; zero-delta consumed versions do
not count. Outputs contain CORE, classic snapshots, phantom likelihood inputs,
controller/policy archives, chronological receipts, and a full OnlineCheckpoint.

After coordinator TRAINING_RELEASE, the runner uses `--phase training`,
`--arm online`, and all four seeds3122910..3122913 for each allocation. Place
runs at `OUT/training/online/ALLOCATION/seed-SEED`. These are independent runs
with2M/4M snapshots, charged separately from evaluation. Reduce each run with:

```text
benchmarks.issue_312.analyse_reflection_tuning --cpu CPU --run RUN_DIRECTORY
```

This reuses the existing evidence reducer and offline chain replay. It first
checks schema3 predispatch information and chronological learning, and writes
CONTROLLER-AUDIT.json. No replay or evidence result enters online feedback.
The frozen arm additionally supplies `--schedule FROZEN_POLICY.json` to both
runner and reducer. Large raw arrays remain remote; do not commit them.

Once all eight training runs pass their audits, run:

```text
benchmarks.issue_312.freeze_reflection_online --cpu 4 \
  --training-root OUT/training --output OUT/frozen
benchmarks.issue_312.validate_reflection_online --cpu 4 \
  --schedule OUT/frozen/uniform.json \
  --evidence-schedule OUT/frozen/evidence_improving.json --output OUT/validation
```

The freezer uses only final classic compression maps and online parameter
tables, including unchanged regions. It never selects on evidence/posterior
accuracy. The held-out validator reuses the independent constrained ensembles
and original thresholds. Raw fields distinguish path versus full-chain choices
and B/h counts; their keys differ. No early-arm equality is assumed for a
learned policy with different independently trained endpoint tails.

Only after coordinator EVALUATION_RELEASE, run all seeds60..89 with
`--phase evaluation` and arms `fixed`, `online`, `frozen`, both allocations,
matched2M/4M snapshots. Keep the source HEAD and inventory unchanged through
sampling/reduction. Start four workers, at most24 after measured resource
checks, preserving64GiB available RAM and20GiB disk headroom. CPU70 is excluded.
Training/evaluation failures are diagnosed, never discarded or retuned.

`state.pkl` is the complete typed experiment checkpoint: core, controller,
frozen policy where applicable, ledgers, policy history, budget/snapshot
position, timings and source inventory. `OnlineSession.from_checkpoint`
constructs a fresh owner; a bare production State cannot resume online
adaptation. Full fresh-session and phantom-off/on parity tests are in
`cicd/tests/test_reflection_online.py`.
