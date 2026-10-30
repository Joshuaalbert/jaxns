# Coordinator source/protocol review 02

Protocol v2 independent actual-lambda frozen policy is approved. Initial source
review covered online_policy/controller/sampler/session, the observer seam and
the shared kernel parameter seam, plus focused tests. Scope and data ownership
look appropriate; no production-boundary crossing is required.

Before source freeze:
- Update allocation_observer module and run_with_observer docstrings: the
  optional sampler-supplier path now intentionally changes the benchmark
  kernel, whereas the default path is still read-only. Keep that distinction
  explicit for future readers.
- The continuation test currently restores a controller into the SAME session,
  whose in-memory rows/history remain present. Also test restoring into a
  freshly constructed session/full experiment checkpoint, so missing ledger,
  policy-history, budget, timing or statistics state cannot be hidden by that
  retained Python object. Original and resumed child/phantom/key/count outputs
  must agree. It is fine to keep benchmark checkpoint ownership local.
- Record real parameter changes separately from version increments or a
  zero-delta consumed update. The preflight learning check should establish
  nontrivial h or T changes, with all suppression reasons still reported.
- Retain requested/effective duration distinction at the B cap; the protocol
  and corrected test now explicitly allow effective T down to .00256.

No source changes selected on inference outcomes have been requested. Training
can be released after the runner, source-freeze receipt and numerical smoke
gates are ready for coordinator inspection. Evaluation still waits for frozen
training policies and corresponding held-out tests.
