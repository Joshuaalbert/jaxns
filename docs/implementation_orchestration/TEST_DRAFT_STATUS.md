# Test Draft Status

This file tracks the first test-first review round.

## Accepted After Review

None yet.


## Accepted Provisionally


These are provisionally accepted pending any later cross-service cleanup, but they are already strong enough to guide
implementation.

- `0001-race-tree-state-and-blocks`: tests cover block construction, plateau
  recurrence, permutation invariance, and invalid `K_g < m_g`; pending final
  reviewer confirmation.
- `0002-bayesian-shrinkage-and-plateaus`: tests cover v3 Dirichlet
  concentrations, evidence sampling, plateau weights, and invalid blocks;
  pending final reviewer confirmation.
- `0004-results-api-and-diagnostics`: tests cover public result block metadata,
  v3 posterior weights, phantom summaries, shrinkage-returned
  `rho_values`/`rho_fit`, resampling alignment, plotting, and malformed-input
  validation.
- `0007-constrained-sampler-contract`: tests cover strict seeds,
  phantom-burn-in validation, and likelihood-only phantom clusters; pending
  final reviewer confirmation.
- `0009-distributed-runtime-and-async-identity`: revised tests cover
  worker-spec parsing, `LoadBalancerClient` run-pattern interface, shared local
  and tcp-address namespaces, serialized worker payload execution, coordinator
  task/attempt/transport identity, acceptance-ledger idempotence, lifecycle
  failure/retry/revoke/replay/stale paths, worker registration/shutdown,
  compile identity/fairness diagnostics, legacy distributed API migration, and
  actual serialized worker retry before acceptance.
- `0008-baseline-directions-straight-line`: revised tests cover ellipsoidal
  integrated-volume selection, adaptation context, frozen per-chain kernels,
  greedy-shrink trace behavior, and tunneling risk.
- `0012-galilean-gradient-sampler`: revised tests cover built-trajectory
  reversibility/symmetry, 2D direction-flip symmetry, path-length sampling,
  phantom exclusion, negative paths, and local/distributed Galilean parity.
- `0011-execution-diagnostics-integration`: revised tests cover public
  diagnostics access, execution-time depth/goal observations, mode-specific
  allocation summaries, coordinator dispatch trace, worker identity, and
  diagnostics/statistics separation.
- `0010-validation-benchmarks`: revised tests cover deterministic fixtures,
  benchmark schemas, producer rollups, multi-seed RMSE/Pareto and posterior
  quality, grouping by full nested `method_setting`, and the minimal public
  collector that runs real v3 toy validations for baseline race-tree,
  phantom-conditioned, dynamic allocation, and Galilean settings.
- `0003-phantom-conditioned-shrinkage`: revised tests cover per-block counts,
  malformed phantom metadata, paper `rho_g` estimation, low-order fitted
  `rho_g`, and fitted-rho usage in the public shrinkage path.
- `0005-core-execution-depth-first`: revised tests cover exact
  allocation-plan targets for `k`/`DeltaK`, public run/resume goal boundaries,
  root initialization, strict parent selection, sentinel fallback, exact
  out-degree updates, weighted parent-block selection, and in-flight parent
  preservation.
- `0006-allocation-utilities`: revised tests cover formula helpers, numerical
  edge cases, exact evidence/posterior utility plan construction from v3 block
  summaries, normalized utilities, and shared integer target construction.

## Drafted Awaiting Review

None currently.

## Sent Back For Revision

None currently.

## Review Rule

No service moves into the implementation loop until its tests are in the accepted or accepted-provisionally state and
are considered strong enough to enforce the final design rather than just current code behavior.
