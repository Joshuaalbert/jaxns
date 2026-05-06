# Implementation Ticket Index

Priority order is based on design risk, hot-path impact, and how much downstream work depends on the service.
Use the dependency DAG below rather than treating the numeric order as a strict
linear blocker. Each ticket should take the minimal perturbation path through
the current code: extend existing `src/jaxns` modules and tests where their
boundaries already match v3, and introduce new helper modules only when they
make the migration smaller and easier to verify.

## Dependency DAG

- `0001-race-tree-state-and-blocks.md`: no dependencies.
- `0002-bayesian-shrinkage-and-plateaus.md`: depends on `0001`.
- `0003-phantom-conditioned-shrinkage.md`: depends on `0001`, `0002`.
- `0004-results-api-and-diagnostics.md`: depends on `0001`, `0002`, `0003`.
- `0007-constrained-sampler-contract.md`: depends on `0001`, `0003`.
- `0005-core-execution-and-depth-first-allocation.md`: depends on `0001`, `0002`, `0007`.
- `0006-allocation-utilities.md`: depends on `0005`; non-blocking for `0008` and `0009`.
- `0008-direction-kernels-and-trajectories.md`: depends on `0005`, `0007`.
- `0009-distributed-runtime-and-async-identity.md`: load-balanced worker runtime and runner interface; depends on `0001`, `0003`, `0005`, `0007`.
- `0010-validation-and-benchmark-suite.md`: deterministic fixtures depend on `0001`, `0002`, `0003`; final roll-up depends on `0004`, `0005`, `0006`, `0008`, `0011`, and applicable feature tickets such as `0009` and `0012`.
- `0011-execution-diagnostics-integration.md`: depends on `0004`, `0005`, `0007`, `0008`, `0009`; include `0006` when evidence/posterior allocation diagnostics are enabled, and include `0012` only if Galilean/gradient-informed diagnostics are enabled.
- `0012-galilean-and-gradient-informed-sampler.md`: depends on `0007`, `0008`.

## Suggested Waves

1. Foundation: `0001`.
2. Statistical core: `0002`, then `0003`, then `0004`; start deterministic validation fixtures from `0010` after `0003`.
3. Sampler/execution: `0007`, then `0005`; `0008` can run after both.
4. Allocation and advanced sampler modes: `0006` and `0012` can run without blocking unrelated work.
5. Load-balanced worker runtime and diagnostics: `0009`, then `0011`.
6. Final validation and benchmark roll-up: remaining `0010` work.

## Branch Names

- `feature/v3-race-tree-state-and-blocks`
- `feature/v3-bayesian-shrinkage-and-plateaus`
- `feature/v3-phantom-conditioned-shrinkage`
- `feature/v3-results-api-and-diagnostics`
- `feature/v3-core-execution-depth-first`
- `feature/v3-allocation-utilities`
- `feature/v3-constrained-sampler-contract`
- `feature/v3-baseline-directions-straight-line`
- `feature/v3-load-balanced-worker-runtime`
- `feature/v3-validation-benchmarks`
- `feature/v3-execution-diagnostics-integration`
- `feature/v3-galilean-gradient-sampler`

## Ticket Files

- `docs/implementation_orchestration/tickets/0001-race-tree-state-and-blocks.md`
- `docs/implementation_orchestration/tickets/0002-bayesian-shrinkage-and-plateaus.md`
- `docs/implementation_orchestration/tickets/0003-phantom-conditioned-shrinkage.md`
- `docs/implementation_orchestration/tickets/0004-results-api-and-diagnostics.md`
- `docs/implementation_orchestration/tickets/0005-core-execution-and-depth-first-allocation.md`
- `docs/implementation_orchestration/tickets/0006-allocation-utilities.md`
- `docs/implementation_orchestration/tickets/0007-constrained-sampler-contract.md`
- `docs/implementation_orchestration/tickets/0008-direction-kernels-and-trajectories.md`
- `docs/implementation_orchestration/tickets/0009-distributed-runtime-and-async-identity.md`
- `docs/implementation_orchestration/tickets/0010-validation-and-benchmark-suite.md`
- `docs/implementation_orchestration/tickets/0011-execution-diagnostics-integration.md`
- `docs/implementation_orchestration/tickets/0012-galilean-and-gradient-informed-sampler.md`
