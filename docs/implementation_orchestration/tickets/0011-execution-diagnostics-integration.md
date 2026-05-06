# Ticket 0011: Execution, Sampler, and Worker-Runtime Diagnostics Integration

Branch: `feature/v3-execution-diagnostics-integration`
Priority: after execution, sampler, and worker-runtime correctness
Depends on: Ticket 0004, Ticket 0005, Ticket 0007, Ticket 0008, Ticket 0009; include Ticket 0006 when evidence/posterior allocation diagnostics are enabled; include Ticket 0012 only if Galilean/gradient-informed diagnostics are enabled
Design docs:

- `docs/design/jaxns-v3-validation-plan.md`
- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`

## Goal

Integrate execution, sampler, and worker-runtime diagnostics into the public
result/reporting surface after those systems exist. Ticket 0004 covers the
block/shrinkage result API; this ticket adds runtime diagnostics that would be
premature before the execution and worker-runtime contracts are implemented.

## Current Code Context

Relevant files likely include:

- `src/jaxns/results.py`
- `src/jaxns/state.py`
- `src/jaxns/core.py`
- `src/jaxns/constrained_sampler.py`
- `src/jaxns/constrained_sampler_distributed.py`
- `src/jaxns/fabric/`
- `tests/test_results_plotting.py`
- `tests/test_distributed_core.py`
- `tests/test_fabric_distributed.py`

## Required Behavior

Expose diagnostics needed by the validation plan and implementation reviews:

- allocation mode and allocation target summaries;
- depth condition values and selected stochastic summaries;
- goal condition values and selected stochastic summaries;
- parent-selection summaries and sentinel fallback counts;
- sampler mode, direction kernel mode, trajectory mode, burn-in, and retained
  phantom counts;
- likelihood evaluations per classic sample and per retained phantom cluster
  where available;
- load-balanced worker count, compute-sector summaries, runner ids, task ids,
  accepted/retried/revoked task counts, model-compilation timings, and async
  identity-preservation diagnostics where worker execution is used.

Diagnostics must not be required to compute the statistical result. They are
audit outputs for validation and performance review.

Diagnostics that depend on transient execution state, such as task ids, retry
attempts, revokes, and in-flight parent targets, cannot be reconstructed after
the fact from current `NestedSamplerResults` fields. They must be captured
during execution in a structured public diagnostics object and attached to, or
otherwise reachable from, public results.

## Out Of Scope

- Core shrinkage result API. Covered by Ticket 0004.
- Feature implementation. Covered by Tickets 0005, 0006, 0007, 0008, 0009, and
  0012.
- Final benchmark production. Covered by Ticket 0010.

## Test Plan

Write tests before implementation.

Required unit tests:

- Runtime diagnostics are available from public result objects or an attached
  public diagnostics object, without private imports.
- No-phantom, phantom, local, and load-balanced worker runs expose consistent
  schemas with explicit empty values where data is not applicable.
- Depth and goal summaries in diagnostics match the values used by execution.
- Sampler diagnostics reflect the actual configured sampler mode, burn-in, and
  phantom retention.
- Worker-runtime diagnostics preserve task ids and in-flight parent targets
  without exposing worker-local state as the source of truth.
- Malformed or misaligned diagnostic arrays fail explicitly.

## Implementation Notes

- Prefer extending `NestedSamplerResults` or attaching a small companion
  diagnostics object over introducing a second result pipeline.
- Reuse values already produced by `State`, `TerminationRegister`, sampler
  returns, and worker-runtime task metadata. Do not recompute diagnostics from
  private logs when structured state is available.

## Acceptance Criteria

- Validation scripts do not need private imports for runtime diagnostics.
- Runtime diagnostics are separated from statistical weights and evidence
  estimates.
- Tests cover unavailable diagnostics explicitly rather than relying on missing
  fields or `None` surprises.
