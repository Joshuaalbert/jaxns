# Review Checklist

Reviews must be strict and design-led.
Read `LEARNINGS.md` before reviewing so repo-specific failure modes stay in scope.

## Order of Review

1. Contract correctness
2. State and lifecycle invariants
3. Test quality
4. Latency and batching implications
5. Code quality and maintainability

## Mandatory Questions

- Does the code match the final design docs exactly where the docs are explicit?
- Did the implementation quietly invent behavior not present in the design?
- Are all enum sets, payload fields, and ordering guarantees enforced by tests?
- Do tests prove the intended invariants, or only the current implementation?
- Are performance-sensitive paths free of obvious avoidable allocations, locks, or over-serialization?
- Does the code preserve deterministic behavior for identical ordered inputs?
- Are error cases explicit and structured where the design requires them?
- Are audit artifacts and manifests complete and machine-checkable?

## Findings Policy

- Findings must cite the ticket, file, and design contract they violate.
- “Looks wrong” is not enough; point to the exact violated invariant.
- If a design ambiguity is discovered, stop and raise a design finding instead of improvising.

## Common Failure Modes To Watch

- payloads omitting ownership fields on later lifecycle events
- tests depending on current helper functions rather than contract-level behavior
- tests proving local stand-in logic instead of forcing the intended future module boundary
- hidden timestamp or timezone assumptions
- batch or pulse bounds left qualitative instead of measured
- replay/rebuild logic that silently tolerates missing data
- audit folders without a complete manifest
