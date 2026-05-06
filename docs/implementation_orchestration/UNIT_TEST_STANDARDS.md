# Unit Test Standards

These are the minimum standards for design-driven tests in this repo.

## General Rules

- Tests must target the design contract, not implementation accidents.
- Prefer deterministic unit tests over integration-heavy tests unless the ticket explicitly asks for integration.
- Use small, legible fixtures. Prefer `NamedTuple` or frozen dataclasses where structure matters.
- State-transition tests must assert both the event outcome and the resulting state snapshot.
- Negative-path tests are required for malformed input, duplicate input, stale input, and unsupported transitions.
- If a service has ordering semantics, tests must assert exact order, not just final state.
- If a service has idempotency semantics, tests must call the same input twice and assert the exact allowed outcome.
- If a service has a source-of-truth rule, tests must assert that consumers do not need hidden prior context.

## Anti-Patterns

- No tests that merely mirror current code internals.
- No local helper or harness logic that implements the very contract the test is supposed to verify.
- If the production surface does not exist yet, tests must force that future surface explicitly rather than proving an
  invented local stand-in.
- No broad snapshot assertions when a tighter invariant is available.
- No hidden dependency on wall clock time.
- No performance claims without a measured threshold and a rationale.
