# Execution Loop

This is the strict implementation workflow.

## Phase 1: Ticket Preparation

1. Read `LEARNINGS.md` before doing implementation or review work.
2. Use the ticket file as the branch charter.
3. Confirm the design doc links in the ticket are still correct.
4. Do not start coding until the test plan is accepted.

## Phase 2: Test-First

1. Spawn a test-writing sub-agent for the ticket branch.
2. Ask it to write tests only, not implementation.
3. Review the tests against:
    - `UNIT_TEST_STANDARDS.md`
    - `PERFORMANCE_TEST_STANDARDS.md`
    - the service ticket & and supporting design docs
4. Iterate until the tests are strong enough.

## Phase 3: Implementation Loop

1. Spawn one implementation sub-agent on the dedicated ticket branch.
2. Require the implementation sub-agent to confirm it has read `LEARNINGS.md`.
3. Spawn one or more review sub-agents on the same branch after implementation work lands.
4. Require review sub-agents to confirm they have read `LEARNINGS.md`.
5. Save review feedback back into the ticket as follow-up actions.
6. If findings remain, repeat implementation -> review until the service passes.

## Branch Discipline

- One active implementation branch per service ticket.
- Test-only work may happen in the same branch before implementation begins.
- Review agents do not silently edit implementation ever. They only comment with findings, which the implementation
  agent must explicitly resolve.

## Exit Criteria Per Ticket

- tests pass and are considered strong
- implementation matches the design
- review findings are resolved
- no known design ambiguity remains in the implemented area
