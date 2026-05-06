# Implementation Orchestration

This folder is the execution workspace for turning the final design into code.

The design docs in [`docs/design`](docs/design) are the
source of truth. Current code is implementation context only. If code and design disagree, follow the design and open
an explicit review finding.

## Contents

- [IMPLEMENTATION_TICKET_INDEX.md](docs/implementation_orchestration/IMPLEMENTATION_TICKET_INDEX.md)
  gives the service-ticket order and branch names.
- [UNIT_TEST_STANDARDS.md](docs/implementation_orchestration/UNIT_TEST_STANDARDS.md)
  defines what good invariants look like.
- [PERFORMANCE_TEST_STANDARDS.md](docs/implementation_orchestration/PERFORMANCE_TEST_STANDARDS.md)
  defines Python-baseline performance and determinism checks.
- [REVIEW_CHECKLIST.md](docs/implementation_orchestration/REVIEW_CHECKLIST.md)
  is the review rubric for both tests and implementations.
- [EXECUTION_LOOP.md](docs/implementation_orchestration/EXECUTION_LOOP.md)
  defines the strict agent workflow.
- [`tickets/`](docs/implementation_orchestration/tickets) contains one service
  ticket per implementation branch.

## Working rule

For each service ticket:

1. Write and review (with a sub-agent) the tests first.
2. Before any implementation or review work, read [`LEARNINGS.md`](LEARNINGS.md) and apply
   the repo-specific lessons there.
3. Do not start implementation until the tests are considered strong enough.
4. Implement in a dedicated git branch branched of develop branch. Be aware you're not alone, others are working on
   other implementation branches at the same time.
5. Review every implementation with separate review sub-agents, who strictly review against ticket and design docs.
6. Feed review findings back into the ticket and iterate until the service passes review.

## Stop condition

This orchestration pass is complete only when each service ticket has:

- explicit code changes merged or ready to merge
- tests that enforce the design and invariants strongly
- review findings either fixed or explicitly accepted
- no remaining ambiguity that would force reviewers to guess what the design intended

Only after this stop condition is asserted merge the final implementation branches into develop and delete the
implementation branches to keep the repo clean. 
