# Common Context for Working on jaxns

This is an opt-in orientation for an agent that has read `AGENTS.md` and the
relevant design material but still does not feel it understands the system or
the human intent. It captures recurring reasoning and collaboration patterns
from prior work with JoshuaAlbert. It is deliberately not a requirements
document. When it conflicts with current code, requirements, invariants, the
paper, or an explicit instruction, those sources win.

## The Shortest Useful Version

Aim for **correctness and completeness first, with performance and
maintainability treated as joint design constraints**. Understand the whole
state flow before changing a local function. Do not call something a bug,
improvement, or performance win without evidence. Once a real bug is found,
fix it narrowly, demonstrate the regression before and after, and record any
new requirement or invariant it revealed.

Write code whose intention a human can assert. Prefer a linear implementation
that reads top to bottom, clear Pytree/module ownership, and comments explaining
non-obvious consequences. Reuse existing mechanisms. Look for sprawl,
duplicated concepts, stale alternatives, and code that has become purposeless.

## How JoshuaAlbert Reasons With Agents

JoshuaAlbert often starts with a proposal while explicitly saying that it may
not be complete. Treat it as a serious model to compare against the current
system, not as either a loose suggestion or an instruction to agree blindly.
Identify sameness and divergence, fill gaps from the code and evidence, and
challenge a detail when the implementation or underlying library proves
otherwise.

> I don't claim my thoughts to be complete; together we can identify sameness
> and divergence.

JoshuaAlbert may add a refinement while the agent is still investigating or
may review an answer by questioning one premise at a time. Integrate those
messages into the same model. Do not treat each correction as an isolated
patch or lose the still-valid parts of the earlier request.

The recurring standard is:

> correctness and completeness of logic and state flow; performance and
> scalability using smart composition that avoids unnecessary movement; and
> maintainability through clear code intent and modularity

JoshuaAlbert supplies domain corrections quickly and expects the agent to
absorb the underlying contract, not merely patch the sentence that triggered
the correction. When he says that an idea is fictitious in the real world,
remove the invented dependency from the model. When he distinguishes a
scientific contract from an implementation detail, tests and docs should
preserve the contract without unnecessarily freezing the implementation.

It is good to say, with evidence, “the coercion is wrong, but the correct type
is integer rather than float.” It is not good to silently defer, guess, or
retrofit evidence around the first proposed answer.

## Reconstruct Intent Before Acting

When context is weak, walk the complete path relevant to the task:

1. Identify the external input, canonical source, ownership boundary, and
   output or side effect.
2. Trace how state is initialized, updated, routed, persisted, and presented.
3. Separate scientific and user-facing contracts from current implementation
   choices and historical leftovers.
4. State the inferred contract concretely enough that it can be tested.
5. Inspect requirements, invariants, design docs, the paper, tests, maintained
   configs, and the real call path—not only the file named in the request.

Do not use an example, stale branch, rendering, or accidental current behavior
as the source of truth when a maintained contract exists. Conversely, do not
preserve an old abstraction merely because it has tests if the project has
deliberately moved on.

## Evidence and Diagnosis

JoshuaAlbert prefers diagnosis in order of likelihood, with instrumentation
and controlled experiments when necessary. A useful investigation normally:

- chooses an independent baseline whose semantics are understood;
- makes the discrepancy reproducible;
- compares like with like, including termination conditions, live or root
  lineage counts, sampler settings, phantom handling, precision, and hardware;
- adds temporary diagnostics at the narrowest useful boundary;
- records experimental results so work can resume after interruption;
- distinguishes a demonstrated cause from noise or a plausible theory; and
- removes temporary diagnostics after the cause is established.

If asked to diagnose, do not rush into a fix. Present the cause and evidence
for review first. Once a fix is approved, use targeted A/B testing so the
result proves that the identified cause—not an incidental change—moved the
outcome.

Phrases such as “strong evidence” and “forensic” are literal expectations.
For performance work, report compilation and steady-state runtime separately
when relevant, compare before and after on realistic shapes and production
composition, and check the complete path rather than relying on a flattering
microbenchmark. Performance work may require several iterations. For
statistical or stochastic behavior, use enough seeds and report dispersion;
three convenient runs are not strong evidence when thirty are practical.

## Correctness Discipline

If work uncovers a real bug, add a regression test. Demonstrate that the test
fails before the fix and passes after it. If the bug reveals a new invariant or
requirement, record it and map its test coverage.

Not every design change needs a new test that freezes its present structure.
Tests should preserve externally meaningful behavior, requirements, and
invariants. Implementation-specific tests are appropriate when a real
implementation requirement exists; incidental decomposition is not a
contract.

Do not add runtime checks merely to compensate for a contract that construction
or tests can guarantee. State ownership, fixed shapes, and schema validation
should live at the boundary that actually owns them. Runtime hot paths should
not accumulate redundant projections and defensive work “just in case.”

Random state is scientific state. When a compiled depth loop returns to Python
for orchestration, resizing, or goal evaluation, verify which key represents
the immediate continuation and which key belongs to the next logical epoch.
A transparent interruption must not silently change the sampler trajectory.

“Surgical” means the smallest coherent change, not a shallow one. If a
correctness premise changes, carry it through every semantically equivalent
path—expectation and Monte Carlo, classic and phantom-conditioned, runtime and
paper/model—so the system does not retain two meanings. Do not opportunistically
refactor unrelated code while doing so.

## Design and Code Intent

Prefer semantic states and transitions over mechanical decomposition. Linear
steps with no conditional gate usually belong in one callback or function,
because an unconditional state transition does not express a decision.

For competing implementations:

- keep the public contract and test preparation shared where possible;
- parameterize contract/invariant tests across implementations;
- keep implementation-specific requirement tests beside that implementation;
- make each implementation modular enough to remove cleanly;
- benchmark realistic end-to-end composition; and
- once a production choice is made, remove legacy choices and naming rather
  than retaining switches with no product purpose.

A parallel implementation is not truly modular if deleting it requires
untangling production code, shared docs, and unrelated tests. “Could we safely
delete one folder and one test folder?” is a useful design pressure even when
the final layout shares contract tests.

Use existing state, result, sample, ordering, termination, and serialization
machinery before creating another version. Duplication is sometimes clearer
than an abstraction with excessive mental plumbing, but duplicated concepts or
sources of truth are dangerous.

Prefer removing unnecessary representation work over making it faster. If data
is already packed, do not flatten and concatenate it again. Before adding an
index, register, projection, diagnostic, or compatibility layer, ask what
consumer needs it and which invariant owns it. Sorting, batching, and resizing
often make redundant identity state actively misleading.

For JAX-heavy code, keep the physical model readable by eye:

- show important array shapes beside fields and transformations;
- give each module a cohesive Pytree state rather than a monolithic union of
  every possible field;
- use structured `lax.cond`/`lax.switch` control flow when it expresses the
  semantic branches more clearly, then inspect compilation rather than
  guessing what it lowers to;
- avoid materialization, movement, and fictitious updates that the chosen
  static composition does not require;
- preserve batched replacement sampling unless evidence supports changing the
  execution model; and
- put rare capacity growth in a transparent orchestration boundary when that
  keeps resizing and recompilation out of the steady-state kernel.

User-facing result and state objects should expose the conceptual model.
Cluster secondary diagnostics in a named Pytree instead of crowding essential
results, and avoid version suffixes or legacy adapters unless a real consumer
needs simultaneous compatibility. Backward compatibility has a cost; do not
pay it for a hypothetical user who would necessarily rebuild or rerun.

## Compact JAXNS Mental Model

The scientific path repeatedly separates concepts that are easy to conflate:

- model: prior transformation and likelihood evaluation;
- constrained sampling: stationary generation above a strict contour;
- scheduling: selection and advancement of logical lineage threads;
- shrinkage: race-tree inference, optionally conditioned on phantom Monte
  Carlo observations; and
- presentation: immutable state and result Pytrees used by scientific users.

A useful high-level execution sketch is:

```text
model + random key + user goal condition
        -> Python goal-level orchestration
        -> jitted depth loop and lineage allocation
        -> vmapped constrained-prior replacement sampling
        -> samples, contours, out-degrees, and phantom clusters
        -> expectation or Monte Carlo shrinkage
        -> evidence and posterior result objects
```

The race-tree formulation makes parent likelihood contours and sample
out-degrees the essential scheduling information; do not add mutable identity
state without a demonstrated consumer. The inner depth loop is pure JAX and
performance sensitive. The outer loop is Pythonic so user goal conditions,
rare storage growth, and future distributed orchestration can remain outside
the steady-state kernel.

Expectation estimates are suitable for cheap depth decisions. Final
user-facing evidence calculations use Monte Carlo shrinkage, with optional
phantom conditioning. Phantom states are Monte Carlo observations, not
independent race participants, and their stationarity and atom/no-atom
semantics are correctness requirements.

This sketch is orientation only. Read the current paper, requirements,
invariants, design documents, and implementation before relying on a detail.

## Collaboration and Operational Hygiene

JoshuaAlbert often says “you are not alone.” Assume other agents may be
working in parallel. PR work belongs in a dedicated worktree and issue-backed
PR branch unless JoshuaAlbert explicitly asks for changes in the primary
checkout. Inspect other worktrees and dirty files before editing, and keep the
task modular enough to avoid conflicts.

Worktree isolation is not architectural coordination. Identify the ownership
and semantic boundary of the feature before borrowing an existing pattern or
touching a neighboring subsystem. A pattern with the wrong intent, lifecycle,
or consumers is worse than a little local duplication. Larger steps are fine
inside a well-understood boundary when JoshuaAlbert and the agent share a
concrete intention and JoshuaAlbert understands and owns the scientific or
product decision. If the work needs a new shared boundary, changes another
feature's contract, or crosses into a colleague's ownership, surface that fact
before implementation and ask JoshuaAlbert to coordinate with the relevant
contributors. Developing that contract in a vacuum risks diffusing a locally
plausible but globally wrong model through the codebase.

An unexpected boundary crossing is also a diagnostic: the agent may not yet
understand the human intent or complete data flow. Help the human see the
crossing, trace it, and clarify it rather than silently expanding the task.
Ephemeral exploration does not need an issue merely because it happened; issue
tracking becomes mandatory when the work is made into a PR. Make long sweeps
and evaluations resumable because the laptop may move or the work may be
interrupted. Give concrete status and ETA updates rather than disappearing.

For a requested PR, “ready for human review” normally means:

- the issue and PR explain intent and evidence;
- behavior, regression, invariant, and implementation-specific tests pass;
- the relevant local CI/autocheck workflow has been reproduced when hosted CI
  cannot run;
- performance-sensitive work includes credible before/after evidence;
- docs and maintained configs agree with the code;
- stale alternatives and dead-purpose code were reviewed; and
- the branch is clean and mergeable.

After merge, remove the completed local worktree and, when its remote branch
has been deleted, delete the local PR branch. Return the primary checkout to
current `develop`. Stop orphaned test supervisors, but do not stop unrelated
user services or another contributor's processes.

## What This File Does Not Authorize

This context does not grant permission to mutate production services,
credentials, or unrelated work. It does not replace a requested review with
implementation, broaden a diagnosis into a fix, or make an old conversational
decision permanent. Use it to ask better questions and form better hypotheses,
then verify the current contract in the repository.
