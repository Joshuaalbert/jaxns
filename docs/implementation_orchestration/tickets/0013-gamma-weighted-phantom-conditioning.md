# Ticket 0013: Gamma-Weighted Phantom Conditioning Target

Branch: `feature/v3-gamma-weighted-phantom-conditioning`
Priority: immediate statistical correction after accepted Ticket 0003
Depends on: Ticket 0001, Ticket 0002, Ticket 0003, Ticket 0004, Ticket 0007, Ticket 0010, Ticket 0011
Design docs:

- `docs/design/jaxns-v3-phantom-conditioning.md`
- `docs/design/jaxns-v3-statistical-core.md`
- `docs/design/jaxns-v3-execution-and-allocation.md`
- `docs/design/jaxns-v3-constrained-sampling.md`
- `docs/design/jaxns-v3-validation-plan.md`
- `docs/design/interface/run_pattern.py`

## Goal

Replace the accepted Ticket 0003 implementation target, which used `rho_g`
effective-count Dirichlet phantom conditioning, with the current paper target:
gamma-weighted per-cluster phantom conditioning with parent-contour-gated
counts, a Kish participating-cluster gate, and Kish/gate diagnostics instead of
`rho_g` curves.

Do not rewrite Ticket 0003 history. This ticket is the branch charter for the
new migration from the old accepted target to the updated design target.

## Current Code Context

This context is from direct inspection of the current repository state.

- `src/jaxns/v3_shrinkage.py` contains
  `phantom_conditioned_dirichlet_concentrations(block_state, phantom_A,
  phantom_B, phantom_E, rho_g)`. It validates finite `0 < rho_g <= 1` and adds
  `rho_g * B_g`, `rho_g * E_g`, and `rho_g * (A_g - B_g - E_g)` to classic
  Dirichlet concentrations. The same module also exposes
  `estimate_raw_rho_g_from_bootstrap_covariance(...)`,
  `fit_low_order_rho_g_curve(...)`, and `sample_dirichlet_probabilities(...)`.
- `src/jaxns/phantom_eval.py` imports the old `rho_g` helpers. Its
  `EvidenceSamples` dataclass exposes `rho_samples`, `rho_values`, `rho_fit`,
  `eta_samples`, `rho_eta_samples`, aggregate `phantom_A/B/E`, and
  conditioned Dirichlet alphas. `sample_mc_shrinkage(...)` computes
  per-cluster `A/B/E` matrices internally, fits raw/smoothed `rho_g`, samples a
  global candlestick-likelihood `rho`, and then conditions through
  `phantom_conditioned_dirichlet_concentrations(...)`.
- `src/jaxns/phantom_eval_ref.py` mirrors the same old NumPy reference target:
  global cluster bootstrap, sampled/fitted `rho`, candlestick likelihood, and
  effective-count Dirichlet concentrations.
- `src/jaxns/state.py` currently collapses
  `samples.phantom_samples.valid_mask` to a one-dimensional per-cluster
  `valid_phantom` mask with `jnp.all(..., axis=-1)`, calls
  `compute_phantom_block_counts(...)`, and stores only block-level
  `block_phantom_A/B/E` on the result. It has no result-side `R_g`, Kish count,
  or gate activation fields.
- `src/jaxns/results.py` exposes `NestedSamplerResults.block_phantom_A/B/E` and
  `A_g/B_g/E_g` aliases. `NestedSamplerResults.sample_mc_shrinkage(...)`
  forwards the current old-rho metadata into `phantom_eval.sample_mc_shrinkage`.
  There is no public `phantom_conditioning_diagnostics()` method matching the
  target in `docs/design/interface/run_pattern.py`.
- `benchmarks/v3_validation/schema_checks.py` defines
  `PHANTOM_DIAGNOSTIC_FIELDS = {"rho_g", "rho_fit"}` and requires those curves
  for phantom-conditioned methods. `benchmarks/v3_validation/collector.py` and
  `benchmarks/v3_validation/producers.py` collect/emit those fields and a
  `rho_bootstrap` timing guardrail. `benchmarks/v3_validation/deterministic_fixtures.py`
  encodes old `phantom_A/B/E` plus `rho_g` fixture expectations.
- Existing tests assert the old target, including rho curve shape/range checks
  in `tests/test_v3_phantom_results.py`, JAX/reference rho agreement in
  `tests/test_phantom_eval_jax.py`, old reference outputs in
  `tests/test_phantom_eval_ref.py`, old Dirichlet fixture expectations in
  `tests/test_v3_validation_deterministic_fixtures.py`, and rho-schema
  requirements in `tests/test_v3_validation_benchmark_schema.py`.
- `tests/test_v3_phantom_conditioning.py` is a major old-target surface. It
  asserts fitted `rho_g`/Dirichlet behavior and rho diagnostics, and must be
  migrated to gamma-weighted count, Kish-gate, and diagnostic assertions.
- `tests/test_state.py` currently asserts no-phantom `rho_values`, `rho_fit`,
  and `rho_eta_samples`; those assertions must be migrated to the explicit
  no-phantom/classic posterior and Kish/gate diagnostic conventions.
- `tests/test_ns_standard_problems.py` still contains old rho diagnostic
  plotting/assertion expectations. The full standard-problem run may remain a
  slower validation gate, but the old rho diagnostic expectations in that file
  must not remain as the target behavior.

## Required Behavior

For each block `g` and phantom cluster `c`, compute parent-contour-gated
per-cluster counts:

```text
A_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_{g-1}} 1{L_{p(c)} <= L_{g-1}}
B_{cg} = sum_{(x,L) in P_c} 1{L(x) > L_g} 1{L_{p(c)} <= L_{g-1}}
E_{cg} = sum_{(x,L) in P_c} 1{L(x) = L_g} 1{L_{p(c)} <= L_{g-1}}
R_{cg} = A_{cg} - B_{cg} - E_{cg}
```

The count implementation must validate shapes, finite values on valid blocks,
non-negative counts, and `B_{cg} + E_{cg} <= A_{cg}`. Clusters generated under
a contour stricter than the block's parent contour must not condition that
block.

For each joint shrinkage draw, first draw independent race posterior gammas:

```text
M_{>g} ~ Gamma(a_{>g}, 1)
M_{=g} ~ Gamma(a_{=g}, 1)
M_{<g} ~ Gamma(a_{<g}, 1)

a_{>g} = K_g - m_g + 1
a_{=g} = m_g + epsilon_g
a_{<g} = 1 - epsilon_g
```

Then draw independent cluster weights:

```text
v_c ~ Gamma(1, 1)
```

`v_c` is independent over clusters and independent of the race gammas. The same
`v_c` for cluster `c` must be reused across every block and every component
within that same joint shrinkage draw. A later joint shrinkage draw receives a
new independent set of race gammas and cluster weights.

Use the Kish participating-cluster gate:

```text
C_g^Kish = (sum_c A_{cg})^2 / sum_c A_{cg}^2
I_g = 1{C_g^Kish >= C_min}
```

`C_min` is the canonical threshold name and defaults to `20`. It should be
configurable through the relevant public or semi-public shrinkage API. If no
clusters participate in a block, the denominator is zero and `I_g = 0`.

For gated blocks, add gamma-weighted per-cluster counts:

```text
M'_{>g} = M_{>g} + I_g sum_c v_c B_{cg}
M'_{=g} = M_{=g} + I_g sum_c v_c E_{cg}
M'_{<g} = M_{<g} + I_g sum_c v_c R_{cg}

p_g = (M'_{>g}, M'_{=g}, M'_{<g})
      / (M'_{>g} + M'_{=g} + M'_{<g})
```

Use `R_{cg}`, not `A_{cg}`, for `M'_{<g}`. The no-phantom fallback remains the
classic race posterior: when there are no retained phantoms, no participating
clusters, or `C_g^Kish < C_min`, the block uses only the non-phantom race
gammas for that joint draw. That fallback must not go through the old
`rho_g`/effective-count Dirichlet phantom path.

The implementation must preserve the two limiting behaviors from the design:
singleton independent phantom clusters recover the iid Dirichlet posterior by
gamma additivity, while correlated multi-sample clusters share one `v_c`,
preserve expected count contribution, and inflate variance relative to
singleton independent observations.

Diagnostics must transition from old `rho_g` curves to Kish/gate diagnostics:

- per-block `kish_participating_cluster_counts`, aligned with block likelihoods;
- per-block `phantom_gate_active`, aligned with block likelihoods;
- aggregate count summaries sufficient for review, including `A_g`, `B_g`,
  `E_g`, and `R_g` or clearly named equivalents;
- a public result path matching `results.phantom_conditioning_diagnostics()` in
  `docs/design/interface/run_pattern.py`.

Old `rho_samples`, `rho_values`, `rho_fit`, `rho_eta_samples`, and
`rho_bootstrap` surfaces are no longer target diagnostics. If any old public
names are temporarily retained for compatibility, they must be explicitly
deprecated, must not be required by validation schemas, and must not be used to
drive the gamma-weighted conditioning target.

## Out Of Scope

- Rewriting the accepted Ticket 0003 file or accepted-history notes.
- Changing constrained-sampler phantom collection rules beyond what is required
  to preserve retained cluster likelihoods, cluster identity, and parent
  contour provenance for shrinkage.
- Final expensive benchmark runs. This ticket updates the collector/schema and
  benchmark hooks so later validation can produce the right diagnostics.
- Adding a different phantom-correlation correction. The paper target is the
  `Gamma(1, 1)` per-cluster weight construction with the Kish gate.

## Test Plan

Write tests before implementation and review them against
`docs/implementation_orchestration/UNIT_TEST_STANDARDS.md` and
`docs/implementation_orchestration/PERFORMANCE_TEST_STANDARDS.md`.

Required unit tests:

- Hand-built `NamedTuple` or frozen-dataclass fixtures for per-cluster count
  matrices, covering parent-contour eligibility, equality atoms, open-interval
  `R_{cg}`, blocks with no participants, and stricter-contour clusters that
  must be excluded.
- Negative count tests for shape mismatch, non-finite counts on valid blocks,
  negative counts, `B_{cg} + E_{cg} > A_{cg}`, stale
  `log_L_constraints`, stale classic-cluster associations, and malformed
  phantom validity masks.
- Kish-gate tests proving:
  `C_min` defaults to `20`; the threshold is configurable; denominator-zero
  blocks fail the gate; one large correlated cluster does not pass merely
  because it contains many retained states; twenty singleton participating
  clusters pass at the default threshold.
- Deterministic gamma-target tests using a helper surface that accepts explicit
  race gamma draws and explicit `v_c` weights, so tests can assert that the
  same `v_c` is reused across blocks and components without reimplementing the
  sampler in the test oracle.
- Randomized, seeded distribution tests showing singleton independent clusters
  recover the iid Dirichlet posterior within tolerance, while an equivalent
  correlated multi-state cluster has larger shrinkage variance.
- Tests proving race gammas are independent between `>`, `=`, and `<`
  components, and that cluster weights are independent of those race gammas.
- No-phantom and gate-inactive tests proving the draw reduces to the classic
  race posterior without using old rho fitting, old candlestick likelihood
  sampling, or old phantom Dirichlet alphas.
- Plateau tests proving `E_{cg}` affects `p_{=g}` and posterior equality-mass
  allocation, and `R_{cg}` affects only the open-interval component.
- JAX and NumPy-reference parity tests for identical count fixtures, including
  plateau and non-plateau cases.
- Result/state tests proving `State.to_result()` preserves enough phantom
  likelihood, cluster-validity, and parent-contour provenance to compute the
  new per-cluster counts and diagnostics. If mixed per-phantom validity is
  collapsed at the cluster boundary, that convention must be explicit and
  tested.
- Public result tests for `phantom_conditioning_diagnostics()` with
  block-aligned `kish_participating_cluster_counts`,
  `phantom_gate_active`, and aggregate count summaries.
- Compatibility/deprecation tests for any retained old rho names. The tests
  must prove those names are not required for validation and are not populated
  as target diagnostics without an explicit deprecation path.
- Migrate `tests/test_v3_phantom_conditioning.py` away from fitted
  `rho_g`/effective-count Dirichlet assertions. It should become one of the
  primary tests for gamma-weighted per-cluster conditioning, Kish-gate behavior,
  and public diagnostics.
- Update `tests/test_state.py` and `tests/test_ns_standard_problems.py` so they
  no longer assert old rho diagnostics as required target behavior. If the
  standard-problem run is deferred for cost, the migrated expectations still
  need focused coverage in the faster test set.
- Validation schema/producer/collector tests replacing required `rho_g` and
  `rho_fit` fields with Kish/gate diagnostics and replacing or renaming the
  `rho_bootstrap` performance timing to a gamma phantom-conditioning timing.
- Plotting tests must run under a non-interactive backend. Use
  `MPLBACKEND=Agg` for any test command that can touch plotting.

Required performance checks:

- Keep phantom counting and gamma-weighted shrinkage on vectorized NumPy/JAX
  paths. Do not add per-sample Python loops to the JAX hot path.
- Update benchmark guardrail names and history fields away from
  `rho_bootstrap` toward the new phantom-conditioning work. Because the design
  defines the numeric default `C_min = 20`, add a tuning benchmark hook or
  guardrail note that lets the threshold be revisited with measured evidence.
- Any performance assertion must include a measured threshold and rationale, as
  required by `PERFORMANCE_TEST_STANDARDS.md`.

Suggested focused acceptance commands after implementation:

```bash
MPLBACKEND=Agg conda run -n jaxns_py pytest \
  tests/test_v3_shrinkage.py \
  tests/test_v3_phantom_conditioning.py \
  tests/test_phantom_eval_jax.py \
  tests/test_phantom_eval_ref.py \
  tests/test_v3_phantom_results.py \
  tests/test_state.py \
  tests/test_results_plotting.py \
  tests/test_v3_validation_deterministic_fixtures.py \
  tests/test_v3_validation_benchmark_schema.py \
  tests/test_v3_validation_benchmark_producers.py \
  tests/test_v3_validation_benchmark_collector.py
MPLBACKEND=Agg conda run -n jaxns_py pytest tests/test_ns_standard_problems.py
conda run -n jaxns_py ruff check \
  src/jaxns/v3_shrinkage.py \
  src/jaxns/phantom_eval.py \
  src/jaxns/phantom_eval_ref.py \
  src/jaxns/state.py \
  src/jaxns/results.py \
  benchmarks/v3_validation
conda run -n jaxns_py flake8 \
  src/jaxns/v3_shrinkage.py \
  src/jaxns/phantom_eval.py \
  src/jaxns/phantom_eval_ref.py \
  src/jaxns/state.py \
  src/jaxns/results.py \
  benchmarks/v3_validation
```

Broaden the pytest command if implementation touches runtime diagnostics,
sampler retention, or public run-pattern behavior outside these files.
`tests/test_ns_standard_problems.py` is part of the acceptance gate. It must run
with the intended local load-balancer setup, using
`LoadBalancerClient(address='local')` and locally added workers as in
`docs/design/interface/run_pattern.py`, rather than only direct local sampler
shortcuts.

## Implementation Notes

- Prefer adding a new gamma-weighted helper API over mutating
  `phantom_conditioned_dirichlet_concentrations(...)` in place. If the old
  helper remains, mark it deprecated or private-to-legacy and remove it from
  the active v3 phantom target.
- Keep a pure NumPy/reference implementation in `phantom_eval_ref.py` and a JAX
  implementation in `phantom_eval.py` or `v3_shrinkage.py`; tests must prove
  parity on small deterministic fixtures.
- Generate race gammas and cluster weights from explicit PRNG splits. The
  cluster-weight array should have a cluster axis and must be reused across all
  blocks/components within one joint draw.
- Carry per-cluster count matrices through the conditioning path long enough to
  compute Kish participation, gate activation, and gamma-weighted additions.
  Aggregate `A_g/B_g/E_g/R_g` is useful for diagnostics but is not sufficient
  by itself to implement the target.
- Preserve parent-contour gating through `log_L_constraints` or an equivalent
  provenance field. Do not infer eligibility from phantom likelihood values
  alone.
- Preserve result compatibility deliberately. The preferred new diagnostics are
  Kish/gate diagnostics. Any old `rho_*` compatibility fields must be
  documented as transitional and must not appear in design-validation records
  as required phantom diagnostics.
- Update validation fixtures, producers, schema checks, and timing guardrails
  in the same branch so validation output cannot continue to report old rho
  curves as if they were paper-target diagnostics.
- Keep classic/no-phantom shrinkage behavior unchanged aside from the removal
  of old rho-only diagnostics on the phantom path.

## Acceptance Criteria

- Test-first draft is accepted against `UNIT_TEST_STANDARDS.md` and
  `PERFORMANCE_TEST_STANDARDS.md` before implementation starts.
- Per-cluster `A_{cg}`, `B_{cg}`, `E_{cg}`, and `R_{cg}` counts are
  parent-contour-gated, validated, and tested in both JAX and NumPy reference
  paths.
- The active phantom-conditioned shrinkage target uses independent race gammas,
  independent `Gamma(1, 1)` cluster weights, and shared per-cluster weights
  across all blocks/components in each joint draw.
- `C_min` defaults to `20`, is configurable where the shrinkage API exposes
  phantom conditioning, and blocks with zero or insufficient Kish
  participation use the classic race posterior for that block.
- `M'_{<g}` uses `R_{cg} = A_{cg} - B_{cg} - E_{cg}`.
- Singleton independent phantom clusters recover the iid Dirichlet limiting
  behavior, and correlated multi-sample clusters produce the expected larger
  variance behavior in tests.
- The old `rho_g` effective-count Dirichlet path is no longer the active v3
  phantom-conditioning target.
- Public diagnostics expose Kish participating-cluster counts and gate
  activation instead of required `rho_g`/`rho_fit` curves.
- Validation schema/producers/collector accept and emit the new diagnostics and
  no longer require old rho curves for phantom-conditioned methods.
- `State.to_result()` and `NestedSamplerResults.sample_mc_shrinkage(...)`
  preserve enough phantom provenance for the new target and fail explicitly on
  malformed stale inputs.
- The focused acceptance pytest command passes with `MPLBACKEND=Agg`, `ruff`
  passes for touched modules, and the constrained local flake8 gate used on
  this branch passes. Default flake8 still reports broad 79-column line-length
  noise across touched and pre-existing files, so that cleanup remains outside
  this ticket.
- `tests/test_ns_standard_problems.py` passes under `MPLBACKEND=Agg` using the
  intended local load-balancer setup with local workers.

## Review Follow-Up Resolution

The first implementation pass was blocked on the findings below. Final review
accepted the remediation after the focused Ticket 0013 suite, the
standard-problem gate, `ruff`, `git diff --check`, and the constrained local
flake8 gate passed.

- Resolved: remove the post-hoc finite-batch moment correction in
  `sample_gamma_weighted_phantom_probabilities(...)`. Each returned draw must
  equal `normalize(race_gamma + sum_c v_c * counts_cg)` for that same draw.
  Tests should include a per-draw invariant, including `num_samples=1`.
- Resolved: make public `EvidenceSamples.p_gt_mean`, `p_eq_mean`, and `p_lt_mean`
  reflect active phantom-conditioned probability samples/target means, or
  rename/remove them if they are intended to be classic-only.
- Resolved: reject all stale rho diagnostics in validation records, including
  `rho_samples`, `rho_values`, `rho_eta_samples`, and `rho_bootstrap`, not only
  `rho_g` and `rho_fit`.
- Resolved: remove or quarantine old deterministic validation fixture paths that encode
  `rho_g` effective-count phantom conditioning. Ticket 0013 validation fixtures
  must use gamma-weighted per-cluster conditioning.
- Resolved: ensure the `gamma_phantom_conditioning` performance timing measures actual
  gamma phantom-conditioning work, or rename the metric so it honestly reflects
  what is measured.
- Resolved: zero-participant blocks fail the Kish gate even when `C_min <= 0`.
- Resolved: the NumPy reference gamma sampler validates count matrices before
  sampling.
- Resolved: `tests/test_ns_standard_problems.py` runs the full original
  `STANDARD_PROBLEM_CASES` list directly in pytest, without subprocess
  isolation, and tears down a local `LoadBalancerClient(address='local')`
  between parametrized cases.
- Resolved: update implementation status text after remediation and keep review findings
  visible until resolved.
- Resolved: add or update tests for the above review findings.
- Resolved: run the focused Ticket 0013 suite and
  `MPLBACKEND=Agg conda run -n jaxns_py pytest tests/test_ns_standard_problems.py`
  using the local `LoadBalancerClient(address='local')` worker setup.
