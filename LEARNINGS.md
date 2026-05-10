# Repository of learnings from agents

Add relevant things we learned about the project, and how to do certain things. If a preexisting "learning" becomes
contradictory, it must be updated to keep this files consistent. This file shall be read by agents to help them not
repeat mistakes specific to this project. Keep learnings compact.

---

- The updated phantom-conditioning paper target is not the old `rho_g A_g`
  effective-count Dirichlet update. Use gamma-weighted per-cluster counts:
  independent race gammas, independent `v_c ~ Gamma(1, 1)` cluster weights, and
  Kish participation gating.
- For phantom conditioning, the open-interval contribution is
  `R_cg = A_cg - B_cg - E_cg`. The current paper equations print `A_cg` in the
  `M'_<g` update, but the Monte Carlo vector and singleton Dirichlet-recovery
  argument require `R_cg`.
- Use `C_min` as the canonical Kish participating-cluster threshold name for
  phantom conditioning, with default `20` unless a ticket/API explicitly
  overrides it. Blocks with no participating clusters fail the phantom gate.
- In this workspace session, `jaxns_py` was restored after initially being
  absent. Use the required `conda run -n jaxns_py ...` commands for checks.
- Ticket 0013 gamma-weighted phantom sampling must not apply post-hoc
  finite-batch moment correction. Each returned probability draw must equal
  `normalize(race_gamma + sum_c v_c * counts_cg)` for that draw.
- Ticket 0013 standard-problem acceptance must run the full original
  `STANDARD_PROBLEM_CASES` list directly in pytest, with no subprocess
  isolation. Each parametrized case must create and close its own local
  `LoadBalancerClient(address='local')` workers so LB teardown is the intended
  memory cleanup.
- Run standard-problem local-LB cases serially. Parallel JAX compilations of
  these tests can exhaust LLVM section memory and produce misleading failures.
- For the GMM-based non-isotropic direction kernel, the paper target is a GMM
  fit to the posterior from samples collected so far. Treat five-shell refits
  and exact fitting thresholds as v3 policy, not paper-derived statements.
- The accepted Ticket 0008 ellipsoidal direction path only freezes a
  one-component bounding ellipsoid from adaptation history. Full
  posterior-weighted GMM fitting is owned by Ticket 0014.
- The earlier full standard-problem local-LB blocker was
  `uniform-basic_mvn` missing the `3 * sample_std` evidence criterion under
  isotropic directions. Ticket 0014's GMM non-isotropic direction path plus the
  allocation-selector starvation fix resolved the gate; do not resurrect the
  old isotropic miss as a current failure.
- Ticket 0014 GMM direction-kernel cadence is measured since the last
  successful direction update. A failed eligible fit attempt increments
  attempts but does not reset the five-distinct-shell clock, so the next
  distinct shell remains eligible for another attempt.
- Ticket 0014 fitting gates are production gates, not test convenience gates:
  require `N_eff >= max(20, 2 * (D_dim + 1))` and per-component
  responsibility `N_eff,k >= D_dim + 1` without small-fixture bypasses.
- For Ticket 0014 MC-shrinkage fitting weights, normalize posterior weights
  within each shrinkage draw first, then average normalized per-sample weights
  across draws. Averaging raw masses and normalizing once is not equivalent
  when the evidence varies by draw.
- Future runtime design should treat process isolation as the durable JAX
  worker boundary: `ZMQActor` load balancer, node coordinator, and workers owned
  by `ProcessManager`; LB talks to node coordinators, which fan out to local
  workers over random `/tmp` `ipc://` endpoints.
- Allocation target utility should be encoded in `build_allocation_plan()`
  through `target_K`; `select_parent_work()` must service positive deficits
  without multiplying by `unit_peak_utility` again, or baseline target deficits
  in low-utility/high-likelihood blocks can be starved.
- Parent selection runs in Python orchestration and must avoid
  variable-length JAX random draws. Use NumPy weighted-CDF selection with a
  scalar keyed uniform; `jax.random.categorical` over changing candidate
  lengths causes repeated JAX compilation in long standard-problem runs.
- Local runtime dispatch should reuse sampler bytes from
  `RuntimeCompileIdentity` construction. Re-serializing the same sampler once
  per accepted local-LB dispatch is avoidable coordinator overhead and should
  stay bounded by the runner identity setup, not by sample count.
- Do not enable a compiled fused `UniDimSliceSampler` runtime worker path until
  strict seed validation and mutable problem identity are designed explicitly.
  A naive jitted wrapper can bypass Python seed checks and reuse stale compiled
  likelihoods when mutable args/params change in place.
- Standard-problem speed benchmarks must keep wall-clock timing fractions
  separate from aggregate worker sampler latency. Summed worker dispatch
  latency can exceed elapsed run wall time when workers overlap, so report it
  as a diagnostic rather than a wall-clock component.
- Benchmark MC-shrinkage timing must materialize or `block_until_ready()` the
  returned evidence samples before stopping the timer; otherwise JAX async
  dispatch can under-report the completed MC sampling cost.
- For v3 `_sample_parent_work`, strict seed selection should use the sorted
  active likelihood prefix and `searchsorted(..., side="right")` semantics
  rather than rebuilding a dynamic boolean candidate array per parent. This
  preserved plateau strictness and reduced full 8D `basic_mvn` benchmark time
  from about `150.89s` to `88.99s` in Ticket 0017.
