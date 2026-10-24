# Repository of learnings from agents

Add relevant things we learned about the project, and how to do certain things. If a preexisting "learning" becomes
contradictory, it must be updated to keep this files consistent. This file shall be read by agents to help them not
repeat mistakes specific to this project. Keep learnings compact.

---

- In the v3 depth core, keep sample/phantom payloads in append order and merge
  only a lightweight likelihood-index pytree. Rebuilding an ordered payload
  every depth iteration adds avoidable device work and memory traffic.
- Allocation magnitude and execution width are separate. Uniform iteration
  `k` targets `d0 * delta_K * k`; utility allocation requests the direct gap
  `ceil(delta_K * unit_peak_utility)`. The static `vmap` width only controls
  how many frozen thread heads execute together.
- Compare v3 with v2 at v2's actual default evidence goal,
  `dlogZ=log1p(1e-3)`; a looser threshold gives misleading performance data.
- Distinct seed selection is stationary only when exclusions are made between
  lanes with the same parent contour. Greedy exclusion across nested contour
  masks biased selected likelihood ranks from 0.50 to 0.47 on the analytic
  weak-curved diagnostic; group equal contours before choosing distinct seeds.
- Keep the replacement vmap narrower than the root population on difficult
  targets. On the issue-247 10D mixture, S=100 took 20.9 s while S=150 took
  215.6 s because data-dependent rejection loops execute to the slowest lane.
- Intersect per-coordinate entry/exit intervals for slice cube bounds.
  Classifying a zero intersection as both signs traps face seeds; zero
  direction components need an unrestricted interval and a safe denominator.
- Singleton likelihood blocks are the paper's exact two-class `Beta(K, 1)`
  race, not a three-class Dirichlet with a tiny equality atom. Sampling them as
  `U**(1/K)` both preserves the model and avoids dominant Gamma-sampling cost.
- Phantom conditioning draws one Gamma(1, 1) weight per retained chain cluster
  and reuses it across that cluster's block counts. Do not merge clusters by
  inferred root ancestry; the paper defines clusters from stationary
  constrained-chain observations, not race-tree parent indices.
- Automatic checkpoints are opt-in full `State` or `DistributedState` Pytree
  saves. The checkpoint layer owns atomic publication and corruption detection;
  callers own model, arguments, sampler, and runner compatibility on resume.
- Sample-buffer growth resumes an unfinished compiled depth epoch. Do not run
  a custom scientific goal at that physical boundary; initial capacity must
  not change the logical allocation round or stopping point.
- GMM directions are immutable `State` data and entirely user staged. Fit from
  every stored classic `(U, log L)` at a coherent Python boundary, calibrate
  component mean likelihoods from those stored values, select by ellipsoid
  volume trimmed above the parent contour, and never fit or probe likelihoods
  inside a depth loop.
- A bounded recent-sample reservoir is appropriate only while one frozen
  thread schedule is active. Once that schedule drains, publish every newly
  stationary classic into the exact seed source before projecting the same
  allocation target. Reusing the older source reduced boundary time but made
  three fixed-seed standard gates fail and raised Jones likelihood work by 73%.
- In the issue-312 oracle experiment, an inverse Hessian at posterior scale
  prevented early broad-contour mixing, even on the straight ridge. Scaling
  by contour volume geometry before imposing a prior-variance cap removed
  that startup failure; the curved marginal remained poorly mixed. See
  `benchmarks/issue_312/HESSIAN_REPORT.md` before generalising this experiment.
- Normalised local Gaussian directions require the angular central Gaussian
  density in a per-transition MH correction. Keep rejected self-transitions;
  the endpoint density ratio alone assumes the fixed-direction slice kernel
  remains reversible, including its bracketing rule.
- In the issue-312 GP experiment, interpolating stored log likelihoods did
  not imply accurate Hessians on the thin curved population. Audit held-out
  values and curvature separately; a high MH acceptance or stationary
  ensemble can coexist with almost perfectly correlated chain endpoints.
  The offline GP refresh hook is not a checkpointable production interface.
- The issue-312 midpoint reflection experiment uses exact cube drifts around
  an involutive velocity reflection and retains invalid endpoints as rejected
  self-transitions. Time reversal does not require finding a contour crossing.
  It improves evidence but still under-mixes flat marginals; count autodiff
  calls separately. Off-contour value probes helped a subsequent NN recover
  local stiff curvature, while full-ridge generalisation remained poor. See
  the reflection and NN reports before treating either as a production kernel.
- Actual issue-312 NN-Hessian runs failed much worse than the GP despite good
  late stiff-axis fits. A Hessian norm dominated by one enormous eigenvalue
  hides false curvature in flat/moderate directions. Audit direction variance
  relative to the true normal and retained-chain movement, not only fit error
  and stationary marginals; see `benchmarks/issue_312/NN_SAMPLING_REPORT.md`.
- In the local-GMM ridge experiment, warm EM fits to successively trimmed
  populations lost all but one substantial component. Fresh 100-iteration
  fits preserved ten components but local MH-corrected directions still
  under-mixed. The correction uses the full marginal angular mixture at
  both endpoints, including the isotropic branch, not a GMM point-density
  ratio. See `benchmarks/issue_312/LOCAL_GMM_REPORT.md` for the tested scope.
