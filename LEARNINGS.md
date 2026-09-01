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
