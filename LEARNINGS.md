# Repository of learnings from agents

Add relevant things we learned about the project, and how to do certain things. If a preexisting "learning" becomes
contradictory, it must be updated to keep this files consistent. This file shall be read by agents to help them not
repeat mistakes specific to this project. Keep learnings compact.

---

- In the v3 depth core, keep sample/phantom payloads in append order and merge
  only a lightweight likelihood-index pytree. Rebuilding an ordered payload
  every depth iteration adds avoidable device work and memory traffic.
- A static `vmap` replacement width must be paired with an allocation increment
  of that width by default. Growing the target by one lineage makes nearly all
  lanes dummy work and adds a Python/device round trip per accepted sample.
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
- Phantom clusters descended from one root race lineage are not independent
  after finite Markov transitions. Share their gamma weight; grouping only
  exact repeated seeds understated uncertainty on the 10D curved release gate.
