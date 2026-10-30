# Phantom evidence extension of the allocation comparison

Replay all 180 frozen runs from `ALLOCATION_PROTOCOL.md` with phantom
collection enabled. Check exact equality with every existing 2M/4M archive:
classic coordinates, likelihoods, constraints, weights, volumes, seed/parent
and batch identities, likelihood counts, evidence and completed epochs.
The original runner and kernel sources remain frozen. Phantoms add no model
calls and never enter allocation or the stopping rule.

Retain the existing sampler-defined prefixes, including repeated rejected
states: 89 slice endpoints for Hessian and eight complete-cycle endpoints
for reflection. Exclude the final child, and retain one cluster per child.
Prior roots have no phantom cluster. Do not treat intermediate reflection
trajectory states as phantom observations.

Use the existing evidence law with 4,096 draws, batch size 64, `C_min=20`,
and all retained phantoms. Preserve its shared per-cluster Gamma weights and
Kish gate. Archive its complete inputs and evidence draws. Keep existing
capacities to reuse event-plan compilations; masked padding is excluded.
Verify the archive path is exactly equivalent to `result.sample_evidence`
in both conditioning modes.

The initial CPU pilot found JAX Gamma sampling too costly for the full cohort
(810 seconds for one Hessian classic ensemble). Before cohort inference,
use the benchmark-only NumPy/SciPy backend for the same distribution:
production event plan and Kish gate, sparse linear maps for A/B counts,
NumPy Gamma draws, and the same evidence quadrature. Since only p_> enters
evidence, combine independent Gamma(alpha_=) and Gamma(alpha_<) into one
Gamma(alpha_= + alpha_<); no equality mass is dropped. Verify explicit
draws against the dense reference with tied blocks, active/inactive gates,
and production quadrature. Compare real-run pilot ensembles to JAX separately.
Production sampling and evidence code remain unchanged.

MC keys start at `PRNGKey(312044)`, then fold in seed, target model-call
budget, kernel index (Hessian/reflection), allocation index
(uniform/evidence/posterior), and conditioning index (classic/phantom).
Use the final key's two uint32 words as NumPy `SeedSequence` entropy for
PCG64. They never consume a constrained-sampling key. Also compute classic Monte
Carlo evidence to distinguish the change in estimator convention from
phantom conditioning. These draws quantify shrinkage uncertainty, not
independent likelihood experiments.

Report RMSE against true log Z = 0 at both budgets for classic moments,
classic MC and phantom MC. Include bias, quoted uncertainty, coverage,
paired-bootstrap RMSE differences, and MC standard errors. Use all 30
seeds in every cell. Conditioning cannot restore missing posterior support;
the original posterior/movement diagnostics still describe these exact runs.
