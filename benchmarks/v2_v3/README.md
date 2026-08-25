# Maintained v2-versus-v3 scientific release benchmark

This is the release gate for comparing the maintained `v2` line with v3. It
preserves the matched standard-problem design and initial 1,200-run evidence
from issue #247 while making later comparisons commit-addressed, resumable,
and reproducible from pinned, isolated environments.

## Scientific contract

- Ten cases: exactly the cases in `tests/test_ns_standard_problems.py`.
- Two conditioning configurations: classic and phantom, with seeds 0–29.
- Both lines use root degree `30D`, five isotropic perfect-slice transitions
  per dimension, float64, CPU, and `dlogZ=log1p(1e-3)`.
- V2 is configured with `c=30D`; its `num_live_points` argument would divide
  independent chains by `k + 1` when phantoms are enabled.
- V3 uses replacement/allocation width `10D`. V2's live array width is `30D`;
  these are different architecture concepts, so both are reported rather than
  falsely described as equal.
- Compilation/lowering, warmed execution, result conversion, MC inference,
  likelihood evaluations, peak RSS, environment, and configuration are
  recorded separately. Every timed JAX value is synchronised.
- Release evidence requires 30 unique seeds in every exact
  implementation/problem/conditioning group. Missing, duplicate, mixed-commit,
  or non-finite records fail validation.
- Gaussian problems report latent posterior-mean RMSE. Mixture problems also
  report analytic versus inferred mode weights, missed modes, and mode-weight
  errors. Spike–Slab mode allocation is a release diagnostic, not merely an
  evidence-integral benchmark.

Phantom retention is evaluated as a distinct final shrinkage conditioning
mode. It does not change v3's classic race tree. Raw phantom counts, retained
states per chain, gate activation, and effective Kish cluster counts remain in
each record.

## Reproduce a release comparison

From a checkout containing the candidate v3 commit:

```bash
workspace=/tmp/jaxns-v2-v3-release
bash benchmarks/v2_v3/prepare_environments.sh \
  "${workspace}" v2 HEAD
bash benchmarks/v2_v3/run_release_matrix.sh \
  v2 "${workspace}" "${workspace}/results"
bash benchmarks/v2_v3/run_release_matrix.sh \
  v3 "${workspace}" "${workspace}/results"
conda run --prefix "${workspace}/env-v3" python \
  benchmarks/v2_v3/summarise.py --release-gate \
  --output "${workspace}/SUMMARY.md" \
  "${workspace}"/results/*.jsonl
```

The preparation command resolves both refs to full commits, creates detached
source worktrees, creates separate pinned conda environments, installs exactly
those source trees, and records import paths and device metadata. Repeating a
matrix command resumes at the first missing seed and refuses to append to a
group from another commit or configuration.

Copy the validated JSONL, `SUMMARY.md`, the two environment JSON files, and
regenerated diagnostics into a versioned release-artifact directory before
approval. Do not overwrite a prior release's evidence. The release reviewer
records the hardware, commit SHAs, runner command, unexpected failures, and
whether the 10% median runtime/IQR review threshold found a material
regression. Wall-clock comparisons are reviewed on controlled hardware and
are not flaky assertions on shared pull-request runners.

## Initial v3.0 evidence

The versioned initial corpus remains in [`benchmarks/issue_247`](../issue_247/):

- [`REPORT.md`](../issue_247/REPORT.md) contains the full per-problem table and
  the known accuracy/performance residuals.
- [`results/final_v2`](../issue_247/results/final_v2/) and
  [`results/final_comparison`](../issue_247/results/final_comparison/) contain
  1,200 raw runs: 30 seeds for all 40 implementation/problem/phantom groups.
- [`diagnostics`](../issue_247/diagnostics/) contains representative
  Spike–Slab corner plots and phantom shrinkage diagnostics selected by the
  declared median-error rule.

Those historical v3 records identify the implementation as `working-tree` and
predate the posterior-mode schema above. The initial corpus
[`PROVENANCE.json`](../issue_247/PROVENANCE.json) records the commit that first
captured the measured code and data together. They establish the v3.0
pre-release scientific baseline but do not substitute for the final
commit-addressed gate or for future release comparisons.
