# Issue 249 bounded final-MC report

## Decision

Use the 64-draw economical streaming program as the default final-evidence
path. It materially lowers measured and compiler-reported memory on every
required production problem, preserves the statistical model, and is faster
in steady execution. The one-time compile-and-first call is also faster on two
problems and 4% slower on one, despite the larger bounded-control program.

This report compares this branch with its exact parent, develop commit
`cb17efc`, rather than with a historical package. Both sources use Python
3.12.9, JAX/JAXLIB 0.10.0, CPU float64, seed 0, 1,000 final evidence draws,
30 root chains per dimension, replacement width 10 per dimension, five
isotropic perfect-slice transitions per dimension, and the same completed
nested-sampling state for each problem. Every reported source record verified
its imported module path. The first timing includes lowering, compilation, and
execution; steady timing is the median of five synchronized repetitions, with
the interquartile range shown.

## Production measurements

| problem | source | blocks | phantoms | first s | steady median [IQR] s | peak RSS MiB | mean log Z |
|---|---|---:|---:|---:|---:|---:|---:|
| basic MVN | develop | 6,049 | 47,104 | 8.739 | 6.432 [6.330, 6.733] | 3,018.9 | -24.18646 |
| basic MVN | bounded 64 | 6,049 | 47,104 | 7.095 | 4.483 [4.399, 5.394] | 825.1 | -24.19262 |
| spike-slab 8D | develop | 5,898 | 45,880 | 9.389 | 5.961 [5.946, 5.973] | 2,986.1 | -18.49912 |
| spike-slab 8D | bounded 64 | 5,898 | 45,880 | 6.337 | 4.064 [3.853, 4.167] | 840.3 | -18.48513 |
| spike-slab 10D | develop | 7,862 | 76,610 | 11.499 | 8.752 [8.708, 8.914] | 4,277.3 | -22.98789 |
| spike-slab 10D | bounded 64 | 7,862 | 76,610 | 8.519 | 5.661 [5.604, 5.737] | 990.3 | -22.98211 |

Peak RSS falls by 72.7%, 71.9%, and 76.8% respectively. Median steady runtime
improves by 30.3%, 31.8%, and 35.3%. Compile-and-first improves by 18.8%,
32.5%, and 25.9%. The paired mean-log-evidence differences are 0.0140 or
smaller and are far below the run-level uncertainties; formal
distributional tests compare independently keyed ensembles at explicit Monte
Carlo tolerances.

The final basic-MVN batch sweep showed the intended trade-off: batch sizes 32,
64, 128, and 256 reached about 741, 817, 966, and 1,272 MiB peak RSS. Batch 64
had the best measured knee at 4.53 seconds steady; 32 saved another 76 MiB but
took 4.75 seconds, while 128 and 256 took 4.77 and 5.90 seconds. A 512-draw
prototype reached about 1,871 MiB. Wall-time measurements were collected on a
shared CPU and are therefore interpreted together with compiler memory and
the large RSS deltas, not in isolation.

Classic-only MC was checked separately because it was already inexpensive.
On the same basic-MVN state, develop's five-run steady median was 0.860 seconds
[IQR 0.693, 1.394] with 1,157.4 MiB process peak; bounded 64 took 0.523
seconds [0.485, 0.534] with 664.0 MiB peak. Its compile-and-first time improved
from 4.272 to 2.354 seconds. Thus the selected default does not trade the
phantom improvement for a classic-only regression.

## Compiler and execution-plan evidence

For the 6,049-block basic-MVN program, XLA's CPU memory analysis reports:

| source | argument MiB | output MiB | temporary MiB | HLO bytes | sorts | scatters | whiles |
|---|---:|---:|---:|---:|---:|---:|---:|
| develop | 0.66 | 277.68 | 2,055.77 | 163,813 | 1 | 14 | 11 |
| bounded 64 | 0.66 | 0.78 | 151.60 | 188,027 | 1 | 14 | 13 |

The output estimate falls by 99.7% because the default no longer returns six
unrequested draw-by-block diagnostic matrices. Temporary memory falls by
92.6% because the scan keeps only one vmapped MC batch live and carries only
the online block sufficient statistics between batches. Deterministic
sparse-event sorting and Kish preparation remain outside that scan, evidenced
by the unchanged single sort. The two additional while calls are the bounded
scan/control structure and explain the larger HLO. Their one-time compilation
effect is measured rather than inferred from HLO size. Replacement sampling in
the nested-sampling depth loop remains fully vmapped and is not changed by
this final-inference scan.

## Statistical and contract evidence

- Fixed-key calls are deterministic for each specialization. If a batch is at
  least the requested draw count, the result is bit-for-bit the old single
  batch for that key.
- Batched and unbatched independently keyed ensembles agree in evidence and
  block-probability moments at explicit Monte Carlo tolerances.
- A final partial batch is weighted by its actual draw count when sufficient
  statistics are merged.
- Tests cover batch size one, a partial batch, a batch no smaller than the
  ensemble, invalid sizes, classic no-phantom mode, singleton two-class
  blocks, plateau three-class blocks, and Kish gating both active and inactive.
- The unchanged standard-problem release gate passes all 20 phantom-off/on
  cases. The existing 30-seed-per-row v2/current accuracy matrix remains in
  `benchmarks/issue_247/REPORT.md`; this change alters only how draws from that
  same final shrinkage model are streamed.
- The final repository suite passes 224 tests in 487.41 seconds; reviewer
  autochecks, focused Ruff, and flake8 syntax/undefined-name checks also pass.

Full per-draw/per-block diagnostics remain opt-in with `diagnostics=True`.
Those arrays have an unavoidable output-memory floor proportional to draws
times blocks, which the API and docstrings state explicitly.

## Performance and intent review

The required review found and fixed two material issues before acceptance:

1. The first compiled scan emitted one `[G]` sufficient-statistic vector per
   batch and reduced them afterwards. Although small at four batches, this
   would asymptotically recreate an `[M, G]` intermediate for batch size one.
   The final scan instead carries five online `[G]` accumulators and emits only
   the required `[M]` evidence and entropy ensembles.
2. The provisional 256-draw default was dominated after that correction.
   Representative 32/64/128/256 measurements selected 64 as the runtime and
   memory knee.

No unresolved correctness or hot-path performance finding remains. At the
execution-layer boundary, the deterministic sparse event plan is prepared once
per compiled call, one `vmap` samples independent draws within a batch, and one
device `scan` serializes batches. This does not introduce `lax.map` or alter the
fully vmapped replacement sampler in the depth loop. Draw count and batch size
remain static specialization inputs, as they determine compiled shapes and
loop trip count; scientific arrays and the frozen, slotted event pytree remain
dynamic children rather than large static constants.

Residual risks are explicit: requesting full diagnostics necessarily retains
the draw-by-block output; changing draw count or batch size may compile a new
specialization; and the measurements cover CPU rather than accelerator
backends. The benchmark runner records source path, versions, platform, device,
precision, production shapes, first-call time, repeated synchronized time,
process RSS, and optional compiler memory/HLO structure so those risks can be
remeasured.
