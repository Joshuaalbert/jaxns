# Issue 255 packed-parameter evaluation

## Decision

Do not adopt `jaxctx.ParameterPack` for JAXNS Bayesian U-space now, and do not
open an implementation PR. The released jaxctx 1.1.5 abstraction is an
optimization-parameter API, not a sampling API, and it violates the normalized
prior-measure contract when called from `Model.sample_U`. A benchmark-only
hypothetical packed-U implementation shows a stable benefit only for a
pathological many-scalar-leaf microcase. It does not clear the pre-registered
end-to-end gate, cannot represent existing dependent priors, and has no current
distributed transport path on which to demonstrate a production benefit.

This evaluation is based on `develop` commit
`4a47c69eec256deb16988f26733cc38c6c2e2365`, Python 3.12.9, JAX/JAXLIB 0.10.0,
jaxctx 1.1.5, CPU float64, and the source-validated worktree at
`/tmp/jaxns-issue-255-parameter-pack`. The decision threshold was recorded on
issue 255 before the final matrix: require a repeatable improvement of at
least 15% in a representative end-to-end nested-sampling or real distributed
path, larger than variability, with no regression greater than 5% in
low-leaf, heterogeneous, or dependent cases. Microbenchmarks, HLO, pickle, and
loopback measurements are explanatory evidence only.

## Scientific compatibility

`ParameterPack` exposes only `.parameter()`. It creates one unconstrained N
parameter, maps it through `quick_unit`, and writes packed U plus independently
transformed X values. JAXNS instead expects `Prior.realise()` to draw normalized
uniform U and later expects caller-supplied U to determine X and likelihood.

The 10,000-sample semantic probe found:

| path | U leaves | mean | standard deviation | range |
|---|---:|---:|---:|---:|
| current `Prior.realise()` | 2 | 0.49980 | 0.28865 | 0.00000–0.99993 |
| public pack, default | 1 | 0.50000 | 0.00000 | 0.50000–0.50000 |
| public pack, random init | 1 | 0.50031 | 0.10807 | 0.13523–0.87147 |

The uniform target standard deviation is 0.28868. `ParameterPack` has no
`realise()` method. Supplying packed U `[0.1, 0.9]` to
`Model.transform_to_X` still returned physical values `[0.5, 0.5]`, because
the optimization parameter path does not consume the supplied Bayesian U.
Using it directly would therefore change the prior measure, not merely its
storage representation.

The public abstraction also cannot statically pack JAXNS's general sequential
models. The standard `basic3` case constructs the distribution of `y` from an
already realized `x`; `ParameterPack` requires all independent prior objects
before execution and applies their transforms independently. Supporting that
case requires a new sequential U-slice allocator or a different model
description, not a local substitution.

For transform-only measurements, the public N path was supplied with
`quick_unit_inverse(U)` outside the model. Current, transport-flat, hypothetical
packed-U, and that out-of-band public transform agreed within `1e-12`. This
validates only the static slicing and transforms; it does not make the public
pack a valid sampler.

## Compiler and runtime evidence

Each microbenchmark retained five tracing samples, three separately timed
lower/compile samples, and 51 synchronized executable calls. Values below are
steady medians in milliseconds from the initial and repeat processes. The
absolute dispatches are only tens of microseconds and were sensitive to the
shared host; the heterogeneous result even changed sign between processes.

| shape | current initial / repeat | hypothetical packed-U initial / repeat | interpretation |
|---|---:|---:|---|
| D64, 1 leaf, B1 | 0.01652 / 0.04518 | 0.01614 / 0.01646 | no structural packing opportunity; timing unstable |
| D1024, 1 leaf, B1 | 0.03699 / 0.06149 | 0.04317 / 0.06793 | packed slower by 17% / 10% |
| D64, 64 scalar leaves, B1 | 0.06513 / 0.08037 | 0.02579 / 0.03357 | stable pathological dispatch win, about 58–60% |
| D96, 8 heterogeneous leaves, B16 | 0.08425 / 0.14367 | 0.11271 / 0.10591 | changed from 34% slower to 26% faster |

The D64/64-leaf case explains the plausible mechanism but does not establish
adoption evidence. Its initial compile median increased from 2.459 s to
3.023 s; the repeat changed to 2.576 s versus 2.518 s. Current and hypothetical
packed programs had 4,095 versus 4,223 JAXPR equations and 460,794 versus
468,466 StableHLO characters. Packing replaced tuple dispatch with 64 static
slices/reshapes; it did not reduce HLO size or compiler-reported temporary
memory. The compile conclusion was itself not repeatable.

The corrected bounded sampler benchmark warms both variants, alternates their
execution order by seed, and runs 30 synchronized seeds. Two independent
processes gave paired median improvements of 14.80% and 14.88%; bootstrap 95%
intervals were 12.25–16.25% and 11.62–17.52%. The packed medians were
1.392/1.396 ms versus current 1.623/1.643 ms. This is just below the 15% gate
on the paired statistic and is a four-dimensional, four-scalar-leaf,
12-sample bounded smoke path rather than a representative production run.

The analytic log evidence for that model is -0.92963. Across the same 30
seeds, current U had mean -0.85989 with standard deviation 0.22507 and the
hypothetical pack had mean -0.89539 with standard deviation 0.18499. Both are
consistent with the deliberately tiny run. Different random-key partitioning
means these are distributional, not pointwise, comparisons. A production
posterior/evidence acceptance matrix was not run because the actual public
API already fails the prior-measure invariant and cannot express the dependent
standard case.

## Serialization and distributed scope

For D64 split into 64 scalar leaves, a flat float64 pickle fell from 2,150 to
635 bytes. Including host ravel, dump, load, and unravel, two processes measured
about 0.094/0.103 ms for flat transport versus 0.233/0.249 ms for the current
pytree. That high-fragmentation result is repeatable. Flattening was slower for
the one-leaf cases after including pack/unpack work, and the eight-leaf
heterogeneous case changed from about 5% faster to about 19% slower.

Current `develop` has no runtime, worker, socket, RPC, or distributed extra.
U remains inside the compiled local sampler and crosses a serialization
boundary only during explicit state persistence. Consequently there is no
real worker or representative loopback transport to optimize. A deleted
prototype once pickled each proposal U, but historical traffic is not current
production evidence.

If distributed likelihood execution returns, evaluate a schema-checked,
dtype-preserving transport-only flat buffer first. It can capture the stable
many-leaf wire benefit while keeping the named scientific U representation.
The existing `jaxns.pytree.pytree_ravel` is not suitable for heterogeneous
transport as written because it records shapes and sizes but not per-leaf
dtypes; JAX's `ravel_pytree` already demonstrates dtype-restoring behavior
elsewhere in the package.

## Performance and code-intent review

The review fixed four methodology/intent problems in the evaluation harness:

1. It labels the released optimization N path separately from the hypothetical
   Bayesian packed-U path, preventing transform timings from being mistaken
   for sampling correctness.
2. It validates the absolute imported source path and fails if the editable
   install resolves outside this worktree.
3. It synchronizes every device timing, separates trace/lower/compile/steady
   phases, retains raw samples, repeats processes, and alternates the full-run
   order to limit thermal and shared-host bias.
4. It forces the `Agg` backend and hard-bounds the documented commands so the
   evaluation cannot open plotting windows or silently become an unbounded
   sampler run.

Changing production U would obscure rather than clarify ownership. U is a
public field of samples and results; one bare packed array would discard member
names, shapes, order, and checkpoint identity unless accompanied by a versioned
schema. It would also diffuse transport concerns through model execution,
sampler state, stored samples, and results. The clearer boundary is to preserve
named U internally and pack only at a proven transport boundary.

No production code was changed because the adoption gate failed. There is no
implementation PR to review. Revisit only when jaxctx provides a genuine,
dependency-aware sampling pack or JAXNS has a concrete distributed boundary
whose end-to-end measurements can beat dtype-safe transport-only flattening.

The final source check passed Ruff and flake8 for this directory. The complete
repository suite passed 224 tests in 499.90 seconds with `MPLBACKEND=Agg`, the
CPU backend selected explicitly, this worktree's `src/` on `PYTHONPATH`, and a
15-minute hard timeout.
