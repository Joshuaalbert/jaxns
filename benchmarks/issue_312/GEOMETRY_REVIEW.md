# Slice geometry and full-cube bracketing

Original review at `b6c45d4`, following the [180-slice experiment](SLICE_COUNT_REPORT.md).
The cube-face bug described below is now fixed on this branch under
[issue #313](https://github.com/Joshuaalbert/jaxns/issues/313). The reported
ridge experiments predate that fix. Reproducible diagnostics are in
[diagnose_geometry.py](diagnose_geometry.py), with numerical output in
[geometry_diagnostics.json](geometry_diagnostics.json) before the fix and
[geometry_diagnostics_fixed.json](geometry_diagnostics_fixed.json) after it.

## What the code brackets

`_slice_bounds` in `src/jaxns/sampling/slice.py` finds the line's intersection
with the unit cube. It does not locate the likelihood-contour intersections.
`_new_proposal` samples uniformly from that bracket, accepts when log L exceeds
the fixed parent contour, and otherwise calls `_shrink_interval`. A rejected
negative displacement replaces the left endpoint; a rejected positive
displacement replaces the right endpoint. The chain anchor stays at t=0
throughout that transition. The continuation implementation uses the same
bounds and shrink primitives.

Thus "perfect bracketing" describes a complete cube chord, not an independent
uniform draw from every valid part of the likelihood-constrained line.

## Connected intervals: cost can change without changing movement

Suppose the valid part of the line is one interval containing the anchor, and
the initial bracket contains that entire interval. Every rejected proposal is
outside the valid interval. Shrinking toward the anchor then removes only
invalid points; it cannot remove an admissible part of the interval. The
accepted point is uniform on that interval in exact arithmetic, regardless of
the excess initial bracket width.

The NumPy reference diagnostic uses the same sign-based shrink rule for
100,000 draws, with valid interval `[-1e-5, 1e-5]` and anchor zero:

| Initial bracket | Mean evaluations | Accepted-point standard deviation |
| --- | ---: | ---: |
| `[-0.5, 0.5]` | 21.242 | 5.763e-6 |
| `[-1e-4, 1e-4]` | 4.411 | 5.781e-6 |

The correct standard deviation is 5.774e-6. A smaller bracket saves work here,
but does not increase the physical step length. Rejection counts alone do not
determine accepted displacement; the original report's exponential conversion
from evaluations to step size should be treated as a heuristic.

## Disconnected intervals: shrinkage can remove distant valid pieces

For a curved or nonconvex contour, the line can intersect several separated
valid intervals. A rejected point in a gap can remove an entire valid interval
on the far side of that gap. Starting in one interval therefore need not
produce an independent uniform draw across the union.

In the reference diagnostic the slice is
`[-0.001, 0.001] union [0.299, 0.301]`, the anchor is zero, and the initial
bracket is `[-0.5, 0.5]`. Only 0.691% of 100,000 transitions land in the distant
interval, although it contains 50% of the slice length. This diagnoses strong
one-step dependence, not a failure of stationarity. The standard shrinkage
kernel can preserve the target while mixing slowly. Neal's analysis distinguishes
these invariant transitions from independent draws and proves the shrink rule's
reversibility; see Sections 4.2–4.4 of [Slice Sampling](https://arxiv.org/pdf/physics/0009028).

A locally sized step-out bracket may save evaluations, but does not
automatically restore jumps to distant components. Correct stepping-out and
acceptance rules matter; arbitrarily truncating a bracket is not a substitute.

## Why global whitening cannot straighten the toy

Write the ridge as `x0 = f(x1) + residual`, with
`f(x1) = 0.5 + 0.3 sin(2 pi x1)`. For a straight displacement `t*d`, the
residual expands locally as

```text
r(t) = r(0) + (d0 - f'(x1)*d1)*t
       - 0.5*f''(x1)*d1**2*t**2 + O(t**3).
```

A generic direction changes the stiff residual at first order. Even a perfectly
tangent direction only cancels that first-order term; curvature still limits
its length. Taking a residual allowance of `1e-5` as an illustration, at a
point with `abs(f'') = 0.3*(2*pi)**2 ~= 11.84`, the tangent displacement along
x1 is only about `sqrt(2e-5/11.84) = 0.0013`. The actual allowance depends on
the contour and the other residual coordinates.

The ideal posterior's global covariance in `(x0, x1)` is, up to negligible
cube truncation,

```text
[[0.3**2 / 2 + sigma_stiff**2, -0.3 / (2*pi)],
 [-0.3 / (2*pi),              1/12]]
```

At `sigma_stiff=1e-5` its eigenvalues are 0.0127168 and 0.115617. Neither
reveals the local residual variance of `1e-10`. A single linear transformation
can rescale this globally curved distribution but cannot make all its local
tangents agree. This is an analytic property of the toy, independent of
finite-population covariance-estimation error.

Because each dense random direction changes the flat coordinates together with
the constrained coordinates, a short allowed line step also limits movement
in otherwise inert directions. A coordinate/block-direction mixture is a
useful diagnostic for that coupling; it would not by itself straighten the
curved `(x0, x1)` ridge.

## Confirmed cube-face boundary bug

The implementation reviewed at `b6c45d4` classified intersections using both
`t >= 0` and `t <= 0`. An intersection at zero was therefore assigned to both
endpoints. For an anchor exactly on a cube face, this collapsed the bracket:

| Anchor | Direction | Correct bounds | Bounds before the fix |
| --- | --- | --- | --- |
| `(0.25, 0.5)` | `(1, 0.1)` | `[-0.25, 0.75]` | `[-0.25, 0.75]` |
| `(0, 0.5)` | `(1, 0.1)` | `[0, 1]` | `[0, 0]` |
| `(1, 0.5)` | `(-1, 0.1)` | `[0, 1]` | `[0, 0]` |

Calling production `_new_proposal` with constant log likelihood 1 and contour
0 returned the unchanged face anchor after one accepted evaluation. Dense random
directions could not release such an anchor from the face with those bounds.

The three final 180-slice checkpoints contain 2, 3 and 1 exact-face classic
rows respectively, out of 98,832, 89,439 and 96,372 rows. Their combined
posterior weights per run are about `3.3e-61`, zero and zero. There are no
out-of-cube classic rows. These observations do not support the boundary bug
as the main explanation for the toy's posterior collapse; an A/B fix would be
needed to quantify any earlier genealogical effect. Boundary-heavy pRT
posteriors have not been checked.

## Boundary fix

The shared `_slice_bounds` routine now orders each coordinate's two face
intersections and intersects those coordinate intervals. An exact-zero
direction component contributes `[-inf, inf]`; its denominator is replaced
before division so a parallel face cannot produce `0/0`. This preserves
genuinely zero-length chords at corners where neither direction enters the cube.
Both scalar and continued chains use this routine.

The original lower-face regression fails against the old implementation and
passes with the fix. All 54 regression cases in
`cicd/tests/test_slice_bounds.py` pass, covering lower/upper faces, both direction
signs, tangent directions, corners, structured coordinates, both float32 and
float64 bounds, full-chord proposals, and scalar/continued chains with isotropic
or covariance directions. The saved post-fix diagnostic now returns `[0, 1]`
and moves away from the seed for both face examples.

This correction does not change the full-cube bracketing policy, direction
law, allocation, or strict likelihood test. It does not establish recovery of
the curved-ridge posterior; the evidence benchmarks have not been rerun with
the boundary fix.

## Reproduction

```bash
PYTHONPATH=src JAX_ENABLE_X64=1 conda run -n jaxns_py \
  python benchmarks/issue_312/diagnose_geometry.py \
  --output /tmp/geometry_diagnostics.json
```

The bracket and boundary examples run without saved states. To reproduce the
checkpoint audit, also pass `--checkpoint-dir` for each retained local checkpoint
directory listed in the saved diagnostic JSON. Checkpoint files themselves are
not committed. The diagnostic calls the production bounds/proposal functions
for the cube-face cases; the connected/disconnected interval simulations are
an explicit NumPy reference for the same shrink rule.
