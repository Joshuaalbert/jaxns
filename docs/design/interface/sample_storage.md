# Sample storage and transparent growth

`NestedSampler` keeps sample-indexed arrays at a static size during each JAX
compilation. The Python goal loop may replace that immutable storage with a
larger copy between compiled depth calls.

Finite storage is the default. If `max_samples` is omitted, its resolved value
is `root_allocation_degree * 1000`, enlarged only when necessary to hold the
root samples and one replacement batch. Physical storage starts at the root
batch plus 64 replacement batches and grows geometrically to that finite
ceiling. An explicitly smaller maximum clamps the initial allocation. The
resolved `initial_capacity` and `max_samples` remain visible on the sampler.

```python
sampler = NestedSampler(
    model=model,
    max_samples=1_000_000,
)
print(sampler.initial_capacity, sampler.max_samples)
```

Unlimited storage requires the explicit `unlimited_samples=True` opt-in and
cannot be combined with a finite `max_samples`. Each time storage fills, the
Python loop normally doubles every sample-indexed buffer, recompiles for that
new shape once, and resumes automatically. This can consume unbounded host or
device memory, so it is not enabled by omission.

```python
sampler = NestedSampler(
    model=model,
    unlimited_samples=True,
)
```

The returned `State` is the sole continuation object. Its scalar status has
exactly one meaning after a compiled depth call:

- `termination_reason != 0`: terminal; both other flags are false.
- `needs_growth`: the same logical depth epoch can continue after resize.
- `depth_reached`: the logical depth epoch completed normally.

A growth boundary preserves `random_key`, does not increment
`goal_loop_iter`, and does not change the allocation target. `trim()` and
`resize()` preserve the status and key. Merging gives terminal status first,
then a pending growth request, otherwise a normal completed-depth status.
`NestedSamplerResults` retains only `termination_reason`; growth is transient
orchestration state and is handled before a completed run is converted.

During an interrupted epoch, `random_key` is the immediate sampling
continuation. The internal `goal_key` retains the already-split key for the
next logical epoch. Keeping both in `State` preserves the established random
stream without replaying work when a physical resize divides one depth epoch
across multiple compiled calls.
