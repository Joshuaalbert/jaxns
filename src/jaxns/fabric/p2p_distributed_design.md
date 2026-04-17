# Distributed Likelihood Evaluation for JAXNS

## Summary

The distributed part of JAXNS should be as small as possible. The nested sampler itself should stay on the
coordinator. What gets distributed is only likelihood evaluation.

The core request/response is:

- input: one proposed `U` value
- output: one scalar log-likelihood

This is enough for both single-machine and cluster execution. It also fits heterogeneous hardware well, because fast
workers can naturally process more evaluations than slow workers.

## What Is Being Distributed

The coordinator owns all sampler state:

- live points
- replacement logic
- termination checks
- evidence accumulation
- diagnostics and results

Workers do not participate in sampling. They only evaluate the model likelihood at proposed points from the
coordinator.

That means the distributed interface should look conceptually like:

```python
logL = worker.evaluate(U)
```

where `U` is a proposal from the coordinator and `logL` is the returned scalar.

## Target Architecture

### Coordinator

The coordinator runs the nested sampler locally and is the only process that knows about sampler state. When the
sampler needs more likelihood evaluations, it sends proposed `U` values to available workers and waits for the scalar
results.

### Scheduler / Load Balancer

A central scheduler/load balancer assigns evaluation requests to ready workers. This should be load balanced rather
than round robin.

Load balancing is important because:

- workers may be on different hardware
- some workers may be CPU-only and others GPU-backed
- different nodes may have different contention or data locality
- likelihood evaluations may have variable latency

The scheduler should therefore match work to whichever worker is ready next, not assume equal worker speed.

### Workers

Each worker should load the model and any static data it needs at startup. Per request, the worker should receive only
the proposal `U`, evaluate the likelihood, and return the scalar log-likelihood.

The design should avoid treating workers as generic RPC endpoints for arbitrary sampler operations. For this task they
are evaluator processes.

## Execution Model

Distributed dispatch will require Python control flow on the coordinator side. In particular, `src/jaxns/core.py`
should gain a parallel implementation that breaks out of fully-JAX control flow so it can dispatch likelihood
evaluations to workers and collect the scalar results.

Similarly, `src/jaxns/constrained_sampler.py` will need a Python-side parallel implementation for distributed use. The
constrained sampler still owns the sampling logic locally, but it must be able to hand off likelihood evaluations
whenever it needs to test proposed `U` values.

This should not become a second independent implementation of the algorithm. The Python path should reuse the same
sampling components and compute kernels that the pure-JAX path already uses. In practice, that means the shared
low-level functions should remain factored so they can be called from either control-flow path, and the compute-heavy
pieces that are reused should be wrapped with `partial(jax.jit, inline=True)` where appropriate.

The duplicated part should therefore be only the orchestration:

- Python control flow for dispatching and collecting remote likelihood evaluations
- JAX kernels reused for the actual sampling and proposal computations

The input and output contracts should stay the same across both paths. The Python-dispatch path should accept and
return the same values as the pure-JAX path, so the difference is execution strategy, not algorithm semantics.

This is an acceptable tradeoff. We cannot cleanly dispatch remote work from inside one large JAX program, and we do not
need to. The algorithm does not have to be expressed as one monolithic JAX computation.

The right split is:

- Python control flow around the distributed dispatch points
- JIT compilation for the compute-heavy blocks between sampling / dispatch steps

As long as the blocks between dispatch points stay JIT compiled, the implementation should still be fast while being
much easier to integrate with a load-balanced worker pool.

## Deployment Modes

### Single Machine

Single-machine execution should use the same design as the cluster case:

- one coordinator
- one local scheduler/load balancer
- a pool of local workers across available CPUs and GPUs

This avoids maintaining a separate execution model just for local runs.

### Multi-Node Cluster

Cluster execution should extend the same pattern:

- one coordinator
- one scheduler/load balancer
- many workers across nodes

Workers can be heterogeneous. The scheduler should simply keep feeding ready workers with `U` evaluations and collecting
scalar results.

## Deliberate Simplifications

This design is intentionally narrower than a general distributed RPC system.

We do **not** need distributed sampling. We do **not** need workers to own live points or sampler state. We do **not**
need a broad service interface if the only required operation is likelihood evaluation.

For this task, simpler is better:

- one distributed operation: `evaluate(U) -> logL`
- one coordinator that owns the sampler
- one load balancer that handles heterogeneous workers well
- one worker role that loads the model and evaluates proposals

## Current Fabric vs Needed Design

The current `jaxns.fabric` code already contains a richer peer-to-peer control/data-plane design in
`zmq_p2p.py`, with leases, revokes, heartbeats, and generic RPC-style request routing.

That may be more machinery than is actually needed for distributed JAXNS likelihood evaluation.

The target design described here is a narrower subset:

- keep the useful part: load-balanced work distribution across heterogeneous workers
- drop the unnecessary part: treating the problem as general distributed RPC
- optimize for the real workload: many `U -> logL` evaluations over a large worker pool

If the current fabric is used as the implementation base, the implementation should still bias toward the smallest
subset that supports reliable load-balanced likelihood evaluation.

## Notes

- This document describes the target design, not current CLI entrypoints.
- Any future CLI or API should reflect the architecture above rather than the older `jaxns.distributed.*` wording.
