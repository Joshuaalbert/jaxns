[![Python](https://img.shields.io/pypi/pyversions/jaxns.svg)](https://badge.fury.io/py/jaxns)
[![PyPI](https://badge.fury.io/py/jaxns.svg)](https://badge.fury.io/py/jaxns)
[![Documentation Status](https://readthedocs.org/projects/jaxns/badge/?version=latest)](https://jaxns.readthedocs.io/en/latest/?badge=latest)

Main: ![Main tests](https://github.com/JoshuaAlbert/jaxns/actions/workflows/unittests.yml/badge.svg?branch=main)

Develop: ![Develop tests](https://github.com/JoshuaAlbert/jaxns/actions/workflows/unittests.yml/badge.svg?branch=develop)

![JAXNS](https://github.com/JoshuaAlbert/jaxns/raw/main/jaxns_logo.png)

## Mission: _To make nested sampling **faster, easier, and more powerful**_

# What is JAXNS?

JAXNS is a nested-sampling library and probabilistic programming interface built
with JAX. It is intended for scientific problems that need Bayesian evidence,
weighted posterior samples, or exploration of difficult constrained priors.

JAXNS v3 implements nested sampling as a race tree with dynamic lineage
allocation. A Python goal loop evaluates a user-defined scientific stopping
condition, while each depth epoch—including batched constrained-prior
replacement—is compiled with JAX. Scientific state and results are immutable,
slotted pytree dataclasses with methods for the operations users perform on
them.

JAXNS can:

1. Estimate the Bayesian evidence of a model or hypothesis.
2. Produce weighted or resampled posterior samples.
3. Explore degenerate and multimodal constrained priors.
4. Model continuous and discrete variables using JAX-compatible distributions.
5. Retain stationary phantom states and optionally condition final Monte Carlo
   evidence estimates on them.
6. Run locally on one JAX device or dispatch constrained-sampling chains to a
   trusted pool of heterogeneous devices on the same machine.

The original JAXNS paper is available on
[arXiv](https://arxiv.org/abs/2012.15286), as is the paper on
[phantom-powered nested sampling](https://arxiv.org/abs/2312.11330).

# Install

JAXNS requires Python 3.10 or later. Install the default package from PyPI:

```bash
pip install jaxns
```

The distributed runtime has two additional dependencies and is opt-in:

```bash
pip install "jaxns[distributed]"
```

For development, clone the repository and install the test and example extras:

```bash
git clone https://www.github.com/JoshuaAlbert/jaxns.git
cd jaxns
pip install -e ".[tests,examples]"
```

# Quick start

## Define a model with JAXCTX

A v3 `Model` contains one prior-model function. Calls to
`Prior(...).realise()` create Bayesian variables, and the scalar returned by
the function is the log likelihood. Calls to `Prior(...).parameter()` create
point parameters managed by JAXCTX.

Pass observations and other runtime inputs through `args`, and initialise and
pass model parameters through `params`. Keeping these values explicit avoids
capturing changing data or parameters in Python closures, gives JAX a stable
program identity, and makes the same model straightforward to serialise for
distributed workers.

```python
import jax
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.model import Model

tfpd = tfp.distributions


def prior_model(observations, measurement_uncertainty):
    location = Prior(
        tfpd.Normal(loc=0.0, scale=2.0),
        name="location",
    ).realise()
    intrinsic_uncertainty = Prior(
        tfpd.Exponential(rate=1.0),
        name="intrinsic_uncertainty",
    ).parameter()
    scale = jnp.sqrt(
        jnp.square(measurement_uncertainty)
        + jnp.square(intrinsic_uncertainty)
    )
    return jnp.sum(tfpd.Normal(location, scale).log_prob(observations))


model = Model(prior_model=prior_model)
args = (
    jnp.asarray([-0.15, 0.05, 0.20, 0.30]),
    jnp.asarray(0.10),
)
params = model.init_params(
    key=jax.random.PRNGKey(0),
    args=args,
)
model.sanity_check(
    key=jax.random.PRNGKey(1),
    args=args,
    params=params,
)
```

The model exposes the same explicit `args` and `params` on its lower-level
operations, including `sample_U`, `transform_to_X`, `log_likelihood`,
`log_prior`, and `log_joint`.

## Run nested sampling locally

```python
from jaxns.core import NestedSampler

sampler = NestedSampler(
    model=model,
    args=args,
    params=params,
    collect_phantom_samples=True,
)
state = sampler.run(key=jax.random.PRNGKey(2))
results = state.to_result().trim()

results.summary()
results.plot_diagnostics()
results.plot_cornerplot()
```

`NestedSampler.run()` uses the default expectation-based termination goal. For
a custom scientific goal, provide a Python condition over the immutable
`State`:

```python
def goal_cond(state):
    return state.to_result().log_Z_uncert < 0.1


state = sampler.run_until_goal(
    goal_cond=goal_cond,
    key=jax.random.PRNGKey(3),
)
```

The lightweight expectation estimates on `State` are suitable for frequent
goal and depth decisions. Final user-facing evidence should be drawn with the
Monte Carlo shrinkage model. Phantom conditioning is explicit, so the same run
can be assessed both ways:

```python
classic = results.sample_evidence_mc(
    num_samples=1_000,
    conditioning="classic",
    key=jax.random.PRNGKey(4),
)
phantom = results.sample_evidence_mc(
    num_samples=1_000,
    conditioning="phantom",
    key=jax.random.PRNGKey(5),
)

print(classic.log_Z_mean, classic.log_Z_uncert)
print(phantom.log_Z_mean, phantom.log_Z_uncert)
posterior = results.resample(
    num_samples=1_000,
    key=jax.random.PRNGKey(6),
)
```

Finite sample capacity is the default. Physical buffers begin smaller and grow
between compiled depth epochs up to `max_samples`. Users who intentionally want
an unbounded geometric growth policy can set `unlimited_samples=True`; each new
buffer shape incurs a one-time compilation pause and may consume unbounded
memory.

The isotropic one-dimensional slice direction is the reference default. An
opt-in, warm-refined Gaussian-mixture direction is documented under
[ellipsoidal directions](docs/user-guide/ellipsoidal_directions.rst).

# Distributed nested sampling

The v3 distributed runtime is useful when constrained likelihood evaluations
are expensive enough to outweigh process and IPC overhead and a machine has
multiple useful devices. The local core remains the preferred path for one
device or cheap likelihoods.

The same runtime covers one machine and multiple nodes. Scientific model code
enters the coordinator through same-user `ipc://`. Workers on the coordinator
node also use IPC automatically; remote workers use authenticated and encrypted
CurveZMQ over TCP. Python model payloads are never accepted over unauthenticated
TCP.

Parallel replacement has two independent dimensions:

1. **Lanes within one sampler call.** The standard local core advances
   `shell_size` chains together with `jax.vmap`. Distributed allocation has no
   shell size: it continuously fills compatible live pool lanes with scalar
   logical lineage threads as soon as their parent and stationary seed are
   known. A worker may combine compatible queued threads up to its configured
   `batch_size`; size one takes the scalar path, while larger batches use
   `jax.vmap`. Every lane in one call waits for its most rejection-heavy chain,
   so worker widths should be chosen from evidence.
2. **Workers in the pool.** Each process is pinned to one configured CPU, GPU,
   or TPU device. Workers complete and receive work independently, nodes may
   join while a session is active, and already queued threads are immediately
   available to them. One logical task credit per live lane fills the pool
   without speculative work beyond its measured capacity. There is no
   pool-wide sampling wave barrier.
For one machine, omit `[network]`; `jaxns-cli up` starts a local coordinator and
the configured IPC workers. For multiple machines, first generate CurveZMQ
identities. Keep secret files on their owning nodes and install only each
worker node's public `.key` file in the coordinator's authorized directory:

```bash
jaxns-cli auth create --directory keys/main --name coordinator
jaxns-cli auth create --directory keys/gpu-node --name gpu-node
mkdir -p keys/authorized
cp keys/gpu-node/gpu-node.key keys/authorized/
```

The main-node configuration names both the bind address and the address other
machines can reach. Local workers may be added with `[[workers]]`; a pure
coordinator needs none:

```toml
[runtime]
stack_id = "science"
node_id = "main"
runtime_dir = ".runtime"
log_dir = ".logs"
program_cache_size = 4
heartbeat_interval_s = 2
missed_heartbeats = 2

[network]
listen = "tcp://0.0.0.0:5555"
advertise = "tcp://main.example.org:5555"
server_public_key = "keys/main/coordinator.key"
server_secret_key = "keys/main/coordinator.key_secret"
authorized_clients = "keys/authorized"
```

Each worker machine has its own config pointing at the advertised coordinator.
Several worker entries can pin processes to different devices on that machine:

```toml
[runtime]
stack_id = "science"
node_id = "gpu-node"
runtime_dir = ".runtime"
log_dir = ".logs"

[network]
coordinator = "tcp://main.example.org:5555"
server_public_key = "keys/main/coordinator.key"
client_public_key = "keys/gpu-node/gpu-node.key"
client_secret_key = "keys/gpu-node/gpu-node.key_secret"

[[workers]]
name = "gpu"
platform = "gpu"
device = 0
batch_size = 4
```

Start the coordinator, then run the same lifecycle command on every worker
node. Additional nodes can be started later without restarting the scientific
run:

```bash
# Main machine
jaxns-cli --config coordinator.toml config validate
jaxns-cli --config coordinator.toml up

# Each worker machine
jaxns-cli --config node.toml config validate
jaxns-cli --config node.toml up
jaxns-cli --config node.toml status
```

The main scientific process uses the coordinator config. Distributed sampling
does not accept `shell_size`; `delta_K` controls scientific allocation depth,
while worker TOMLs control execution widths:

```python
from jaxns.distributed_core import DistributedNestedSampler

distributed_sampler = DistributedNestedSampler(
    model=model,
    config="coordinator.toml",
    args=args,
    params=params,
    collect_phantom_samples=True,
)
checkpoint = distributed_sampler.run(
    key=jax.random.PRNGKey(7),
)
results = checkpoint.to_result().trim()
results.summary()
```

`down` on a worker node first marks its workers draining, lets current tasks
finish, then removes the node. New work continues on the remaining pool. Each
worker sends heartbeats independently of JAX sampling; after two missed
heartbeats the coordinator fences its lease, requeues the exact task IDs and
random keys, and rejects late results. The worker also exits after missing two
lease acknowledgements, so a partition cannot create two accepted owners.
Before model registration, the coordinator also verifies Python, JAXNS,
JAX/JAXLIB, x64, and measure-dtype compatibility. CPU, GPU, and TPU platforms
may differ intentionally within one heterogeneous pool.

Lifecycle commands are idempotent for the same config:

```bash
jaxns-cli --config node.toml down
jaxns-cli --config coordinator.toml down
```

`jaxns-cli status` distinguishes connecting, ready, busy, draining, dropped,
disconnected, and exited workers and reports lease generation, heartbeat age,
device, batch capacity, compilation/execution time, and RSS high-water mark.
Worker processes are not silently restarted. After a coordinator restart, old
leases are invalid, workers self-quarantine, and the operator restarts the node
stacks before resuming the immutable `DistributedRunError.checkpoint`. This
keeps recovery explicit; the ownership lock prevents duplicate local owners
for one configured stack.

See the executable [run-pattern design](docs/design/interface/run_pattern.py)
for local IPC, checkpoint resumption, and custom-goal examples.

# Documentation and support

Read the [JAXNS documentation](https://jaxns.readthedocs.io/en/latest/) and the
[repository examples](docs/examples). Likelihood and prior-model calculations
must use JAX-compatible operations.

For scientific questions, use the
[GitHub discussion forum](https://github.com/JoshuaAlbert/jaxns/discussions).
Bug reports and contributions are welcome through GitHub issues and pull
requests.

# Change Log

26 Aug, 2026 -- JAXNS 3.0.0. Major paper-driven core and public API update:

- Added race-tree nested sampling with dynamic lineage allocation, a Python
  user-goal loop, and JIT-compiled depth epochs with `vmap` replacement.
- Made model data and parameters explicit through JAXCTX `args` and `params`,
  and made scientific state and result objects immutable pytree dataclasses.
- Added plateau-correct shrinkage and bounded final Monte Carlo evidence draws,
  with explicit classic or phantom conditioning using retained early-chain
  phantom states.
- Added opt-in warm-refined ellipsoidal slice directions and transparent
  finite or explicitly unlimited sample-buffer growth.
- Added an opt-in asynchronous worker runtime and `jaxns-cli` for local IPC or
  authenticated multi-node TCP pools with dynamic membership, heartbeat
  leases, scalar logical-thread dispatch, and device-local `vmap` batching.
- Removed v2-only APIs, including evidence maximisation. Public v3 classes are
  imported from their defining modules.

3 Aug, 2025 -- JAXNS 2.6.9 released. Fix sdist, and TFP dependency.

7 Dec, 2024 -- JAXNS 2.6.7 released. Fix pip dependencies install.

13 Nov, 2024 -- JAXNS 2.6.6 released. Minor improvements to plotting.

9 Nov, 2024 -- JAXNS 2.6.5 released. Added gradient guided nested sampling. Removed `num_parallel_workers` in favour
`devices`.

4 Nov, 2024 -- JAXNS 2.6.4 released. Resolved bias when using phantom points.

1 Oct, 2024 -- JAXNS 2.6.3 released. Enable pytrees in context.

25 Sep, 2024 -- JAXNS 2.6.2 released. Fixed some important (not so edge) cases. Made faster. Handle no seed scenarios.

24 Sep, 2024 -- JAXNS 2.6.1 released. Sharded parallel JAXNS. Rewrite of internals to support sharded parallelisation.

20 Aug, 2024 -- JAXNS 2.6.0 released. Removed haiku dependency. Implemented our own
context. `jaxns.framework.context.convert_external_params` enables interfacing with any external NN libary.

24 Jul, 2024 -- JAXNS 2.5.3 released. Replacing framework U-space with W-space. Maintained external API in U space.

23 Jul, 2024 -- JAXNS 2.5.2 released. Added explicit density prior. Sped up parametrisation. Scan associative
implemented.

27 May, 2024 -- JAXS 2.5.1 released. Fixed minor accuracy degradation introduced in 2.4.13.

15 May, 2024 -- JAXNS 2.5.0 released. Added ability to handle non-JAX likelihoods, e.g. if you have a simulation
framework with python bindings you can now use it for likelihoods in JAXNS. Small performance improvements.

22 Apr, 2024 -- JAXNS 2.4.13 released. Fixes bug where slice sampling not invariant to monotonic transforms of
likelihood.

20 Mar, 2024 -- JAXNS 2.4.12 released. Minor bug fixes, and readability improvements. Added Empirical special prior.

5 Mar, 2024 -- JAXNS 2.4.11/b released. Add `random_init` to parametrised variables. Enable special priors to be
parametrised.

23 Feb, 2024 -- JAXNS 2.4.10 released. Hotfix for import error.

21 Feb, 2024 -- JAXNS 2.4.9 released. Minor improvements to some priors, and bug fixes.

31 Jan, 2024 -- JAXNS 2.4.8 released. Improved global optimisation performance using gradient slicing.
Improved evidence maximisation.

25 Jan, 2024 -- JAXNS 2.4.6/7 released. Added logging. Use L-BFGS for Evidence Maximisation M-step. Fix bug in finetune.

24 Jan, 2024 -- JAXNS 2.4.5 released. Gradient based finetuning global optimisation using L-BFGS. Added ability to
simulate prior models without bulding model (for data generation.)

15 Jan, 2024 -- JAXNS 2.4.4 released. Fix performance issue for larger `max_samples`. Fixed bug in termination
conditions. Improved parallel performance.

10 Jan, 2024 -- JAXNS 2.4.2/3 released. Another performance boost, and experimental global optimiser.

9 Jan, 2024 -- JAXNS 2.4.1 released. Improve performance slightly for larger `max_samples`, still a performance issue.

8 Jan, 2024 -- JAXNS 2.4.0 released. Python 3.9+ becomes supported. Migrate parametrised models to stable.
All models are now default able to be parametrised, so you can use hk.Parameter anywhere in the model.

21 Dec, 2023 -- JAXNS 2.3.4 released. Correction for ESS and logZ uncert. `parameter_estimation` mode.

20 Dec, 2023 -- JAXNS 2.3.2/3 released. Improved default parameters. `difficult_model` mode. Improve plotting.

18 Dec, 2023 -- JAXNS 2.3.1 released. Paper open science release. Default parameters from paper.

11 Dec, 2023 -- JAXNS 2.3.0 released. Release of Phantom-Powered Nested Sampling algorithm.

5 Oct, 2023 -- JAXNS 2.2.6 released. Minor update to evidence maximisation.

3 Oct, 2023 -- JAXNS 2.2.5 released. Parametrised priors, and evidence maximisation added.

24 Sept, 2023 -- JAXNS 2.2.4 released. Add marginalising from saved U samples.

28 July, 2023 -- JAXNS 2.2.3 released. Bug fix for singular priors.

26 June, 2023 -- JAXNS 2.2.1 released. Multi-ellipsoidal sampler added back in. Adaptive refinement disabled, as a bias
has been detected in it.

15 June, 2023 -- JAXNS 2.2.0 released. Added support to allow TFP bijectors to defined transformed distributions. Other
minor improvements.

15 April, 2023 -- JAXNS 2.1.0 released. pmap used on outer-most loops allowing efficient device-device communication
during parallel runs.

8 March, 2023 -- JAXNS 2.0.1 released. Changed how we're doing annotations to support python 3.8 again.

3 January, 2023 -- JAXNS 2.0 released. Complete overhaul of components. New way to build models.

5 August, 2022 -- JAXNS 1.1.1 released. Pytree shaped priors.

2 June, 2022 -- JAXNS 1.1.0 released. Dynamic sampling takes advantage of adaptive refinement. Parallelisation. Bayesian
opt and global opt modules.

30 May, 2022 -- JAXNS 1.0.1 released. Improvements to speed, parallelisation, and structure of code.

9 April, 2022 -- JAXNS 1.0.0 released. Parallel sampling, dynamic search, and adaptive refinement. Global optimiser
released.

2 Jun, 2021 -- JAXNS 0.0.7 released.

13 May, 2021 -- JAXNS 0.0.6 released.

8 Mar, 2021 -- JAXNS 0.0.5 released.

8 Mar, 2021 -- JAXNS 0.0.4 released.

7 Mar, 2021 -- JAXNS 0.0.3 released.

28 Feb, 2021 -- JAXNS 0.0.2 released.

28 Feb, 2021 -- JAXNS 0.0.1 released.

1 January, 2021 -- Paper submitted

## Star History

<a href="https://star-history.com/#joshuaalbert/jaxns&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/jaxns&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/jaxns&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=joshuaalbert/jaxns&type=Date" />
  </picture>
</a>
