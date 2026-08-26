# Dependency Decisions

This audit records why each install requirement exists and which public
operation consumes it. The source of truth for package metadata remains
`pyproject.toml`.

## Runtime classifications

| Dependency | Classification | Decision and evidence |
|---|---|---|
| `jax` | Core runtime | Keep in base. Every sampler, state, result, pytree, and numerical path uses JAX. JAX also selects a compatible `jaxlib`, NumPy, and SciPy. |
| `jaxlib` | Transitive JAX runtime | Remove the independent declaration. Directly constraining it duplicates JAX metadata and can interfere with accelerator-specific JAX installations. |
| `jaxctx` | Public modeling API | Keep in base. `Model`, state, result, and type APIs carry `CtxParams`; the documented prior workflow uses JAXCTX transforms and priors. |
| `numpy` | Core and host validation | Keep in base. Public validation, result summaries, serialization helpers, and reference calculations use NumPy directly. |
| `scipy` | Public diagnostics | Keep in base. Corner diagnostics use `scipy.stats.gaussian_kde` and `insert_index_diagnostic` uses `scipy.stats.kstwobign`; JAX also requires SciPy. Moving it would not produce a genuinely SciPy-free core install. |
| `tfp-nightly` | Public model authoring | Keep in base. The README, installation demo, standard problems, and JAXCTX `Prior` implementation use TensorFlow Probability JAX distributions. JAXCTX 1.1.5 imports TFP for its public prior module but does not declare that dependency itself, so JAXNS must supply it for the documented workflow. |
| `matplotlib` | Public plotting API | Keep in base. Diagnostic and corner plots are part of the ordinary scientific-results workflow. Plotting entry points still import it lazily so importing JAXNS does not eagerly initialise a plotting backend. |
| `tomli` | Python 3.10 configuration parser | Keep conditionally in base for Python below 3.11. The installed CLI validates TOML without requiring the distributed extra; newer Python uses `tomllib`. |
| `pyzmq` | Distributed process transport | Keep only in `distributed`. Local `NestedSampler` and CLI config validation do not import it; coordinator/node lifecycle, CurveZMQ authentication, and worker routing do. The similarly named `zmq` shim is not used. |
| `cloudpickle` | Distributed model registration | Keep only in `distributed`. It serializes notebook, script, and closure-defined model code once per authorized worker session; per-task array payloads retain standard pickle. |

## Extras

- `examples`: scikit-learn and Optax, matching the maintained notebook and
  example feature set. Base modeling and plotting dependencies are inherited
  from JAXNS itself.
- `tests`: scikit-learn, NetworkX, psutil, pytest, flake8, Ruff, Cloudpickle,
  and PyZMQ,
  matching imports and tooling used by the test and review
  workflows. Matplotlib is inherited from the base installation.
- `distributed`: PyZMQ for local IPC and authenticated multi-node CurveZMQ,
  plus Cloudpickle for one-time model registration. JAX and scientific
  dependencies are inherited from JAXNS itself.

## Public entry-point behavior

The base environment must import `jaxns.core`, `jaxns.model`, `jaxns.state`,
`jaxns.results`, and `jaxns.utils`, author the README-style TFP/JAXCTX model,
complete a small local nested-sampling run, and write diagnostic plots under a
non-interactive backend. Importing results or multi-ellipsoid utilities must
not eagerly import Matplotlib. If a broken environment lacks Matplotlib, the
plotting call itself must identify that the default installation is incomplete.
The base wheel installs `jaxns-cli`; `config validate` works from that base
environment, while `up`, `status`, and `down` explain that
`jaxns[distributed]` is required when PyZMQ is absent.

Wheel validation inspects `Requires-Dist` rather than only the source table,
because the built metadata is what pip consumes. Clean-environment smokes are
run for every supported Python version so a dependency is not classified as
optional merely because it happens to be present transitively in the
development environment.

## Issue 256 validation record

This historical record describes the dependency audit before issue 260
restored Matplotlib to the base installation. The built wheel was 113,385 bytes
and declared only JAX, JAXCTX, NumPy, SciPy, and TFP-nightly in base metadata.
It had no console-entry-point metadata for the absent fabric implementation.

Clean base-wheel environments passed imports of `jaxns`, `core`, `model`,
`results`, `state`, and `utils`, followed by the maintained local
nested-sampling demo, on Python 3.10, 3.11, 3.12, 3.13, and 3.14. Matplotlib
was absent in each base environment. Cold no-cache installs took 31--42 seconds
on the review machine; the range is dominated by wheel download variability.
The Python 3.12 base environment occupied 752 MiB. The parent develop wheel
installed into an otherwise equivalent environment occupied 982 MiB and took
42.3 seconds because it additionally downloaded and installed the Matplotlib
stack and the `zmq` shim plus PyZMQ. The candidate wheel itself is 4,387 bytes
larger due to the lazy-import helper and bounded-MC implementation; wheel size
is not the install-size driver.

| Python | Resolved JAX/JAXLIB | Cold install s | Environment MiB | Base import and demo |
|---|---|---:|---:|---|
| 3.10 | 0.6.2 | 31.00 | 797 | pass |
| 3.11 | 0.10.2 | 31.76 | 812 | pass |
| 3.12 | 0.11.1 | 41.81 | 752 | pass |
| 3.13 | 0.11.1 | 32.26 | 815 | pass |
| 3.14 | 0.11.1 | 37.47 | 848 | pass |

The version differences are pip's supported-wheel resolution for each Python
interpreter, not JAXNS pins. Removing the independent `jaxlib` declaration
still produced the matching JAXLIB version in every row.

After the base smoke, the former `jaxns[plotting]` extra supported an Agg
backend plot and file write. Issue 260 moved that requirement back into the
base metadata, so the same smoke now runs immediately after the base install.
The missing-Matplotlib unit test confirms that base import remains lazy and a
plotting request explains that the installation is incomplete. Issue 252 later
accepted a separate trusted-local distributed extra and `jaxns-cli`; its real
worker smoke and lifecycle checks now live under `cicd/tests/` rather than
being retroactively attributed to the issue 256 evidence above.

The pull-request unit-test matrix installs `jaxns[tests]`; its pip cache is
keyed by `pyproject.toml`. The maintained demo workflow now deliberately
installs base JAXNS because the demo is itself the clean scientific workflow.
Read the Docs installs only its documentation tool requirements and parses the
source tree without executing notebooks. The repository has no separate
release-time dependency list or installer: wheel construction and publishing
consume `pyproject.toml`, so the inspected wheel metadata is also the release
tooling contract.
