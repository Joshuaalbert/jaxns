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
| `matplotlib` | Plotting | Move to `plotting`. Scientific result imports and local runs do not need it. Plotting entry points import it lazily and explain how to install `jaxns[plotting]` when absent. |
| `zmq` | Unimplemented distributed execution | Remove. There is no `jaxns.fabric` package in this release, so the old dependency and console entry points described functionality that cannot run. A `distributed` extra will be introduced only after the tracked design validation and implementation exist; the eventual package would be `pyzmq`, not the stale `zmq` name. |

## Extras

- `plotting`: Matplotlib and only the dependency needed by result and
  ellipsoid plotting entry points.
- `examples`: Matplotlib, scikit-learn, and Optax, matching the maintained
  notebook and example feature set. Base modeling dependencies are inherited
  from JAXNS itself.
- `tests`: Matplotlib, scikit-learn, NetworkX, psutil, pytest, flake8, and
  Ruff, plus TOMLI on Python 3.10, matching imports and tooling used by the
  test and review workflows.

Distributed execution is intentionally not an extra yet. Publishing an empty
or ZMQ-only extra would imply that a distributed JAXNS entry point exists when
it does not.

## Public entry-point behavior

The base environment must import `jaxns.core`, `jaxns.model`, `jaxns.state`,
`jaxns.results`, and `jaxns.utils`, author the README-style TFP/JAXCTX model,
and complete a small local nested-sampling run. Importing results or
multi-ellipsoid utilities must not import Matplotlib. Plotting with the extra
installed must continue to write files under a non-interactive backend;
without the extra, the plotting call itself must fail with the
`jaxns[plotting]` installation command.

Wheel validation inspects `Requires-Dist` rather than only the source table,
because the built metadata is what pip consumes. Clean-environment smokes are
run for every supported Python version so a dependency is not classified as
optional merely because it happens to be present transitively in the
development environment.

## Issue 256 validation record

The built wheel is 113,385 bytes and declares only JAX, JAXCTX, NumPy, SciPy,
and TFP-nightly in base metadata. Its three extras contain exactly the packages
listed above, and it has no console-entry-point metadata for the absent fabric
implementation.

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

After the base smoke, installing `jaxns[plotting]` alone supported an Agg
backend plot and file write. Installing `jaxns[examples]` supported Matplotlib,
Optax, and scikit-learn imports together. The missing-Matplotlib unit test
confirms that base import remains successful and a plotting request names the
exact `jaxns[plotting]` remedy. There is intentionally no distributed smoke:
no distributed public operation or extra exists yet, and advertising one
before its separately tracked design validation would be misleading.

The pull-request unit-test matrix installs `jaxns[tests]`; its pip cache is
keyed by `pyproject.toml`. The maintained demo workflow now deliberately
installs base JAXNS because the demo is itself the clean scientific workflow.
Read the Docs installs only its documentation tool requirements and parses the
source tree without executing notebooks. The repository has no separate
release-time dependency list or installer: wheel construction and publishing
consume `pyproject.toml`, so the inspected wheel metadata is also the release
tooling contract.
