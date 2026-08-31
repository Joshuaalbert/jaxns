# Source layout

JAXNS keeps its user-facing scientific objects and Python runners at the top
level of `src/jaxns`. Internal modules are grouped by the scientific or runtime
responsibility they own. Package `__init__.py` files stay empty: consumers
import the concrete owner rather than relying on a second re-export surface.

## User-facing modules

| Module | Ownership |
| --- | --- |
| `model.py` | JAXCTX-backed model evaluation and U-to-X transformation. |
| `constrained_sampler.py` | Public sampler configurations and sampler API. |
| `core.py` | Local runner configuration and the Python goal loop. |
| `distributed_core.py` | Distributed runner configuration and asynchronous orchestration. |
| `samples.py`, `state.py`, `results.py` | Object-oriented scientific data, state transitions, and result methods. |
| `depth_condition.py`, `checkpoint.py` | Compiled depth policy and durable run continuation. |

These modules may delegate work, but they remain meaningful API owners rather
than compatibility-only facades.

## Internal subsystems

| Package | Ownership |
| --- | --- |
| `algorithm/` | Allocation targets, race-tree blocks, initial state construction, and the complete pure-JAX depth loop. |
| `sampling/` | Request schemas, scalar/vmapped batching, slice transitions, continuation scheduling, stationary seeding, GMM fitting, and ellipsoidal geometry. |
| `shrinkage/` | Classic Bayesian shrinkage, phantom-conditioned Monte Carlo evidence, the independent reference implementation, and online expectation estimates. |
| `runtime/` | Distributed configuration, transport protocol, worker sessions, coordinator/client processes, and node/worker lifecycle. |
| `diagnostics/` | Presentation, plotting, insertion diagnostics, and brute-force scientific reference calculations. |

The local and distributed runners share `algorithm/` and `sampling/`; a runtime
transport must not grow its own scientific implementation. Likewise,
`results.py` keeps methods with the data while delegating presentation to
`diagnostics/` and evidence draws to `shrinkage/`.

## Flat foundations

High-fan-in numerical and schema foundations remain flat so they do not depend
on higher-level subsystems: `types.py`, `mixed_precision.py`, `pytree.py`,
`log_semiring.py`, `cumulative_ops.py`, `random_utils.py`, `stats_utils.py`,
`logging.py`, and `optional.py`.

When adding code, choose the module that owns the behavior rather than the
caller that first needs it. New cross-subsystem imports should follow the
directions above and must not create an alternative implementation of the
race, sampler, or shrinkage model.
