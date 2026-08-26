"""Supported local and trusted-process run patterns.

This file is executable-shaped design documentation. Distributed execution is
local same-user IPC in its first release; it is not a remote pickle protocol.
"""

import jax
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
from jaxctx import CtxParams
from jaxctx.priors.prior import Prior

from jaxns.core import NestedSampler
from jaxns.distributed_core import DistributedNestedSampler, DistributedState
from jaxns.model import Model
from jaxns.phantom_eval import EvidenceSamples
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition

tfpd = tfp.distributions


# Model code is registered once with trusted worker processes. Module scope is
# easiest to audit and cache; the distributed extra also supports notebook and
# closure definitions through Cloudpickle.
def prior_model(a, b):
    x = Prior(tfpd.Uniform(0.0, a), name="x").realise()
    y = Prior(tfpd.Uniform(0.0, b), name="y").parameter()
    return x.sum() + y


model = Model(prior_model=prior_model)
model_args = (1.0, 2.0)
model_params: CtxParams = model.init_params(
    key=jax.random.PRNGKey(0),
    args=model_args,
)


def goal_cond(state: State) -> bool:
    # The Python goal loop receives only complete immutable scientific states.
    return state.to_result().log_Z_uncert < 0.1


# The established local path keeps the complete depth epoch in one JIT and
# vmaps replacement chains on the selected local JAX device.
local = NestedSampler(
    model=model,
    args=model_args,
    params=model_params,
    collect_phantom_samples=True,
)
local_state = local.run_until_goal(
    goal_cond=goal_cond,
    depth_cond=TerminationCondition(),
    key=jax.random.PRNGKey(1),
)


# The opt-in distributed path uses a stack started separately with:
#
#   jaxns-cli --config docs/design/interface/workers.toml config validate
#   jaxns-cli --config docs/design/interface/workers.toml up
#
# The main process still owns allocation, random keys, immutable checkpoints,
# capacity growth, and goal evaluation. Workers receive complete constrained-
# sampling requests, so data-dependent likelihood loops remain inside JAX. The
# one-time finite root prior batch is bootstrapped locally in this first design.
distributed = DistributedNestedSampler(
    nested_sampler=local,
    config="docs/design/interface/workers.toml",
)
checkpoint: DistributedState = distributed.run_until_goal(
    goal_cond=goal_cond,
    depth_cond=TerminationCondition(),
    key=jax.random.PRNGKey(2),
)

# A checkpoint contains retry-stable pending tasks if execution was interrupted.
# Resumption reconnects the same session, resubmits those exact payloads, and
# deduplicates any result retained by the supervisor.
checkpoint = distributed.resume_until_goal(
    checkpoint,
    goal_cond=goal_cond,
    depth_cond=TerminationCondition(),
)
results = checkpoint.to_result()
results.summary()
results.plot_diagnostics()
results.plot_cornerplot()


def some_fn(parameters: CtxParams):
    return {"x2": parameters["x"] ** 2, "y2": parameters["y"] ** 2}


expected_post_predictive = results.integrate_fn_over_posterior(
    some_fn,
    semi_positive=True,
)
shrinkage_samples: EvidenceSamples = results.sample_mc_shrinkage(
    num_samples=1000,
)
plt.plot(results.log_L_blocks, shrinkage_samples.rho_values)
plt.plot(results.log_L_blocks, shrinkage_samples.rho_fit)

# Stack shutdown is explicit and idempotent:
#
#   jaxns-cli --config docs/design/interface/workers.toml down
