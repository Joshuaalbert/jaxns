Migrating from JAXNS v2
=======================

Start by porting the model, then update the sampler and result calls. The
:doc:`advanced_usage` guide explains the new model and run-control interfaces
in more detail.

Replace yielded priors with realised values
-------------------------------------------

A typical v2 model uses a generator to define its prior and a separate
log-likelihood function:

.. code-block:: python

   # v2
   def prior_model_v2():
       x = yield Prior(tfpd.Normal(0.0, 1.0), name="x")
       y = yield Prior(tfpd.HalfNormal(1.0), name="y")
       return x, y

   def log_likelihood_v2(x, y):
       return tfpd.Normal(x, y).log_prob(0.5)

   model_v2 = Model(prior_model_v2, log_likelihood_v2)

In v3, replace each ``yield`` with ``realise()`` and call the existing
log-likelihood before returning. You can keep that likelihood function:

.. code-block:: python

   import tensorflow_probability.substrates.jax as tfp

   from jaxns.model import Model
   from jaxns.priors import Prior

   tfpd = tfp.distributions

   def log_likelihood_v2(x, y):
       return tfpd.Normal(x, y).log_prob(0.5)

   def prior_model_v3():
       x = Prior(tfpd.Normal(0.0, 1.0), name="x").realise()
       y = Prior(tfpd.HalfNormal(1.0), name="y").realise()
       return log_likelihood_v2(x, y)

   model_v3 = Model(prior_model=prior_model_v3)

The function must return a scalar **log-likelihood**, not a probability density
or a tuple of variables. Sum independent observations' log densities. Preserve
normalisation constants when comparing absolute evidences. Prior density is
already accounted for by the prior transform and should not be added to this
return value.

Give each realised prior a stable name, unique within its model scope. Add
``name="x"`` explicitly when a v2 example used an unnamed prior. The ordinary
``Prior`` class is available through ``jaxns.priors``. Special discrete priors
without TFP quantiles still use the specialised JAXCTX classes, as illustrated
by the logic and polynomial-order notebooks.

NaN and zero likelihoods
------------------------

V2 allowed its default NaN-to-``-inf`` conversion to be bypassed with
``allow_nan``. V3 removes that option: ``model.log_likelihood`` always maps NaN
to ``-inf`` (zero likelihood), and ``model.log_joint`` uses that same likelihood.
The policy applies to every local and worker evaluation throughout a run.

We recommend ``model.sanity_check(...)`` before sampling. It reads raw outputs
and reports NaNs before conversion, as well as non-finite prior values,
positive-infinite log-likelihoods, and non-scalar likelihoods. It accepts an
explicit ``-inf``. Passing this sampled check cannot guarantee that an
unvisited part of the prior has no model errors.

Root samples in v3 are unconditional prior draws, with one likelihood
evaluation each. Both explicit zero likelihoods and mapped NaNs retain their
prior mass without redrawing. For a uniform prior on [0, 1], a likelihood that
is zero below 1/2 and one above it therefore has evidence 1/2.

A helper for the common model pattern
-------------------------------------

Download :download:`migrate_v2_model.py <../examples/migrate_v2_model.py>` and run:

.. code-block:: bash

   python migrate_v2_model.py old_model.py \
       --prior prior_model_v2 --likelihood log_likelihood_v2 \
       > migrated_model_snippet.py

The helper reads the original file without executing or modifying it. It emits
public imports, a new prior function, and ``model_v3 = Model(...)``. Keep the
original distribution imports, helper functions, and likelihood definition
alongside this snippet. It preserves comments and formatting within the prior
function, turns yielded prior objects into realised values, and passes the
returned positional arguments to the named likelihood.

Review the generated snippet and run ``model_v3.sanity_check`` before sampling.
The helper does not rewrite an entire experiment, sampler settings, or result
analysis. It rejects delegated generators (``yield from``), nested definitions,
decorators, and return annotations that need manual migration. The supported
v2 convention is to return the likelihood's positional arguments as a tuple,
including a one-element tuple for a single argument.

Pass fixed data when starting a run
-----------------------------------

``Model`` stores the callable. Model inputs belong to the run, so use
``args`` on ``run`` or ``run_until_goal``, not on the ``Model`` constructor:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from jaxns.core import NestedSampler

   def prior_model(observations):
       x = Prior(tfpd.Normal(0.0, 1.0), name="x").realise()
       y = Prior(tfpd.HalfNormal(1.0), name="y").realise()
       return jnp.sum(tfpd.Normal(x, y).log_prob(observations))

   model = Model(prior_model=prior_model)
   args = (jnp.asarray([-0.2, 0.5, 0.8]),)
   model.sanity_check(jax.random.PRNGKey(0), args=args)
   sampler = NestedSampler(model=model)
   state = sampler.run(key=jax.random.PRNGKey(1), args=args)
   results = state.to_result().trim()
   results.summary()
   results.plot_diagnostics()
   results.plot_cornerplot()

If the model needs no inputs, use a zero-argument function and omit ``args``.
There is no need to accept and discard ``*args``. For fitted parameters, use
``Prior.parameter()`` and pass the pytree from ``model.init_params`` to the
run-start method. Ordinary Bayesian models that only use ``realise()`` need no
``params``. See :doc:`advanced_usage` for a complete optimisation example.

Update stopping and continuation
--------------------------------

The runner owns a default expected-depth condition. A Python goal condition
expresses the scientific stopping criterion, such as log-evidence uncertainty:

.. code-block:: python

   state = sampler.run_until_goal(
       goal_cond=lambda state: state.expected_log_Z_uncert <= 0.1,
       args=args,
       key=jax.random.PRNGKey(2),
   )
   state.save("model-state.pkl")
   state = sampler.resume_until_goal(
       state,
       goal_cond=lambda state: state.expected_log_Z_uncert <= 0.05,
   )

Do not translate a v2 remaining-evidence ``dlogZ`` tolerance directly into a
log-evidence uncertainty goal: they measure different things. The default v3
depth uses ``dlogZ=log1p(1e-3)``. ``run()`` reaches the configured depth, while
``run_until_goal`` can add allocation until the supplied goal is satisfied.
Save a full state for continuation; result objects are intended for analysis.

.. list-table:: Common API changes
   :header-rows: 1
   :widths: 38 62

   * - Previous pattern
     - Current pattern
   * - Yielded priors and a separate ``Model`` likelihood argument
     - One callable using ``realise()`` and returning scalar log-likelihood.
   * - Imports from the package root or dependency internals
     - ``Model`` from ``jaxns.model``, ``NestedSampler`` from ``jaxns.core``,
       and ordinary ``Prior`` from ``jaxns.priors``.
   * - A sampler call returning termination information and state
     - ``state = sampler.run(...)`` or ``sampler.run_until_goal(...)``.
   * - Constructing results through the runner
     - ``results = state.to_result().trim()``.
   * - Free-standing summary and plot utilities
     - ``results.summary()``, ``plot_diagnostics()``, and ``plot_cornerplot()``.
   * - Posterior resampling treated as a full run result
     - ``results.resample(n, key=...)`` returns equally weighted
       ``PosteriorSamples`` with ``X_samples`` and posterior integration methods.
   * - Fitted parameters stored on a sampler
     - Supply ``params`` at run start. Resumption uses the parameters saved on
       the state.

For users of earlier v3 development snapshots, ``root_allocation_degree``
replaces ``target_num_live_points``, and ``replacement_width`` replaces the
local runner's ``shell_size``. The former controls initial scientific
resolution; the latter controls execution batching. Distributed workers own
their own batch widths. Evidence draws use ``sample_evidence`` and the Boolean
``phantom_conditioning`` argument, replacing older MC/shrinkage spellings and
a string ``conditioning`` argument.

Recheck the scientific comparison
---------------------------------

Compare old and new runs using the same model, precision, depth tolerance,
root/live-point budget, and chain length. A fixed seed does not promise the same
samples between major versions. Test the model's scalar likelihood and prior
transform before using evidence differences to assess the sampler.

The stored result evidence and posterior weights use the classic expectation
calculation. Retaining phantoms enables a separate Monte Carlo evidence
analysis through ``results.sample_evidence(..., phantom_conditioning=True)``;
it does not silently replace those fields. See :doc:`phantom_conditioning`.
Old result or checkpoint schemas should not be treated as v3 continuation
states. Recreate runs with the ported model, then use the current state-saving
and checkpoint APIs described in :doc:`laptop_to_cluster`.
