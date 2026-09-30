Stopping conditions and return status
=====================================

A depth condition, a scientific goal, and a storage limit answer different
questions. The same distinction applies to local and distributed runs.

Depth conditions
----------------

``DepthCondition`` controls how deeply one allocation epoch follows the
expected classic shrinkage path before the Python goal is evaluated. The
runner's default is ``DepthCondition(dlogZ=log1p(1e-3))``.

``dlogZ`` compares the estimated remaining contribution :math:`L_g X_g` with
the sum of that contribution and the evidence accumulated through contour g.
Despite its historical name, it is a fraction threshold, not a requested
uncertainty on log evidence. ``cummax_XL_frac`` instead compares :math:`L_g X_g`
with its running maximum, cutting off the declining posterior tail. Values
between zero and one are useful, with smaller values requesting deeper tails.
Both use expected classic volumes and the likelihood structure already seen.

``None`` disables a cutoff. If both are set, either cutoff can end the
traversal. ``DepthCondition()`` completes the allocation target without a tail
cutoff. Filling storage partway through an epoch is a physical boundary, not
an opportunity to evaluate the scientific goal early.

Goals and hard limits
---------------------

``run_until_goal`` and ``resume_until_goal`` evaluate a Python function on a
complete ``State``. For example:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   def precision_goal(state):
       return bool(state.expected_log_Z_uncert <= 0.05)

   state = runner.run_until_goal(precision_goal, key=jax.random.PRNGKey(0))
   if int(state.termination_reason) == 1:
       print("The maximum sample count was reached before the next goal check.")
   elif bool(state.depth_reached) and precision_goal(state):
       print("The requested goal is satisfied.")

``termination_reason == 0`` means no hard stop was reported. It does not, by
itself, mean a particular precision goal was satisfied. Code 1 means the finite
``max_samples`` limit was reached. ``results.summary()`` also displays this
status. A capacity-limited result can still be inspected, but its reported
uncertainty is not evidence that the requested goal was reached.

``run()`` uses the default depth-based goal. It does not request a particular
evidence uncertainty. ``run_single_iteration()`` performs one depth epoch and
can return earlier if physical sample storage fills. In that case
``needs_growth=True`` and ``depth_reached=False`` identify an unfinished epoch.
The full goal loop handles permitted growth automatically. With
``unlimited_samples=True``, there is no finite scientific sample cap, though
physical memory remains finite.

Resuming and changing a budget
------------------------------

A state stopped by a satisfied goal can be passed directly to
``resume_until_goal`` with a tighter goal. A state with ``needs_growth=True``
can also be handed to that method to grow storage and finish the same epoch.
Keep the sampler and allocation settings unchanged while an epoch is unfinished.

A hard sample limit is deliberately sticky. For a local state that stopped
with code 1, explicitly increase the limit and clear that reason before
resuming:

.. code-block:: python

   from dataclasses import replace

   assert int(state.termination_reason) == 1
   larger_runner = replace(runner, max_samples=2 * runner.max_samples)
   continuation = replace(
       state, termination_reason=jnp.zeros_like(state.termination_reason),
   )
   continued = larger_runner.resume_until_goal(
       continuation, precision_goal,
       checkpoint_dir="checkpoints/larger-budget",
   )

This example assumes ``runner.max_samples`` was set explicitly. Use a new
checkpoint directory for this deliberate change: an existing checkpoint takes
precedence over the state supplied to a run method. Increasing only the limit
does not clear a saved hard-stop reason, and resizing alone does not change the
scientific budget.

All local entry points save the latest coherent continuation on Ctrl-C when
checkpointing is enabled, then raise ``KeyboardInterrupt``. An unfinished depth
retains its schedule and random stream and resumes without an extra goal
evaluation. Without checkpointing, the interruption propagates without a save.
See :doc:`advanced_usage` for staged scientific goals and :doc:`laptop_to_cluster`
for distributed checkpoints and worker cleanup.
