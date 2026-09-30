Phantom collection and evidence conditioning
============================================

Phantom states are eligible intermediate transitions from a constrained slice
chain. They condition the Monte Carlo shrinkage model, but they are not classic
race-tree samples and do not contribute posterior coordinates or posterior
effective sample size.

Collection owns memory
----------------------

Enable collection on :class:`jaxns.core.NestedSampler`:

.. code-block:: python

   nested_sampler = NestedSampler(
       model=model,
       collect_phantom_samples=True,
   )
   state = nested_sampler.run(key=jax.random.PRNGKey(0))
   results = state.to_result().trim()

The runner stores every intermediate transition from each generated chain.
The final transition remains the classic replacement and is never stored as a
phantom. A chain with ``num_slices=s*D`` therefore retains ``s*D - 1`` states.
Root prior draws do not produce phantoms. Keeping these likelihoods increases
state and checkpoint memory in proportion to the retained count.

The runner owns collection even when an explicit ``UniDimSliceSampler`` is
supplied. It requests either all intermediate states or zero, according to
``collect_phantom_samples``. For direct low-level calls, the sampler accepts
``num_phantom_samples`` and validates that it lies between zero and
``num_slices - 1``. A shorter evidence-time prefix remains an independent choice.

Continuation requires the saved phantom width to match this policy. Older runs
that retained only a shorter prefix remain readable for analysis, but should be
continued with the code and collection settings that produced them. Missing
intermediate states cannot be reconstructed retrospectively.

Conditioning owns computation
-----------------------------

The completed state or results can reuse any leading part of the retained
prefix:

.. code-block:: python

   all_saved = results.sample_evidence(
       num_samples=4096,
       phantom_conditioning=True,
       num_phantoms=None,
       key=jax.random.PRNGKey(1),
   )
   first_four = results.sample_evidence(
       num_samples=4096,
       phantom_conditioning=True,
       num_phantoms=4,
       key=jax.random.PRNGKey(1),
   )
   classic = results.sample_evidence(
       num_samples=4096,
       phantom_conditioning=False,
       key=jax.random.PRNGKey(1),
   )

``None`` uses all saved states. An explicit positive count uses
``log_L_phantom[:, :num_phantoms]`` before the MC kernel is compiled, so an
unused suffix does not add device work. Classic conditioning is the default,
independent of phantom storage. Phantom conditioning requires
``phantom_conditioning=True``.

``state.expected_log_Z_mean`` and ``state.expected_log_Z_uncert`` provide the
classic expectation calculation for goal conditions. Results carry those
estimates in ``log_Z_mean`` and ``log_Z_uncert``. Calling
``sample_evidence`` leaves them unchanged and returns a separate ensemble
with its own Monte Carlo mean and uncertainty.
