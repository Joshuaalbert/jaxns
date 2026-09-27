Phantom collection and evidence conditioning
============================================

Phantom states are eligible intermediate transitions from a constrained slice
chain. They condition the Monte Carlo shrinkage model, but they are not classic
race-tree samples and do not contribute posterior coordinates or posterior
effective sample size.

Collection owns memory
----------------------

Enable collection on :class:`jaxns.core.NestedSampler` and optionally bound
the static phantom axis:

.. code-block:: python

   nested_sampler = NestedSampler(
       model=model,
       collect_phantom_samples=True,
       max_phantom_samples=8,
   )
   state = nested_sampler.run(key=jax.random.PRNGKey(0))
   results = state.to_result().trim()

The sampler stores the first eligible transitions from each generated chain.
The final transition remains the classic replacement and is never stored as a
phantom. With no explicit bound, the default slice sampler retains up to one
model dimension of states, capped by ``num_slices - 1``. A larger bound can be
useful for later sensitivity checks, at the cost of wider result and checkpoint
arrays. The same dimension-sized default is resolved when an otherwise
unbounded ``UniDimSliceSampler`` is supplied to ``NestedSampler``. Set a direct
capacity on that low-level sampler when it should take precedence instead.

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
