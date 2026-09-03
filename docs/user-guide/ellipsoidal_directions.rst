Ellipsoidal directions
======================

The one-dimensional slice sampler starts with exact isotropic directions. GMM
direction fitting is an explicit operation on an immutable ``State``; neither
``NestedSampler`` nor the constrained sampler automatically fits, enables, or
refines it.

.. code-block:: python

   import jax

   from jaxns.core import NestedSampler
   from jaxns.depth_condition import DepthCondition


   # Empty DepthCondition exposes regular complete states to these user goals.
   depth_condition = DepthCondition()

   def fit_goal(state):
       return bool(state.expected_log_Z_uncert < 0.2)

   def final_goal(state):
       return bool(state.expected_log_Z_uncert < 0.1)

   nested_sampler = NestedSampler(model=model)
   state = nested_sampler.run_until_goal(
       fit_goal,
       depth_cond=depth_condition,
       key=jax.random.PRNGKey(42),
   )

   # Fit once from every classic sample currently stored on state. This does
   # not call the likelihood function. Later sampling uses this frozen fit.
   state = state.fit_gmm_directions(iso_prob=1e-2)
   state = nested_sampler.resume_until_goal(
       state,
       final_goal,
       depth_cond=depth_condition,
   )

The two uncertainty targets are user choices, not built-in staging rules. A
caller may stop at any other goal boundary, fit or refine from all classics
stored at that point, and resume. The fit remains frozen until the caller
explicitly invokes ``fit_gmm_directions`` again.

The conservative first-fit default is one component. Pass
``num_components=...`` when the scientific problem supports a multimodal
surrogate. Later calls preserve the retained component count by default.

Expected posterior weights fit the full-covariance component locations and
geometry in homogeneous U-space. The already stored likelihood observations
fit each component's value at its mean. A component is eligible only while
that fitted value is strictly above the parent contour. Eligible components
are selected in proportion to the volume of their fitted Gaussian ellipsoid
above that contour. The empirical maximum of assigned samples, the Gaussian
density normalization, and an untrimmed sample hull do not enter this rule.
``iso_prob`` is the independent isotropic safety probability used by future
transitions. If no component is eligible, the transition is isotropic.

Direction mode also remains under explicit user control:

.. code-block:: python

   state = state.iso_directions()  # Exact isotropic directions.
   state = state.gmm_directions()  # Re-enable the retained successful fit.

``iso_directions`` retains fitted geometry but does not use it. Consequently,
``gmm_directions`` can re-enable that same frozen geometry without another fit;
it fails if the state has no successful fit. These transformations do not
evaluate likelihoods and do not themselves run or resume nested sampling.

Fitted data and the selected direction mode are stored on the immutable state,
so checkpoint/resume and capacity growth preserve the future transition
kernel. The default isotropic mode remains the reference choice. The
maintained benchmark under ``benchmarks/issue_246`` shows how callers can stage
an explicit fit and compare the resulting likelihood work on anisotropic
problems.
