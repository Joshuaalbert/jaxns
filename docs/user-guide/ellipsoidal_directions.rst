Ellipsoidal directions
======================

The one-dimensional slice sampler uses isotropic directions by default. An
opt-in warm Gaussian-mixture policy can instead adapt directions to the
weighted classic-sample population:

.. code-block:: python

   from jaxns.constrained_sampler import EllipsoidalDirection
   from jaxns.constrained_sampler import UniDimSliceSampler
   from jaxns.core import NestedSampler

   sampler = UniDimSliceSampler(
       model=model,
       num_slices=5 * model.U_ndims(),
       direction=EllipsoidalDirection(),
   )
   nested_sampler = NestedSampler(model=model, sampler=sampler)

Sampling begins isotropically. Once there are enough weighted observations,
JAXNS fits a persistent full-covariance mixture and bounding ellipsoids. A
component is eligible only while its observed peak lies strictly above the
parent contour, and eligible components are selected in proportion to their
ellipsoidal volumes. Every transition retains an independent one-percent
isotropic safety probability by default. If no component is eligible, the
transition is isotropic.

The model is warm-refined only when a scheduled parent contour has passed all
currently eligible component peaks. One scalar update is considered before a
replacement batch; the fit remains fixed for every complete constrained chain
in that batch, and the chains themselves remain ``vmap``-parallel. Fitted data
is stored on the immutable sampler state, so checkpoint/resume and automatic
capacity growth preserve the future transition kernel.

``EllipsoidalDirection`` exposes the component count, effective-sample gate,
bounded EM iteration count, fixed fit-population size, isotropic probability,
and scale-relative covariance regularisation. The default isotropic sampler
remains the reference choice. The maintained benchmark report under
``benchmarks/issue_246`` shows that fitted directions can reduce likelihood
evaluations on anisotropic problems, while their coordination cost is visible
when likelihoods themselves are very cheap.
