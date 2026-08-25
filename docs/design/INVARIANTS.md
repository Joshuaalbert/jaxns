# Design Invariants

This file is the implementation-agnostic scientific and user-facing contract
for JAXNS. Every entry is intended to be testable without depending on a
particular module boundary, loop primitive, buffer layout, or compilation
strategy. Current implementation and release constraints live in
`REQUIREMENTS.md`.

The exact text following each `- Invariant:` prefix is the stable key used by
`cicd/coverage_record.json`.

## Model And Prior Measure

- Invariant: A model defines a likelihood over a normalized prior measure, and the evidence is
  the prior expectation of that likelihood.
- Invariant: Sampling the model's unit-hypercube representation and transforming it to parameter
  space produces draws from the declared prior, including declared parameter dependencies.
- Invariant: Evaluating a model through its unit-hypercube representation gives the same
  likelihood as evaluating the corresponding transformed parameter values.
- Invariant: Model arguments and parameters used to create a run remain the arguments and
  parameters associated with every state and result derived from that run.
- Invariant: A model's reported unit-hypercube dimensionality equals the number of scalar
  unit-hypercube coordinates consumed by its prior transformation.
- Invariant: Invalid model outputs fail visibly rather than entering nested-sampling state as
  apparently valid scientific samples.

## Strict Constrained-Prior Sampling

- Invariant: Every nested-sampling contour is strict: a point satisfies contour lambda exactly
  when its likelihood is greater than lambda.
- Invariant: A classic child associated with a parent contour has the constrained-prior marginal
  distribution above that parent contour.
- Invariant: A constrained-sampling transition leaves the parent constrained-prior distribution
  invariant.
- Invariant: Every emitted classic child satisfies its recorded parent contour strictly.
- Invariant: A seed used for a constrained chain is stationary for the requested parent contour
  before the chain starts.
- Invariant: Changing how valid constrained-sampling work is batched or scheduled does not
  change the marginal law of any classic child.
- Invariant: Likelihood-evaluation accounting for a generated sample counts every likelihood
  evaluation used to generate that sample and no evaluations belonging to another sample.

## Race Tree

- Invariant: Every classic sample has exactly one parent contour, while the sentinel root has no
  parent and has strictly positive out-degree.
- Invariant: A sample's out-degree is exactly the number of classic children generated from that
  sample's contour.
- Invariant: The sentinel out-degree plus all classic-sample out-degrees equals the number of
  classic samples in a complete stored race tree.
- Invariant: The first race has active lineage count equal to the sentinel out-degree.
- Invariant: After one sample arrives with out-degree d, the active lineage count changes from K
  to K minus one plus d.
- Invariant: Every valid race arrival has a strictly positive incoming active-lineage count.
- Invariant: The likelihood contours and out-degrees determine the race shrinkage law
  independently of the physical storage order of samples.
- Invariant: Permuting samples within an equal-likelihood block does not change block lineage
  counts, shrinkage, evidence, or posterior mass.

## Likelihood Blocks And Plateaus

- Invariant: All classic samples at the same likelihood form one block, and distinct valid
  blocks are ordered by strictly increasing likelihood.
- Invariant: A block's multiplicity is the number of classic samples at its likelihood and its
  out-degree is the sum of their out-degrees.
- Invariant: A block cannot contain more race arrivals than its incoming active lineage count.
- Invariant: The probability beyond a block is the strict prior-volume ratio across that block,
  the equality probability is the atom at the block likelihood, and the remaining probability is
  the open race interval.
- Invariant: The beyond, equality, and open-interval probabilities of every valid block are
  non-negative and sum to one.
- Invariant: A singleton block is not evidence for a likelihood atom and uses the exact
  two-class shrinkage law Beta(K, 1).
- Invariant: A plateau block with multiplicity greater than one uses a three-class model that
  separates strict volume beyond the plateau, equality-atom mass, and the open race interval.
- Invariant: Equality-atom posterior mass for a plateau is divided equally among the classic
  samples in that plateau.
- Invariant: Strict prior volume after a plateau excludes the plateau's equality-atom mass.

## Classic Shrinkage, Evidence, And Posterior Mass

- Invariant: The initial prior volume is one and the valid strict prior-volume sequence is
  non-increasing and remains between zero and one.
- Invariant: Every valid evidence contribution and unnormalized posterior weight is
  non-negative.
- Invariant: Evidence and posterior calculations use the strict volume beyond each block rather
  than a race rank censored inside a plateau.
- Invariant: Normalized posterior weights over valid samples sum to one whenever the evidence is
  finite and positive.
- Invariant: Classic Monte Carlo shrinkage draws follow the same block probability model used by
  the corresponding classic expectation calculation.
- Invariant: Increasing the number of independent shrinkage draws makes Monte Carlo evidence
  summaries converge to the moments implied by the classic shrinkage model.
- Invariant: Evidence and posterior results are invariant to a compatible reordering of stored
  classic samples.

## Phantom Samples And Conditioning

- Invariant: Phantom samples are auxiliary constrained-prior observations and never participate
  as arrivals or lineages in the classic race tree.
- Invariant: Every phantom cluster is associated with one classic child and contains only
  retained chain states generated under that child's recorded parent contour.
- Invariant: Retained phantom states have the constrained-prior marginal distribution of their
  cluster's parent contour.
- Invariant: Dependence within a phantom cluster is preserved by sharing one random cluster
  weight across all observations and all blocks contributed by that cluster in one shrinkage
  draw.
- Invariant: Phantom clusters treated as independent contributions originate from independently
  initialized stationary chains.
- Invariant: A phantom contributes to a block only when its cluster parent contour is no
  stricter than the preceding block contour and the phantom lies beyond that preceding contour.
- Invariant: For a plateau block, an eligible phantom is classified exactly once as beyond the
  block, equal to the block, or in the open race interval.
- Invariant: For a singleton block, phantom conditioning remains a two-class model: an eligible
  phantom is either beyond the block or in its complement, and no equality class is introduced.
- Invariant: Phantom conditioning is applied to a block only when the participating clusters
  provide the configured minimum effective independent information.
- Invariant: When phantom conditioning is inactive for a block, that block has exactly the
  classic shrinkage law.
- Invariant: When all eligible phantom clusters are independent singletons, phantom conditioning
  reduces to ordinary conjugate categorical-count conditioning.
- Invariant: Collecting or retaining phantoms does not change the classic samples, race tree, or
  classic-conditioned result for the same random run.
- Invariant: Phantom-conditioned evidence is an optional final inference over a completed state
  and does not alter the state used to generate the race.

## Lineage Allocation And Work Scheduling

- Invariant: A generated edge from parent likelihood L_i to child likelihood L_j adds one active
  lineage to every existing block whose likelihood satisfies L_i less than L_g less than or
  equal to L_j.
- Invariant: The allocation gap at a block is the positive part of its target lineage count
  minus its current lineage count.
- Invariant: Advancing a logical thread fills allocation gaps only along contours crossed by the
  generated lineage.
- Invariant: A parent selected for new work is chosen from a contour with an outstanding
  allocation gap or from a scientifically equivalent seedable reparenting contour.
- Invariant: A seed is eligible for parent contour lambda only when its likelihood is greater
  than lambda and the contour from which it was generated is no greater than lambda.
- Invariant: Concurrent children requested from the same parent contour use distinct eligible
  seeds whenever enough distinct eligible seeds exist.
- Invariant: Reparenting caused by a lack of stationary seeds preserves the constrained-prior
  law required by the newly recorded parent contour.
- Invariant: Allocation strategy may change where precision is added but cannot change the
  shrinkage law implied by the completed race tree.

## Run Control, State, And Resumption

- Invariant: A state contains all scientific data and random-stream position needed to continue
  its run without hidden mutable sampler state.
- Invariant: Resuming a state with its stored random-stream position is scientifically
  equivalent to uninterrupted continuation under the same conditions.
- Invariant: The user-supplied goal condition is evaluated against complete immutable run states
  and determines successful user-goal termination.
- Invariant: A depth condition bounds one allocation epoch without being reported as successful
  satisfaction of the user's outer goal.
- Invariant: Depth-loop evidence and posterior-tail conditions use online expectation estimates
  from classic samples and do not use phantom-conditioned Monte Carlo draws.
- Invariant: A finite scientific sample limit is never exceeded, including by a partially filled
  replacement batch.
- Invariant: A run that cannot satisfy its goal returns its resumable state rather than silently
  claiming that the goal was met.

## User-Facing Results

- Invariant: Converting a state to results preserves every valid classic sample and excludes
  every unused capacity entry from user-facing sample statistics.
- Invariant: User-facing evidence sampling requires an explicit choice between classic and
  phantom conditioning.
- Invariant: Classic evidence sampling remains available whether or not phantom samples were
  collected.
- Invariant: Phantom evidence sampling fails clearly when the state lacks the metadata required
  for valid conditioning.
- Invariant: Posterior resampling draws from the normalized posterior weights represented by the
  result.
- Invariant: Integrating a function over the posterior uses the result's normalized posterior
  measure and preserves the function's pytree structure.
- Invariant: Summary and plotting operations do not mutate the scientific result from which they
  are produced.

## Randomness, Numerics, And Validation

- Invariant: Independent logical random choices use non-overlapping random streams derived from
  the run key.
- Invariant: Repeating a run with the same model, configuration, inputs, and random key
  reproduces the same scientific state on the same supported execution platform.
- Invariant: Changing only the random key may change a realization but does not change the
  target nested-sampling law.
- Invariant: Invalid shapes, invalid lineage capacity, inconsistent block membership, or
  incompatible phantom metadata fail visibly before producing scientific summaries.
- Invariant: Numerically stable evidence arithmetic represents values across the supported
  likelihood dynamic range without converting avoidable underflow or overflow into finite
  scientific conclusions.
- Invariant: Public effective-sample-size, uncertainty, and diagnostic quantities are finite
  whenever their mathematical inputs are finite and non-degenerate, and otherwise report the
  degenerate case explicitly.

## Scientific Calibration

- Invariant: On a problem with known evidence, repeated normalized evidence errors are centered
  at zero and have unit-scale dispersion when the reported uncertainty is calibrated.
- Invariant: On a problem with a known posterior, posterior estimates converge to the known
  posterior as independent nested-sampling information increases.
- Invariant: On a likelihood with known equality atoms, inferred strict volume and atom mass
  converge to their known values without treating singleton observations as atom evidence.
- Invariant: Enabling valid phantom conditioning may reduce uncertainty but does not introduce
  systematic evidence or posterior bias.
