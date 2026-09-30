# Local Run And Evidence Round Trip

The local system test builds a one-dimensional probabilistic model through the
public model API, runs nested sampling with phantom retention, converts the
returned state to a trimmed result, and samples final evidence in both classic
and phantom-conditioned modes.

The test proves that the main locally supported user workflow composes across
model construction, constrained sampling, state production, result
construction, and both final shrinkage modes. It does not set an accuracy or
performance release threshold; those belong to the maintained standard
problem benchmark.
