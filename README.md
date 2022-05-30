![JAXNS](https://github.com/Joshuaalbert/jaxns/raw/master/jaxns_logo.png)

[![build status](https://travis-ci.com/Joshuaalbert/jaxns.svg?branch=master)](https://travis-ci.com/github/Joshuaalbert/jaxns)

# What is it?
Provides a probabilistic programming framework based on nested sampling. It's coded in JAX in a manner that allows lowering the entire inference algorithm to XLA primitives, which are JIT-compiled for high performance. You can read about it here: (https://arxiv.org/abs/2012.15286)

JAXNS uses a unique adaptive algorithm that enables arbitrary precision computing of evidence. Stay tuned for the paper that explains it.

# Install
Make sure you have JAX and the usual suspects with `pip install jax jaxlib numpy matplotlib scipy`, optionally also `scikit-learn` and `haiku-dm` for some examples. 
Install with `pip install jaxns` or `pip install git+http://github.com/Joshuaalbert/jaxns.git` for the (no promises) bleeding-edge.

# Getting help and contributing examples

I'm really encourage anyone in the scientific community to get involved and join the discussion forum.
Please use the [github discussion forum](https://github.com/Joshuaalbert/jaxns/discussions) for getting help, or contributing examples/neat use cases.

# Quick start

JAXNS is really fast because it uses JAX. 
The caveat is that you need to be able to define your likelihood function with JAX. This is usually no big deal because JAX is just a replacement for numpy and many likelihoods can be expressed such. 
If you're unfamiliar, take a quick tour of JAX (https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).
Check out the examples for a starter.

# Speed test comparison with other nested sampling packages

JAXNS is much faster than PolyChord, MultiNEST, and dynesty, typically achieving two to three orders of magnitude improvement in speed on cheap likelihood evaluations.
This is shown in (https://arxiv.org/abs/2012.15286). With regards to how efficiently JAXNS used likeihood evaluations, JAXNS prizes exactness over efficiency, however since it employs an adaptive strategy, users can control efficiency by controlling some precision parameters.

# Change Log

30 May, 2022 -- JAXNS 1.0.1 released. Improvements to speed, parallelisation, and structure of code.

9 April, 2022 -- JAXNS 1.0.0 released. Parallel sampling, dynamic search, and adaptive refinement. Global optimiser released.

2 Jun, 2021 -- JAXNS 0.0.7 released.

13 May, 2021 -- JAXNS 0.0.6 released.

8 Mar, 2021 -- JAXNS 0.0.5 released.

8 Mar, 2021 -- JAXNS 0.0.4 released.

7 Mar, 2021 -- JAXNS 0.0.3 released. 

28 Feb, 2021 -- JAXNS 0.0.2 released. 

28 Feb, 2021 -- JAXNS 0.0.1 released. 

1 January, 2021 -- Paper submitted
