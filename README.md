![JAXNS](https://github.com/Joshuaalbert/jaxns/blob/master/jaxns_logo.png)

[![build status](https://travis-ci.com/Joshuaalbert/jaxns.svg?branch=master)](https://travis-ci.com/github/Joshuaalbert/jaxns)
# What is it?
Enables probabilistic programming using nested sampling. It's coded in JAX in a manner that allows lowering the entire inference algorithm to XLA primitives, which are JIT-compiled for high performance. You can read about it here: (https://arxiv.org/abs/2012.15286)

JAXNS provides a constrained likelihood sampler which combines and modifies ideas from MultiNest (F. Feroz et al. 2008; https://arxiv.org/pdf/0809.3437.pdf) and PolyChord (W.J. Handley et al. 2015; https://arxiv.org/abs/1506.00171).
There are two samplers available provided by setting `sampler_name=slice` for slice sampling, and `sampler_name=multi_ellipsoid` for rejection sampling.

# Install
Make sure you have JAX and the usual suspects with `pip install jax jaxlib numpy matplotlib scipy`. 
Install with `pip install jaxns` or `pip install git+http://github.com/Joshuaalbert/jaxns.git`.

# Getting help and contributing examples

Please use the [github discussion forum](https://github.com/Joshuaalbert/jaxns/discussions) for getting help, or contributing examples/neat use cases. 

# Quick start

JAXNS is really fast because it uses JAX. 
The caveat is that you need to be able to define your likelihood function with JAX. This is usually no big deal because JAX is just a replacement for numpy and many likelihoods can be expressed such. 
If you're unfamiliar, take a quick tour of JAX (https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).
For a simple example see the simple [multivariate normal likelihood with conjugate prior example](https://github.com/Joshuaalbert/jaxns/blob/master/examples/mvn_data_mvn_prior.py)

# Speed test comparison with other nested sampling packages

JAXNS is much faster than PolyChord, MultiNEST, and dynesty, typically achieving two to three orders of magnitude improvement in speed.
I show this in (https://arxiv.org/abs/2012.15286).
