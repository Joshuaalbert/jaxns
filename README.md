[![Python](https://img.shields.io/pypi/pyversions/jaxns.svg)](https://badge.fury.io/py/jaxns)
[![PyPI](https://badge.fury.io/py/jaxns.svg)](https://badge.fury.io/py/jaxns)
[![Documentation Status](https://readthedocs.org/projects/jaxns/badge/?version=latest)](https://jaxns.readthedocs.io/en/latest/?badge=latest)


Main
Status: ![Workflow name](https://github.com/JoshuaAlbert/jaxns/actions/workflows/unittests.yml/badge.svg?branch=main)

Develop
Status: ![Workflow name](https://github.com/JoshuaAlbert/jaxns/actions/workflows/unittests.yml/badge.svg?branch=develop)

![JAXNS](https://github.com/Joshuaalbert/jaxns/raw/main/jaxns_logo.png)

## Mission: _To make nested sampling **faster, easier, and more powerful**_

# What is it?

JAXNS is:

1) a probabilistic programming framework using nested sampling as the engine;
2) coded in JAX in a manner that allows lowering the entire inference algorithm to XLA primitives, which are
   JIT-compiled for high performance;
3) continuously improving on its mission of making nested sampling faster, easier, and more powerful; and
4) citable, and you can read an (old) pre-print here: (https://arxiv.org/abs/2012.15286).

# Documentation

You can read the documentation [here](https://jaxns.readthedocs.io/en/latest/#).

# Install

**Notes:**

1. JAXNS requires >= Python 3.8.
2. It is always highly recommended to use a unique virtual environment for each project.
   To use `miniconda`, have it installed, and run

```bash
# To create a new env, if necessary
conda create -n jaxns_py python=3.11
conda activate jaxns_py
```

## For end users

Install directly from PyPi,

```bash
pip install jaxns
```

## For development

Clone repo `git clone https://www.github.com/JoshuaAlbert/jaxns.git`, and install:

```bash
cd jaxns
pip install -r requirements.txt
pip install -r requirements-tests.txt
pip install -r requirements-examples.txt
pip install .
```

# Getting help and contributing examples

Do you have a neat Bayesian problem, and want to solve it with JAXNS?
I'm really encourage anyone in either the scientific community or industry to get involved and join the discussion
forum.
Please use the [github discussion forum](https://github.com/Joshuaalbert/jaxns/discussions) for getting help, or
contributing examples/neat use cases.

# Quick start

Checkout the examples [here](https://jaxns.readthedocs.io/en/latest/#).

## Caveats

The caveat is that you need to be able to define your likelihood function with JAX. This is usually no big deal because
JAX is just a replacement for NumPy and many likelihoods can be expressed such.
If you're unfamiliar, take a quick tour of JAX (https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).

# Speed test comparison with other nested sampling packages

JAXNS is really fast because it uses JAX.
JAXNS is much faster than PolyChord, MultiNEST, and dynesty, typically achieving two to three orders of magnitude
improvement in speed on cheap likelihood evaluations.
This is shown in (https://arxiv.org/abs/2012.15286). With regards to how efficiently JAXNS used likelihood evaluations,
JAXNS prizes exactness over efficiency, however since it employs an adaptive strategy, users can control efficiency by
controlling some precision parameters.

# Change Log

11 Dec, 2023 -- JAXNS 2.3.0 released. Released of Phantom-Powered Nested Sampling algorithm.

5 Oct, 2023 -- JAXNS 2.2.6 released. Minor update to evidence maximisation.

3 Oct, 2023 -- JAXNS 2.2.5 released. Parametrised priors, and evidence maximisation added.

24 Sept, 2023 -- JAXNS 2.2.4 released. Add marginalising from saved U samples.

28 July, 2023 -- JAXNS 2.2.3 released. Bug fix for singular priors.

26 June, 2023 -- JAXNS 2.2.1 released. Multi-ellipsoidal sampler added back in. Adaptive refinement disabled, as a bias
has been detected in it.

15 June, 2023 -- JAXNS 2.2.0 released. Added support to allow TFP bijectors to defined transformed distributions. Other
minor improvements.

15 April, 2023 -- JAXNS 2.1.0 released. pmap used on outer-most loops allowing efficient device-device communication
during parallel runs.

8 March, 2023 -- JAXNS 2.0.1 released. Changed how we're doing annotations to support python 3.8 again.

3 January, 2023 -- JAXNS 2.0 released. Complete overhaul of components. New way to build models.

5 August, 2022 -- JAXNS 1.1.1 released. Pytree shaped priors.

2 June, 2022 -- JAXNS 1.1.0 released. Dynamic sampling takes advantage of adaptive refinement. Parallelisation. Bayesian
opt and global opt modules.

30 May, 2022 -- JAXNS 1.0.1 released. Improvements to speed, parallelisation, and structure of code.

9 April, 2022 -- JAXNS 1.0.0 released. Parallel sampling, dynamic search, and adaptive refinement. Global optimiser
released.

2 Jun, 2021 -- JAXNS 0.0.7 released.

13 May, 2021 -- JAXNS 0.0.6 released.

8 Mar, 2021 -- JAXNS 0.0.5 released.

8 Mar, 2021 -- JAXNS 0.0.4 released.

7 Mar, 2021 -- JAXNS 0.0.3 released.

28 Feb, 2021 -- JAXNS 0.0.2 released.

28 Feb, 2021 -- JAXNS 0.0.1 released.

1 January, 2021 -- Paper submitted


## Star History

<a href="https://star-history.com/#joshuaalbert/jaxns&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/jaxns&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/jaxns&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=joshuaalbert/jaxns&type=Date" />
  </picture>
</a>
