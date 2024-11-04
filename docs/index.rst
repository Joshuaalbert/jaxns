.. jaxns documentation master file, created by sphinx-quickstart on Sat Aug 13 12:39:28 2022.

Welcome to JAXNS's documentation!
=================================

JAXNS is a probabilistic programming framework and advanced nested sampling algorithm.
It's goal is to empower researchers and scientists of all types, from early career to seasoned professionals,
from small jupyter notebooks to massive HPC problem.
Initially, I developed JAXNS to solve my own problems during my PhD. However, it has since grown into a
full-fledged probabilistic programming framework.
JAXNS has been applied in numerous domains from cosmology, astrophysics, gravitational waves, interferometry,
exoplanets, particle physics, meta materials, epidemiology, climate modelling, and beyond. Not to mention it has been
used in industry for a variety of applications. All of this is welcomed and gladly supported.
JAXNS is citable, use the [(outdated) pre-print here](https://arxiv.org/abs/2012.15286).

Here are 10 things you can do with JAXNS:

1. Build probabilistic models in an easy to use, high-level language, that can be used anywhere in the JAX ecosystem.

2. Compute the Bayesian evidence of a model or hypothesis (the ultimate scientific method);

3. Produce high-quality samples from the posterior distribution;

4. Easily handle degenerate difficult multi-modal posteriors;

5. Model both discrete and continuous priors;

6. Encode complex constraints on the prior space;

7. Easily embed your neural networks or ML model in the likelihood/prior;

8. Easily embed JAXNS in your ML model;

9. Use JAXNS in a distributed computing environment;

10. Solve global optimisation problems.


JAXNS's Mission Statement
-------------------------
Our mission is to make nested sampling faster, easier, and more powerful.

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :glob:

   user-guide/*
   Change Log <https://github.com/Joshuaalbert/jaxns#change-log>


.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :glob:

   api/jaxns/index


.. TOC trees for example notebooks

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :glob:

   examples/*

.. toctree::
   :maxdepth: 2
   :caption: Papers
   :glob:

   papers/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
