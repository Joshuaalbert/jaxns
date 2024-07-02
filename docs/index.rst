.. jaxns documentation master file, created by sphinx-quickstart on Sat Aug 13 12:39:28 2022.

Welcome to JAXNS's documentation!
=================================

JAXNS is a probabilistic programming framework, built on top of JAX for high performance.
Initially, JAXNS was designed for nested sampling powered Bayesian computations,
however it has since grown into a full-fledged probabilistic programming framework.

Here are some of the things you can do with JAXNS:

1. Build Bayesian models in an easy to use, high-level language.

2. Compute and sample the posterior distribution of your model.

3. Compute the Bayesian evidence of your model.

4. Use deep learning models in your Bayesian models.

5. Use Bayesian models in your deep learning models.

6. Maximise the Bayesian evidence of your model.

7. Global optimisation (maximum likelihood determination).

8. Use JAX's automatic differentiation to compute gradients of your model.


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
