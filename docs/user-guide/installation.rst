Installation
============

Install the stable version with,

.. code-block:: bash

   pip install jaxns

This base installation includes JAXNS model authoring, local nested sampling,
and plotting.

The maintained examples additionally use scikit-learn and Optax:

.. code-block:: bash

   pip install 'jaxns[examples]'

Distributed execution is not part of the current release API. It will receive
an installation extra only after its process and serialization design has been
validated.

or the latest release (after appropriate dependencies) with

.. code-block:: bash
   
   pip install git+http://github.com/Joshuaalbert/jaxns.git
