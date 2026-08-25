Installation
============

Install the stable version with,

.. code-block:: bash

   pip install jaxns

This base installation includes JAXNS model authoring and local nested
sampling. Plotting is optional and can be installed with,

.. code-block:: bash

   pip install 'jaxns[plotting]'

The maintained examples additionally use plotting, scikit-learn, and Optax:

.. code-block:: bash

   pip install 'jaxns[examples]'

Distributed execution is not part of the current release API. It will receive
an installation extra only after its process and serialization design has been
validated.

or the latest release (after appropriate dependencies) with

.. code-block:: bash
   
   pip install git+http://github.com/Joshuaalbert/jaxns.git
