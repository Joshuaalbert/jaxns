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

For distributed execution, install the runtime dependencies on the coordinator
and worker nodes:

.. code-block:: bash

   pip install 'jaxns[distributed]'

See :doc:`laptop_to_cluster` for continuing a local run on CPU and GPU workers.

or the latest release (after appropriate dependencies) with

.. code-block:: bash
   
   pip install git+http://github.com/Joshuaalbert/jaxns.git
