From a laptop to a cluster
==========================

You can start a JAXNS run on a laptop, pause it, and continue the same experiment
on a changing pool of CPU and GPU workers. The scientific process holds the
full run state and decides what to sample. A coordinator routes its requests
to workers, which perform the likelihood evaluations and constrained sampling.

This tutorial starts the scientific process before any workers exist. It waits
until the first worker joins, then starts sampling. More workers can join later.

Prepare the environment
-----------------------

Install JAXNS on the laptop, coordinator host, and worker hosts. The cluster
hosts also need the distributed extra:

.. code-block:: bash

   pip install 'jaxns[distributed]'

Use matching Python, JAXNS, JAX, and jaxlib versions, and the same precision
settings on all hosts. On GPU hosts, follow the
`JAX installation instructions <https://docs.jax.dev/en/latest/installation.html>`_
for the accelerator software provided by your cluster. Workers may mix CPU and
GPU devices within one run.

Choose a coordinator host that every worker can reach on TCP port 5555. Run the
scientific Python process on that same host, under the same operating-system
user as the coordinator. Worker nodes initiate their connections to this port,
so the coordinator does not need permission to open inbound connections to
worker nodes. Use a trusted cluster network.

Save these downloadable files in one experiment directory:

* :download:`problem.py <../examples/laptop_to_cluster/problem.py>`
* :download:`laptop.py <../examples/laptop_to_cluster/laptop.py>`
* :download:`cluster.py <../examples/laptop_to_cluster/cluster.py>`
* :download:`coordinator.toml <../examples/laptop_to_cluster/coordinator.toml>`
* :download:`workers.toml <../examples/laptop_to_cluster/workers.toml>`

Run the commands below from that directory. Keep model modules and any data
they read available in the same Python environment on the cluster hosts.

Start on the laptop
-------------------

The example uses a small Gaussian likelihood. Replace ``prior_model`` with
your own model when using the workflow for your experiment.

.. literalinclude:: ../examples/laptop_to_cluster/problem.py
   :language: python
   :caption: problem.py

The allocation settings are shared by the local and distributed runners, so
moving the run changes its execution resources while preserving its scientific
configuration. ``unlimited_samples=True`` lets the sample buffers grow as
needed instead of stopping at a fixed sample limit. The machine holding the
state still needs enough memory for the growing sample collection.

.. literalinclude:: ../examples/laptop_to_cluster/laptop.py
   :language: python
   :caption: laptop.py

Start the run:

.. code-block:: bash

   python laptop.py

When you decide to move to the cluster, create a pause file in a second terminal:

.. code-block:: bash

   touch PAUSE

The goal condition checks this file after a complete sampling round. Wait for
``laptop.py`` to return and write ``laptop-state.pkl``. If the accuracy goal was
already reached, that file is already ready to transfer. Save the full
``State`` for continuation. A result object alone contains analysis outputs
but does not retain everything needed to resume sampling.

Automatic checkpoints are also written to ``checkpoints/laptop``. Re-running
``laptop.py`` loads its latest checkpoint automatically. To continue locally,
remove ``PAUSE`` first.

Transfer the paused experiment
------------------------------

Copy the scripts, configuration files, and saved state to the coordinator host.
For example, after creating ``~/jaxns-experiment`` there:

.. code-block:: bash

   scp problem.py laptop.py cluster.py coordinator.toml workers.toml laptop-state.pkl \
       coordinator.example.org:~/jaxns-experiment/

The commands in the remaining steps run on cluster hosts. The pause file is
local to the experiment directory and should be absent when resuming.

Start the coordinator and scientific process
--------------------------------------------

This coordinator configuration starts with no workers:

.. literalinclude:: ../examples/laptop_to_cluster/coordinator.toml
   :language: toml
   :caption: coordinator.toml

On the coordinator host:

.. code-block:: bash

   cd ~/jaxns-experiment
   jaxns-cli --config coordinator.toml config validate
   jaxns-cli --config coordinator.toml up
   jaxns-cli --config coordinator.toml status

The status should initially show an empty worker list. ``up`` starts the
coordinator in the background. Keep this host and its allocation running for
the duration of the experiment.

The cluster script loads the laptop state and uses
``DistributedState.from_state`` to prepare it for distributed continuation.
It asks for a tighter evidence uncertainty than the laptop run:

.. literalinclude:: ../examples/laptop_to_cluster/cluster.py
   :language: python
   :caption: cluster.py

Start this scientific job on the coordinator host, for example in a persistent
terminal session or a scheduler allocation pinned to that host:

.. code-block:: bash

   JAX_PLATFORMS=cpu python cluster.py

The process now waits for workers without a worker-arrival deadline. No model
likelihoods are evaluated in the scientific process. It keeps the experiment
state on the coordinator host's CPU, leaving worker GPUs for sampling.

Add one worker per GPU and one CPU worker
-----------------------------------------

On a compute node with two allocated GPUs, edit ``workers.toml`` to use the
coordinator's reachable hostname and a unique ``node_id``:

.. literalinclude:: ../examples/laptop_to_cluster/workers.toml
   :language: toml
   :caption: workers.toml

There is one worker process for each listed device. Use device indices visible
inside the node's allocation, starting at zero for each platform. Add or remove
GPU entries to match the resources you have allocated. ``batch_size`` is the
maximum number of chains a worker can process together. It controls execution
width, not the scientific allocation target.

On that compute node, in an active interactive allocation:

.. code-block:: bash

   jaxns-cli --config workers.toml config validate
   jaxns-cli --config workers.toml up
   jaxns-cli --config workers.toml status

As soon as a worker registers the model, the waiting scientific process starts
dispatching work. The first tasks include JAX compilation. It does not wait for
every configured worker to become ready. On the coordinator host, inspect the
whole pool with:

.. code-block:: bash

   jaxns-cli --config coordinator.toml status

Repeat the worker setup on additional nodes, with a different ``node_id`` for
each. Keep their allocations alive: ``up`` returns after starting background
processes, so it must not be the last command of a batch job whose exit releases
the node. For a worker batch job, run its supervisor in the foreground instead:

.. code-block:: bash

   python -m jaxns.runtime.node --config workers.toml

Put that command after your scheduler directives, environment activation, and
change into the experiment directory. Use either this foreground command or
``jaxns-cli ... up`` for a given node.

Checkpoint, pause, and resume
-----------------------------

Both scripts enable automatic checkpointing with a 60-second cadence. The
cadence is checked after complete sampling rounds, so a long round can take
longer than 60 seconds before its next checkpoint. A changed final state is
always saved when the goal condition returns true. Checkpoints are published
atomically, and an interrupted process resumes from the latest saved state.
Work after that checkpoint may need to be repeated.

On the cluster, ``touch PAUSE`` requests a clean pause in the same way as on the
laptop. After the script returns, remove the file and run ``cluster.py`` again
to continue. Its ``checkpoints/cluster`` directory takes precedence over the
original laptop state. Keep ``laptop-state.pkl`` and the shared configuration
alongside the script for subsequent invocations. Preserve separate local and
distributed checkpoint directories.

Worker processes may join or leave while the scientific process is running.
If all workers disappear, it retains its state and waits for capacity to return.
If the coordinator itself must be restarted, stop the worker nodes, restart the
coordinator and nodes, then rerun ``cluster.py`` to load the saved continuation.
Automatic sample-buffer growth is also enabled on the cluster, so you do not
need to choose a final sample count in advance.

Move back to local sampling
---------------------------

After the distributed run reaches its goal or a requested pause, convert its
completed state back with ``to_state``:

.. code-block:: python

   completed.to_state().save("return-to-laptop.pkl")

Copy that file and the model code back to the laptop. Then continue with a new
local checkpoint directory and your next goal:

.. code-block:: python

   from jaxns.core import NestedSampler
   from jaxns.state import State
   from problem import run_settings

   state = State.load("return-to-laptop.pkl")
   runner = NestedSampler(model=state.model, **run_settings)
   continued = runner.resume_until_goal(
       state,
       goal_cond=lambda state: state.expected_log_Z_uncert <= 0.01,
       checkpoint_dir="checkpoints/returned-local",
       checkpoint_cadence=60.0,
   )

Both conversions preserve samples, model inputs, and continuation keys.
``to_state`` rejects a distributed state with pending work. Finish resuming that
work through the distributed runner before converting it to local execution.

Stop the pool
-------------

Once the scientific script has returned, stop the workers on each node, then
the coordinator on its host:

.. code-block:: bash

   jaxns-cli --config workers.toml down
   jaxns-cli --config coordinator.toml down

The saved checkpoints remain available for the next run.
