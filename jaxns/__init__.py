import os
import logging
logging.basicConfig(format='%(levelname)s[%(asctime)s]: %(message)s', level=logging.INFO)

# _JAXNS_IMPORTED = False
# if ('jax' in sys.modules) and not _JAXNS_IMPORTED:
#     raise ImportError("JAX was already imported, but this should be imported first to set device_count.")
#
# _JAXNS_IMPORTED = True

try:
    ncpu = os.cpu_count()
    if 'sched_getaffinity' in dir(os):
        ncpu = len(os.sched_getaffinity(0))
except:
    import multiprocessing

    ncpu = multiprocessing.cpu_count()

os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
# import jax
#
# if jax.device_count('cpu') != ncpu:
#     raise ImportError(f"Importing JAX with 'XLA_FLAGS=--xla_force_host_platform_device_count={ncpu}' failed.")

from jax.config import config
#TODO: explore if we can turn this off, and ensure that logaddsum and similar works.
config.update("jax_enable_x64", True)

from jaxns.nested_sampler.nested_sampling import NestedSampler
from jaxns.plotting import plot_cornerplot,plot_diagnostics
