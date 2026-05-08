"""Local compatibility shims for the test environment."""

import jax
from jax.interpreters import xla


if not hasattr(xla, "pytype_aval_mappings"):
    xla.pytype_aval_mappings = jax.core.pytype_aval_mappings
