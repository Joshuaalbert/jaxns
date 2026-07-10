import dataclasses
from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp, random
from jax.scipy import special as jsp

from jaxns.log_semiring import LogSpace
from jaxns.pytree import PureDataclassPytree
from jaxns.types import FloatArray, IntArray, BoolArray, PRNGKey


@dataclasses.dataclass(slots=True, frozen=True)
class EvidenceSamples(PureDataclassPytree):
    log_Z_samples: FloatArray  # [num_Z_samples] samples of the evidence log Z from the MC shrinkage sampling
    H_samples: FloatArray  # [num_Z_samples] the information E[log_L - log_Z]
    log_dZ_mean: FloatArray  # [num_blocks] L_{g} * (X_{g-1} - X_g) averaged over MC chains
    log_dZ_var: FloatArray  # [num_blocks] variance of L_{g} * (X_{g-1} - X_g) over MC chains
    log_L_blocks: FloatArray  # [num_blocks] block levels derived from log_L_classic, padded with +inf
    block_first_idx: IntArray  # [num_blocks] first classic index per block, -1 for padded blocks


EvidenceSamples.register_pytree()

