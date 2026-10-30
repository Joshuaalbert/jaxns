"""Archive the existing evidence API's inputs without retaining phantom coordinates."""

from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp

from jaxns.algorithm.race_tree import BlockState
from jaxns.results import NestedSamplerResults, _incoming_lineages_per_sample
from jaxns.shrinkage.phantom import EvidenceSamples, sample_evidence


class EvidenceInputs(NamedTuple):
    """One frozen race and its retained chain clusters, including storage padding."""

    log_L_constraints: jax.Array
    log_L_classic: jax.Array
    K_classic: jax.Array
    valid_phantom: jax.Array
    log_L_phantom: jax.Array
    num_samples: jax.Array
    block_state: BlockState

    @classmethod
    def from_result(cls, result: NestedSamplerResults) -> "EvidenceInputs":
        return cls(
            result.log_L_constraints, result.log_L,
            _incoming_lineages_per_sample(result), result.valid_phantom,
            result.log_L_phantom, result.total_num_samples,
            result.block_data.to_block_state(),
        )

    def save(self, path: Path) -> None:
        """Keep the original capacity to reuse MC compilations across seeds."""
        block = self.block_state
        np.savez_compressed(
            path, log_L_constraints=self.log_L_constraints, log_L_classic=self.log_L_classic,
            K_classic=self.K_classic, valid_phantom=self.valid_phantom,
            log_L_phantom=self.log_L_phantom, num_samples=self.num_samples,
            log_L_blocks=block.log_L_blocks, block_first_idx=block.block_first_idx,
            block_size=block.block_size, incoming_K=block.incoming_K,
            block_out_degree=block.block_out_degree, valid=block.valid,
            block_start=block.block_start, block_stop=block.block_stop,
            block_sample_indices=block.block_sample_indices,
        )

    @classmethod
    def load(cls, path: Path) -> "EvidenceInputs":
        with np.load(path) as saved:
            arrays = {name: jnp.asarray(saved[name]) for name in saved.files}
        block = BlockState(
            log_L_blocks=arrays.pop("log_L_blocks"),
            block_first_idx=arrays.pop("block_first_idx"),
            block_size=arrays.pop("block_size"), incoming_K=arrays.pop("incoming_K"),
            block_out_degree=arrays.pop("block_out_degree"), valid=arrays.pop("valid"),
            block_start=arrays.pop("block_start"), block_stop=arrays.pop("block_stop"),
            block_sample_indices=arrays.pop("block_sample_indices"),
        )
        return cls(**arrays, block_state=block)

    def draw(self, key: jax.Array, *, phantoms: bool, draws: int = 4096) -> EvidenceSamples:
        """Use the same shared implementation and defaults as result.sample_evidence."""
        return sample_evidence(
            key=key, log_L_constraints=self.log_L_constraints,
            log_L_classic=self.log_L_classic, K_classic=self.K_classic,
            valid_phantom=(self.valid_phantom if phantoms
                           else jnp.zeros_like(self.valid_phantom)),
            log_L_phantom=(self.log_L_phantom if phantoms else self.log_L_phantom[:, :0]),
            num_samples=self.num_samples, num_Z_samples=draws,
            block_state=self.block_state, batch_size=min(draws, 64),
            C_min=20, diagnostics=False,
        )
