"""Typed benchmark integration of a frozen controller with the original kernel."""

from dataclasses import dataclass, replace

import jax
from jax import numpy as jnp

from benchmarks.issue_312.online_frozen import FrozenLikelihoodPolicy
from benchmarks.issue_312.online_policy import OnlinePolicy, choose_parameters
from benchmarks.issue_312.reflection_sampler import ReflectionSampler
from jaxns.algorithm.depth import CoreWorkBatch
from jaxns.algorithm.race_tree import build_block_state
from jaxns.constrained_sampler import AbstractSampler
from jaxns.pytree import PureDataclassPytree
from jaxns.shrinkage.classic import classic_dirichlet_concentrations
from jaxns.state import State


@dataclass(frozen=True, slots=True)
class PreparedContours(PureDataclassPytree):
    """Exact pre-dispatch parent information, not a model/phantom backchannel."""

    contours: jax.Array  # [width], sorted; unused entries +inf
    compression: jax.Array  # [width], corresponding classic -log E[X]
    incoming_k: jax.Array  # [width]

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [])


PreparedContours.register_pytree()


@jax.jit
def prepare_contours(state: State, work: CoreWorkBatch) -> PreparedContours:
    """Derive frozen compression from existing classic observations only.

    Rebuilding blocks has a measurable controller cost. No model evaluation,
    posterior transform, phantom access or seed-position statistic is involved.
    """
    blocks = build_block_state(state.samples, root_out_degree=state.root_out_degree,
                               num_samples=state.num_samples)
    concentrations = classic_dirichlet_concentrations(blocks)
    total = concentrations.alpha_gt + concentrations.alpha_eq + concentrations.alpha_lt
    ratio = jnp.where(blocks.valid, concentrations.alpha_gt, 1.) / jnp.where(
        blocks.valid, total, 1.,
    )
    compression = -jnp.cumsum(jnp.log(ratio))  # [capacity], ordered classic blocks
    locations = jnp.where(blocks.valid, blocks.log_L_blocks, jnp.inf)
    block_indices = jnp.searchsorted(locations, work.log_L_constraint, side='right') - 1
    parent_compression = jnp.where(
        block_indices >= 0, compression[jnp.maximum(block_indices, 0)], 0.,
    )
    incoming = jnp.where(block_indices >= 0, blocks.incoming_K[jnp.maximum(block_indices, 0)],
                         state.root_out_degree)
    contours = jnp.where(work.valid, work.log_L_constraint, jnp.inf)
    order = jnp.argsort(contours, stable=True)
    return PreparedContours(contours[order], parent_compression[order], incoming[order])


@dataclass(frozen=True, slots=True)
class OnlineReflectionSampler(AbstractSampler, PureDataclassPytree):
    """Immutable per-batch data; adaptation remains outside compiled sampling."""

    kernel: ReflectionSampler
    policy: OnlinePolicy
    prepared: PreparedContours
    explore: bool = True
    frozen_policy: FrozenLikelihoodPolicy | None = None

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, ['explore'])

    def num_phantom(self) -> int:
        return self.kernel.num_phantom()

    def max_num_phantom(self) -> int:
        return self.kernel.max_num_phantom()

    def _with_phantom_count(self, num_phantom_samples: int):
        return replace(self, kernel=self.kernel._with_phantom_count(num_phantom_samples))

    def parameters(self, key: jax.Array, contour: jax.Array):
        if self.frozen_policy is not None:
            # Deliberately no prepared/evaluation population access in this arm.
            return self.frozen_policy.choose(key, contour, explore=self.explore)
        index = jnp.minimum(jnp.searchsorted(self.prepared.contours, contour, side='left'),
                            self.prepared.contours.shape[0] - 1)
        return choose_parameters(self.policy, key, self.prepared.compression[index],
                                 explore=self.explore)

    def get_sample(self, key, log_L_constraint, seed_point, *, model, args=(), params=None):
        chosen = self.parameters(key, log_L_constraint)
        return self.kernel.get_sample(
            key, log_L_constraint, seed_point, model=model, args=args, params=params,
            contour_step_size=chosen.step_size, contour_num_bounces=chosen.num_bounces,
        )


OnlineReflectionSampler.register_pytree()
