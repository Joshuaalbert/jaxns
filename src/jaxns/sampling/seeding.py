"""Stationary seed populations for constrained chains."""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

from jaxns.mixed_precision import mp_policy
from jaxns.pytree import PureDataclassPytree
from jaxns.random_utils import sample_uniformly_masked
from jaxns.samples import Samples, SeedPoint
from jaxns.types import BoolArray, FloatArray, IntArray, PRNGKey, UType


@dataclasses.dataclass(slots=True, frozen=True)
class PhantomSeedBank(PureDataclassPytree):
    """One bounded set of phantom representatives and their source clusters."""

    U_samples: UType  # [R, ...] one homogeneous point per retained cluster
    log_L: FloatArray  # [R]
    log_L_birth: FloatArray  # [R]
    cluster_idx: IntArray  # [R] classic row identifying the source chain
    priority: FloatArray  # [R] value-independent reservoir priorities
    valid: BoolArray  # [R]

    @classmethod
    def empty(cls, capacity: int, U_template: UType) -> "PhantomSeedBank":
        """Create an empty bank with the same Pytree point structure."""
        return cls(
            U_samples=jax.tree.map(
                lambda value: jnp.zeros(
                    (capacity,) + value.shape,
                    dtype=value.dtype,
                ),
                U_template,
            ),
            log_L=jnp.full(
                (capacity,),
                -jnp.inf,
                dtype=mp_policy.measure_dtype,
            ),
            log_L_birth=jnp.full(
                (capacity,),
                jnp.inf,
                dtype=mp_policy.measure_dtype,
            ),
            cluster_idx=jnp.full(
                (capacity,),
                -1,
                dtype=mp_policy.index_dtype,
            ),
            priority=jnp.full(
                (capacity,),
                -jnp.inf,
                dtype=mp_policy.measure_dtype,
            ),
            valid=jnp.zeros((capacity,), dtype=mp_policy.bool_dtype),
        )


PhantomSeedBank.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class PhantomSeedPool(PureDataclassPytree):
    """Bounded phantom representatives staged between source publications.

    ``active`` is immutable while one generation of constrained calls is in
    flight. Completed chains enter ``staging`` and become selectable only at
    the next existing seed-source publication boundary. This delay and direct
    source-cluster exclusion prevent a logical chain from immediately
    continuing from its own correlated intermediate state.
    """

    active: PhantomSeedBank
    staging: PhantomSeedBank

    @classmethod
    def empty(cls, capacity: int, U_template: UType) -> "PhantomSeedPool":
        """Create an empty active/staging reservoir pair."""
        return cls(
            active=PhantomSeedBank.empty(capacity, U_template),
            staging=PhantomSeedBank.empty(capacity, U_template),
        )

    def stage(
            self,
            *,
            U_samples: UType,
            log_L: FloatArray,
            log_L_birth: FloatArray,
            cluster_idx: IntArray,
            priority: FloatArray,
            valid: BoolArray,
    ) -> "PhantomSeedPool":
        """Return a pool with completed-cluster candidates staged."""
        return _stage_phantom_seeds(
            self,
            U_samples,
            log_L,
            log_L_birth,
            cluster_idx,
            priority,
            valid,
        )

    def promote(self) -> "PhantomSeedPool":
        """Return a pool with staged candidates published atomically."""
        return _promote_phantom_seeds(self)


PhantomSeedPool.register_pytree()


@partial(jax.jit, inline=True)
def phantom_seed_is_eligible(
        log_L_birth: FloatArray,
        log_L: FloatArray,
        log_L_constraint: FloatArray,
) -> BoolArray:
    """Return whether a stationary representative crosses a strict contour."""
    return (
        (log_L_birth <= log_L_constraint)
        & (log_L > log_L_constraint)
    )


@partial(jax.jit, inline=True)
def _stage_phantom_seeds(
        pool: PhantomSeedPool,
        U_samples: UType,
        log_L: FloatArray,
        log_L_birth: FloatArray,
        cluster_idx: IntArray,
        priority: FloatArray,
        valid: BoolArray,
) -> PhantomSeedPool:
    """Offer one preselected representative from each completed chain.

    Priorities and representative positions are chosen by work planning before
    constrained sampling. The update therefore cannot prefer a candidate for
    having a larger realised likelihood or longer observed contour coverage.
    """
    capacity = pool.staging.valid.shape[0]
    selection_priority = jnp.concatenate(
        (
            jnp.where(
                pool.staging.valid,
                pool.staging.priority,
                -jnp.inf,
            ),
            jnp.where(valid, priority, -jnp.inf),
        )
    )  # [R + S]
    _, selected = jax.lax.top_k(selection_priority, capacity)  # [R]
    combined = PhantomSeedBank(
        U_samples=jax.tree.map(
            lambda retained, candidates: jnp.concatenate(
                (retained, candidates)
            ),
            pool.staging.U_samples,
            U_samples,
        ),
        log_L=jnp.concatenate((pool.staging.log_L, log_L)),
        log_L_birth=jnp.concatenate(
            (pool.staging.log_L_birth, log_L_birth)
        ),
        cluster_idx=jnp.concatenate(
            (pool.staging.cluster_idx, cluster_idx)
        ),
        priority=jnp.concatenate((pool.staging.priority, priority)),
        valid=jnp.concatenate((pool.staging.valid, valid)),
    )
    staging = jax.tree.map(lambda value: value[selected], combined)
    staging = dataclasses.replace(
        staging,
        valid=staging.valid & jnp.isfinite(selection_priority[selected]),
    )
    return dataclasses.replace(pool, staging=staging)


@partial(jax.jit, inline=True)
def _promote_phantom_seeds(pool: PhantomSeedPool) -> PhantomSeedPool:
    """Publish the bounded, value-independent cluster reservoir."""
    capacity = pool.active.valid.shape[0]
    # Selection depends only on priorities fixed before constrained sampling.
    # Consequently the retained set is invariant to staging batches and task
    # completion order for the same completed clusters.
    selection_priority = jnp.concatenate(
        (
            jnp.where(pool.staging.valid, pool.staging.priority, -jnp.inf),
            jnp.where(pool.active.valid, pool.active.priority, -jnp.inf),
        )
    )  # [2R]
    _, selected = jax.lax.top_k(selection_priority, capacity)  # [R]
    combined = PhantomSeedBank(
        U_samples=jax.tree.map(
            lambda active, staging: jnp.concatenate((staging, active)),
            pool.active.U_samples,
            pool.staging.U_samples,
        ),
        log_L=jnp.concatenate((pool.staging.log_L, pool.active.log_L)),
        log_L_birth=jnp.concatenate(
            (pool.staging.log_L_birth, pool.active.log_L_birth)
        ),
        cluster_idx=jnp.concatenate(
            (pool.staging.cluster_idx, pool.active.cluster_idx)
        ),
        priority=jnp.concatenate(
            (pool.staging.priority, pool.active.priority)
        ),
        valid=jnp.concatenate((pool.staging.valid, pool.active.valid)),
    )
    active = jax.tree.map(lambda value: value[selected], combined)
    # ``top_k`` fills unused slots with invalid -inf entries. Carry the
    # explicit mask so those placeholders can never replace a classic row.
    active = dataclasses.replace(
        active,
        valid=active.valid & jnp.isfinite(active.priority),
    )
    U_template = jax.tree.map(lambda value: value[0], active.U_samples)
    return PhantomSeedPool(
        active=active,
        staging=PhantomSeedBank.empty(capacity, U_template),
    )


@partial(jax.jit, inline=True)
def get_seed_point(
        key: PRNGKey,
        samples: Samples,
        log_L_constraint: FloatArray,
) -> SeedPoint:
    """Choose uniformly among existing samples above the strict contour."""
    select_mask = samples.log_likelihoods > log_L_constraint
    return sample_uniformly_masked(
        key=key,
        v=SeedPoint(U0=samples.U_samples, log_L0=samples.log_likelihoods),
        select_mask=select_mask,
        num_samples=1,
        squeeze=True,
    )
