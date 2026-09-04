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
    """One bounded set of phantom representatives and their contour slots."""

    U_samples: UType  # [R, ...] one homogeneous point per retained cluster
    log_L: FloatArray  # [R]
    log_L_birth: FloatArray  # [R]
    log_L_slot: FloatArray  # [R] contour fixed before observing the phantom
    cluster_idx: IntArray  # [R] classic row identifying the source chain
    priority: FloatArray  # [R] value-independent reservoir priorities
    slot_valid: BoolArray  # [R] slot was assigned from planned work
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
            log_L_slot=jnp.full(
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
            slot_valid=jnp.zeros(
                (capacity,),
                dtype=mp_policy.bool_dtype,
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
            slot_idx: IntArray,
            log_L_slot: FloatArray,
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
            slot_idx,
            log_L_slot,
        )

    def assign_staging_slots(
            self,
            log_L_slot: FloatArray,
            valid: BoolArray,
    ) -> "PhantomSeedPool":
        """Reset staging with contours derived from the current planned work."""
        capacity = self.staging.valid.shape[0]
        U_template = jax.tree.map(
            lambda value: value[0],
            self.staging.U_samples,
        )
        staging = PhantomSeedBank.empty(capacity, U_template)
        staging = dataclasses.replace(
            staging,
            log_L_slot=log_L_slot,
            slot_valid=valid,
            valid=jnp.zeros_like(valid),
        )
        return dataclasses.replace(self, staging=staging)

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
        slot_idx: IntArray,
        log_L_slot: FloatArray,
) -> PhantomSeedPool:
    """Offer one preselected representative to each planned contour slot.

    Slot, priority, and representative position are chosen before constrained
    sampling. A candidate is admitted only if it crosses its assigned contour.
    Conditional on that crossing it is a draw from the constrained prior at
    the slot, while its independent priority makes completion order irrelevant.
    """
    capacity = pool.staging.valid.shape[0]
    lanes = jnp.arange(valid.shape[0], dtype=mp_policy.index_dtype)  # [S]

    def offer(lane_idx, staging):
        safe_slot = jnp.clip(slot_idx[lane_idx], 0, capacity - 1)
        assigned_log_L = staging.log_L_slot[safe_slot]
        crosses_slot = phantom_seed_is_eligible(
            log_L_birth[lane_idx],
            log_L[lane_idx],
            assigned_log_L,
        )
        replace = (
            valid[lane_idx]
            & (slot_idx[lane_idx] >= 0)
            & (slot_idx[lane_idx] < capacity)
            & staging.slot_valid[safe_slot]
            & (log_L_slot[lane_idx] == assigned_log_L)
            & crosses_slot
            & (
                jnp.logical_not(staging.valid[safe_slot])
                | (priority[lane_idx] > staging.priority[safe_slot])
            )
        )

        def replace_slot(current):
            return PhantomSeedBank(
                U_samples=jax.tree.map(
                    lambda retained, candidates: retained.at[safe_slot].set(
                        candidates[lane_idx]
                    ),
                    current.U_samples,
                    U_samples,
                ),
                log_L=current.log_L.at[safe_slot].set(log_L[lane_idx]),
                log_L_birth=current.log_L_birth.at[safe_slot].set(
                    log_L_birth[lane_idx]
                ),
                log_L_slot=current.log_L_slot.at[safe_slot].set(
                    assigned_log_L
                ),
                cluster_idx=current.cluster_idx.at[safe_slot].set(
                    cluster_idx[lane_idx]
                ),
                priority=current.priority.at[safe_slot].set(
                    priority[lane_idx]
                ),
                slot_valid=current.slot_valid,
                valid=current.valid.at[safe_slot].set(True),
            )

        return jax.lax.cond(
            replace,
            replace_slot,
            lambda current: current,
            staging,
        )

    staging = jax.lax.fori_loop(
        0,
        lanes.shape[0],
        offer,
        pool.staging,
    )
    return dataclasses.replace(pool, staging=staging)


@partial(jax.jit, inline=True)
def _promote_phantom_seeds(pool: PhantomSeedPool) -> PhantomSeedPool:
    """Publish the completed planned-slot cohort atomically."""
    capacity = pool.active.valid.shape[0]
    # A slot with no crossing candidate keeps its preceding representative
    # under that representative's original effective contour. No value is
    # relabelled, and a temporarily sparse cohort cannot erase useful seeds.
    active = jax.tree.map(
        lambda staging, preceding: jnp.where(
            jnp.reshape(
                pool.staging.valid,
                pool.staging.valid.shape
                + (1,) * (staging.ndim - pool.staging.valid.ndim),
            ),
            staging,
            preceding,
        ),
        pool.staging,
        pool.active,
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
