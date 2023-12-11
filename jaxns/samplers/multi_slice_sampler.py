import logging
from typing import TypeVar, NamedTuple, Tuple, Optional

from jax import numpy as jnp, random, tree_map, lax

from jaxns.framework.bases import BaseAbstractModel
from jaxns.samplers.abc import SamplerState
from jaxns.samplers.bases import SeedPoint, BaseAbstractMarkovSampler
from jaxns.internals.shrinkage_statistics import _cumulative_op_static
from jaxns.internals.types import PRNGKey, FloatArray, BoolArray, Sample, int_type, StaticStandardNestedSamplerState, UType, \
    IntArray, float_type

__all__ = [
    'MultiDimSliceSampler'
]

logger = logging.getLogger('jaxns')

T = TypeVar('T')


def _slice_bounds(key: PRNGKey, point_U0: FloatArray, num_restrict_dims: int) -> Tuple[FloatArray, FloatArray]:
    """
    Get the slice bounds, randomly selecting which dimensions to slice in.

    Args:
        key: PRNGKey
        point_U0: the seed point
        num_restrict_dims: the number of dimensions to slice in

    Returns:
        left, and right bounds of slice
    """
    if num_restrict_dims is not None:
        slice_dims = random.choice(key, point_U0.size, shape=(num_restrict_dims,), replace=False)
        left = point_U0.at[slice_dims].set(jnp.zeros(num_restrict_dims, point_U0.dtype))
        right = point_U0.at[slice_dims].set(jnp.ones(num_restrict_dims, point_U0.dtype))
    else:
        left = jnp.zeros_like(point_U0)
        right = jnp.ones_like(point_U0)
    return left, right


def _new_sample(key: PRNGKey, left: FloatArray, right: FloatArray) -> UType:
    return random.uniform(key=key, shape=left.shape, dtype=left.dtype, minval=left, maxval=right)


def _shrink_region(point_U: UType, point_U0: UType, left: FloatArray, right: FloatArray) -> Tuple[
    FloatArray, FloatArray]:
    """
    Shrink the region to the left and right of the point_U0.

    Args:
        point_U: the point to shrink to
        point_U0: the origin of the slice
        left: the left bound of the slice
        right: the right bound of the slice

    Returns:
        new left, and right bounds of slice
    """
    # if point_U is on the 'right' side then we shrink the 'right' side to it.
    # same of 'left'
    left = jnp.where(point_U < point_U0,
                     jnp.maximum(left, point_U),
                     left)
    right = jnp.where(point_U > point_U0,
                      jnp.minimum(right, point_U),
                      right)

    return left, right


def _new_proposal(key: PRNGKey, seed_point: SeedPoint, num_restrict_dims: int, log_L_constraint: FloatArray,
                  model: BaseAbstractModel) -> Tuple[FloatArray, FloatArray, IntArray]:
    """
    Sample from a slice about a seed point.

    Args:
        key: PRNG key
        seed_point: the seed point to sample from
        num_restrict_dims: how many dimensions to restrict the slice to
        log_L_constraint: the constraint to sample within
        model: the model to sample from

    Returns:
        point_U: the new sample
        log_L: the log-likelihood of the new sample
        num_likelihood_evaluations: the number of likelihood evaluations performed
    """

    class Carry(NamedTuple):
        key: PRNGKey
        left: FloatArray
        right: FloatArray
        point_U: UType
        log_L: FloatArray
        num_likelihood_evaluations: IntArray

    def cond(carry: Carry) -> BoolArray:
        not_close_to_zero = jnp.any((carry.right - carry.left) > 2 * jnp.finfo(carry.right.dtype).eps)
        return jnp.bitwise_and(carry.log_L <= log_L_constraint, not_close_to_zero)

    def body(carry: Carry) -> Carry:
        key, t_key, shrink_key = random.split(carry.key, 3)
        left, right = _shrink_region(
            point_U=carry.point_U,
            point_U0=seed_point.U0,
            left=carry.left,
            right=carry.right
        )
        point_U = _new_sample(
            key=t_key,
            left=left,
            right=right
        )
        log_L = model.forward(point_U)
        num_likelihood_evaluations = carry.num_likelihood_evaluations + jnp.ones_like(carry.num_likelihood_evaluations)
        return Carry(
            key=key,
            left=left,
            right=right,
            point_U=point_U,
            log_L=log_L,
            num_likelihood_evaluations=num_likelihood_evaluations,
        )

    key, slice_key, t_key = random.split(key, 3)
    num_likelihood_evaluations = jnp.full((), 0, int_type)
    (left, right) = _slice_bounds(
        key=slice_key,
        point_U0=seed_point.U0,
        num_restrict_dims=num_restrict_dims
    )
    point_U = _new_sample(
        key=t_key,
        left=left,
        right=right
    )
    log_L = model.forward(point_U)
    init_carry = Carry(
        key=key,
        left=left,
        right=right,
        point_U=point_U,
        log_L=log_L,
        num_likelihood_evaluations=num_likelihood_evaluations
    )

    carry = lax.while_loop(
        cond_fun=cond,
        body_fun=body,
        init_val=init_carry
    )
    return carry.point_U, carry.log_L, carry.num_likelihood_evaluations


class MultiDimSliceSampler(BaseAbstractMarkovSampler):

    def __init__(self, model: BaseAbstractModel, num_slices: int, num_phantom_save: int,
                 num_restrict_dims: Optional[int] = None):
        """
        Multi-dimensional slice sampler, with exponential shrinkage. Produces correlated (non-i.i.d.) samples.

        Notes: Not very efficient.

        Args:
            model: AbstractModel
            num_slices: number of slices between acceptance, in units of 1, unlike other software which does it in
                units of prior dimension.
            num_phantom_save: number of phantom samples to save. Phantom samples are samples that meeting the constraint
                but are not accepted. They can be used for numerous things, e.g. to estimate the evidence uncertainty.
            num_restrict_dims: size of subspace to slice along. Setting to 1 would be like UniDimSliceSampler,
                but far less efficient.
        """
        super().__init__(model=model)
        if num_slices < 1:
            raise ValueError(f"num_slices must be > 0.")
        if num_phantom_save < 0:
            raise ValueError(f"num_phantom_save should be >= 0, got {num_phantom_save}.")
        if num_phantom_save >= num_slices:
            raise ValueError(f"num_phantom_save should be < num_slices, got {num_phantom_save} >= {num_slices}.")
        self.num_slices = int(num_slices)
        self.num_phantom_save = int(num_phantom_save)
        if num_restrict_dims is not None:
            if num_restrict_dims == 1:
                raise ValueError(f"If restricting to 1 dimension, then you should use UniDimSliceSampler.")
            if not (1 < num_restrict_dims <= model.U_ndims):
                raise ValueError(f"Expected num_restriction dim in (1, {model.U_ndims}], got {num_restrict_dims}.")
        self.num_restrict_dims = int(num_restrict_dims)

    def num_phantom(self) -> int:
        return self.num_phantom_save

    def pre_process(self, state: StaticStandardNestedSamplerState) -> SamplerState:
        return (state,)

    def post_process(self, state: StaticStandardNestedSamplerState, sampler_state: SamplerState) -> SamplerState:
        return (state,)

    def get_seed_point(self, key: PRNGKey, sampler_state: SamplerState,
                       log_L_constraint: FloatArray) -> SeedPoint:

        state: StaticStandardNestedSamplerState
        (state,) = sampler_state

        unnorm_select_prob = (state.sample_collection.log_L[state.front_idx] > log_L_constraint).astype(float_type)
        # Choose randomly where mask is True
        g = random.gumbel(key, shape=unnorm_select_prob.shape)
        sample_idx = state.front_idx[jnp.argmax(g + jnp.log(unnorm_select_prob))]

        return SeedPoint(
            U0=state.sample_collection.U_samples[sample_idx],
            log_L0=state.sample_collection.log_L[sample_idx]
        )

    def get_sample_from_seed(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray,
                             sampler_state: SamplerState) -> Tuple[Sample, Sample]:

        def propose_op(sample: Sample, key: PRNGKey) -> Sample:
            U_sample, log_L, num_likelihood_evaluations = _new_proposal(
                key=key,
                seed_point=SeedPoint(
                    U0=sample.U_sample,
                    log_L0=sample.log_L
                ),
                num_restrict_dims=self.num_restrict_dims,
                log_L_constraint=log_L_constraint,
                model=self.model
            )
            return Sample(
                U_sample=U_sample,
                log_L_constraint=log_L_constraint,
                log_L=log_L,
                num_likelihood_evaluations=num_likelihood_evaluations + sample.num_likelihood_evaluations
            )

        init_sample = Sample(
            U_sample=seed_point.U0,
            log_L_constraint=log_L_constraint,
            log_L=seed_point.log_L0,
            num_likelihood_evaluations=jnp.asarray(0, int_type)
        )
        final_sample, cumulative_samples = _cumulative_op_static(
            op=propose_op,
            init=init_sample,
            xs=random.split(key, self.num_slices),
            unroll=1
        )

        # Last sample is the final sample, the rest are potential phantom samples
        # Take only the last num_phantom_save phantom samples
        phantom_samples: Sample = tree_map(lambda x: x[-(self.num_phantom_save + 1):-1], cumulative_samples)

        return final_sample, phantom_samples
