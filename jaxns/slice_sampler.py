from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple, Any

from etils.array_types import FloatArray, IntArray, PRNGKey, BoolArray
from jax import random, numpy as jnp, tree_map
from jax._src.lax.control_flow import while_loop

from jaxns.model import Model
from jaxns.types import NestedSamplerState, LivePoints, Sample, int_type

__all__ = ['UniDimSliceSampler']


class SeedPoint(NamedTuple):
    U0: FloatArray
    log_L0: FloatArray


class UniDimProposalState(NamedTuple):
    key: jnp.ndarray  # PRNG key
    process_step: jnp.ndarray  # what stage of the sampling each independent point is at
    proposal_count: jnp.ndarray  # how many successful proposals have happened for each independent point
    num_likelihood_evaluations: jnp.ndarray  # how many  likelihood evaluations have been made for each independent point
    point_U0: jnp.ndarray  # the origin of current slice of each independent point
    log_L0: jnp.ndarray  # the value of log-likelihood at each indpendent point origin
    direction: jnp.ndarray  # the direction of slicing
    left: jnp.ndarray  # the left bound of slice
    right: jnp.ndarray  # the right bound of slice
    point_U: jnp.ndarray  # the point up for likelihood computation
    t: jnp.ndarray  # the parameter measuring between left and right bounds.


PreprocessType = Optional[Any]


class AbstractSliceSampler(ABC):
    @abstractmethod
    def get_sample(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray, num_slices: IntArray,
                   preprocess_data: PreprocessType) -> Sample:
        """
        Get a sample from a given seed point that already satisfied the likelihood constraint.

        Args:
            key: PRNGKey
            seed_point: the seed point
            log_L_constraint: likelihood constraint
            num_slices: how many slices to take per sample
            preprocess_data: auxilary data specific to the slice sampler that is needed to perform the op.

        Returns:
            a new sample
        """
        ...

    @abstractmethod
    def preprocess(self, state: NestedSamplerState) -> PreprocessType:
        """
        Compute an auxillary object that is needed to perform the slice sampling.

        Args:
            state: state

        Returns:
            an object specfic to this sampler's needs
        """
        ...


class UniDimSliceSampler(AbstractSliceSampler):
    def __init__(self, model: Model, midpoint_shrink: bool, multi_ellipse_bound: bool = False):
        self.model = model
        self.midpoint_shrink = midpoint_shrink
        self.multi_ellipse_bound = multi_ellipse_bound

    def compute_multi_ellipse_bound(self, state: NestedSamplerState) -> Any:
        raise NotImplementedError()

    def preprocess(self, state: NestedSamplerState) -> PreprocessType:
        if self.multi_ellipse_bound:
            return self.compute_multi_ellipse_bound(state=state)
        return None

    def get_seed_point(self, key: PRNGKey, live_points: LivePoints, log_L_constraint: FloatArray) -> SeedPoint:
        sample_idx = random.randint(key, (), minval=0, maxval=live_points.reservoir.log_L.size)
        return SeedPoint(
            U0=live_points.reservoir.point_U[sample_idx],
            log_L0=live_points.reservoir.log_L[sample_idx]
        )

    def sample_direction(self, n_key: PRNGKey, ndim: int) -> FloatArray:
        """
        Choose a direction randomly from S^(D-1).

        Args:
            n_key: PRNG key
            ndim: int, number of dimentions

        Returns:
            direction: [D] direction from S^(D-1)
        """
        if ndim == 1:
            return jnp.ones(())
        direction = random.normal(n_key, shape=(ndim,))
        direction /= jnp.linalg.norm(direction)
        return direction

    def slice_bounds(self, point_U0: FloatArray, direction: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Compute the slice bounds, t, where point_U0 + direction * t intersects uit cube boundary.

        Args:
            point_U0: [D]
            direction: [D]

        Returns:
            left_bound: left most point (<= 0).
            right_bound: right most point (>= 0).
        """
        t1 = (1. - point_U0) / direction
        t1_right = jnp.min(jnp.where(t1 >= 0., t1, jnp.inf))
        t1_left = jnp.max(jnp.where(t1 <= 0., t1, -jnp.inf))
        t0 = -point_U0 / direction
        t0_right = jnp.min(jnp.where(t0 >= 0., t0, jnp.inf))
        t0_left = jnp.max(jnp.where(t0 <= 0., t0, -jnp.inf))
        right_bound = jnp.minimum(t0_right, t1_right)
        left_bound = jnp.maximum(t0_left, t1_left)
        return left_bound, right_bound

    def pick_point_in_interval(self, t_key: PRNGKey, point_U0: FloatArray, direction: FloatArray, left: FloatArray,
                               right: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Select a point along slice in [point_U0 + direction * left, point_U0 + direction * right]

        Args:
            t_key: PRNG key
            point_U0: [D]
            direction: [D]
            left: left most point (<= 0).
            right: right most point (>= 0).

        Returns:
            point_U: [D]
            t: selection point between [left, right]
        """
        t = random.uniform(t_key, minval=left, maxval=right)
        point_U = point_U0 + t * direction
        return point_U, t

    def change_direction(self, from_proposal_state: UniDimProposalState,
                         log_L_proposal: jnp.ndarray) -> UniDimProposalState:
        """
        Successful proposal, but not enough proposals to de-correlate.
        Pick a new direction and propose from current point.
        """
        proposal_key, n_key, t_key = random.split(from_proposal_state.key, 3)
        point_U0 = from_proposal_state.point_U
        log_L0 = log_L_proposal
        direction = self.sample_direction(n_key, point_U0.size)
        # project out the previous direction to sample in orthogonal slice
        if point_U0.size > 1:
            direction = direction - direction * (direction @ from_proposal_state.direction)
        (left, right) = self.slice_bounds(point_U0, direction)
        point_U, t = self.pick_point_in_interval(t_key, point_U0, direction, left, right)

        return from_proposal_state._replace(key=proposal_key,
                                            process_step=jnp.full((), 3, int_type),
                                            point_U0=point_U0,
                                            log_L0=log_L0,
                                            direction=direction,
                                            left=left,
                                            right=right,
                                            point_U=point_U,
                                            t=t)

    def shrink_interval(self, from_proposal_state: UniDimProposalState, log_L_proposal: jnp.ndarray,
                        log_L_contour: jnp.ndarray,
                        midpoint_shrink: bool, destructive_shrink: bool) -> UniDimProposalState:
        """
        Not successful proposal, so shrink, optionally apply exponential shrinkage.
        """
        # witout exponential shrinkage, we shrink to failed proposal point, which is 100% correct.
        left = jnp.where(from_proposal_state.t < 0., from_proposal_state.t, from_proposal_state.left)
        right = jnp.where(from_proposal_state.t > 0., from_proposal_state.t, from_proposal_state.right)
        key, t_key, midpoint_key = random.split(from_proposal_state.key, 3)
        if destructive_shrink:
            # shrink to constraint using linear approximation
            # Line logic:
            # logL(t) = m * t + b
            # logL(0) = b
            # (logL(t_R) - logL(0))/t_R
            # logL(t_R*alpha) = (logL(t_R) - logL(0))*alpha + logL(0)
            # alpha = (logL_contour - logL(0))/(logL(t_R) - logL(0))
            alpha = (log_L_contour - from_proposal_state.log_L0) / (log_L_proposal - from_proposal_state.log_L0)
            left = jnp.where((from_proposal_state.t < 0.), alpha * left, left)
            right = jnp.where((from_proposal_state.t > 0.), alpha * right, right)
        if midpoint_shrink:
            # we take two points along lines from origin:
            #  - alpha: one from a satisfying point (t=0) to non-satisfying proposal,
            #  - beta: and, one from non-satisfying proposal to the constraint.
            # We shrink to alpha point if beta point is above alpha point.
            # Intuitively, the tangent from constraint is more accurate than proposal and should be a supremum of heights
            # in reasonable cases.
            # An extension is to make alpha shrink to constraint line, which would shrink very fast, but introduce
            # auto-correlation which must be later refined away.
            # Line logic:
            # logL(t) = m * t + b
            # logL(0) = b
            # (logL(t_R) - logL(0))/t_R
            # logL(t_R*alpha) = (logL(t_R) - logL(0))*alpha + logL(0)
            alpha_key, beta_key = random.split(midpoint_key, 2)
            alpha = random.uniform(alpha_key)
            beta = random.uniform(beta_key)
            logL_alpha = from_proposal_state.log_L0 + alpha * (log_L_proposal - from_proposal_state.log_L0)
            logL_beta = log_L_proposal + beta * (log_L_contour - log_L_proposal)
            do_mid_point_shrink = logL_alpha < logL_beta
            left = jnp.where((from_proposal_state.t < 0.) & do_mid_point_shrink, alpha * left, left)
            right = jnp.where((from_proposal_state.t > 0.) & do_mid_point_shrink, alpha * right, right)
        point_U, t = self.pick_point_in_interval(t_key, from_proposal_state.point_U0, from_proposal_state.direction,
                                                 left, right)
        return from_proposal_state._replace(key=key,
                                            process_step=jnp.full((), 3, int_type),
                                            left=left,
                                            right=right,
                                            point_U=point_U,
                                            t=t)

    def get_sample(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray, num_slices: IntArray,
                   preprocess_data: PreprocessType) -> Sample:

        slice_sampler_key, select_key, proposal_key, n_key, t_key = random.split(key, 5)

        direction = self.sample_direction(n_key, seed_point.U0.size)
        num_likelihood_evaluations = jnp.full((), 0, int_type)
        (left, right) = self.slice_bounds(seed_point.U0, direction)
        point_U, t = self.pick_point_in_interval(t_key, seed_point.U0, direction, left, right)
        init_proposal_state = UniDimProposalState(key=proposal_key,
                                                  process_step=jnp.full((), 3, int_type),
                                                  proposal_count=jnp.zeros((), int_type),
                                                  num_likelihood_evaluations=num_likelihood_evaluations,
                                                  point_U0=seed_point.U0,
                                                  log_L0=seed_point.log_L0,
                                                  direction=direction,
                                                  left=left,
                                                  right=right,
                                                  point_U=point_U,
                                                  t=t)

        CarryType = Tuple[UniDimProposalState, FloatArray]

        def slice_sampler_body(body_state: CarryType) -> CarryType:
            """
            Iteratively samples num_parallel_samplers new points using a sampler, then replaces that number of points in
            the reservoir.
            """
            (proposal_state, _) = body_state
            log_L_point_U = self.model.forward(proposal_state.point_U)
            num_likelihood_evaluations = proposal_state.num_likelihood_evaluations + jnp.ones_like(
                proposal_state.num_likelihood_evaluations)
            # assumes that log_L0 > log_L_constraint
            good_proposal = jnp.greater(log_L_point_U, log_L_constraint)
            proposal_count = jnp.where(good_proposal,
                                       proposal_state.proposal_count + jnp.ones_like(proposal_state.proposal_count),
                                       proposal_state.proposal_count)

            enough_proposals = proposal_count >= num_slices

            # 0: successful proposal & enough proposals -> done
            # 1: successful proposal & not enough proposals -> change direction
            # 2: unsuccessful proposal -> shrink interval

            process_step = jnp.where(good_proposal & enough_proposals,
                                     jnp.full(good_proposal.shape, 0, int_type),
                                     jnp.where(good_proposal & ~enough_proposals,
                                               jnp.full(good_proposal.shape, 1, int_type),
                                               jnp.full(good_proposal.shape, 2, int_type)
                                               )
                                     )

            def _map_where(cond, a_tree, b_tree):
                return tree_map(lambda a, b: jnp.where(cond, a, b), a_tree, b_tree)

            proposal_state_from_1 = self.change_direction(proposal_state, log_L_point_U)
            proposal_state_from_2 = self.shrink_interval(proposal_state, log_L_point_U,
                                                         log_L_contour=log_L_constraint,
                                                         midpoint_shrink=self.midpoint_shrink,
                                                         destructive_shrink=False)

            # replace with the proposal state as appropriate

            proposal_state = _map_where(process_step == 1,
                                        proposal_state_from_1, proposal_state)
            proposal_state = _map_where(process_step == 2,
                                        proposal_state_from_2, proposal_state)

            proposal_state = proposal_state._replace(process_step=process_step,
                                                     proposal_count=proposal_count,
                                                     num_likelihood_evaluations=num_likelihood_evaluations)
            return (proposal_state, log_L_point_U)

        def slice_sampler_cond(body_state: CarryType) -> BoolArray:
            """
            Stops when there have been enough proposals.
            """
            (proposal_state, _) = body_state
            return jnp.bitwise_not(proposal_state.proposal_count >= num_slices)

        (proposal_state, log_L) = while_loop(slice_sampler_cond,
                                             slice_sampler_body,
                                             (init_proposal_state, -jnp.inf))
        # passthrough when num_slices==0
        pass_through = num_slices == jnp.zeros_like(num_slices)
        log_L = jnp.where(pass_through, seed_point.log_L0, log_L)
        point_U = jnp.where(pass_through, seed_point.U0, proposal_state.point_U)
        sample = Sample(point_U=point_U,
                        log_L_constraint=log_L_constraint,
                        log_L=log_L,
                        num_likelihood_evaluations=proposal_state.num_likelihood_evaluations,
                        num_slices=num_slices,
                        iid=jnp.asarray(False, jnp.bool_))
        return sample


class MultiDimProposalState(NamedTuple):
    key: jnp.ndarray  # PRNG key
    process_step: jnp.ndarray  # what stage of the sampling each independent point is at
    proposal_count: jnp.ndarray  # how many successful proposals have happened for each independent point
    num_likelihood_evaluations: jnp.ndarray  # how many  likelihood evaluations have been made for each independent point
    point_U0: jnp.ndarray  # the origin of current slice of each independent point
    log_L0: jnp.ndarray  # the value of log-likelihood at each indpendent point origin
    left: jnp.ndarray  # the left bound of slice
    right: jnp.ndarray  # the right bound of slice
    point_U: jnp.ndarray  # the point up for likelihood computation


class MultiDimSliceSampler(AbstractSliceSampler):
    def __init__(self, model: Model, num_restrict_dims: Optional[int] = None):
        self.model = model
        if num_restrict_dims is not None:
            if num_restrict_dims == 1:
                raise ValueError(f"If restricting to 1 dimension, then you should use UniDimSliceSampler.")
            if not (1 < num_restrict_dims <= model.U_ndims):
                raise ValueError(f"Expected num_restriction dim in (1, {model.U_ndims}], got {num_restrict_dims}.")
        self.num_restrict_dims = num_restrict_dims

    def preprocess(self, state: NestedSamplerState) -> PreprocessType:
        return None

    def get_seed_point(self, key: PRNGKey, live_points: LivePoints, log_L_constraint: FloatArray) -> SeedPoint:
        sample_idx = random.randint(key, (), minval=0, maxval=live_points.reservoir.log_L.size)
        return SeedPoint(
            U0=live_points.reservoir.point_U[sample_idx],
            log_L0=live_points.reservoir.log_L[sample_idx]
        )

    def slice_bounds(self, slice_key: PRNGKey, point_U0: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Get the slice bounds
        Args:
            slice_key:
            point_U0:

        Returns:

        """
        if self.num_restrict_dims is not None:
            slice_dims = random.choice(slice_key, point_U0.size, shape=(self.num_restrict_dims,), replace=False)
            left = point_U0.at[slice_dims].set(jnp.zeros(self.num_restrict_dims, point_U0.dtype))
            right = point_U0.at[slice_dims].set(jnp.ones(self.num_restrict_dims, point_U0.dtype))
        else:
            left = jnp.zeros_like(point_U0)
            right = jnp.ones_like(point_U0)
        return left, right

    def new_slice(self, proposal_state: MultiDimProposalState, log_L_U: FloatArray) -> MultiDimProposalState:
        key, sample_key = random.split(proposal_state.key)

        left = jnp.zeros_like(proposal_state.left)
        right = jnp.ones_like(proposal_state.right)

        point_U = random.uniform(key=sample_key, shape=left.shape, dtype=left.dtype, minval=left, maxval=right)

        next_proposal_state = MultiDimProposalState(
            key=key,
            process_step=jnp.full((), 3, int_type),
            proposal_count=proposal_state.proposal_count,
            num_likelihood_evaluations=proposal_state.num_likelihood_evaluations,
            point_U0=proposal_state.point_U,
            log_L0=log_L_U,
            left=left,
            right=right,
            point_U=point_U
        )
        return next_proposal_state

    def shrink_region(self, proposal_state: MultiDimProposalState) -> MultiDimProposalState:
        # witout exponential shrinkage, we shrink to failed proposal point, which is 100% correct.
        left = jnp.minimum(jnp.maximum(proposal_state.left, proposal_state.point_U), proposal_state.point_U0)
        right = jnp.maximum(jnp.minimum(proposal_state.right, proposal_state.point_U), proposal_state.point_U0)

        key, sample_key = random.split(proposal_state.key, 2)

        point_U = random.uniform(key=sample_key, shape=left.shape, dtype=left.dtype, minval=left, maxval=right)

        next_proposal_state = proposal_state._replace(
            key=key,
            process_step=jnp.full((), 3, int_type),
            left=left,
            right=right,
            point_U=point_U
        )

        return next_proposal_state

    def get_sample(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray, num_slices: IntArray,
                   preprocess_data: PreprocessType) -> Sample:
        slice_sampler_key, sample_key, proposal_key = random.split(key, 3)

        num_likelihood_evaluations = jnp.full((), 0, int_type)

        left = jnp.zeros_like(seed_point.U0)
        right = jnp.ones_like(seed_point.U0)

        point_U = random.uniform(key=sample_key, shape=left.shape, dtype=left.dtype, minval=left, maxval=right)

        init_proposal_state = MultiDimProposalState(key=proposal_key,
                                                    process_step=jnp.full((), 3, int_type),
                                                    proposal_count=jnp.zeros((), int_type),
                                                    num_likelihood_evaluations=num_likelihood_evaluations,
                                                    point_U0=seed_point.U0,
                                                    log_L0=seed_point.log_L0,
                                                    left=left,
                                                    right=right,
                                                    point_U=point_U)

        CarryType = Tuple[MultiDimProposalState, FloatArray]

        def slice_sampler_body(body_state: CarryType) -> CarryType:
            """
            Iteratively samples num_parallel_samplers new points using a sampler, then replaces that number of points in
            the reservoir.
            """
            (proposal_state, _) = body_state
            log_L_point_U = self.model.forward(proposal_state.point_U)
            num_likelihood_evaluations = proposal_state.num_likelihood_evaluations + jnp.ones_like(
                proposal_state.num_likelihood_evaluations)
            # assumes that log_L0 > log_L_constraint
            good_proposal = jnp.greater(log_L_point_U, log_L_constraint)
            proposal_count = jnp.where(good_proposal,
                                       proposal_state.proposal_count + jnp.ones_like(proposal_state.proposal_count),
                                       proposal_state.proposal_count)

            enough_proposals = proposal_count >= num_slices

            # 0: successful proposal & enough proposals -> done
            # 1: successful proposal & not enough proposals -> change direction
            # 2: unsuccessful proposal -> shrink interval

            process_step = jnp.where(good_proposal & enough_proposals,
                                     jnp.full(good_proposal.shape, 0, int_type),
                                     jnp.where(good_proposal & ~enough_proposals,
                                               jnp.full(good_proposal.shape, 1, int_type),
                                               jnp.full(good_proposal.shape, 2, int_type)
                                               )
                                     )

            def _map_where(cond, a_tree, b_tree):
                return tree_map(lambda a, b: jnp.where(cond, a, b), a_tree, b_tree)

            proposal_state_from_1 = self.new_slice(proposal_state, log_L_point_U)
            proposal_state_from_2 = self.shrink_region(proposal_state)

            # replace with the proposal state as appropriate

            proposal_state = _map_where(process_step == 1, proposal_state_from_1, proposal_state)
            proposal_state = _map_where(process_step == 2, proposal_state_from_2, proposal_state)

            proposal_state = proposal_state._replace(process_step=process_step,
                                                     proposal_count=proposal_count,
                                                     num_likelihood_evaluations=num_likelihood_evaluations)
            return (proposal_state, log_L_point_U)

        def slice_sampler_cond(body_state: CarryType) -> BoolArray:
            """
            Stops when there have been enough proposals.
            """
            (proposal_state, _) = body_state
            return jnp.bitwise_not(proposal_state.proposal_count >= num_slices)

        (proposal_state, log_L) = while_loop(slice_sampler_cond,
                                             slice_sampler_body,
                                             (init_proposal_state, -jnp.inf))
        # passthrough when num_slices==0
        pass_through = num_slices == jnp.zeros_like(num_slices)
        log_L = jnp.where(pass_through, seed_point.log_L0, log_L)
        point_U = jnp.where(pass_through, seed_point.U0, proposal_state.point_U)
        sample = Sample(point_U=point_U,
                        log_L_constraint=log_L_constraint,
                        log_L=log_L,
                        num_likelihood_evaluations=proposal_state.num_likelihood_evaluations,
                        num_slices=num_slices,
                        iid=jnp.asarray(False, jnp.bool_))
        return sample
