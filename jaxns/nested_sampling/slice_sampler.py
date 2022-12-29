from typing import NamedTuple, Optional, Union, Tuple

from etils.array_types import FloatArray, IntArray, PRNGKey, BoolArray
from jax import random, numpy as jnp, value_and_grad, tree_map
from jax._src.lax.control_flow import while_loop

from jaxns.nested_sampling.likelihood_samplers.parallel_slice_sampling import sample_direction, slice_bounds, \
    pick_point_in_interval, \
    ProposalState, change_direction, shrink_interval
from jaxns.nested_sampling.model import Model
from jaxns.nested_sampling.types import NestedSamplerState, LivePoints, Sample, int_type

__all__ = ['SliceSampler']


class MultiEllipses(NamedTuple):
    cluster_id: IntArray
    mu: FloatArray
    radii: FloatArray
    rotation: FloatArray
    num_k: IntArray


class SeedPoint(NamedTuple):
    U0: FloatArray
    log_L0: FloatArray


PreprocessType = Optional[Union[MultiEllipses]]


class SliceSampler:
    def __init__(self, model: Model, midpoint_shrink: bool, destructive_shrink: bool, gradient_boost: bool,
                 multi_ellipse_bound: bool = False):
        self.model = model
        self.midpoint_shrink = midpoint_shrink
        self.destructive_shrink = destructive_shrink
        self.gradient_boost = gradient_boost
        self.multi_ellipse_bound = multi_ellipse_bound

    def compute_multi_ellipse_bound(self, state: NestedSamplerState) -> MultiEllipses:
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

    def get_sample(self, key: PRNGKey, seed_point: SeedPoint, log_L_constraint: FloatArray, num_slices: IntArray,
                   preprocess_data: PreprocessType) -> Sample:

        slice_sampler_key, select_key, proposal_key, n_key, t_key = random.split(key, 5)

        direction = sample_direction(n_key, seed_point.U0.size)
        num_likelihood_evaluations = jnp.full((), 0, int_type)
        if self.gradient_boost:
            # sets the initial direction of search up-hill (biasing away from uniform sampling, but pulling toward peak)
            _, grad_direction = value_and_grad(lambda U: self.model.forward(U)[1])(seed_point.U0)
            grad_direction /= jnp.linalg.norm(grad_direction)
            direction = jnp.where(jnp.isnan(grad_direction), direction, grad_direction)
            num_likelihood_evaluations += jnp.full((), 1, int_type)
        (left, right) = slice_bounds(seed_point.U0, direction)
        point_U, t = pick_point_in_interval(t_key, seed_point.U0, direction, left, right)
        init_proposal_state = ProposalState(key=proposal_key,
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

        CarryType = Tuple[ProposalState, FloatArray]

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

            proposal_state_from_1 = change_direction(proposal_state, log_L_point_U)
            proposal_state_from_2 = shrink_interval(proposal_state, log_L_point_U,
                                                    log_L_contour=log_L_constraint,
                                                    midpoint_shrink=self.midpoint_shrink,
                                                    destructive_shrink=self.destructive_shrink)

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
