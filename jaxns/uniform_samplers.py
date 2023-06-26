from typing import NamedTuple

from etils.array_types import BoolArray, IntArray
from etils.array_types import PRNGKey, FloatArray
from jax import random, numpy as jnp
from jax.lax import while_loop

from jaxns.model import Model
from jaxns.static_nested_sampler import PreProcessType, RejectionSampler
from jaxns.types import NestedSamplerState, LivePoints, Sample, int_type

__all__ = ['UniformSampler']


class UniformSampler(RejectionSampler):
    def preprocess(self, state: NestedSamplerState, live_points: LivePoints) -> PreProcessType:
        return ()

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, live_points: LivePoints,
                   preprocess_data: PreProcessType) -> Sample:
        class CarryState(NamedTuple):
            done: BoolArray
            key: PRNGKey
            U: FloatArray
            log_L: FloatArray
            log_L_constraint: FloatArray
            num_likelihood_evals: IntArray

        def body(carry_state: CarryState):
            key, sample_key = random.split(carry_state.key, 2)
            log_L = self.model.forward(U=carry_state.U)
            num_likelihood_evals = carry_state.num_likelihood_evals + jnp.asarray(1, int_type)
            # backoff by one e-fold per attempt after efficiency threshold reached
            log_L_constraint = jnp.where(num_likelihood_evals > 1. / self.efficiency_threshold,
                                         carry_state.log_L_constraint - 0.1, carry_state.log_L_constraint)
            done = log_L > log_L_constraint
            U = jnp.where(done, carry_state.U, self.model.sample_U(key=sample_key))
            return CarryState(done=done, key=key, U=U, log_L=log_L, num_likelihood_evals=num_likelihood_evals,
                              log_L_constraint=log_L_constraint)

        key, sample_key = random.split(key, 2)
        init_carry_state = CarryState(done=jnp.asarray(False),
                                      key=key,
                                      U=self.model.sample_U(key=sample_key),
                                      log_L=log_L_constraint,
                                      log_L_constraint=log_L_constraint,
                                      num_likelihood_evals=jnp.asarray(0, int_type))

        carry_state = while_loop(lambda s: jnp.bitwise_not(s.done), body, init_carry_state)

        sample = Sample(point_U=carry_state.U,
                        log_L_constraint=carry_state.log_L_constraint,
                        log_L=carry_state.log_L,
                        num_likelihood_evaluations=carry_state.num_likelihood_evals,
                        num_slices=jnp.asarray(0, int_type),
                        iid=jnp.asarray(True, jnp.bool_))
        return sample


class BadUniformSampler(RejectionSampler):
    def __init__(self, mis_fraction: float, model: Model):
        super().__init__(model=model, efficiency_threshold=None)
        self.mis_fraction = mis_fraction

    def preprocess(self, state: NestedSamplerState, live_points: LivePoints) -> PreProcessType:
        return ()

    def get_sample(self, key: PRNGKey, log_L_constraint: FloatArray, live_points: LivePoints,
                   preprocess_data: PreProcessType) -> Sample:
        class CarryState(NamedTuple):
            done: BoolArray
            key: PRNGKey
            U: FloatArray
            log_L: FloatArray
            log_L_constraint: FloatArray
            num_likelihood_evals: IntArray

        def body(carry_state: CarryState):
            key, sample_key = random.split(carry_state.key, 2)
            log_L = self.model.forward(U=carry_state.U)
            num_likelihood_evals = carry_state.num_likelihood_evals + jnp.asarray(1, int_type)
            bad_log_L_constraint = carry_state.log_L_constraint + jnp.log(1. + self.mis_fraction)
            done = log_L > bad_log_L_constraint
            U = jnp.where(done, carry_state.U, self.model.sample_U(key=sample_key))
            return CarryState(done=done, key=key, U=U, log_L=log_L,
                              num_likelihood_evals=num_likelihood_evals,
                              log_L_constraint=carry_state.log_L_constraint)

        key, sample_key = random.split(key, 2)
        init_carry_state = CarryState(done=jnp.asarray(False),
                                      key=key,
                                      U=self.model.sample_U(key=sample_key),
                                      log_L=log_L_constraint,
                                      log_L_constraint=log_L_constraint,
                                      num_likelihood_evals=jnp.asarray(0, int_type))

        carry_state = while_loop(lambda s: jnp.bitwise_not(s.done), body, init_carry_state)

        sample = Sample(point_U=carry_state.U,
                        log_L_constraint=carry_state.log_L_constraint,
                        log_L=carry_state.log_L,
                        num_likelihood_evaluations=carry_state.num_likelihood_evals,
                        num_slices=jnp.asarray(0, int_type),
                        iid=jnp.asarray(self.mis_fraction == 0., jnp.bool_))
        return sample
