from etils.array_types import PRNGKey, FloatArray
from jax import random, numpy as jnp
from jax.lax import while_loop

from jaxns.internals.types import int_type
from jaxns.new_code.model import Model
from jaxns.new_code.types import Sample

__all__ = ['UniformSampler']


class UniformSampler:
    def __init__(self, model: Model):
        self.model = model

    def single_sample(self, key: PRNGKey, log_L_constraint: FloatArray) -> Sample:
        """
        Produces a single sample from the joint-prior and computes the likelihood.

        Args:
            key: PRNG key
            log_L_constraint: the log likelihood constraint to sample above

        Returns:
            a sample
        """
        key, sample_key = random.split(key, 2)
        U = self.model.sample_U(key=sample_key)
        log_L = self.model.forward(U=U)
        num_likelihood_evals = jnp.asarray(1, int_type)
        done = (log_L > log_L_constraint)

        def body(state):
            (_, key, _, _, num_likelihood_evals) = state
            key, sample_key = random.split(key, 2)
            U = self.model.sample_U(key=sample_key)
            log_L = self.model.forward(U=U)
            num_likelihood_evals += jnp.asarray(1, int_type)
            done = (log_L > log_L_constraint)
            return (done, key, U, log_L, num_likelihood_evals)

        (_, _, U, log_L, num_likelihood_evals) = while_loop(lambda s: jnp.bitwise_not(s[0]),
                                                            body,
                                                            (done, key, U, log_L, num_likelihood_evals))
        sample = Sample(point_U=U,
                        log_L_constraint=log_L_constraint,
                        log_L=log_L,
                        num_likelihood_evaluations=num_likelihood_evals,
                        num_slices=jnp.asarray(0, int_type),
                        iid=jnp.asarray(True, jnp.bool_))
        return sample
