from typing import NamedTuple, Any, Tuple

import jax
from jax import numpy as jnp, random, lax

from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.mixed_precision import mp_policy
from jaxns.internals.types import PRNGKey, UType, MeasureType, IntArray
from jaxns.nested_samplers.common.types import Sample


def _single_uniform_sample(key: PRNGKey, model: BaseAbstractModel) -> Sample:
    """
    Gets a single sample strictly within -inf bound (the entire prior), accounting for forbidden regions.

    Args:
        key: PRNGKey
        model: the model to use.

    Returns:
        a sample
    """

    log_L_constraint = jnp.asarray(-jnp.inf, mp_policy.measure_dtype)

    class CarryState(NamedTuple):
        key: PRNGKey
        U: UType
        log_L: MeasureType
        num_likelihood_evals: IntArray

    def cond(carry_state: CarryState):
        return carry_state.log_L <= log_L_constraint

    def body(carry_state: CarryState) -> CarryState:
        key, sample_key = random.split(carry_state.key, 2)
        U = model.sample_U(key=sample_key)
        log_L = model.forward(U=U)
        num_likelihood_evals = carry_state.num_likelihood_evals + jnp.ones_like(carry_state.num_likelihood_evals)
        return CarryState(key=key, U=U, log_L=log_L, num_likelihood_evals=num_likelihood_evals)

    key, sample_key = random.split(key, 2)
    init_U = model.sample_U(key=sample_key)
    init_log_L = model.forward(init_U)
    init_carry_state = CarryState(
        key=key,
        U=init_U,
        log_L=init_log_L,
        num_likelihood_evals=jnp.asarray(1, mp_policy.count_dtype)
    )

    carry_state = lax.while_loop(cond_fun=cond, body_fun=body, init_val=init_carry_state)

    sample = Sample(
        U_sample=carry_state.U,
        log_L_constraint=log_L_constraint,
        log_L=carry_state.log_L,
        num_likelihood_evaluations=carry_state.num_likelihood_evals
    )
    return sample


def draw_uniform_samples(keys: PRNGKey, model: BaseAbstractModel, method: str = 'vmap') -> Sample:
    """
    Get initial live points from uniformly sampling the entire prior.

    Args:
        keys: PRNGKey
        model: the model
        method: which way to draw the init points. vmap is vectorised, and for performant but uses more memory.

    Returns:
        uniformly drawn samples within -inf bound
    """

    if method == 'vmap':
        return jax.vmap(lambda _key: _single_uniform_sample(key=_key, model=model))(keys)
    elif method == 'scan':

        def body(carry_unused: Any, key: PRNGKey) -> Tuple[Any, Sample]:
            return carry_unused, _single_uniform_sample(key=key, model=model)

        _, samples = lax.scan(body, (), keys)

        return samples
    else:
        raise ValueError(f'Invalid method {method}')
