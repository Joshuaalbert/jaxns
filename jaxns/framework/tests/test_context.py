import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxns.framework.context import wrap_random, get_parameter, get_state, transform_with_state, global_context, \
    next_rng_key


def test_context():
    def my_model(x):
        # Example of parameter usage
        w_init = wrap_random(jax.nn.initializers.glorot_normal())
        w = get_parameter("w", (x.shape[-1], 128), init=w_init)

        # Example of state usage
        reg_loss = 0.01 * jnp.sum(w ** 2)
        get_state("reg_loss", init=reg_loss)

        param_pytree = get_parameter("param_pytree", init={"a": jnp.ones(()), "b": jnp.ones(())})
        state_pytree = get_state("state_pytree", init={"a": jnp.ones(()), "b": jnp.ones(())})

        return jnp.dot(x, w) + param_pytree["a"] + param_pytree["b"] + reg_loss + state_pytree["a"] + state_pytree["b"]

    assert len(global_context.stack) == 0

    # Transform the model
    init, apply = transform_with_state(my_model)
    assert len(global_context.stack) == 0

    # Initialize parameters and state
    rng = jax.random.PRNGKey(42)
    x = jnp.ones([1, 32])
    f0, params, states = init(rng, x)

    # Apply the model
    f1, states1 = apply(params, states, jax.random.PRNGKey(0), x)

    np.testing.assert_allclose(f1, f0)
    np.testing.assert_allclose(states['reg_loss'], states1['reg_loss'])

    with pytest.raises(ValueError, match="No context available."):
        _ = get_parameter("w", init=jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="No context available."):
        _ = get_state("reg_loss", init=jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="No context available."):
        next_rng_key()
