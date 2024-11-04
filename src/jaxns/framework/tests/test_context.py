import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxns.framework.context import wrap_random, get_parameter, get_state, global_context, transform_with_state, \
    next_rng_key, scope, ScopedDict


def test_context():
    def my_model(x):
        # Example of parameter usage
        w_init = wrap_random(jax.nn.initializers.glorot_normal())
        w = get_parameter("w", (x.shape[-1], 128), init=w_init)

        # Example of state usage
        reg_loss = 0.01 * jnp.sum(w ** 2)
        get_state("reg_loss", init=reg_loss)

        param_pytree = get_parameter("param_pytree", init={"a": jnp.ones(()), "b": jnp.ones(())})
        state_pytree = get_state("state_pytree", init=lambda *args: {"a": jnp.ones(()), "b": jnp.ones(())})

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


def test_scope():
    def f():
        x = get_parameter("x", init=jnp.ones(()))
        x_state = get_state("X", init=jnp.ones(()))
        with scope('a'):
            y = get_parameter("y", init=jnp.ones(()))
            with scope('b'):
                z = get_parameter("z", init=jnp.ones(()))
                return x + y + z

    transformed_fn = transform_with_state(f)
    init = transformed_fn.init(jax.random.PRNGKey(42))
    print(init)
    assert '.x' in init.params.keys()
    assert '.X' in init.states.keys()
    assert 'a.y' in init.params.keys()
    assert 'a.b.z' in init.params.keys()
    output = jax.jit(transformed_fn.apply)(init.params, init.states, jax.random.PRNGKey(42))
    print(output)


def test_scope_dict_flatten():
    sd = ScopedDict()
    sd.push_scope("scope")
    sd["key"] = 1
    leaves, treedef = jax.tree_util.tree_flatten(sd)
    sd2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert sd2["key"] == 1
    assert sd2.scopes == ["scope"]

    sd = ScopedDict()
    sd.push_scope("scope")
    sd["key"] = jnp.ones(1)

    def f(sd):
        return sd


    f_jit = jax.jit(f).lower(sd).compile()
    sd2 = f_jit(sd)
    np.testing.assert_allclose(sd2["key"], jnp.ones(1))
    assert sd2.scopes == ["scope"]
