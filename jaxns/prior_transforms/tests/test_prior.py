from jax import random, numpy as jnp

from jaxns.prior_transforms import PriorChain, UniformPrior, DeltaPrior


def test_builtins():
    with PriorChain() as prior_chain:
        x = UniformPrior('x',0., 1.)
        x_squared = x.square(name='x_squared')
        x_squared_ = x.multiply(x, name='x_times_x')
        x_plus_1 = x.add(1., name='x_plus_1')
        x_exp = x.exp(name='x_exp')

    prior_chain.build()

    samples = prior_chain(prior_chain.sample_U_flat(random.PRNGKey(42)))
    assert (samples['x_squared'] == samples['x_times_x']) and (samples['x_times_x'] == samples['x']**2)
    assert samples['x_plus_1'] == samples['x'] + 1.
    assert samples['x_exp'] == jnp.exp(samples['x'])

    # interp test
    with PriorChain() as prior_chain:
        xp = DeltaPrior('xp',jnp.linspace(0., 1., 2))
        fp = DeltaPrior('fp',jnp.linspace(0., 1., 2))
        x = DeltaPrior('x',jnp.linspace(0.25, 0.75, 2))
        # for a linear function, the interp value should be y=x
        f = x.interp(xp, fp, name='f')
        f_static_input = x.interp(xp, jnp.linspace(0., 1., 2), name='f_s')

    prior_chain.build()

    samples = prior_chain(prior_chain.sample_U_flat(random.PRNGKey(42)))
    assert jnp.allclose(samples['f'], samples['x'])
    assert jnp.allclose(samples['f_s'], samples['x'])