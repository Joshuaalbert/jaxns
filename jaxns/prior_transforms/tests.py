from jax import numpy as jnp, vmap

from jaxns.prior_transforms import PriorChain, NormalPrior, LaplacePrior, DiagGaussianWalkPrior, HalfLaplacePrior, \
    ForcedIdentifiabilityPrior


def test_prior_chain():
    from jax import random
    chain = PriorChain()
    mu = NormalPrior('f', jnp.array([0., 0.]), 1.)
    gamma = jnp.array([1.])
    X = NormalPrior('x', mu, gamma)
    chain.push(mu).push(X)
    print(chain)
    U = random.uniform(random.PRNGKey(0), shape=(chain.U_ndims,))
    y = chain(U)
    print(y)

    chain = PriorChain()
    mu = NormalPrior('f', jnp.array([0., 0.]), 1.)
    gamma = jnp.array([1.])
    X = LaplacePrior('x', mu, gamma)
    chain.push(mu).push(X)
    print(chain)
    U = random.uniform(random.PRNGKey(0), shape=(chain.U_ndims,))
    y = chain(U)
    print(y)

    chain = PriorChain()
    x0 = NormalPrior('x0', jnp.array([0., 0.]), 1.)
    gamma = 1.
    X = DiagGaussianWalkPrior('W', 2, x0, gamma)
    chain.push(mu).push(X)
    print(chain)
    U = random.uniform(random.PRNGKey(0), shape=(chain.U_ndims,))
    y = chain(U)
    print(y)


def test_half_laplace():
    p = PriorChain().push(HalfLaplacePrior('x', 1.))
    U = jnp.linspace(0., 1., 100)[:, None]
    assert ~jnp.any(jnp.isnan(vmap(p)(U)['x']))


def test_forced_identifiability_prior():
    from jax import random
    prior = PriorChain().push(ForcedIdentifiabilityPrior('x', 10, 0., 10.))
    for i in range(10):
        out = prior(random.uniform(random.PRNGKey(i), shape=(prior.U_ndims,)))
        assert jnp.all(jnp.sort(out['x'], axis=0) == out['x'])
        assert jnp.all((out['x'] >= 0.) & (out['x'] <= 10.))
    prior = PriorChain().push(ForcedIdentifiabilityPrior('x', 10, jnp.array([0., 0.]), 10.))
    for i in range(10):
        out = prior(random.uniform(random.PRNGKey(i), shape=(prior.U_ndims,)))
        assert out['x'].shape == (10, 2)
        assert jnp.all(jnp.sort(out['x'], axis=0) == out['x'])
        assert jnp.all((out['x'] >= 0.) & (out['x'] <= 10.))