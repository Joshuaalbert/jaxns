from jax import numpy as jnp, vmap, random

from jaxns.prior_transforms import PriorChain, NormalPrior, UniformPrior, HalfLaplacePrior, \
    ForcedIdentifiabilityPrior, DeltaPrior, GumbelBernoulliPrior, GumbelCategoricalPrior


def test_prior_chain():
    # two subspaces: a -> b, c -> d
    a = DeltaPrior('a', 0.)
    b = NormalPrior('b', a, 1.)
    chain1 = PriorChain(b)
    for name in chain1.prior_chain.keys():
        assert name in ['a', 'b', '_b_sigma']

    y = chain1(chain1.compactify_U(chain1.sample_U(random.PRNGKey(235))))

    chain2 = PriorChain()
    c = UniformPrior('f2', 0., 1.)
    d = GumbelBernoulliPrior('x2', c)
    chain2.push(d)
    chain2.push(chain1)
    # [{'f2', '_f2_low', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}, {'_b_sigma', 'b', 'a'}]
    assert len(chain2.subspaces) == 2
    assert chain2.subspaces[0] == {'f2', '_f2_low', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}
    assert chain2.subspaces[1] == {'_b_sigma', 'b', 'a'}

    # priors in chain1 added before the chain stay in those subspaces
    chain2 = PriorChain()
    c = UniformPrior('f2', a, 1.)
    d = GumbelBernoulliPrior('x2', c)
    chain2.push(d)
    chain2.push(chain1)
    # [{'f2', '_f2_low', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}, {'_b_sigma', 'b', 'a'}]
    assert len(chain2.subspaces) == 2
    assert chain2.subspaces[0] == {'f2', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}
    assert chain2.subspaces[1] == {'_b_sigma', 'b', 'a'}

    # priors in chain1 added after the chain stay in those subspaces
    chain2 = PriorChain(chain1)
    c = UniformPrior('f2', a, 1.)
    d = GumbelBernoulliPrior('x2', c)
    chain2.push(d)
    # [{'f2', '_f2_low', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}, {'_b_sigma', 'b', 'a'}]
    assert len(chain2.subspaces) == 2
    assert chain2.subspaces[0] == {'f2', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}
    assert chain2.subspaces[1] == {'_b_sigma', 'b', 'a'}

    try:
        chain2.push(chain1)
    except ValueError:
        #intersection
        assert True







def test_discrete():
    for p in [0., 0.05, 0.5, 0.95, 1.]:
        x = GumbelBernoulliPrior('x', p)
        chain = PriorChain(x)
        y = vmap(lambda key: chain(chain.compactify_U(chain.sample_U(key))))(
            random.split(random.PRNGKey(245263), 100000))
        assert jnp.abs(y['x'].mean() - p) < 0.01

    p = jnp.array([0.1, 0.6, 0.1, 0.2])
    x = GumbelCategoricalPrior('x', jnp.log(p))
    chain = PriorChain(x)
    y = vmap(lambda key: chain(chain.compactify_U(chain.sample_U(key))))(
        random.split(random.PRNGKey(245263), 100000))
    for i in range(4):
        assert jnp.all(jnp.abs(jnp.mean(y['x'] == i) - p[i]) < 0.01)


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
