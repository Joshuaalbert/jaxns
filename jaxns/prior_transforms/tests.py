from jax import numpy as jnp, vmap, random
from jax._src.scipy.special import logsumexp

from jaxns.prior_transforms import PriorChain, NormalPrior, UniformPrior, HalfLaplacePrior, \
    ForcedIdentifiabilityPrior, DeltaPrior, GumbelBernoulliPrior, GumbelCategoricalPrior, PoissonPrior, CategoricalPrior


def test_prior_chain():
    # two subspaces: a -> b, c -> d
    a = DeltaPrior('a', 0.)
    b = NormalPrior('b', a, [1.])
    chain1 = PriorChain(b)
    for name in chain1.prior_chain.keys():
        assert name in ['a', 'b', '_b_sigma']

    U = chain1.sample_U(random.PRNGKey(235))
    U_compact = chain1.compactify_U(U)
    X = chain1(U_compact)
    assert jnp.size(X['b']) == 1
    assert jnp.shape(X['b']) == (1,)
    y = chain1.disperse_U(U_compact)
    for _U,_y in zip(U.values(), y.values()):
        assert jnp.allclose(_U,_y)

    chain2 = PriorChain()
    c = UniformPrior('var', 0., 1.)
    d = GumbelBernoulliPrior('x2', c)
    chain2.push(d)
    chain2.push(chain1)
    # [{'var', '_f2_low', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}, {'_b_sigma', 'b', 'a'}]
    assert len(chain2.subspaces) == 2
    assert chain2.subspaces[0] == {'var', '_var_low', 'x2', '_var_high', '_x2_gumbel', '_log_var'}
    assert chain2.subspaces[1] == {'_b_sigma', 'b', 'a'}

    # priors in chain1 added before the chain stay in those subspaces
    chain2 = PriorChain()
    c = UniformPrior('var', a, 1.)
    d = GumbelBernoulliPrior('x2', c)
    chain2.push(d)
    chain2.push(chain1)
    # [{'var', '_f2_low', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}, {'_b_sigma', 'b', 'a'}]
    assert len(chain2.subspaces) == 2
    assert chain2.subspaces[0] == {'var', 'x2', '_var_high', '_x2_gumbel', '_log_var'}
    assert chain2.subspaces[1] == {'_b_sigma', 'b', 'a'}

    # priors in chain1 added after the chain stay in those subspaces
    chain2 = PriorChain(chain1)
    c = UniformPrior('var', a, 1.)
    d = GumbelBernoulliPrior('x2', c)
    chain2.push(d)
    # [{'var', '_f2_low', 'x2', '_f2_high', '_x2_gumbel', '_log_f2'}, {'_b_sigma', 'b', 'a'}]
    assert len(chain2.subspaces) == 2
    assert chain2.subspaces[0] == {'var', 'x2', '_var_high', '_x2_gumbel', '_log_var'}
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
    X = vmap(lambda u: p((u,)))(U)['x']
    assert ~jnp.any(jnp.isnan(X))


def test_poisson():
    from jax import disable_jit
    p = PriorChain().push(PoissonPrior('x', 1000.))
    U = jnp.linspace(0., 1., 1000)[:, None]
    # p((jnp.asarray([1.]),))
    X = vmap(lambda u: p((u,)))(U)['x']
    print(X)
    print(X.mean())
    print(X.var())
    assert ~jnp.any(jnp.isnan(X))


def test_forced_identifiability_prior():
    from jax import random
    prior = PriorChain(ForcedIdentifiabilityPrior('x', 100, 0., 10.))
    for i in range(10):
        out = prior(prior.compactify_U(prior.sample_U(random.PRNGKey(i))))
        assert jnp.all(jnp.sort(out['x'], axis=0) == out['x'])
        assert jnp.all((out['x'] >= 0.) & (out['x'] <= 10.))
    prior = PriorChain().push(ForcedIdentifiabilityPrior('x', 100, jnp.array([0., 0.]), 10.))
    for i in range(10):
        out = prior(prior.compactify_U(prior.sample_U(random.PRNGKey(i))))
        assert out['x'].shape == (100, 2)
        assert jnp.all(jnp.sort(out['x'], axis=0) == out['x'])
        assert jnp.all((out['x'] >= 0.) & (out['x'] <= 10.))



def test_gh_20():
    prior_a = UniformPrior('a', -20., 20.)
    prior_b = UniformPrior('b', -20., 20.)
    prior_c = UniformPrior('c', 0, 40)
    prior_d = UniformPrior('d', -20., 20.)
    prior_sigma = HalfLaplacePrior('sigma', 1)

    prior_chain = PriorChain(prior_a, prior_b, prior_c, prior_d, prior_sigma)

    assert prior_chain.U_ndims == 5 #should be 5


def test_CategoricalPrior():
    from jax import random, vmap
    logits = jnp.log(jnp.arange(4))
    b = CategoricalPrior('b', logits, False)
    prior_chain = b.prior_chain()
    samples = vmap(lambda key:prior_chain(prior_chain.compactify_U(prior_chain.sample_U(key))))(random.split(random.PRNGKey(42), 100000))
    freqs, _ = jnp.histogram(samples['b'].flatten(), bins =jnp.arange(5), density=True)
    assert jnp.allclose(jnp.exp(logits - logsumexp(logits)), freqs, atol=1e-2)