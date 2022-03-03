from jax import numpy as jnp, vmap, random
from jax.scipy.special import logsumexp

from jaxns.prior_transforms import PriorChain, NormalPrior, UniformPrior, HalfLaplacePrior, \
    ForcedIdentifiabilityPrior, DeltaPrior, GumbelBernoulliPrior, GumbelCategoricalPrior, BernoulliPrior, \
    CategoricalPrior, PoissonPrior
from jaxns.prior_transforms.prior_chain import iterative_topological_sort


def test_prior_chain():
    # two subspaces: a -> b, c -> d
    with PriorChain() as chain1:
        a = DeltaPrior('a', 0.)
        b = NormalPrior('b', a, [1.])
        for name in chain1._prior_chain.keys():
            assert name in ['a', 'b', '_b_sigma']
    chain1.build()
    U_compact = chain1.sample_U_flat(random.PRNGKey(42))
    X = chain1(U_compact)
    assert jnp.size(X['b']) == 1
    assert jnp.shape(X['b']) == (1,)


def test_discrete():
    for p in [0., 0.05, 0.5, 0.95, 1.]:
        with PriorChain() as chain:
            x = GumbelBernoulliPrior('x', p)
        chain.build()
        y = vmap(lambda key: chain(chain.sample_U_flat(key)))(
            random.split(random.PRNGKey(245263), 100000))
        assert jnp.abs(y['x'].mean() - p) < 0.01

    for p in [0., 0.05, 0.5, 0.95, 1.]:
        with PriorChain() as chain:
            x = BernoulliPrior('x', p)
        chain.build()
        y = vmap(lambda key: chain(chain.sample_U_flat(key)))(
            random.split(random.PRNGKey(245263), 100000))
        assert jnp.abs(y['x'].mean() - p) < 0.01

    p = jnp.array([0.1, 0.6, 0.1, 0.2])
    with PriorChain() as chain:
        x = GumbelCategoricalPrior('x', jnp.log(p))
    chain.build()
    y = vmap(lambda key: chain(chain.sample_U_flat(key)))(
        random.split(random.PRNGKey(245263), 100000))
    for i in range(4):
        assert jnp.all(jnp.abs(jnp.mean(y['x'] == i) - p[i]) < 0.01)

    p = jnp.array([0.1, 0.6, 0.1, 0.2])
    with PriorChain() as chain:
        x = CategoricalPrior('x', jnp.log(p))
    chain.build()
    y = vmap(lambda key: chain(chain.sample_U_flat(key)))(
        random.split(random.PRNGKey(245263), 100000))
    for i in range(4):
        assert jnp.all(jnp.abs(jnp.mean(y['x'] == i) - p[i]) < 0.01)


def test_half_laplace():
    with PriorChain() as p:
        x = HalfLaplacePrior('x', 1.)
    p.build()
    assert p.sample_U_flat(random.PRNGKey(42)).shape == (p.U_ndims,)
    U = jnp.linspace(0., 1., 100)[:,None]
    X = vmap(p)(U)['x']
    assert ~jnp.any(jnp.isnan(X))


def test_poisson():
    with PriorChain() as p:
        x = PoissonPrior('x', 1000.)
    p.build()
    assert p.sample_U_flat(random.PRNGKey(42)).shape == (p.U_ndims,)
    U = jnp.linspace(0., 1., 10000)[:, None]
    X = vmap(p)(U)['x']
    assert jnp.isclose(X.mean(), 1000., atol=0.9)
    assert ~jnp.any(jnp.isnan(X))


def test_forced_identifiability_prior():
    with PriorChain() as prior_chain:
         ForcedIdentifiabilityPrior('x', 100, 0., 10.)
    prior_chain.build()
    for i in range(10):
        out = prior_chain(prior_chain.sample_U_flat(random.PRNGKey(i)))
        assert jnp.all(jnp.sort(out['x'], axis=0) == out['x'])
        assert jnp.all((out['x'] >= 0.) & (out['x'] <= 10.))
    with PriorChain() as prior_chain:
       ForcedIdentifiabilityPrior('x', 100, jnp.array([0., 0.]), 10.)
    prior_chain.build()
    for i in range(10):
        out = prior_chain(prior_chain.sample_U_flat(random.PRNGKey(i)))
        assert out['x'].shape == (100, 2)
        assert jnp.all(jnp.sort(out['x'], axis=0) == out['x'])
        assert jnp.all((out['x'] >= 0.) & (out['x'] <= 10.))



def test_gh_20():
    with PriorChain() as prior_chain:
        prior_a = UniformPrior('a', -20., 20.)
        prior_b = UniformPrior('b', -20., 20.)
        prior_c = UniformPrior('c', 0, 40)
        prior_d = UniformPrior('d', -20., 20.)
        prior_sigma = HalfLaplacePrior('sigma', 1)

    #shouldn't need to be built
    assert prior_chain.U_ndims == 5 #should be 5

    prior_chain.build()
    assert prior_chain.U_ndims == 5  # should be 5



def test_CategoricalPrior():
    logits = jnp.log(jnp.arange(4))
    with PriorChain() as prior_chain:
        b = CategoricalPrior('b', logits, False)
    prior_chain.build()
    samples = vmap(lambda key:prior_chain(prior_chain.sample_U_flat(key)))(random.split(random.PRNGKey(42), 100000))
    freqs, _ = jnp.histogram(samples['b'].flatten(), bins=jnp.arange(5), density=True)
    assert jnp.allclose(jnp.exp(logits - logsumexp(logits)), freqs, atol=1e-2)


def test_iterative_topological_sort():
    dsk = {'a': [],
           'b': ['a'],
           'c': ['a', 'b']}
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a']
    dsk = {'a': [],
           'b': ['a', 'd'],
           'c': ['a', 'b']}
    # print(iterative_topological_sort(dsk, ['a', 'b', 'c']))
    assert iterative_topological_sort(dsk, ['a', 'b', 'c']) == ['c', 'b', 'a', 'd']
    assert iterative_topological_sort(dsk) == ['c', 'b', 'a', 'd']