from jax import random, numpy as jnp, vmap
from jax.scipy.special import logsumexp

from jaxns.prior_transforms import PriorChain, HalfLaplacePrior, GumbelBernoulliPrior, BernoulliPrior, \
    GumbelCategoricalPrior, CategoricalPrior, PoissonPrior, ForcedIdentifiabilityPrior, PiecewiseLinearPrior

def test_piecewise_linear():
    with PriorChain() as prior_chain:
        PiecewiseLinearPrior('a', 10, low=-2., high=2.)
    prior_chain.build()
    samples = vmap(lambda key: prior_chain(prior_chain.sample_U_flat(key)))(random.split(random.PRNGKey(42), 10000))
    assert not jnp.any(jnp.isnan(samples['a']))
    assert jnp.all(samples['a']<=2.)
    assert jnp.all(samples['a']>=-2.)
    # bins = jnp.linspace(-2., 2., 100)
    # f, _ = jnp.histogram(samples['a'], bins=bins)
    # import pylab as plt
    # plt.plot(samples['_a_x'][1], samples["_a_y"][1])
    # plt.show()
    # plt.scatter(bins[1:], f)
    # plt.show()

def test_half_laplace():
    with PriorChain() as p:
        x = HalfLaplacePrior('x', 1.)
    p.build()
    assert p.sample_U_flat(random.PRNGKey(42)).shape == (p.U_ndims,)
    U = jnp.linspace(0., 1., 100)[:,None]
    X = vmap(p)(U)['x']
    assert ~jnp.any(jnp.isnan(X))


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

    with PriorChain() as chain:
        x = BernoulliPrior('x', jnp.ones(2)*0.5)
    chain.build()
    y = vmap(lambda key: chain(chain.sample_U_flat(key)))(
        random.split(random.PRNGKey(245263), 100000))
    assert jnp.all(jnp.abs(y['x'].mean(0) - 0.5) < 0.01)

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


def test_poisson():
    with PriorChain() as p:
        x = PoissonPrior('x', 100.)
    p.build()
    assert p.sample_U_flat(random.PRNGKey(42)).shape == (p.U_ndims,)
    U = jnp.linspace(0., 1., 10000)[:, None]
    X = vmap(p)(U)['x']
    assert jnp.isclose(X.mean(), 100., atol=1.)
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


def test_CategoricalPrior():
    logits = jnp.log(jnp.arange(4))
    with PriorChain() as prior_chain:
        b = CategoricalPrior('b', logits, False)
    prior_chain.build()
    samples = vmap(lambda key:prior_chain(prior_chain.sample_U_flat(key)))(random.split(random.PRNGKey(42), 100000))
    freqs, _ = jnp.histogram(samples['b'].flatten(), bins=jnp.arange(5), density=True)
    assert jnp.allclose(jnp.exp(logits - logsumexp(logits)), freqs, atol=1e-2)