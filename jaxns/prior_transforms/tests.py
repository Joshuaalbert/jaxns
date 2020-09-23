from jax import numpy as jnp, vmap

from jaxns.prior_transforms import PriorChain, MVNDiagPrior, LaplacePrior, DiagGaussianWalkPrior, HalfLaplacePrior, \
    ForcedIdentifiabilityPrior, GMMMarginalPrior


def test_prior_chain():
    from jax import random
    chain = PriorChain()
    mu = MVNDiagPrior('mu', jnp.array([0., 0.]), 1.)
    gamma = jnp.array([1.])
    X = MVNDiagPrior('x', mu, gamma)
    chain.push(mu).push(X)
    print(chain)
    U = random.uniform(random.PRNGKey(0), shape=(chain.U_ndims,))
    y = chain(U)
    print(y)

    chain = PriorChain()
    mu = MVNDiagPrior('mu', jnp.array([0., 0.]), 1.)
    gamma = jnp.array([1.])
    X = LaplacePrior('x', mu, gamma)
    chain.push(mu).push(X)
    print(chain)
    U = random.uniform(random.PRNGKey(0), shape=(chain.U_ndims,))
    y = chain(U)
    print(y)

    chain = PriorChain()
    x0 = MVNDiagPrior('x0', jnp.array([0., 0.]), 1.)
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


def test_unit_cube_mixture_prior():
    import jax.numpy as jnp
    from jax import random
    from jaxns.nested_sampling import NestedSampler
    from jaxns.plotting import plot_cornerplot, plot_diagnostics

    # prior_chain = PriorChain().push(MultiCubeMixturePrior('x', 2, 1, -5., 15.))
    prior_chain = PriorChain().push(GMMMarginalPrior('x', 2, -5., 15.))

    def loglikelihood(x, **kwargs):
        return jnp.log(0.5 * jnp.exp(-0.5 * jnp.sum(x) ** 2) / jnp.sqrt(2. * jnp.pi)
                       + 0.5 * jnp.exp(-0.5 * jnp.sum(x - 10.) ** 2) / jnp.sqrt(2. * jnp.pi))

    ns = NestedSampler(loglikelihood, prior_chain, sampler_name='ellipsoid')
    results = ns(random.PRNGKey(0), 100, max_samples=1e5,
                 collect_samples=True,
                 termination_frac=0.05,
                 stoachastic_uncertainty=True)
    plot_diagnostics(results)
    plot_cornerplot(results)