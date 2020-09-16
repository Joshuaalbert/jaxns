from jax import numpy as jnp
from jax import vmap
from jax.lax import scan
from jax.scipy.special import ndtri
from jaxns.utils import broadcast_shapes, iterative_topological_sort, tuple_prod
from jaxns.gaussian_process.kernels import Kernel
from collections import OrderedDict


def get_shape(v):
    if isinstance(v, PriorTransform):
        return v.to_shape
    if isinstance(v, jnp.ndarray):
        return v.shape
    return ()


class PriorTransform(object):
    def __init__(self, name, U_ndims, parents, tracked):
        self._name = name
        self._U_ndims = U_ndims
        self._parents = list(parents)
        self._tracked = tracked

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ",".join([p.name for p in self.parents]))

    @property
    def parents(self):
        return self._parents

    @property
    def name(self):
        return self._name

    @property
    def U_ndims(self):
        return self._U_ndims

    @property
    def tracked(self):
        return self._tracked

    @property
    def to_shape(self):
        """
        tuple providing the shape of the output.
        """
        raise NotImplementedError()

    def forward(self, U, *parents, **kwargs):
        """
        Transforms a vector of i.i.d. Uniform random variables to another random variable.
        The transformation may be dependent on any ancestor RVs.
        Args:
            U: [U_ndims] flat vector drawn from U[0,1]
            **ancestors: arbitrary dict of RVs early in the chain.
        Returns:
            jnp.ndarray with shape_dict given by to_shape
        """
        raise NotImplementedError()

    def __call__(self, U, *parents, **kwargs):
        """
        Calls the forward method and checks that names and shape_dict match to_shapes.
        """
        transformed_prior = self.forward(U, *parents, **kwargs)
        if transformed_prior.shape != self.to_shape:
            raise ValueError("Expected shape_dict {}, got {}.".format(self.to_shape, transformed_prior.shape))
        return transformed_prior


class PriorChain(object):
    def __init__(self):
        self.prior_chain = OrderedDict()

    def push(self, prior: PriorTransform):
        """
        Pushs a prior onto the chain that can be used by all descendents.
        """
        if not isinstance(prior, PriorTransform):
            raise ValueError("Trying to push {} onto chain".format(type(prior)))
        self.prior_chain[prior.name] = prior
        for parent in prior.parents:
            self.push(parent)
        return self

    @property
    def U_ndims(self):
        return sum([p.U_ndims for p in self.prior_chain.values()])

    @property
    def to_shapes(self):
        shape = OrderedDict()
        for p in self.prior_chain.values():
            shape[p.name] = p.to_shape
        return shape

    def __repr__(self):
        dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(dsk)[::-1]
        s = []
        for name in forward_topo_sort:
            s.append("{} ~ {} : {}".format(name, self.prior_chain[name].to_shape, self.prior_chain[name]))
        return "\n".join(s)

    def __call__(self, U, **kwargs):
        dsk = {p.name: [parent.name for parent in p.parents] for p in self.prior_chain.values()}
        forward_topo_sort = iterative_topological_sort(dsk)[::-1]
        idx = 0
        for name in forward_topo_sort:
            p = self.prior_chain[name]
            input = U[idx:idx + p.U_ndims]
            idx += p.U_ndims
            parents = [dsk[parent] for parent in dsk[name]]
            dsk[p.name] = p(input, *parents, **kwargs)
        transformed_prior = dsk
        return transformed_prior


class DeltaPrior(PriorTransform):
    def __init__(self, name, value, tracked=False):
        super(DeltaPrior, self).__init__(name, 0, [], tracked)
        self.value = jnp.atleast_1d(jnp.asarray(value))

    def __repr__(self):
        return "DeltaPrior({})".format(self.value if self.value.size == 1 else "array<{}>".format(self.value.shape))

    @property
    def to_shape(self):
        return self.value.shape

    def forward(self, U, **kwargs):
        return self.value


class MVNDiagPrior(PriorTransform):
    def __init__(self, name, mu, gamma, tracked=True):
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), mu, False)
        if not isinstance(gamma, PriorTransform):
            gamma = DeltaPrior('_{}_gamma'.format(name), gamma, False)
        # replaces mu and gamma when parents injected
        U_dims = broadcast_shapes(get_shape(mu), get_shape(gamma))[0]
        super(MVNDiagPrior, self).__init__(name, U_dims, [mu, gamma], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims,)

    def forward(self, U, mu, gamma, **kwargs):
        return ndtri(U) * gamma + mu


class MVNPrior(PriorTransform):
    def __init__(self, name, mu, Gamma, tracked=True):
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), mu, False)
        if not isinstance(Gamma, PriorTransform):
            Gamma = DeltaPrior('_{}_Gamma'.format(name), Gamma, False)
        # replaces mu and gamma when parents injected
        U_dims = broadcast_shapes(get_shape(mu), get_shape(Gamma)[0:1])[0]
        super(MVNPrior, self).__init__(name, U_dims, [mu, Gamma], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims,)

    def forward(self, U, mu, Gamma, **kwargs):
        L = jnp.linalg.cholesky(Gamma)
        return L @ ndtri(U) + mu


class MixturePrior(PriorTransform):
    def __init__(self, name, pi, *components, tracked=True):
        if not isinstance(pi, PriorTransform):
            pi = DeltaPrior('_{}_pi'.format(name), pi, False)
        assert (get_shape(pi)[0] == len(components))
        shape = ()
        for component in components:
            assert isinstance(component, PriorTransform)
            shape = broadcast_shapes(shape, component.to_shape)
        self._shape = shape
        # replaces mu and gamma when parents injected
        U_dims = 1
        super(MixturePrior, self).__init__(name, U_dims, [pi] + components, tracked)

    @property
    def to_shape(self):
        return self._shape

    def forward(self, U, pi, *components, **kwargs):
        j = jnp.argmax(U[0] <= jnp.cumsum(pi) / jnp.sum(pi))
        components = jnp.stack(components, axis=0)  # each components must be the same size
        return components[j, ...]


class GMMDiagPrior(PriorTransform):
    """
    More efficient version of a mixture of diagonal Gaussians because it avoids computing and stacking 
    all components before selecting.
    """

    def __init__(self, name, pi, mu, gamma, tracked=True):
        if not isinstance(pi, PriorTransform):
            pi = DeltaPrior('_{}_pi'.format(name), pi, False)
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), jnp.atleast_2d(mu), False)
        if not isinstance(gamma, PriorTransform):
            gamma = DeltaPrior('_{}_gamma'.format(name), jnp.atleast_2d(gamma), False)
        assert (get_shape(pi)[0] == get_shape(mu)[0]) and (get_shape(pi)[0] == get_shape(gamma)[0]) \
               and (get_shape(mu)[1] == get_shape(gamma)[1])
        # replaces mu and gamma when parents injected
        U_dims = 1 + broadcast_shapes(get_shape(mu)[-1:], get_shape(gamma)[-1:])[0]
        super(GMMDiagPrior, self).__init__(name, U_dims, [pi, mu, gamma], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims - 1,)

    def forward(self, U, pi, mu, gamma, **kwargs):
        j = jnp.argmax(U[0] <= jnp.cumsum(pi) / jnp.sum(pi))
        gamma = gamma[j, ...]
        mu = mu[j, ...]
        return gamma * ndtri(U[1:]) + mu


class GMMMarginalPrior(GMMDiagPrior):
    def __init__(self, name, num_components, low, high, tracked=True):
        pi = UniformPrior(f'_{name}_pi', jnp.zeros(num_components), jnp.ones(num_components),
                          tracked=True)
        mean = ForcedIdentifiabilityPrior(f'_{name}_mean',
                                          num_components,
                                          low,
                                          high,
                                          tracked=True)
        gamma_max = (high - low) / num_components * jnp.ones((num_components, 1))
        # gamma = ForcedIdentifiabilityPrior(f'_{name}_gamma',
        #                                   num_components,
        #                                   jnp.zeros_like(gamma_max),
        #                                   gamma_max,
        #                                   tracked=False)
        super(GMMMarginalPrior, self).__init__(name, pi, mean, gamma_max, tracked=tracked)


class GMMPrior(PriorTransform):
    """
    More efficient version of a mixture of Gaussians because it avoids computing and stacking
    all components before selecting.
    """

    def __init__(self, name, pi, mu, Gamma, tracked=True):
        if not isinstance(pi, PriorTransform):
            pi = DeltaPrior('_{}_pi'.format(name), pi, False)
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), jnp.atleast_2d(mu), False)
        if not isinstance(Gamma, PriorTransform):
            Gamma = DeltaPrior('_{}_Gamma'.format(name), jnp.atleast_3d(Gamma), False)
        assert (get_shape(pi)[0] == get_shape(mu)[0]) and (get_shape(pi)[0] == get_shape(Gamma)[0]) \
               and (get_shape(mu)[1] == get_shape(Gamma)[2])
        # replaces mu and gamma when parents injected
        U_dims = 1 + broadcast_shapes(get_shape(mu)[-1:], get_shape(Gamma)[-1:])[0]
        super(GMMPrior, self).__init__(name, U_dims, [pi, mu, Gamma], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims - 1,)

    def forward(self, U, pi, mu, Gamma, **kwargs):
        j = jnp.argmax(U[0] <= jnp.cumsum(pi) / jnp.sum(pi))
        Gamma = Gamma[j, ...]
        mu = mu[j, ...]
        L = jnp.linalg.cholesky(Gamma)
        return L @ ndtri(U[1:]) + mu


class UniformMixturePrior(PriorTransform):
    def __init__(self, name, pi, low, high, tracked=True):
        if not isinstance(pi, PriorTransform):
            pi = DeltaPrior('_{}_pi'.format(name), pi, False)
        if not isinstance(low, PriorTransform):
            low = DeltaPrior('_{}_low'.format(name), jnp.atleast_2d(low), False)
        if not isinstance(high, PriorTransform):
            high = DeltaPrior('_{}_high'.format(name), jnp.atleast_2d(high), False)
        assert (get_shape(pi)[0] == get_shape(low)[0]) and (get_shape(pi)[0] == get_shape(high)[0]) \
               and (get_shape(low)[1] == get_shape(high)[1])
        # replaces mu and high when parents injected
        U_dims = 1 + broadcast_shapes(get_shape(low)[-1:], get_shape(high)[-1:])[0]
        super(UniformMixturePrior, self).__init__(name, U_dims, [pi, low, high], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims - 1,)

    def forward(self, U, pi, low, high, **kwargs):
        j = jnp.argmax(U[0] <= jnp.cumsum(pi) / jnp.sum(pi))
        low = low[j, ...]
        high = high[j, ...]
        return low + (high - low) * U[1:]


class MultiCubeMixturePrior(UniformMixturePrior):
    """
    Creates a UniformMixturePrior with num_components components.

    The mixture parameter, pi, is ordered from smallest to largest such that pi[i] <= pi[i+1].
    pi ~ ForcedIdentifiabilityPrior(num_components, 0., 1.)

    Each components is a Uniform on some retangular regions of [0,1]^D
    low
    X[i] ~ U[low[i], high[i]]

    j ~ MultiNomial[pi]
    Y ~ X[j]

    """

    def __init__(self, name, num_components, num_dims, low, high):
        pi = ForcedIdentifiabilityPrior(f'_{name}_pi', num_components, 0., 1., tracked=True)
        low_high = ForcedIdentifiabilityPrior(f'_{name}_low_high',
                                              2,
                                              low * jnp.ones((num_components, num_dims)),
                                              high * jnp.ones((num_components, num_dims)), tracked=False)
        low = SlicePrior(f'_{name}_low', 0, low_high, tracked=True)
        high = SlicePrior(f'_{name}_high', 1, low_high, tracked=True)
        super(MultiCubeMixturePrior, self).__init__(name, pi, low, high, tracked=True)


class UnitCubeMixturePrior(MultiCubeMixturePrior):
    """
    Creates a UniformMixturePrior with num_components components.

    The mixture parameter, pi, is ordered from smallest to largest such that pi[i] <= pi[i+1].
    pi ~ ForcedIdentifiabilityPrior(num_components, 0., 1.)

    Each components is a Uniform on some retangular regions of [0,1]^D
    low
    X[i] ~ U[low[i], high[i]]

    j ~ MultiNomial[pi]
    Y ~ X[j]

    """

    def __init__(self, name, num_components, num_dims):
        super(UnitCubeMixturePrior, self).__init__(name, num_components,
                                                   num_dims, 0., 1., tracked=True)


def test_unit_cube_mixture_prior():
    import jax.numpy as jnp
    from jax import random
    import pylab as plt
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


class LaplacePrior(PriorTransform):
    def __init__(self, name, mu, b, tracked=True):
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), mu, False)
        if not isinstance(b, PriorTransform):
            b = DeltaPrior('_{}_b'.format(name), b, False)
        # replaces mu and gamma when parents injected
        U_dims = broadcast_shapes(get_shape(mu), get_shape(b))[0]
        super(LaplacePrior, self).__init__(name, U_dims, [mu, b], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims,)

    def forward(self, U, mu, b, **kwargs):
        return mu - b * jnp.sign(U - 0.5) * jnp.log(1. - 2. * jnp.abs(U - 0.5))


class HalfLaplacePrior(PriorTransform):
    def __init__(self, name, b, tracked=True):
        if not isinstance(b, PriorTransform):
            b = DeltaPrior('_{}_b'.format(name), b, False)
        # replaces mu and gamma when parents injected
        U_dims = get_shape(b)[0]
        super(HalfLaplacePrior, self).__init__(name, U_dims, [b], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims,)

    def forward(self, U, b, **kwargs):
        return - b * jnp.sign(0.5 * U) * jnp.log(1. - 2. * jnp.abs(0.5 * U))


def test_half_laplace():
    p = PriorChain().push(HalfLaplacePrior('x', 1.))
    U = jnp.linspace(0., 1., 100)[:, None]
    assert ~jnp.any(jnp.isnan(vmap(p)(U)['x']))


class SlicePrior(PriorTransform):
    def __init__(self, name, idx, dist, tracked=True):
        self._idx = idx
        self._shape = get_shape(dist)[1:]
        U_dims = 0
        super(SlicePrior, self).__init__(name, U_dims, [dist], tracked)

    @property
    def to_shape(self):
        return self._shape

    def forward(self, U, dist, **kwargs):
        return dist[self._idx, ...]


# class SlicePrior(PriorTransform):
#     def __init__(self, name, idx, dist, tracked=True):
#         if not isinstance(idx, PriorTransform):
#             idx = DeltaPrior('_{}_idx'.format(name), idx, False)
#         if not isinstance(dist, PriorTransform):
#             dist = DeltaPrior('_{}_dist'.format(name), dist, False)
#
#         self._shape = get_shape(dist)[1:]
#         U_dims = 0
#         super(SlicePrior, self).__init__(name, U_dims, [idx, dist], tracked)
#
#     @property
#     def to_shape(self):
#         return self._shape
#
#     def forward(self, U, idx, dist, **kwargs):
#         return dist[idx,...]

class TransposePrior(PriorTransform):
    def __init__(self, name, dist, tracked=True):
        if not isinstance(dist, PriorTransform):
            dist = DeltaPrior('_{}_dist'.format(name), dist, False)

        self._shape = get_shape(dist)[::-1]
        U_dims = 0
        super(TransposePrior, self).__init__(name, U_dims, [dist], tracked)

    @property
    def to_shape(self):
        return self._shape

    def forward(self, U, dist, **kwargs):
        return jnp.transpose(dist)


class UniformPrior(PriorTransform):
    def __init__(self, name, low, high, tracked=True):
        if not isinstance(low, PriorTransform):
            low = DeltaPrior('_{}_low'.format(name), low, False)
        if not isinstance(high, PriorTransform):
            high = DeltaPrior('_{}_high'.format(name), high, False)
        # replaces mu and gamma when parents injected

        self._broadcast_shape = broadcast_shapes(get_shape(low), get_shape(high))
        U_dims = tuple_prod(self._broadcast_shape)
        super(UniformPrior, self).__init__(name, U_dims, [low, high], tracked)

    @property
    def to_shape(self):
        return self._broadcast_shape

    def forward(self, U, low, high, **kwargs):
        return low + jnp.reshape(U, self.to_shape) * (high - low)


class ForcedIdentifiabilityPrior(PriorTransform):
    """
    Prior for a sequence of `n` random variables uniformly distributed on U[low, high] such that X[i,...] <= X[i+1,...].
    For broadcasting the resulting random variable is sorted on the first dimension elementwise.
    """

    def __init__(self, name, n, low, high, tracked=True):
        if not isinstance(low, PriorTransform):
            low = DeltaPrior('_{}_low'.format(name), low, False)
        if not isinstance(high, PriorTransform):
            high = DeltaPrior('_{}_high'.format(name), high, False)
        self._n = n
        # replaces mu and gamma when parents injected

        self._broadcast_shape = (self._n,) + broadcast_shapes(get_shape(low), get_shape(high))
        U_dims = tuple_prod(self._broadcast_shape)
        super(ForcedIdentifiabilityPrior, self).__init__(name, U_dims, [low, high], tracked)

    @property
    def to_shape(self):
        return self._broadcast_shape

    def forward(self, U, low, high, **kwargs):
        log_x = jnp.log(jnp.reshape(U, self.to_shape))

        # theta[i] = theta[i-1] * (1 - x[i]) + theta_max * x[i]
        def body(state, X):
            (log_theta,) = state
            (log_x, i) = X
            log_theta = log_x / i + log_theta
            return (log_theta,), (log_theta,)

        log_init_theta = jnp.zeros(broadcast_shapes(low.shape, high.shape))
        _, (log_theta,) = scan(body, (log_init_theta,), (log_x, jnp.arange(1, self._n + 1)), reverse=True)
        theta = low + (high - low) * jnp.exp(log_theta)
        return theta


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


class DiagGaussianWalkPrior(PriorTransform):
    def __init__(self, name, T, x0, omega, tracked=True):
        if not isinstance(x0, PriorTransform):
            x0 = DeltaPrior('_{}_x0'.format(name), x0, False)
        if not isinstance(omega, PriorTransform):
            omega = DeltaPrior('_{}_omega'.format(name), omega, False)
        # replaces mu and gamma when parents injected
        self.dim = broadcast_shapes(get_shape(x0), get_shape(omega))[0]
        self.T = T
        super(DiagGaussianWalkPrior, self).__init__(name, self.dim * self.T, [x0, omega], tracked)

    @property
    def to_shape(self):
        return (self.T, self.dim)

    def forward(self, U, x0, omega, **kwargs):
        return x0 + omega * jnp.cumsum(ndtri(U).reshape((self.T, -1)), axis=0)


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


class GaussianProcessKernelPrior(PriorTransform):
    def __init__(self, name, kernel: Kernel, X, *gp_params, tracked=False):
        if not isinstance(X, PriorTransform):
            X = DeltaPrior('_{}_X'.format(name), X, False)
        gp_params = list(gp_params)
        for i, gp_param in enumerate(gp_params):
            if not isinstance(gp_param, PriorTransform):
                gp_params[i] = DeltaPrior('_{}_param[{:d}]'.format(name, i), gp_param, False)
        # replaces mu and gamma when parents injected
        self._kernel = kernel
        self._to_shape = (get_shape(X)[0], get_shape(X)[0])
        super(GaussianProcessKernelPrior, self).__init__(name, 0, [X] + gp_params, tracked)

    @property
    def to_shape(self):
        return self._to_shape

    def forward(self, U, X, *gp_params, **kwargs):
        return self._kernel(X, X, *gp_params) + 1e-6 * jnp.eye(X.shape[0])
