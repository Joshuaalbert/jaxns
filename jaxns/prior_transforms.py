from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import ndtri
from jaxns.utils import broadcast_shapes, iterative_topological_sort
from collections import OrderedDict

class Support(object):
    def check(self, value):
        """
        Check if value is inside support.
        """
        raise NotImplementedError()

class Reals(Support):
    def check(self, value):
        return jnp.all(jnp.isreal(value))

class PositiveReals(Reals):
    def check(self, value):
        return jnp.all(jnp.isreal(value) & (value >= 0))


class StrictlyPositiveReals(PositiveReals):
    def check(self, value):
        return jnp.all(jnp.isreal(value) & (value > 0))

class Integers(Support):
    def check(self, value):
        return jnp.all(jnp.floor(value) == value)

class PositiveIntegers(Integers):
    def check(self, value):
        return jnp.all((jnp.floor(value) == value) & value >= 0)

class StrictlyPositiveIntegers(PositiveIntegers):
    def check(self, value):
        return jnp.all((jnp.floor(value) == value) & value > 0)

class Boolean(PositiveIntegers):
    def check(self, value):
        return jnp.all((jnp.floor(value) == value) & ((value == 0) | (value == 1)))

class PriorTransform(object):
    #@TODO(joshuaalbert): add support of RVs (Reals, Reals>0, int, boolean, interval,...)
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
        Tuple describing the structure of the output.
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
            jnp.ndarray with shape given by to_shape
        """
        raise NotImplementedError()

    def __call__(self, U, *parents, **kwargs):
        """
        Calls the forward method and checks that names and shape match to_shapes.
        @TODO(joshuaalbert): add nan/type/support checking later.
        """
        transformed_prior = self.forward(U, *parents, **kwargs)
        if transformed_prior.shape != self.to_shape:
            raise ValueError("Expected shape {}, got {}.".format(self.to_shape, transformed_prior.shape))
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
            s.append("{} ~ {}".format(name, self.prior_chain[name]))
        return "\n".join(s)

    def __call__(self, U, **kwargs):
        dsk = {p.name : [parent.name for parent in p.parents] for p in self.prior_chain.values()}
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
        return "DeltaPrior({})".format(self.value)

    @property
    def to_shape(self):
        return self.value.shape

    def forward(self, U, **kwargs):
        return self.value

def get_shape(v):
    if isinstance(v, PriorTransform ):
        return v.to_shape
    if isinstance(v, jnp.ndarray):
        return v.shape
    return ()

class MVNDiagPrior(PriorTransform):
    def __init__(self, name, mu, gamma, tracked=True):
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), mu, False)
        if not isinstance(gamma, PriorTransform):
            gamma = DeltaPrior('_{}_gamma'.format(name), gamma, False)
        #replaces mu and gamma when parents injected
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
        #replaces mu and gamma when parents injected
        U_dims = broadcast_shapes(get_shape(mu), get_shape(Gamma)[0:1])[0]
        super(MVNPrior, self).__init__(name, U_dims, [mu, Gamma], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims,)

    def forward(self, U, mu, Gamma, **kwargs):
        L = jnp.linalg.cholesky(Gamma)
        return L @ ndtri(U) + mu

class GMMDiagPrior(PriorTransform):
    def __init__(self, name, pi, mu, gamma, tracked=True):
        if not isinstance(pi, PriorTransform):
            pi = DeltaPrior('_{}_pi'.format(name), mu, False)
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), mu, False)
        if not isinstance(gamma, PriorTransform):
            gamma = DeltaPrior('_{}_gamma'.format(name), gamma, False)
        assert (get_shape(pi)[0] == get_shape(mu)[0]) and (get_shape(pi)[0] == get_shape(gamma)[0])
        #replaces mu and gamma when parents injected
        U_dims = 1 + broadcast_shapes(get_shape(mu)[-1:], get_shape(gamma)[-1:])[0]
        super(GMMDiagPrior, self).__init__(name, U_dims, [pi, mu, gamma], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims-1,)

    def forward(self, U, pi, mu, gamma, **kwargs):
        j = jnp.argmax(U[0]<=jnp.cumsum(pi)/jnp.sum(pi))
        gamma = gamma[j,...]
        mu = mu[j,...]
        return gamma * ndtri(U[1:]) + mu

class GMMPrior(PriorTransform):
    def __init__(self, name, pi, mu, Gamma, tracked=True):
        if not isinstance(pi, PriorTransform):
            pi = DeltaPrior('_{}_pi'.format(name), mu, False)
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), mu, False)
        if not isinstance(Gamma, PriorTransform):
            Gamma = DeltaPrior('_{}_Gamma'.format(name), Gamma, False)
        assert (get_shape(pi)[0] == get_shape(mu)[0]) and (get_shape(pi)[0] == get_shape(Gamma)[0])
        #replaces mu and gamma when parents injected
        U_dims = 1 + broadcast_shapes(get_shape(mu)[-1:], get_shape(Gamma)[-1:])[0]
        super(GMMPrior, self).__init__(name, U_dims, [pi, mu, Gamma], tracked)

    @property
    def to_shape(self):
        return (self.U_ndims-1,)

    def forward(self, U, pi, mu, Gamma, **kwargs):
        j = jnp.argmax(U[0]<=jnp.cumsum(pi)/jnp.sum(pi))
        Gamma = Gamma[j,...]
        mu = mu[j,...]
        L = jnp.linalg.cholesky(Gamma)
        return L @ ndtri(U[1:]) + mu

class LaplacePrior(PriorTransform):
    def __init__(self, name, mu, b, tracked=True):
        if not isinstance(mu, PriorTransform):
            mu = DeltaPrior('_{}_mu'.format(name), mu, False)
        if not isinstance(b, PriorTransform):
            b = DeltaPrior('_{}_b'.format(name), b, False)
        #replaces mu and gamma when parents injected
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
        #replaces mu and gamma when parents injected
        U_dims = get_shape(b)[0]
        super(HalfLaplacePrior, self).__init__(name, U_dims, [b], tracked)


    @property
    def to_shape(self):
        return (self.U_ndims,)

    def forward(self, U, b, **kwargs):
        return  - b * jnp.sign(0.5*U) * jnp.log(1. - 2. * jnp.abs(0.5*U))

def test_half_laplace():
    p = PriorChain().push(HalfLaplacePrior('x',1.))
    U = jnp.linspace(0., 1., 100)[:, None]
    assert ~jnp.any(jnp.isnan(vmap(p)(U)['x']))


class UniformPrior(PriorTransform):
    def __init__(self, name, low, high, tracked=True):
        if not isinstance(low, PriorTransform):
            low = DeltaPrior('_{}_low'.format(name), low, False)
        if not isinstance(high, PriorTransform):
            high = DeltaPrior('_{}_high'.format(name), high, False)
        #replaces mu and gamma when parents injected
        U_dims = broadcast_shapes(get_shape(low), get_shape(high))[0]
        super(UniformPrior, self).__init__(name, U_dims, [low, high], tracked)


    @property
    def to_shape(self):
        return (self.U_ndims,)

    def forward(self, U, low, high, **kwargs):
        return low + U * (high - low)


class DiagGaussianWalkPrior(PriorTransform):
    def __init__(self, name, T, x0, omega, tracked=True):
        if not isinstance(x0, PriorTransform):
            x0 = DeltaPrior('_{}_x0'.format(name), x0, False)
        if not isinstance(omega, PriorTransform):
            omega = DeltaPrior('_{}_omega'.format(name), omega, False)
        #replaces mu and gamma when parents injected
        self.dim = broadcast_shapes(get_shape(x0), get_shape(omega))[0]
        self.T = T
        super(DiagGaussianWalkPrior, self).__init__(name, self.dim*self.T, [x0, omega], tracked)

    @property
    def to_shape(self):
        return (self.T, self.dim)

    def forward(self, U, x0, omega, **kwargs):
        return x0 + omega * jnp.cumsum(ndtri(U).reshape((self.T, -1)), axis=0)

def test_prior_chain():
    from jax import random
    chain = PriorChain()
    mu = MVNDiagPrior('mu', jnp.array([0.,0.]), 1.)
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