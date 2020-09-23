from jax import numpy as jnp
from jax.scipy.special import ndtri

from jaxns.prior_transforms import ForcedIdentifiabilityPrior, SlicePrior
from jaxns.prior_transforms.common import DeltaPrior, UniformPrior
from jaxns.prior_transforms.prior_chain import PriorTransform
from jaxns.prior_transforms.prior_utils import get_shape
from jaxns.utils import broadcast_shapes, msqrt


class MixturePrior(PriorTransform):
    def __init__(self, name, transform, pi, *components, tracked=True):
        self._transform = transform
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
        j = jnp.argmax(U[0] * jnp.sum(pi)<= jnp.cumsum(pi))
        components = [c[j,...] for c in components]  # each components must be at least 1d
        return self._transform(*components)


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


class GMMPrior(PriorTransform):
    """
    More efficient version of a mixture of Gaussians because it avoids computing and stacking
    all components before selecting.
    """

    def __init__(self, name, pi, mu, Gamma, ill_cond:bool=False, tracked=True):
        self._ill_cond = ill_cond
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
        j = jnp.argmax(U[0] * jnp.sum(pi) <= jnp.cumsum(pi))
        Gamma = Gamma[j, ...]
        mu = mu[j, ...]
        if self._ill_cond:
            L = msqrt(Gamma)
        else:
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


