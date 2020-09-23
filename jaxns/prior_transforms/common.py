from jax import numpy as jnp
from jax.scipy.special import ndtri

from jaxns.prior_transforms.prior_chain import PriorTransform
from jaxns.prior_transforms.prior_utils import get_shape
from jaxns.utils import broadcast_shapes, msqrt, tuple_prod


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
    def __init__(self, name, mu, Gamma, ill_cond=False, tracked=True):
        self._ill_cond = ill_cond
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
        if self._ill_cond:
            L = msqrt(Gamma)
        else:
            L = jnp.linalg.cholesky(Gamma)
        return L @ ndtri(U) + mu


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


# class CategoricalPrior(PriorTransform):
#     def __init__(self, name, pi, tracked=True):
#         if not isinstance(pi, PriorTransform):
#             pi = DeltaPrior('_{}_pi'.format(name), jnp.atleast_1d(pi), False)
#         shape = ()
#         self._shape = ()
#         # replaces mu and gamma when parents injected
#         U_dims = 1
#         super(CategoricalPrior, self).__init__(name, U_dims, [pi], tracked)
#
#     @property
#     def to_shape(self):
#         return self._shape
#
#     def forward(self, U, pi, **kwargs):
#         j = jnp.argmax(U[0] * jnp.sum(pi)<= jnp.cumsum(pi))
#         return j

class CategoricalPrior(PriorTransform):
    def __init__(self, name, logits, tracked=True):
        if not isinstance(logits, PriorTransform):
            logits = DeltaPrior('_{}_logits'.format(name), jnp.atleast_1d(logits), False)
        self._shape = (1,)
        U_dims = get_shape(logits)[0]
        super(CategoricalPrior, self).__init__(name, U_dims, [logits], tracked)

    @property
    def to_shape(self):
        return self._shape

    def forward(self, U, logits, **kwargs):
        return jnp.argmax(logits - jnp.log(-jnp.log(jnp.maximum(U, jnp.finfo(U.dtype).eps))))[None]

class Gumbel(PriorTransform):
    def __init__(self, name, logits, tracked=True):
        if not isinstance(logits, PriorTransform):
            logits = DeltaPrior('_{}_logits'.format(name), jnp.atleast_1d(logits), False)
        self._shape = (get_shape(logits)[0],)
        U_dims = get_shape(logits)[0]
        super(Gumbel, self).__init__(name, U_dims, [logits], tracked)

    @property
    def to_shape(self):
        return self._shape

    def forward(self, U, logits, **kwargs):
        return -jnp.log(-jnp.log(jnp.maximum(U, jnp.finfo(U.dtype).eps)))
