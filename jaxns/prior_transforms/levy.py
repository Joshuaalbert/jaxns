from jax import numpy as jnp
from jax.scipy.special import ndtri

from jaxns.prior_transforms.common import DeltaPrior
from jaxns.prior_transforms.prior_chain import PriorTransform
from jaxns.prior_transforms.prior_utils import get_shape
from jaxns.utils import broadcast_shapes


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

class SymmetricUniformWalkPrior(PriorTransform):
    def __init__(self, name, T, x0, half_width, tracked=True):
        if not isinstance(x0, PriorTransform):
            x0 = DeltaPrior('_{}_x0'.format(name), x0, False)
        if not isinstance(half_width, PriorTransform):
            half_width = DeltaPrior('_{}_half_width'.format(name), half_width, False)
        # replaces mu and gamma when parents injected
        self.dim = broadcast_shapes(get_shape(x0), get_shape(half_width))[0]
        self.T = T
        super(SymmetricUniformWalkPrior, self).__init__(name, self.dim * self.T, [x0, half_width], tracked)

    @property
    def to_shape(self):
        return (self.T, self.dim)

    def forward(self, U, x0, half_width, **kwargs):
        U_centered = (2.*half_width)*U - half_width
        return x0 + jnp.cumsum(U_centered.reshape((self.T, -1)), axis=0)