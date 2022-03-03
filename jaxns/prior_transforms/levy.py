from jax import numpy as jnp
from jax.scipy.special import ndtri

from jaxns.prior_transforms.common import ContinuousPrior
from jaxns.prior_transforms import prior_docstring, get_shape
from jaxns.internals.shapes import broadcast_shapes


class DiagGaussianWalkPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, T, x0, omega, tracked=True):
        """
        Random walk where the innovations are uncorrelated Gaussian.

        Args:
            T: int, number of steps
            x0: initial point
            omega: std-dev per step
        """
        x0 = self._prepare_parameter(name, 'x0', x0)
        omega = self._prepare_parameter(name, 'omega', omega)
        shape = (T,) + broadcast_shapes(get_shape(x0), get_shape(omega))
        self.T = T
        super(DiagGaussianWalkPrior, self).__init__(name, shape, [x0, omega], tracked)

    def transform_U(self, U, x0, omega, **kwargs):
        return x0 + omega*jnp.cumsum(ndtri(U), axis=0)


class SymmetricUniformWalkPrior(ContinuousPrior):
    @prior_docstring
    def __init__(self, name, T, x0, half_width, tracked=True):
        """
        Random walk where the innovations are symmetric Uniform steps.

        Args:
            T: int, number of steps
            x0: initial point
            half_width: half-interval
        """
        x0 = self._prepare_parameter(name, 'x0', x0)
        half_width = self._prepare_parameter(name, 'half_width', half_width)
        shape = (T,) + broadcast_shapes(get_shape(x0), get_shape(half_width))
        self.T = T
        super(SymmetricUniformWalkPrior, self).__init__(name, shape, [x0, half_width], tracked)

    def transform_U(self, U, x0, half_width, **kwargs):
        U_centered = (2. * half_width) * U - half_width
        return x0 + jnp.cumsum(U_centered, axis=0)
