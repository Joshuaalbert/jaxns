from jax import numpy as jnp

from jaxns.gaussian_process.kernels import Kernel
from jaxns.prior_transforms.common import HierarchicalPrior
from jaxns.prior_transforms.prior_chain import PriorBase
from jaxns.prior_transforms.prior_utils import get_shape, prior_docstring


class DeterministicTransformPrior(HierarchicalPrior):
    @prior_docstring
    def __init__(self, name, transform, to_shape, *params, tracked=True, dtype=jnp.float_):
        """
        Transform a set of Prior RVs, deterministically.

        Args:
            transform: callable(*params, **kwargs)
            to_shape: shape that transform returns
            *params: list of Priors or arrays that transform accepts.
        """
        params = [self._prepare_parameter(name, 'param{}'.format(i), params[i]) for i in range(len(params))]
        self._to_shape = to_shape
        self._transform = transform
        super(DeterministicTransformPrior, self).__init__(name, to_shape, params, tracked, PriorBase((), dtype), dtype=dtype)

    def transform_U(self, U, *params, **kwargs):
        del U
        return self._transform(*params)


class GaussianProcessKernelPrior(DeterministicTransformPrior):
    @prior_docstring
    def __init__(self, name, kernel: Kernel, X, *gp_params, tracked=False):
        """
        Gaussian process kernel with priors on hyper-parameters. A small diagonal term is added to the kernel for
        stability.

        Args:
            kernel: Kernel to be applied to the below params.
            X: locations
            *gp_params: parameters that the kernel accepts.
        """
        gp_params = [X] + list(gp_params)

        def _transform(X, *gp_params):
            return kernel(X, X, *gp_params) + 1e-6 * jnp.eye(X.shape[0])

        to_shape = (get_shape(X)[0], get_shape(X)[0])
        super(GaussianProcessKernelPrior, self).__init__(name, _transform, to_shape, *gp_params, tracked=tracked)
