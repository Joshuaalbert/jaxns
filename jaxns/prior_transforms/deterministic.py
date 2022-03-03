from jax import numpy as jnp

from jaxns.modules.gaussian_process.kernels import Kernel
from jaxns.prior_transforms.common import HierarchicalPrior
from jaxns.prior_transforms import prior_docstring, PriorBase


class DeterministicTransformPrior(HierarchicalPrior):
    @prior_docstring
    def __init__(self, name, transform, *params, tracked=True):
        """
        Transform a set of Prior RVs, deterministically.

        Args:
            transform: callable(*params, **kwargs)
            to_shape: shape that transform returns
            *params: list of Priors or arrays that transform accepts.
        """
        params = [self._prepare_parameter(name, 'param{}'.format(i), params[i]) for i in range(len(params))]
        self._transform = transform
        super(DeterministicTransformPrior, self).__init__(name, params, tracked, PriorBase())

    def transform_U(self, U, *params, **kwargs):
        del U
        return self._transform(*params)


class GaussianProcessKernelPrior(DeterministicTransformPrior):
    @prior_docstring
    def __init__(self, name, kernel: Kernel, X, *gp_params, tracked=False):
        """
        Gaussian process Q_kernel with priors on hyper-parameters. A small diagonal term is added to the Q_kernel for
        stability.

        Args:
            kernel: Kernel to be applied to the below params.
            X: locations
            *gp_params: parameters that the Q_kernel accepts.
        """
        gp_params = [X] + list(gp_params)

        def _transform(X, *gp_params):
            return kernel(X, X, *gp_params) + 1e-6 * jnp.eye(X.shape[0])

        super(GaussianProcessKernelPrior, self).__init__(name, _transform, *gp_params, tracked=tracked)
