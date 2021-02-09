from jax import numpy as jnp

from jaxns.gaussian_process.kernels import Kernel
from jaxns.prior_transforms.common import DeltaPrior
from jaxns.prior_transforms.prior_chain import PriorTransform
from jaxns.prior_transforms.prior_utils import get_shape


class DeterministicTransformPrior(PriorTransform):
    def __init__(self, name, transform, to_shape, *params, tracked=True):
        params = list(params)
        for i, param in enumerate(params):
            if not isinstance(param, PriorTransform):
                params[i] = DeltaPrior('_{}_param[{:d}]'.format(name, i), param, False)
        self._to_shape = to_shape
        self._transform = transform
        super(DeterministicTransformPrior, self).__init__(name, 0, params, tracked)

    @property
    def to_shape(self):
        return self._to_shape

    def forward(self, U, *params, **kwargs):
        return self._transform(*params)


class TransposePrior(DeterministicTransformPrior):
    def __init__(self, name, dist, tracked=True):
        if not isinstance(dist, PriorTransform):
            dist = DeltaPrior('_{}_dist'.format(name), dist, False)
        self._shape = get_shape(dist)[::-1]
        super(TransposePrior, self).__init__(name, lambda x: jnp.transpose(x), get_shape(dist)[::-1],
                                             [dist], tracked=tracked)


class GaussianProcessKernelPrior(DeterministicTransformPrior):
    def __init__(self, name, kernel: Kernel, X, *gp_params, Xstar=None, tracked=False):
        gp_params = [X] + list(gp_params)
        def _transform(X, *gp_params):
            if Xstar is None:
                return kernel(X, X, *gp_params) + 1e-6 * jnp.eye(X.shape[0])
            else:
                return kernel(X, Xstar, *gp_params)
        to_shape = (get_shape(X)[0], get_shape(X)[0])
        super(GaussianProcessKernelPrior, self).__init__(name, _transform, to_shape, *gp_params, tracked=tracked)