from jax import numpy as jnp

from jaxns.gaussian_process import TomographicKernel
from jaxns.prior_transforms import GaussianProcessKernelPrior, UniformPrior, MVNPrior, DeterministicTransformPrior, \
    PriorChain


def build_prior(X, kernel, tec_to_dtec, x0, tec_conv):
    K = GaussianProcessKernelPrior('K',
                                   TomographicKernel(x0, kernel, S_marg=100, S_gamma=10), X,
                                   UniformPrior('height', 100., 300.),
                                   UniformPrior('width', 50., 150.),
                                   UniformPrior('l', 7., 20.),
                                   UniformPrior('sigma', 0.3, 2.), tracked=False)
    tec = MVNPrior('tec', jnp.zeros((X.shape[0],)), K, ill_cond=True, tracked=False)
    dtec = DeterministicTransformPrior('dtec', tec_to_dtec, tec.to_shape, tec, tracked=False)
    Y = DeterministicTransformPrior('Y', lambda dtec: jnp.concatenate([jnp.cos(dtec[:,None]*tec_conv), jnp.sin(dtec[:,None]*tec_conv)], axis=-1),
                                    dtec.to_shape+(tec_conv.size*2,), dtec, tracked=False)
    prior_chain = PriorChain() \
        .push(Y) \
        .push(UniformPrior('uncert', 0.01, 1.))
    return prior_chain