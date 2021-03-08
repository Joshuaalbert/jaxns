from jax import numpy as jnp

from jaxns.gaussian_process import TomographicKernel
from jaxns.prior_transforms import DeterministicTransformPrior, NormalPrior, UniformPrior, GaussianProcessKernelPrior, \
    MVNPrior, PriorChain


def build_frozen_flow_prior(X, kernel, tec_to_dtec, x0):
    v_dir = DeterministicTransformPrior('v_dir', lambda n: n / jnp.linalg.norm(n), (3,),
                                        NormalPrior('num_options', jnp.zeros(3), jnp.ones(3),
                                                    tracked=False), tracked=False)
    v_mag = UniformPrior('v_mag', 0., 0.5, tracked=False)
    v = DeterministicTransformPrior('v', lambda v_dir, v_mag: v_mag * v_dir,
                                    (3,), v_dir, v_mag, tracked=True)
    X_frozen_flow = DeterministicTransformPrior('X',
                                                lambda v: X[:, 0:6] - jnp.concatenate([v, jnp.zeros(3)]) * X[:, 6:7],
                                                X[:, 0:6].shape, v, tracked=False)
    K = GaussianProcessKernelPrior('K',
                                   TomographicKernel(x0, kernel, S_marg=20, S_gamma=10),
                                   X_frozen_flow,
                                   UniformPrior('height', 100., 300.),
                                   UniformPrior('width', 50., 150.),
                                   UniformPrior('l', 0., 20.),
                                   UniformPrior('sigma', 0., 2.), tracked=False)
    tec = MVNPrior('tec', jnp.zeros((X.shape[0],)), K, ill_cond=True, tracked=False)
    dtec = DeterministicTransformPrior('dtec', tec_to_dtec, tec.to_shape, tec, tracked=False)
    prior_chain = PriorChain() \
        .push(dtec) \
        .push(UniformPrior('uncert', 0., 5.))
    return prior_chain