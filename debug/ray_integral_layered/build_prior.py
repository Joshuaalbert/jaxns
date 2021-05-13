from jax import numpy as jnp

from jaxns.gaussian_process import TomographicKernel
from jaxns.prior_transforms import GaussianProcessKernelPrior, DeltaPrior, UniformPrior, CategoricalPrior, \
    DeterministicTransformPrior, MVNPrior, PriorChain


def build_layered_prior(X, kernel, x0, tec_to_dtec):

    layer_edges = jnp.linspace(80., 500., int((500. - 80.) / 50.) + 1)
    layer_kernels = []
    for i in range(len(layer_edges) - 1):
        height = 0.5 * (layer_edges[i] + layer_edges[i + 1])
        width = layer_edges[i + 1] - layer_edges[i]
        #Efficiency 0.39664684771546416
        # Time to run (including compile) 246.36920081824064
        # 0.39198953960498245
        # Time to run (no compile) 130.1565416753292
        # Efficiency normalised time 51.020025508433804
        K = GaussianProcessKernelPrior('K{}'.format(i),
                                       TomographicKernel(x0, kernel, S_marg=100, S_gamma=20), X,
                                       DeltaPrior('height{}'.format(i), height, tracked=False),
                                       DeltaPrior('width{}'.format(i), width, tracked=False),
                                       UniformPrior('l{}'.format(i), 7., 20., tracked=False),
                                       UniformPrior('sigma{}'.format(i), 0.3, 2., tracked=False), tracked=False)
        layer_kernels.append(K)
    logits = jnp.zeros(len(layer_kernels))
    select = CategoricalPrior('j', logits, tracked=True)
    K = DeterministicTransformPrior('K', lambda j, *K: jnp.stack(K, axis=0)[j[0], :, :], layer_kernels[0].to_shape,
                                    select, *layer_kernels, tracked=False)
    tec = MVNPrior('tec', jnp.zeros((X.shape[0],)), K, ill_cond=True, tracked=False)
    dtec = DeterministicTransformPrior('dtec', tec_to_dtec, tec.to_shape, tec, tracked=False)
    prior_chain = PriorChain() \
        .push(dtec) \
        .push(UniformPrior('uncert', 2., 3.))
    return prior_chain