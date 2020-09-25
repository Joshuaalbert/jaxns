from jax import numpy as jnp

from jaxns.prior_transforms.prior_chain import PriorTransform


def get_shape(v):
    if isinstance(v, PriorTransform):
        return v.to_shape
    if isinstance(v, jnp.ndarray):
        return v.shape
    return ()
