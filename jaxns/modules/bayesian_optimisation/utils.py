from jax import random, numpy as jnp, vmap



def latin_hypercube(key, num_samples, num_dim, cube_scale):
    """
    Sample from the latin-hypercube defined as the continuous analog of the discrete latin-hypercube.
    That is, if you partition each dimension into `num_samples` equal volume intervals then there is (conditionally)
    exactly one point in each interval. We guarantee that uniformity by randomly assigning the permutation of each dimension.
    The degree of randomness is controlled by `cube_scale`. A value of 0 places the sample at the center of the grid point,
    and a value of 1 places the value randomly inside the grid-cell.

    Args:
        key: PRNG key
        num_samples: number of samples in total to draw
        num_dim: number of dimensions in each sample
        cube_scale: The scale of randomness, in (0,1).

    Returns:
        latin-hypercube samples of shape [num_samples, num_dim]
    """
    key1, key2 = random.split(key, 2)
    cube_scale = jnp.clip(cube_scale, 0., 1.)
    samples = vmap(lambda key: random.permutation(key, num_samples))(random.split(key2, num_dim)).T
    samples += random.uniform(key1, shape=samples.shape, minval=0.5 - cube_scale / 2., maxval=0.5 + cube_scale / 2.)
    samples /= num_samples
    return samples