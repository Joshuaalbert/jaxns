from jax import numpy as jnp


def make_coord_array(*X, flat=True, coord_map=None):
    """
    Create the design matrix from a list of coordinates
    :param X: list of length p of float, array [Ni, D]
        Ni can be different for each coordinate array, but D must be the same.
    :param flat: bool
        Whether to return a flattened representation
    :param coord_map: callable(coordinates), optional
            If not None then get mapped over the coordinates
    :return: float, array [N0,...,Np, D] if flat=False else [N0*...*Np, D]
        The coordinate design matrix
    """

    if coord_map is not None:
        X = [coord_map(x) for x in X]

    def add_dims(x, where, sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return jnp.tile(jnp.reshape(x, shape), tiles)

    N = [x.shape[0] for x in X]
    X_ = []

    for i, x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:, dim], [i], N))
    X = jnp.stack(X_, axis=-1)
    if not flat:
        return X
    return jnp.reshape(X, (-1, X.shape[-1]))


def product_log(w):
    """
    fifth order approximation to lambertw between -1/e and 0.
    Args:
        w:

    Returns:

    """
    Q = jnp.array([0., 1., -1., 3./2., -8./3.])
    E = jnp.exp(1.)

    P = jnp.array([-1., jnp.sqrt(2*E), - (2*E)/3., (11*E**1.5)/(18.*jnp.sqrt(2)), - (43*E**2)/135.,
                   + (769*E**2.5)/(2160.*jnp.sqrt(2)), -  (1768*E**3)/8505., (680863*E**3.5)/(2.7216e6*jnp.sqrt(2)),
                   - (3926*E**4)/25515., (226287557*E**4.5)/(1.1757312e9*jnp.sqrt(2)), -  (23105476*E**5)/1.89448875e8])
    return jnp.where(w > -0.5/E, jnp.polyval(Q[::-1], w), jnp.polyval(P[::-1], jnp.sqrt(jnp.exp(-1.) + w)))

