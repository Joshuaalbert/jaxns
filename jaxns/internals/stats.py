from jax import numpy as jnp, vmap

from jaxns.internals.log_semiring import LogSpace
from jaxns.internals.types import float_type


def normal_to_lognormal(mu, std):
    """
    Convert normal parameters to log-normal parameters.

    Args:
        mu: mean of normal RV
        std: standard deviation of normal RV

    Returns:
        mu, sigma of log-normal RV
    """
    var = std ** 2
    ln_mu = 2. * jnp.log(mu) - 0.5 * jnp.log(var)
    ln_var = jnp.log(var) - 2. * jnp.log(mu)
    return ln_mu, jnp.sqrt(ln_var)


def density_estimation(xstar, x, alpha=1. / 3., order=1):
    """
    Estimates the density of xstar given x using a trick.

    Args:
        xstar: array of points to estimate density at
        x: array of points to estimate density from
        alpha: power law exponent
        order: order of norm to use

    Returns:
        density at xstar
    """
    assert len(x.shape) == 2

    N = x.shape[0]
    m = int(pow(N, 1. - alpha))
    s = N // m
    N = m * s
    x = x[:N]

    def single_density(xstar):
        dist = jnp.linalg.norm(x - xstar, ord=order, axis=-1)  # N
        dist = jnp.reshape(dist, (s, m))  # s,m
        min_dist = jnp.min(dist, axis=0)  # m
        avg_dist = jnp.mean(min_dist)  # scalar
        return 0.5 / ((1. + s) * avg_dist)

    return vmap(single_density)(xstar)


def linear_to_log_stats(log_f_mean, *, log_f2_mean=None, log_f_var=None):
    """
    Converts normal to log-normal stats.
    Args:
        log_f_mean: log(E(f))
        log_f2_mean: log(E(f**2))
        log_f_var: log(Var(f))
    Returns:
        E(log(f))
        Var(log(f))
    """
    f_mean = LogSpace(log_f_mean)
    if log_f_var is not None:
        f_var = LogSpace(log_f_var)
        f2_mean = f_var + f_mean.square()
    else:
        f2_mean = LogSpace(log_f2_mean)
    mu = f_mean.square() / f2_mean.sqrt()
    sigma2 = f2_mean / f_mean.square()
    return mu.log_abs_val, jnp.maximum(sigma2.log_abs_val, jnp.finfo(float_type).eps)


def effective_sample_size(log_Z_mean, log_dZ2_mean):
    """
    Computes Kish's ESS = [sum dZ]^2 / [sum dZ^2]

    :param log_Z_mean:
    :param log_dZ2_mean:
    :return:
    """
    ess = LogSpace(log_Z_mean).square() / LogSpace(log_dZ2_mean)
    return ess.value
