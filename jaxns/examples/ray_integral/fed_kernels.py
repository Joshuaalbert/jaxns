from jaxns.examples.ray_integral.utils import squared_norm
import jax.numpy as jnp


def diagonal(x1, x2, sigma):
    return jnp.square(sigma) * jnp.eye(x1.shape[0])

def rbf_act(r2, sigma):
    return 2. * jnp.log(sigma) - 0.5 * r2

def rbf(x1, x2, sigma, hpd):
    l = hpd * jnp.sqrt(2 * jnp.log(2.))
    r2 = squared_norm(x1/l, x2/l)
    return jnp.exp(rbf_act(r2, sigma))

def m12_act(r2, sigma):
    return 2. * jnp.log(sigma) - jnp.sqrt(r2)

def m12(x1, x2, sigma, hpd):
    l = hpd * jnp.log(2.)
    r2 = squared_norm(x1/l, x2/l)
    return jnp.exp(m12_act(r2, sigma))


def m32_act(r2, sigma):
    r = jnp.sqrt(r2)
    r *= jnp.sqrt(3.)
    log_res = 2. * jnp.log(sigma) + jnp.log(1. + r) - r
    return log_res

def m32(x1, x2, sigma, hpd):
    l = hpd / 1.032
    r2 = squared_norm(x1/l, x2/l)
    return jnp.exp(m32_act(r2, sigma))

def m52_act(r2, sigma):
    r = jnp.sqrt(r2)
    r *= jnp.sqrt(5.)
    r2 = r2 / 3.
    log_res = 2. * jnp.log(sigma) + jnp.log(1. + r + r2) - r
    return log_res

def m52(x1, x2, sigma, hpd):
    l = hpd / 0.95958
    r2 = squared_norm(x1/l, x2/l)
    return jnp.exp(m52_act(r2, sigma))

def rational_quadratic_act(r2, alpha, sigma):
    log_res = 2. * jnp.log(sigma) - alpha * jnp.log(1. + jnp.sqrt(r2) / (2. * alpha))
    return log_res

def rational_quadratic(x1, x2, sigma, alpha, hpd):
    l = hpd * (jnp.sqrt(2.) * jnp.sqrt(pow(jnp.sqrt(2.), 1. / alpha) - 1.) * jnp.sqrt(alpha))
    r2 = squared_norm(x1/l, x2/l)
    return jnp.exp(rational_quadratic_act(r2, alpha, sigma))



