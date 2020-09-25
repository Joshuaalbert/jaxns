from jaxns.gaussian_process.tomographic_kernel.tomographic_kernel_utils import log_tomographic_weight_function, \
    gamma_min_max
import jax.numpy as jnp
from jax import vmap
from jaxns.gaussian_process.kernels import Kernel
from jax.scipy.special import logsumexp


class TomographicKernel(Kernel):
    def __init__(self, x0, fed_kernel, S_gamma=25, S_marg=25):
        self.S_gamma = S_gamma
        self.S_marg = S_marg
        self.x0 = x0
        self.fed_kernel = fed_kernel

    def __call__(self, x1, x2, height, width, l, sigma, *fed_kernel_params):
        a1 = x1[:, 0:3]
        k1 = x1[:, 3:6]
        a2 = x2[:, 0:3]
        k2 = x2[:, 3:6]
        return tomographic_kernel(a1, a2, k1, k2, self.x0, self.fed_kernel, height, width, l, sigma, *fed_kernel_params,
                                  S_gamma=self.S_gamma, S_marg=self.S_marg)


def log_tomographic_kernel(a1, a2, k1, k2, x0, fed_kernel, height, width, l, sigma, *fed_kernel_params, S_gamma=25,
                           S_marg=25):
    """
    Computes the tomographic kernel,
        I = b^2 secphi1 secphi2 int_0^1 int_0^1 K(x1+t1 p1 - x2 - t2 p2) dt1 dt2
        I = b^2 secphi1 secphi2 int_0^infty w(gamma) K(gamma) dgamma
    Args:
        key: jnp.ndarray
        a1: [N, 3]
        a2: [M, 3]
        k1: [N, 3]
        k2: [M, 3]
        x0: [3]
        fed_kernel: callable(r2, **params)
        height: ionosphere height, units of a1,a2
        width: ionosphere width, units of a1, a2
        l: FED kernel lengthscale
        **fed_kernel_params: kernel specific parameters

    Returns: [N, M]

    """
    kernel_act = fed_kernel.act
    kernel_inverse_x = fed_kernel.inverse_x
    # N
    s1 = (height - 0.5 * width - (a1[:, 2] - x0[2])) / k1[:, 2]
    # N, 3
    x1 = (a1 + s1[:, None] * k1)
    p1 = (k1 * width / k1[:, 2:3])
    # M
    s2 = (height - 0.5 * width - (a2[:, 2] - x0[2])) / k2[:, 2]
    # M, 3
    x2 = (a2 + s2[:, None] * k2)
    p2 = (k2 * width / k2[:, 2:3])

    # s = t*b sec phi + s0
    # ds = dt*b*sec phi
    # N,M
    log_conv = 2. * jnp.log(width) - (jnp.log(k1[:, 2:3]) + jnp.log(k2[None, :, 2]))

    # out until r = 5*half_width where the kernel should be really small
    # U
    def get_bins(x1,x2,p1,p2):
        g_min, g_max = gamma_min_max(x1,p1,x2,p2)
        bins = jnp.linspace(g_min,g_max,S_gamma)
        gamma = 0.5 * (bins[:-1] + bins[1:])
        log_w = log_tomographic_weight_function(bins, x1, x2, p1, p2, S=S_marg)
        log_dgamma = jnp.log(jnp.diff(bins))
        return log_dgamma, gamma, log_w

    #N, M, S_gamma
    log_dgamma, gamma, log_w = vmap(lambda x1,p1:
         vmap(lambda x2, p2: get_bins(x1,x2,p1,p2)
              )(x2,p2)
         )(x1,p1)

    # N,M,U
    log_K = kernel_act(gamma / l ** 2, sigma, *fed_kernel_params)
    # N,M
    return logsumexp(log_K + log_w + log_dgamma + log_conv[:, :, None], axis=-1)


def tomographic_kernel(a1, a2, k1, k2, x0, fed_kernel, height, width, l, sigma, *fed_kernel_params, S_gamma=25,
                       S_marg=25):
    """
    Computes the tomographic kernel,
        I = int_0^1 int_0^1 K(x1+t1 p1 - x2 - t2 p2) dt1 dt2
    Args:
        key: jnp.ndarray
        a1: [N, 3]
        a2: [M, 3]
        k1: [N, 3]
        k2: [M, 3]
        x0: [3]
        fed_kernel: callable(r2, **params)
        height: ionosphere height, units of a1,a2
        width: ionosphere width, units of a1, a2
        l: FED kernel lengthscale
        **fed_kernel_params: kernel specific parameters

    Returns: [N, M]

    """
    return jnp.exp(
        log_tomographic_kernel(a1, a2, k1, k2, x0, fed_kernel, height, width, l, sigma, *fed_kernel_params,
                               S_gamma=S_gamma, S_marg=S_marg))


def dtec_tomographic_kernel(key, ref_ant, a1, a2, k1, k2, x0, fed_kernel, height, width, l, *fed_kernel_params,
                            S_gamma=25, S_marg=25):
    return tomographic_kernel(key, a1, a2, k1, k2, x0, fed_kernel, height, width, l, *fed_kernel_params,
                              S_gamma=S_gamma, S_marg=S_marg) + \
           tomographic_kernel(key, ref_ant[None, :], ref_ant[None, :], k1, k2, x0, fed_kernel, height, width, l,
                              *fed_kernel_params, S_gamma=S_gamma, S_marg=S_marg) - \
           tomographic_kernel(key, ref_ant[None, :], a2, k1, k2, x0, fed_kernel, height, width, l, *fed_kernel_params,
                              S_gamma=S_gamma, S_marg=S_marg) - \
           tomographic_kernel(key, a1, ref_ant[None, :], k1, k2, x0, fed_kernel, height, width, l, *fed_kernel_params,
                              S_gamma=S_gamma, S_marg=S_marg)


def ddtec_tomographic_kernel(key, ref_dir, ref_ant, a1, a2, k1, k2, x0, fed_kernel, height, width, l,
                             *fed_kernel_params, S_gamma=25, S_marg=25, max_l=40.):
    return dtec_tomographic_kernel(key, ref_ant, a1, a2, k1, k2, x0, fed_kernel, height, width, l, *fed_kernel_params,
                                   S_gamma=S_gamma, S_marg=S_marg) + \
           dtec_tomographic_kernel(key, ref_ant, a1, a2, ref_dir[None, :], ref_dir[None, :], x0, fed_kernel, height,
                                   width, l, *fed_kernel_params, S_gamma=S_gamma, S_marg=S_marg) - \
           dtec_tomographic_kernel(key, ref_ant, a1, a2, ref_dir[None, :], k2, x0, fed_kernel, height, width, l,
                                   *fed_kernel_params, S_gamma=S_gamma, S_marg=S_marg) - \
           dtec_tomographic_kernel(key, ref_ant, a1, a2, k1, ref_dir[None, :], x0, fed_kernel, height, width, l,
                                   *fed_kernel_params, S_gamma=S_gamma, S_marg=S_marg)
