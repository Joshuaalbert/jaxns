from jaxns.modules.gaussian_process.utils import product_log
from jaxns.internals.linalg import squared_norm
import jax.numpy as jnp


class Kernel(object):
    def __call__(self, x1, x2, *args):
        raise NotImplemented()


class Diagonal(Kernel):
    def __call__(self, x1, x2, sigma):
        return jnp.full((x1.shape[0], x1.shape[0]), sigma ** 2)


class StationaryKernel(Kernel):

    def inverse_x(self, p, l, sigma, *params):
        """
        Find x where K(x,0) == p
        Args:
            p:
            l:
            sigma:

        Returns:

        """
        return l * jnp.sqrt(2. * jnp.log(sigma ** 2 / p))

    def inverse_l(self, p, x, sigma):
        """
        Find l where K(x/l,0) == p
        Args:
            p:
            l:
            sigma:

        Returns:

        """
        return x / jnp.sqrt(2. * jnp.log(sigma ** 2 / p))

    def act(self, r2, sigma, *args):
        """
        Given the squared Euclidean norm of the outer difference between two coordinate sets, scaled by the lengthscale,
        return the log of the Q_kernel. Only valid for strictly positive kernels.

        Args:
            r2: [N, M]
            *args: other Q_kernel arguments, e.g. sigma, alpha, ...

        Returns: log(K(r2)) [N, M]

        """
        raise NotImplemented()

    def __call__(self, x1, x2, l, *args):
        r2 = squared_norm(x1/l, x2/l)
        r2 = jnp.maximum(r2, 1e-36)
        return jnp.exp(self.act(r2, *args))


class RBF(StationaryKernel):
    def inverse_x(self, p, l, sigma):
        return l * jnp.sqrt(2. * jnp.log(sigma ** 2 / p))

    def inverse_l(self, p, x, sigma):
        return x / jnp.sqrt(2. * jnp.log(sigma ** 2 / p))

    def act(self, r2, sigma):
        return 2. * jnp.log(sigma) - 0.5 * r2


class RationalQuadratic(StationaryKernel):
    def inverse_x(self, p, l, sigma, alpha):
        return l * jnp.sqrt(2. * alpha * ((p / sigma ** 2) ** (-1. / alpha) - 1.))

    def inverse_l(self, p, x, sigma, alpha):
        return x / jnp.sqrt(2. * alpha * ((p / sigma ** 2) ** (-1. / alpha) - 1.))

    def act(self, r2, sigma, alpha):
        log_res = 2. * jnp.log(sigma) - alpha * jnp.log(1. + jnp.sqrt(r2) / (2. * alpha))
        return log_res


class M12(StationaryKernel):
    def inverse_x(self, p, l, sigma):
        return l * jnp.log(sigma ** 2 / p)

    def inverse_l(self, p, x, sigma):
        return x / jnp.log(sigma ** 2 / p)

    def act(self, r2, sigma):
        return 2. * jnp.log(sigma) - jnp.sqrt(r2)


class M32(StationaryKernel):
    def inverse_x(self, p, l, sigma):
        return l * (product_log(-p / jnp.exp(1.) / sigma ** 2) + 1.) / (-jnp.sqrt(3.))

    def inverse_l(self, p, x, sigma):
        return x * (-jnp.sqrt(3.)) / (product_log(-p / jnp.exp(1.) / sigma ** 2) + 1.)

    def act(self, r2, sigma):
        r = jnp.sqrt(r2)
        r *= jnp.sqrt(3.)
        log_res = 2. * jnp.log(sigma) + jnp.log(1. + r) - r
        return log_res


class M52(StationaryKernel):
    def inverse_x(self, p, l, sigma):
        return l * (product_log(-p / jnp.exp(1.) / sigma ** 2) + 1.) / (-jnp.sqrt(3.))

    def inverse_l(self, p, x, sigma):
        return x * (-jnp.sqrt(3.)) / (product_log(-p / jnp.exp(1.) / sigma ** 2) + 1.)

    def act(self, r2, sigma):
        r = jnp.sqrt(r2)
        r *= jnp.sqrt(3.)
        log_res = 2. * jnp.log(sigma) + jnp.log(1. + r) - r
        return log_res
