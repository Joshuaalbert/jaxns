from jax import numpy as jnp
from jax.lax import scan

from jaxns.prior_transforms.common import DeltaPrior
from jaxns.prior_transforms.prior_chain import PriorTransform
from jaxns.prior_transforms.prior_utils import get_shape
from jaxns.utils import broadcast_shapes, tuple_prod


class ForcedIdentifiabilityPrior(PriorTransform):
    """
    Prior for a sequence of `n` random variables uniformly distributed on U[low, high] such that X[i,...] <= X[i+1,...].
    For broadcasting the resulting random variable is sorted on the first dimension elementwise.
    """

    def __init__(self, name, n, low, high, tracked=True):
        if not isinstance(low, PriorTransform):
            low = DeltaPrior('_{}_low'.format(name), low, False)
        if not isinstance(high, PriorTransform):
            high = DeltaPrior('_{}_high'.format(name), high, False)
        self._n = n
        self._broadcast_shape = (self._n,) + broadcast_shapes(get_shape(low), get_shape(high))
        U_dims = tuple_prod(self._broadcast_shape)
        super(ForcedIdentifiabilityPrior, self).__init__(name, U_dims, [low, high], tracked)

    @property
    def to_shape(self):
        return self._broadcast_shape

    def forward(self, U, low, high, **kwargs):
        log_x = jnp.log(jnp.reshape(U, self.to_shape))

        # theta[i] = theta[i-1] * (1 - x[i]) + theta_max * x[i]
        def body(state, X):
            (log_theta,) = state
            (log_x, i) = X
            log_theta = log_x / i + log_theta
            return (log_theta,), (log_theta,)

        log_init_theta = jnp.zeros(broadcast_shapes(low.shape, high.shape))
        _, (log_theta,) = scan(body, (log_init_theta,), (log_x, jnp.arange(1, self._n + 1)), reverse=True)
        theta = low + (high - low) * jnp.exp(log_theta)
        return theta
