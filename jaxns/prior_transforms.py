from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import ndtri

class PriorTransform(object):

    @property
    def U_ndims(self):
        raise NotImplementedError()

    @property
    def to_shapes(self):
        raise NotImplementedError()

    def forward(self, U):
        raise NotImplementedError()

    def __call__(self, U):
        transformed_prior = self.forward(U)
        for name, v in transformed_prior.items():
            if name not in self.to_shapes.keys():
                raise ValueError("{} not a valid key.".format(name))
            if v.shape != self.to_shapes[name]:
                raise ValueError("Expected shape {}, got {}.".format(self.to_shapes[name], v.shape))
        return transformed_prior

class MVNDiagPrior(PriorTransform):
    def __init__(self, mu, gamma):
        mu, gamma = jnp.broadcast_arrays(mu, gamma)
        self.ndims = mu.size
        self.mu = mu
        self.gamma = gamma

    @property
    def U_ndims(self):
        return self.ndims

    @property
    def to_shapes(self):
        return dict(x=(self.ndims,))

    def forward(self, x):
        return dict(x=ndtri(x) * self.gamma + self.mu)

class LaplacePrior(PriorTransform):
    def __init__(self, mu, b):
        mu, b = jnp.broadcast_arrays(mu, b)
        self.ndims = mu.size
        self.mu = mu
        self.b = b

    @property
    def U_ndims(self):
        return self.ndims

    @property
    def to_shapes(self):
        return dict(x=(self.ndims,))

    def forward(self, x):
        return dict(x=self.mu - self.b * jnp.sign(x - 0.5)*jnp.log(1. - 2.*jnp.abs(x - 0.5)))

class UniformPrior(PriorTransform):
    def __init__(self, low, high):
        low, high = jnp.broadcast_arrays(low, high)
        self.ndims = low.size
        self.low = low
        self.high = high

    @property
    def U_ndims(self):
        return self.ndims

    @property
    def to_shapes(self):
        return dict(x=(self.ndims,))

    def forward(self, x):
        return dict(x=self.low + (self.high - self.low)*x)

class UniformUncertNormalParam(PriorTransform):
    def __init__(self, uncert_low, uncert_high, mu, gamma):
        self.uncert_prior = UniformPrior(uncert_low, uncert_high)
        self.param_prior = MVNDiagPrior(mu, gamma)

    @property
    def U_ndims(self):
        return self.uncert_prior.ndims + self.param_prior.ndims

    @property
    def to_shapes(self):
        return dict(x=(self.param_prior.ndims,), uncert=(self.uncert_prior.ndims,))

    def forward(self, u):
        print(self.param_prior.ndims)
        return dict(x=self.param_prior(u[:self.param_prior.ndims])['x'], uncert=self.uncert_prior(u[self.param_prior.ndims:])['x'])

class HMMPrior(PriorTransform):
    def __init__(self, x0_low, x0_high, omega_scale, T, dim):
        self.T = T
        self.dim = dim
        # self.x0_prior_transform = UniformPrior(x0_low, x0_high)
        self.x0_prior_transform = MVNDiagPrior(x0_low, x0_high)
        self.omega_prior_transform = MVNDiagPrior(0., omega_scale)

    @property
    def U_ndims(self):
        return self.dim + self.dim*(self.T-1)

    @property
    def to_shapes(self):
        return dict(x=(self.T, self.dim))

    def forward(self, U):
        """
        x[0] = U[0]
        [1:T+1] = U[1:T+1]
        Returns:

        """
        x0 = self.x0_prior_transform(U[:self.dim])['x']
        omega = vmap(self.omega_prior_transform)(U[self.dim:].reshape((self.T-1, self.dim)))['x']
        x = jnp.concatenate([x0[None,:], omega], axis=0)
        x = jnp.cumsum(x, axis=0)
        return dict(x=x)