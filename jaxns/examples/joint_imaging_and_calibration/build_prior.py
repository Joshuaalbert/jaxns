from jaxns.prior_transforms import PriorChain, NormalPrior, DeterministicTransformPrior
import jax.numpy as jnp

def build_prior(nant, ndir):
    theta = NormalPrior('theta', jnp.zeros(nant * ndir), jnp.ones(nant * ndir), tracked=True)
    gamma = NormalPrior('gamma', jnp.zeros(ndir), 0. * jnp.ones(ndir), tracked=True)

    def vis(theta, gamma, **kwargs):
        theta = theta.reshape((nant, ndir))
        diff = 1j * (theta[:, None, :] - theta)
        delta = jnp.mean(jnp.exp(-gamma + diff), axis=-1)
        return delta

    delta = DeterministicTransformPrior('delta', vis, (nant, nant), theta, gamma, tracked=False)
    prior = PriorChain().push(delta)
    return prior