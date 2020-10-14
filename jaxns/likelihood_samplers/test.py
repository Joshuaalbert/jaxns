from jaxns.prior_transforms import UniformPrior, PriorChain
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering
from jax import numpy as jnp, vmap,random,jit, disable_jit
import pylab as plt

def main():
    def log_likelihood(theta, **kwargs):
        return (2. + jnp.prod(jnp.cos(0.5 * theta))) ** 5

    prior_chain = PriorChain() \
        .push(UniformPrior('theta', low=jnp.zeros(2), high=jnp.pi * 10. * jnp.ones(2)))

    U = vmap(lambda key: random.uniform(key, (prior_chain.U_ndims,)))(random.split(random.PRNGKey(0), 700))
    theta = vmap(lambda u: prior_chain(u))(U)
    lik = vmap(lambda theta: log_likelihood(**theta))(theta)

    select = lik > 150.
    print("Selecting", jnp.sum(select), "need", 18*3)
    log_VS = jnp.log(jnp.sum(select)/select.size)
    print("V(S)",jnp.exp(log_VS))

    U = U[select, :]

    with disable_jit():
        cluster_id, ellipsoid_parameters = \
            jit(lambda key, points, log_VS: ellipsoid_clustering(random.PRNGKey(0), points, 7, log_VS)
                )(random.PRNGKey(0), U, log_VS)
        mu, radii, rotation = ellipsoid_parameters

    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)

    for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
        y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        plt.plot(y[0, :], y[1, :])
        mask = cluster_id == i
        plt.scatter(U[mask, 0], U[mask, 1], c=jnp.atleast_2d(plt.cm.jet(i / len(ellipsoid_parameters))))

    plt.show()



if __name__ == '__main__':
    main()