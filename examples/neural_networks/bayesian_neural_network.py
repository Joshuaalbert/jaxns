try:
    import haiku as hk
except ImportError:
    print("You must `pip install dm-haiku` first.")

from jax import numpy as jnp, random, jit, vmap
import jax
from jax.flatten_util import ravel_pytree

from jaxns.prior_transforms import UniformPrior, PriorChain
from jaxns.nested_sampler.nested_sampling import NestedSampler
from jaxns.plotting import plot_diagnostics
from jaxns.utils import summary
from jax.scipy.optimize import minimize
from itertools import product


def boolean_func(x):
    """
    Computes the XOR operation on a sequence of bits.

    Examples:
        100 -> xor(xor(1,0),0) = 1
        001 -> xor(xor(0,0),1) = 1
        110 -> xor(xor(1,1),0) = 0
        011 -> xor(xor(0,1),1) = 0

    Args:
        x: boolean vector of bits.

    Returns:
        bool
    """
    output = x[0]
    for i in range(1, x.shape[-1]):
        output = jnp.logical_xor(output, x[i])
    return output

def main():
    num_variables = 8
    options = [True, False]
    x = jnp.asarray(list(product(options, repeat=num_variables)))#N,2
    y = vmap(boolean_func)(x)[:,None]#N, 1
    x = x.astype(jnp.float32)
    print("Data:")
    for input, output in zip(x,y):
        print(f"{input} -> {output}")

    def model(x, is_training=False):
        mlp = hk.Sequential([hk.Linear(4),
                             jax.nn.sigmoid,
                             hk.Linear(1)])
        return mlp(x)

    model = hk.without_apply_rng(hk.transform(model))
    init_params = model.init(random.PRNGKey(2345), x)

    init_params, unravel_func = ravel_pytree(init_params)
    n_dims = init_params.size
    print("initial params", init_params)

    def softplus(x):
        return jnp.log1p(jnp.exp(x))

    def log_likelihood(params, **kwargs):
        """
        log(P(y|p))
        p = exp(logits)/1 - exp(logits)
        = log(p) * y + log(1-p) * (1-y)
        = logits * y1 - log(exp(-logits)/(exp(-logits) - 1)) * y0
        """
        params_dict = unravel_func(params)
        logits = model.apply(params_dict, x)
        log_prob0, log_prob1 = -softplus(logits), -softplus(-logits)
        #log(p) * y + log(1-p) * (1-y)
        log_prob = jnp.mean(jnp.where(y, log_prob1, log_prob0))
        return jnp.asarray(log_prob, jnp.float64)



    params_bfgs = minimize(lambda p: -log_likelihood(p),
                       random.normal(random.PRNGKey(2435), shape=(n_dims,)),
                       method='BFGS').x
    print("Params BFGS", params_bfgs)
    print("log L_BFGS", log_likelihood(params_bfgs))

    with PriorChain() as prior_chain:
        UniformPrior('params', -10.*jnp.ones(n_dims), 10.*jnp.ones(n_dims))

    ns = NestedSampler(loglikelihood=log_likelihood, prior_chain=prior_chain, num_live_points=prior_chain.U_ndims*2,
                       sampler_kwargs=dict(num_slices=prior_chain.U_ndims//2,
                                           midpoint_shrink=False)
                       )

    # prior_chain.test_prior(random.PRNGKey(42), 100, log_likelihood)

    #we use termination_likelihood_frac to set a termination criterion, effectively turning nested sampling into a
    # powerful global maximum finder. Note, this can use a lot more iterations than normal nested sampling.
    ns = jit(ns)
    results = ns(random.PRNGKey(42),
                      termination_likelihood_frac=0.00001,
                      termination_evidence_frac=None)

    i_max = jnp.argmax(results.log_L_samples)
    params_max = results.samples['params'][i_max]
    print("Params max(L)", params_max)
    print("log L_max(L)", log_likelihood(params_max))

    summary(results)
    plot_diagnostics(results)
    # plot_cornerplot(results)

if __name__ == '__main__':
    main()