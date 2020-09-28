![JAXNS](./jaxns_logo.png)

# What is it?
Enables probabilistic programming using nested sampling. It's coded in JAX in a manner that allows lowering the entire inference algorithm to XLA primatives, which are JIT compiled for high performance. 

JAXNS provides a modified version of the MultiNest constrained sampler (F. Feroz et al. 2008; https://arxiv.org/pdf/0809.3437.pdf).
In order to allow JIT compiling the MultiNest algorithm was redesigned as an iterative algorithm with fixed maximum depth.

# Install
Make sure you have JAX and the usual suspects with `pip install jax jaxlib numpy matplotlib scipy tensorflow tensorboard_plugin_profile`. The last two `tensorflow tensorboard_plugin_profile` is is for profiling, and can be neglected if you don't want to profile.
Install with `python setup.py install` or `pip install git+http://github.com/Joshuaalbert/jaxns.git`.

# Quick start

JAXNS is really fast because it uses JAX. I've found it's 3-4 orders of magnitude faster than other nested sampling packages.
The caveat is that you should define your likelihood function with JAX. This is no big deal because JAX is just a replacement for numpy. 
If you're unfamiliar, take a quick tour of JAX (https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).
See more examples in `jaxns/examples`.

```python
from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax.scipy.linalg import solve_triangular
from jax import random, jit, disable_jit
from jax import numpy as jnp
import pylab as plt

ndims = 2

# define highly correlated data
data_mu = jnp.ones(ndims)
data_cov = jnp.diag(jnp.ones(ndims)) ** 2
data_cov = jnp.where(data_cov == 0., 0.95, data_cov)


# define prior which is a diagonal MVN
prior_mu = 2 * jnp.ones(ndims)
prior_cov = jnp.diag(jnp.ones(ndims)) ** 2
# "push" on each prior
prior_chain = PriorChain().push(MVNDiagPrior('x', prior_mu, jnp.sqrt(jnp.diag(prior_cov))))

# ground truth is analytic for comparison
true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
    prior_cov + data_cov) @ prior_mu
post_cov = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_cov

# define the likelihood (note you need the include **unused_kwargs to consume unused dummy variables)
# The variable "x" will be passed in from the prior chain defined above.

def log_normal(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    dx = x - mean
    dx = solve_triangular(L, dx, lower=True)
    return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
           - 0.5 * dx @ dx
  
log_likelihood = lambda x, **unused_kwargs: log_normal(x, data_mu, data_cov)

# define the sampler you want to use.
ns = NestedSampler(log_likelihood, prior_chain, sampler_name='multi_ellipsoid')

# run with options
results = ns(key=random.PRNGKey(0),
                  num_live_points=50 * ndims,
                  max_samples=1e5,
                  collect_samples=True,
                  termination_frac=0.01,
                  stoachastic_uncertainty=True)

plot_diagnostics(results)
plot_cornerplot(results)

print("True logZ={} | calculated logZ = {:.2f} +- {:.2f}".format(true_logZ, results.logZ, results.logZerr))
print("True posterior m={}\nCov={}".format(post_mu, post_cov))
```
