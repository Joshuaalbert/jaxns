![JAXNS](./jaxns_logo.png)

# What is it?
Enables probabilistic programming using nested sampling. It's coded in JAX in a manner that allows lowering the entire inference algorithm to XLA primitives, which are JIT-compiled for high performance. You can read about it here: (https://arxiv.org/abs/2012.15286)

JAXNS provides a constrained likelihood sampler which combines and modifies ideas from MultiNest (F. Feroz et al. 2008; https://arxiv.org/pdf/0809.3437.pdf) and PolyChord (W.J. Handley et al. 2015; https://arxiv.org/abs/1506.00171).
In particular we perform a sequence of 1D slice sampling with a step-out procedure (https://projecteuclid.org/euclid.aos/1056562461), using clustering to initialise the step-out procedure at each step.

# Install
Make sure you have JAX and the usual suspects with `pip install jax jaxlib numpy matplotlib scipy`. 
Install with `python setup.py install` or `pip install git+http://github.com/Joshuaalbert/jaxns.git`.

# Quick start

JAXNS is really fast because it uses JAX. I've found it's 2-4 orders of magnitude faster than other nested sampling packages.
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
prior_chain = PriorChain().push(NormalPrior('x', prior_mu, jnp.sqrt(jnp.diag(prior_cov))))

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
ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice')

# run with options
results = ns(key=random.PRNGKey(0),
                  num_live_points=300,
                  max_samples=1e5,
                  collect_samples=True,
                  termination_frac=0.01)

plot_diagnostics(results)
plot_cornerplot(results)

print("True logZ={} | calculated logZ = {:.2f} +- {:.2f}".format(true_logZ, results.logZ, results.logZerr))
print("True posterior m={}\nCov={}".format(post_mu, post_cov))
```

# Speed test comparison with other nested sampling packages

JAXNS is much faster than PolyChord, MultiNEST, and dynesty, typically achieve two to three orders of magnitude improvement in speed.
I show this in (https://arxiv.org/abs/2012.15286).
