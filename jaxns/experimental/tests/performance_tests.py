import jax.random
import jax.random
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from jaxns import Prior, Model
from jaxns.experimental.global_optimisation import GlobalOptimisationTerminationCondition
from jaxns.experimental.public import GlobalOptimisation

tfpd = tfp.distributions

def xin_she_yang_1_problem_nd(ndims: int):

    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-2. * jnp.pi * jnp.ones(ndims), high=2. * jnp.pi * jnp.ones(ndims)), name='z')
        return z

    def log_likelihood(z):
        return - jnp.sum(jnp.abs(z)) * jnp.exp(-jnp.sum(jnp.sin(z ** 2)))

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    optimum = 0. * jnp.ones(ndims)
    a_tol = 0.01
    log_L_tol = -0.01
    return model, optimum, a_tol, log_L_tol


def drop_wave_problem_nd(ndims: int):
    def prior_model():
        z = yield Prior(tfpd.Uniform(low=-5.12 * jnp.ones(ndims), high=5.12 * jnp.ones(ndims)), name='z')
        return z

    def log_likelihood(z):
        return (1 + jnp.cos(12. * jnp.sqrt(jnp.sum(z ** 2)))) / (0.5 * jnp.sum(z ** 2) + 2) - 1.

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    optimum = 0. * jnp.ones(ndims)
    a_tol = 0.01
    log_L_tol = -0.01
    return model, optimum, a_tol, log_L_tol


def run_performance_test_no_gradient_slice():
    (model, optimum, a_tol, log_L_tol) = drop_wave_problem_nd(5)
    keys = jax.random.split(jax.random.PRNGKey(42), 100)
    # gradient_slice=False, s=1, c=10, k=4, shell_frac=0.5:
    # mean_res=0.3747073956625793, success_prob=0.0
    # gradient_slice=False, s=1, c=50, k=4, shell_frac=0.5:
    # mean_res=0.3720548526096466, success_prob=0.019999999552965164
    # gradient_slice=False, s=1, c=100, k=4, shell_frac=0.5:
    # mean_res=0.35544749388887315, success_prob=0.08999999612569809
    # gradient_slice=False, s=5, c=10, k=24, shell_frac=0.5:
    # mean_res=0.37351151565613305, success_prob=0.019999999552965164
    # gradient_slice=False, s=5, c=50, k=24, shell_frac=0.5:
    # mean_res=0.33936949884649364, success_prob=0.11999999731779099
    # gradient_slice=False, s=5, c=100, k=24, shell_frac=0.5:
    # mean_res=0.31854922674033903, success_prob=0.17000000178813934
    # gradient_slice=False, s=10, c=10, k=49, shell_frac=0.5:
    # mean_res=0.35583731215755904, success_prob=0.07000000029802322
    # gradient_slice=False, s=10, c=50, k=49, shell_frac=0.5:
    # mean_res=0.32152508919560535, success_prob=0.1599999964237213
    # gradient_slice=False, s=10, c=100, k=49, shell_frac=0.5: <---- Best
    # mean_res=0.28122300728873983, success_prob=0.26999998092651367

    for s in [1, 5, 10]:
        for c in [10, 50, 100]:
            for k in [s * model.U_ndims - 1]:
                for shell_frac in [0.5]:
                    @jax.jit
                    def run(key):
                        go = GlobalOptimisation(
                            model,
                            gradient_slice=False,
                            shell_frac=shell_frac,
                            s=s,
                            k=k,
                            num_search_chains=model.U_ndims * c
                        )
                        results = go(
                            key=key,
                            term_cond=GlobalOptimisationTerminationCondition(log_likelihood_contour=log_L_tol),
                            finetune=False
                        )
                        return jnp.max(jnp.abs(results.solution[0] - optimum))

                    res = jax.block_until_ready(jnp.stack([run(key) for key in keys]))
                    mean_res = jnp.mean(res)
                    success = res < a_tol
                    success_prob = jnp.mean(success)
                    print(
                        f"gradient_slice={False}, s={s}, c={c}, k={k}, shell_frac={shell_frac}:\n"
                        f"mean_res={mean_res}, success_prob={success_prob}"
                    )


def run_performance_test_gradient_slice():
    (model, optimum, a_tol, log_L_tol) = xin_she_yang_1_problem_nd(2)
    # (model, optimum, a_tol, log_L_tol) = drop_wave_problem_nd(20)
    keys = jax.random.split(jax.random.PRNGKey(42), 100)

    # Can solve 100D dropwave with 100% success rate with about 275k likelihood evaluations, i.e. 2.75k per dimension
    # gradient_slice=True, s=1, c=1, k=0, shell_frac=0.5:
    # mean_res=1.7066995784509587e-05, success_prob=1.0, n_like=274194.42

    for s in [1, 2, 3, 4, 5]:
        for c in [1, 5, 10, 15]:
            for k in [s * model.U_ndims - 1]:
                for shell_frac in [0.5]:
                    @jax.jit
                    def run(key):
                        go = GlobalOptimisation(
                            model,
                            gradient_slice=True,
                            shell_frac=shell_frac,
                            s=s,
                            k=k,
                            num_search_chains=model.U_ndims * c
                        )
                        results = go(
                            key=key,
                            term_cond=GlobalOptimisationTerminationCondition(log_likelihood_contour=log_L_tol),
                            finetune=False
                        )
                        return jnp.max(jnp.abs(results.solution[0] - optimum)), results.num_likelihood_evaluations

                    res = []
                    n_like = []
                    for key in keys:
                        r, n = run(key)
                        res.append(r)
                        n_like.append(n)
                    res = jnp.stack(res)
                    n_like = jnp.stack(n_like)
                    mean_res = jnp.mean(res)
                    success = res < a_tol
                    success_prob = jnp.mean(success)
                    print(
                        f"gradient_slice={True}, s={s}, c={c}, k={k}, shell_frac={shell_frac}:\n"
                        f"mean_res={mean_res}, success_prob={success_prob}, n_like={jnp.mean(n_like)}"
                    )


if __name__ == '__main__':
    # run_performance_test_no_gradient_slice()
    run_performance_test_gradient_slice()
