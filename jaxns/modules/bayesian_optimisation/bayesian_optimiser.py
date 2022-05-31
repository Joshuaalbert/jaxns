from typing import List, Tuple

from jaxns import NestedSampler, PriorChain, UniformPrior, HalfLaplacePrior, resample, GlobalOptimiser
from jaxns.modules.gaussian_process.kernels import RBF
from jaxns.modules.bayesian_optimisation.utils import latin_hypercube
from jaxns.internals.maps import prepare_func_args
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr
from jax.scipy.optimize import minimize
from jax import random, vmap
from jax import numpy as jnp, jit
from functools import partial
from timeit import default_timer


def log_normal(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    # U, S, Vh = jnp.linalg.svd(cov)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))  # jnp.sum(jnp.log(S))#
    dx = x - mean
    dx = solve_triangular(L, dx, lower=True)
    # U S Vh V 1/S Uh
    # pinv = (Vh.T.conj() * jnp.where(S!=0., jnp.reciprocal(S), 0.)) @ U.T.conj()
    maha = dx @ dx  # dx @ pinv @ dx#solve_triangular(L, dx, lower=True)
    log_likelihood = -0.5 * x.size * jnp.log(2. * jnp.pi) \
                     - log_det \
                     - 0.5 * maha
    return log_likelihood


def build_aquisition(U, Y, kernel, sigma, lengthscale, uncert):
    def f(x):
        return x * ndtr(x) + jnp.exp(-0.5 * x ** 2) / jnp.sqrt(2. * jnp.pi)

    Kxx = kernel(U, U, lengthscale, sigma)
    data_cov = jnp.square(uncert) * jnp.eye(U.shape[0])
    h = jnp.linalg.solve(Kxx + data_cov, Y)
    post_mu_x_max = jnp.max(Kxx @ h)

    def moments(U_star):
        Kxs = kernel(U, U_star[None,:], lengthscale, sigma)
        H = jnp.linalg.solve(Kxx + data_cov, Kxs)
        # mu(U_star) = mu_X + Ksx @ (Kxx + Sigma)^(-1) @ (Y - mu_Y)
        # var(U_star) = Kss - Ksx @ (Kxx + Sigma)^(-1) @ Kxs
        post_mu_s = H.T @ Y
        post_var_s = sigma ** 2 - (Kxs.T @ H).reshape(())
        return post_mu_s, post_var_s

    def _aquisition(U_star):

        post_mu_s, post_var_s = moments(U_star)
        sigma_s = jnp.sqrt(post_var_s)
        return sigma_s * f((post_mu_s - post_mu_x_max) / sigma_s)

    return _aquisition

def build_top_two_aquisition(U_top1, U, Y, kernel, sigma, lengthscale, uncert):
    def f(x):
        return x * ndtr(x) + jnp.exp(-0.5 * x ** 2) / jnp.sqrt(2. * jnp.pi)

    Kxx = kernel(U, U, lengthscale, sigma)
    data_cov = jnp.square(uncert) * jnp.eye(U.shape[0])

    def moments(U_star):
        S = jnp.stack([U_star, U_top1], axis=0)
        Kxs = kernel(U, S, lengthscale, sigma)
        H = jnp.linalg.solve(Kxx + data_cov, Kxs)
        # mu(U_star) = mu_X + Ksx @ (Kxx + Sigma)^(-1) @ (Y - mu_Y)
        # var(U_star) = Kss - Ksx @ (Kxx + Sigma)^(-1) @ Kxs
        post_mu_s = H.T @ Y
        post_K_s = kernel(S, S, lengthscale, sigma) - Kxs.T @ H
        return post_mu_s, post_K_s

    def _aquisition(U_star):
        post_mu_s, post_K_s = moments(U_star)
        sigma = jnp.sqrt(post_K_s[0,0] + post_K_s[1,1] - 2.  * post_K_s[0,1])
        return sigma * f((post_mu_s[0] - post_mu_s[1]) / sigma)

    return _aquisition


def build_marginalised_aquisition(U, Y, kernel, samples):

    def __aquisition(U_star):
        @prepare_func_args
        def _aquisition(sigma, lengthscale, uncert):
            aquisition = build_aquisition(U, Y, kernel, sigma=sigma, lengthscale=lengthscale, uncert=uncert)
            return aquisition(U_star=U_star)
        return jnp.mean(vmap(_aquisition)(**samples))

    return __aquisition


def build_marginalised_top_two_aquisition(U_top1, U, Y, kernel, samples):
    def __aquisition(U_star):
        @prepare_func_args
        def _aquisition(sigma, lengthscale, uncert):
            aquisition = build_top_two_aquisition(U_top1, U, Y, kernel, sigma=sigma, lengthscale=lengthscale, uncert=uncert)
            return aquisition(U_star=U_star)
        return jnp.mean(vmap(_aquisition)(**samples))

    return __aquisition


def posterior_solve(key, U, Y, kernel):
    with PriorChain() as prior_chain:
        UniformPrior('lengthscale', jnp.zeros(U.shape[1]), jnp.ones(U.shape[1]))
        HalfLaplacePrior('uncert', 0.1)
        HalfLaplacePrior('sigma', 1.)

    def log_likelihood(sigma, lengthscale, uncert):
        """
        P(Y|sigma, half_width) = N[Y, f, K]
        Args:
            sigma:
            l:

        Returns:

        """
        data_cov = jnp.square(uncert) * jnp.eye(U.shape[0])
        mu = jnp.zeros_like(Y)
        K = kernel(U, U, lengthscale, sigma)
        return log_normal(Y, mu, K + data_cov)

    ns = NestedSampler(log_likelihood, prior_chain)
    results = jit(ns)(key=key, adaptive_evidence_patience=1, termination_live_evidence_frac=1e-4)
    ns.summary(results)
    ns.plot_diagnostics(results)
    ns.plot_cornerplot(results)
    return results

def random_restart_bfgs(key, func, prior_chain:PriorChain, num_restarts:int):
    def _minimize(key):
        result = minimize(lambda U: func(prior_chain(U)), prior_chain.sample_U_flat(key), method='BFGS')
        return result.fun, result.x
    result = [_minimize(key) for key in random.split(key, num_restarts)]
    fun = jnp.asarray([r[0] for r in result])
    x = jnp.asarray([r[1] for r in result])
    return x[jnp.argmin(fun)]


def choose_next_U(key, U, Y, top_two: bool = False,
                  *,
                  termination_patience=3,
                  termination_frac_likelihood_improvement=1e-3,
                  termination_likelihood_contour=None,
                  termination_max_num_steps=None,
                  termination_max_num_likelihood_evaluations=None):
    kernel = RBF()
    key1, key2, key3, key4 = random.split(key, 4)

    U_mean = jnp.nanmean(U, axis=0)
    U_scale = jnp.nanstd(U, axis=0) + 1e-6
    U -= U_mean
    U /= U_scale

    Y_mean = jnp.nanmean(Y, axis=0)
    Y_scale = jnp.nanstd(Y, axis=0) + 1e-6
    Y -= Y_mean
    Y /= Y_scale

    results = posterior_solve(key1, U, Y, kernel)

    # search over U-domain space

    samples = resample(key=key2,
                       samples=results.samples,
                       log_weights=results.log_dp_mean,
                       S=100,
                       replace=True)

    aquisition = build_marginalised_aquisition(U, Y, kernel, samples)

    with PriorChain() as search_prior_chain:
        # we'll effectively place no prior on the parameters, other than requiring them to be within [-10,10]
        UniformPrior('U_star', jnp.zeros(U.shape[1]), jnp.ones(U.shape[1]))

    search_prior_chain.build()

    # U_star = random_restart_bfgs(key3, aquisition, search_prior_chain, num_restarts=10)

    go = GlobalOptimiser(aquisition, prior_chain=search_prior_chain,
                         samples_per_step=search_prior_chain.U_ndims * 10)

    go_result = go(key=key3,
                   termination_patience=termination_patience,
                   termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                   termination_likelihood_contour=termination_likelihood_contour,
                   termination_max_num_steps=termination_max_num_steps,
                   termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)
    U_star = go_result.sample_L_max['U_star']

    if top_two:
        U_top1 = U_star
        top_two_aquisition = build_marginalised_top_two_aquisition(U_top1, U, Y, kernel, samples)
        # U_star = random_restart_bfgs(key4, top_two_aquisition, search_prior_chain, num_restarts=10)
        go = GlobalOptimiser(top_two_aquisition, prior_chain=search_prior_chain,
                             samples_per_step=search_prior_chain.U_ndims * 10)

        go_result = go(key=key4,
                       termination_patience=termination_patience,
                       termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                       termination_likelihood_contour=termination_likelihood_contour,
                       termination_max_num_steps=termination_max_num_steps,
                       termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)
        U_star = go_result.sample_L_max['U_star']

    next_U = U_star * U_scale + U_mean
    return next_U


class BayesianOptimiser(object):
    def __init__(self, prior_chain: PriorChain, U=None, X=None, Y=None, key=None):
        self.prior_chain = prior_chain
        self.prior_chain.build()
        self.U = U or []
        self.X = X or []
        self.Y = Y or []
        assert len(self.Y) == len(self.U) == len(self.X)
        if key is None:
            key = random.PRNGKey(42)
        self.key = key
        self.beta = 0.5

    # @partial(jit, static_argnums=(0,))
    def choose_next_U(self,key, U, Y,
                  *,
                  termination_patience=3,
                  termination_frac_likelihood_improvement=1e-3,
                  termination_likelihood_contour=None,
                  termination_max_num_steps=None,
                  termination_max_num_likelihood_evaluations=None):
        return choose_next_U(key, U, Y, top_two=False,
                  termination_patience=termination_patience,
                  termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                  termination_likelihood_contour=termination_likelihood_contour,
                  termination_max_num_steps=termination_max_num_steps,
                  termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)

    # @partial(jit, static_argnums=(0,))
    def choose_next_U_top_two(self, key, U, Y,
                      *,
                      termination_patience=3,
                      termination_frac_likelihood_improvement=1e-3,
                      termination_likelihood_contour=None,
                      termination_max_num_steps=None,
                      termination_max_num_likelihood_evaluations=None):

        return choose_next_U(key, U, Y, top_two=True,
                             termination_patience=termination_patience,
                             termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                             termination_likelihood_contour=termination_likelihood_contour,
                             termination_max_num_steps=termination_max_num_steps,
                             termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations)
    def __repr__(self):
        return f"BayesianOptimiser(num_measurements={len(self.Y)}, max.obj.={max(self.Y)})"

    def initialise_experiment(self, num_samples) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        self.key, key = random.split(self.key)
        U_test = list(latin_hypercube(key, num_samples, self.prior_chain.U_ndims, cube_scale=0))
        X_test = list(map(lambda U: self.prior_chain(U), U_test))
        return U_test, X_test

    def add_result(self, U, X, Y):
        self.U.append(U)
        self.X.append(X)
        self.Y.append(Y)

    def choose_next_sample_location(self, termination_patience=2,
                                    termination_frac_likelihood_improvement=1e-2,
                                    termination_likelihood_contour=None,
                                    termination_max_num_steps=None,
                                    termination_max_num_likelihood_evaluations=None):
        self.key, key1, key2 = random.split(self.key, 3)
        choose_I1 = random.uniform(key1) < self.beta
        t0 = default_timer()
        if choose_I1:
            print("Top-one selected")
            U_test = self.choose_next_U(key=key2,
                                   U=jnp.asarray(self.U),
                                   Y=jnp.asarray(self.Y),
                                   termination_patience=termination_patience,
                                   termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                                   termination_likelihood_contour=termination_likelihood_contour,
                                   termination_max_num_steps=termination_max_num_steps,
                                   termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations
                                   )
        else:
            print("Top-two selected")
            U_test = self.choose_next_U_top_two(key=key2,
                                        U=jnp.asarray(self.U),
                                        Y=jnp.asarray(self.Y),
                                        termination_patience=termination_patience,
                                        termination_frac_likelihood_improvement=termination_frac_likelihood_improvement,
                                        termination_likelihood_contour=termination_likelihood_contour,
                                        termination_max_num_steps=termination_max_num_steps,
                                        termination_max_num_likelihood_evaluations=termination_max_num_likelihood_evaluations
                                        )
        print(f"Choice of next point took: {default_timer() - t0} seconds")
        X_test = self.prior_chain(U_test)
        return U_test, X_test
