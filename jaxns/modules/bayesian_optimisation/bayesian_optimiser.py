import datetime
from typing import List, Dict

from jaxns import NestedSampler, PriorChain, UniformPrior, HalfLaplacePrior, GlobalOptimiser, \
    marginalise_static
from jaxns.modules.gaussian_process.kernels import RBF
from jaxns.modules.bayesian_optimisation.utils import latin_hypercube
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr
from jax import random, tree_map
from jax import numpy as jnp, jit
from jax.lax import cummax
import pylab as plt
import numpy as np
from timeit import default_timer
from dataclasses import dataclass, field, asdict
from functools import partial


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
        Kxs = kernel(U, U_star[None, :], lengthscale, sigma)
        H = jnp.linalg.solve(Kxx + data_cov, Kxs)
        # mu(U_star) = mu_X + Ksx @ (Kxx + Sigma)^(-1) @ (Y - mu_Y)
        # var(U_star) = Kss - Ksx @ (Kxx + Sigma)^(-1) @ Kxs
        post_mu_s = (H.T @ Y).reshape(())
        post_var_s = sigma ** 2 - (Kxs.T @ H).reshape(())
        return post_mu_s, post_var_s

    def _aquisition(U_star):
        post_mu_s, post_var_s = moments(U_star)
        sigma_s = jnp.sqrt(jnp.maximum(1e-6, post_var_s))
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
        sigma = jnp.sqrt(jnp.maximum(1e-6, post_K_s[0, 0] + post_K_s[1, 1] - 2. * post_K_s[0, 1]))
        return sigma * f((post_mu_s[0] - post_mu_s[1]) / sigma)

    return _aquisition


def build_marginalised_aquisition(key, U, Y, kernel, samples, log_weights):
    def __aquisition(U_star):
        def _aquisition(sigma, lengthscale, uncert):
            aquisition = build_aquisition(U, Y, kernel, sigma=sigma, lengthscale=lengthscale, uncert=uncert)
            return aquisition(U_star=U_star)

        ESS = 1000
        return marginalise_static(key, samples, log_weights, ESS, _aquisition)

    return __aquisition


def build_marginalised_top_two_aquisition(key, U_top1, U, Y, kernel, samples, log_weights):
    def __aquisition(U_star):
        def _aquisition(sigma, lengthscale, uncert):
            aquisition = build_top_two_aquisition(U_top1, U, Y, kernel, sigma=sigma, lengthscale=lengthscale,
                                                  uncert=uncert)
            return aquisition(U_star=U_star)

        ESS = 1000
        return marginalise_static(key, samples, log_weights, ESS, _aquisition)

    return __aquisition


@dataclass
class Observation:
    _U: jnp.ndarray
    _X: Dict[str, jnp.ndarray]
    _create_time: float = field(default_factory=lambda: datetime.datetime.now().timestamp())
    _Y: jnp.ndarray = None
    _update_time: float = None

    @property
    def sample_point(self) -> Dict[str, jnp.ndarray]:
        """
        The point where to compute next objective value.
        """
        return self._X

    def set_response(self, obj_val: jnp.ndarray):
        """
        Set the objective value at the `sample_point`.

        Args:
            obj_val: scalar value at self.sample_point
        """
        assert jnp.size(obj_val) == 1, "only scalar objective values allowed."
        obj_val = obj_val.reshape(())
        self._Y = obj_val
        self._update_time = datetime.datetime.now().timestamp()


class BayesianOptimiser(object):
    def __init__(self, prior_chain: PriorChain, key=None, load_file: str = None,
                 num_parallel_solvers: int = 1, kernel=RBF()):
        self.prior_chain = prior_chain
        self.prior_chain.build()
        self.observations: Observation = None
        self.num_parallel_solvers = num_parallel_solvers
        self.kernel = kernel
        if load_file is not None:
            self.load_state(load_file)
        if key is None:
            key = random.PRNGKey(42)
        self.key = key
        self.beta = 0.5

    @property
    def U(self):
        return self.observations._U

    @property
    def X(self):
        return self.observations._X

    @property
    def Y(self):
        return self.observations._Y

    @property
    def create_time(self):
        return self.observations._create_time

    @property
    def update_time(self):
        return self.observations._update_time

    def save_state(self, save_file: str):
        np.savez(save_file,
                 U=np.asarray(self.U),
                 X=tree_map(lambda x: np.asarray(x), self.X),
                 Y=np.asarray(self.Y),
                 create_time=np.asarray(self.create_time),
                 update_time=np.asarray(self.update_time))

    def load_state(self, save_file: str):
        data = np.load(save_file, allow_pickle=True)
        self.observations = Observation(jnp.asarray(data['U']),
                                        tree_map(lambda x: jnp.asarray(x), data['X'][()]),
                                        jnp.asarray(data['create_time']),
                                        jnp.asarray(data['Y']),
                                        jnp.asarray(data['update_time']))

    def posterior_solve(self, key, U, Y):
        with PriorChain() as prior_chain:
            HalfLaplacePrior('lengthscale', jnp.ones(U.shape[1]))
            # HalfLaplacePrior('lengthscale', 1.)
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
            K = self.kernel(U, U, lengthscale, sigma)
            return log_normal(Y, mu, K + data_cov)

        ns = NestedSampler(log_likelihood, prior_chain, num_parallel_samplers=self.num_parallel_solvers)
        results = ns(key=key, adaptive_evidence_patience=2)
        # ns.summary(results)
        # ns.plot_diagnostics(results)
        # ns.plot_cornerplot(results)
        return results

    def search_U_top1(self, key, U, Y, samples, log_dp_mean, U_mean, U_scale):
        key1, key2 = random.split(key, 2)
        aquisition = build_marginalised_aquisition(key1, U, Y, self.kernel, samples, log_weights=log_dp_mean)

        with PriorChain() as search_prior_chain:
            # we'll effectively place no prior on the parameters, other than requiring them to be within [-10,10]
            UniformPrior('_U', jnp.zeros(U.shape[1]), jnp.ones(U.shape[1]), tracked=False).subtract(U_mean).true_divide(
                U_scale, name='U_star', tracked=True)

        # U_star = random_restart_bfgs(key3, aquisition, search_prior_chain, num_restarts=10)

        go = GlobalOptimiser(aquisition, prior_chain=search_prior_chain, sampler_kwargs=dict(gradient_boost=True),
                             num_parallel_samplers=self.num_parallel_solvers)

        go_result = go(key=key2)
        U_star = go_result.sample_L_max['U_star']

        return U_star

    def search_U_top2(self, key, U_top1, U, Y, samples, log_dp_mean, U_mean, U_scale):
        key1, key2 = random.split(key, 2)
        top_two_aquisition = build_marginalised_top_two_aquisition(key1, U_top1, U, Y, self.kernel, samples,
                                                                   log_weights=log_dp_mean)

        with PriorChain() as search_prior_chain:
            # we'll effectively place no prior on the parameters, other than requiring them to be within [-10,10]
            UniformPrior('_U', jnp.zeros(U.shape[1]), jnp.ones(U.shape[1]), tracked=False).subtract(U_mean).true_divide(
                U_scale, name='U_star', tracked=True)

        # U_star = random_restart_bfgs(key4, top_two_aquisition, search_prior_chain, num_restarts=10)
        go = GlobalOptimiser(top_two_aquisition, prior_chain=search_prior_chain,
                             sampler_kwargs=dict(gradient_boost=True),
                             num_parallel_samplers=self.num_parallel_solvers)

        go_result = go(key=key2)
        U_star = go_result.sample_L_max['U_star']

        return U_star

    def _choose_next_U(self, key, U, Y, top_two: bool = False):
        key1, key2, key3 = random.split(key, 3)

        U_mean = jnp.nanmean(U, axis=0)
        U_scale = jnp.nanstd(U, axis=0) + 1e-6
        U -= U_mean
        U /= U_scale

        Y_mean = jnp.nanmean(Y, axis=0)
        Y_scale = jnp.nanstd(Y, axis=0) + 1e-6
        Y -= Y_mean
        Y /= Y_scale

        results = self.posterior_solve(key1, U, Y)

        # search over U-domain space for top1

        U_star = self.search_U_top1(key2, U, Y, results.samples, results.log_dp_mean, U_mean, U_scale)

        # uvec = jnp.linspace(0., 1., 100)
        # U1, U2 = jnp.meshgrid((uvec - U_mean[0]) / U_scale[0], (uvec - U_mean[1]) / U_scale[1], indexing='ij')
        # _U = jnp.stack([U1.flatten(), U2.flatten()], axis=1)
        # _aq = vmap(aquisition)(_U).reshape(U1.shape)
        # import pylab as plt
        # plt.imshow(_aq.T, extent=(U1.min(), U1.max(), U2.min(), U2.max()), origin='lower', aspect='auto')
        # plt.colorbar()
        # plt.scatter(U[:, 0], U[:, 1], c='black')
        # plt.scatter(U_star[0], U_star[1], c='red',label='U_top')
        # max_idx = jnp.where(_aq == jnp.max(_aq))
        # plt.scatter(U1[max_idx], U2[max_idx], c='cyan')
        # plt.legend()
        # plt.title("Aq.")
        # plt.show()

        if top_two:
            # search for top2
            U_top1 = U_star

            U_star = self.search_U_top2(key2, U_top1, U, Y, results.samples, results.log_dp_mean, U_mean, U_scale)

            # uvec = jnp.linspace(0., 1., 100)
            # U1, U2 = jnp.meshgrid((uvec - U_mean[0]) / U_scale[0], (uvec - U_mean[1]) / U_scale[1], indexing='ij')
            # _U = jnp.stack([U1.flatten(), U2.flatten()], axis=1)
            # _aq = vmap(top_two_aquisition)(_U).reshape(U1.shape)
            # import pylab as plt
            # plt.imshow(_aq.T, extent=(U1.min(), U1.max(), U2.min(), U2.max()), origin='lower', aspect='auto')
            # plt.colorbar()
            # plt.scatter(U[:, 0], U[:, 1], c='black')
            # plt.scatter(U_top1[0], U_top1[1], c='red', label='U_top1')
            # plt.scatter(U_star[0], U_star[1], c='white', label='U_top2')
            # max_idx = jnp.where(_aq == jnp.max(_aq))
            # plt.scatter(U1[max_idx], U2[max_idx], c='cyan')
            # plt.legend()
            # plt.title("Top Two Aq.")
            # plt.show()

        next_U = U_star * U_scale + U_mean
        return next_U

    def choose_next_U(self, key, U, Y):
        return self._choose_next_U(key, U, Y, top_two=False)

    def choose_next_U_top_two(self, key, U, Y):
        return self._choose_next_U(key, U, Y, top_two=True)

    def __repr__(self):
        return f"BayesianOptimiser(num_measurements={len(self.Y)}, max.obj.={max(self.Y)})"

    def plot_progress(self, save_file: str = None):
        """
        Plot the progress.

        Args:
            save_file: name of file to save fig to.
        """
        idx = np.arange(len(self.Y))
        best = cummax(self.Y)
        plt.scatter(idx, self.Y, c='black')
        plt.plot(idx, best, c='red')
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()

    def initialise_experiment(self, num_samples) -> List[Observation]:
        """
        Initialise with latin hyper cube experimental setup.

        Args:
            num_samples: int, number of initial sample points.

        Returns: list of Observation points, for the user to set the response and give back.
        """
        self.key, key = random.split(self.key)
        U_test = list(latin_hypercube(key, num_samples, self.prior_chain.U_ndims, cube_scale=0))
        X_test = list(map(lambda U: self.prior_chain.filter_sample(self.prior_chain(U)), U_test))
        observations = [Observation(u, x) for u, x in zip(U_test, X_test)]
        return observations

    def add_results(self, observations: List[Observation]):
        """
        Add a list of observations with response set.
        Args:
            observations: List[Observation]
        """
        for obs in observations:
            self.add_result(obs)

    def add_result(self, obs: Observation):
        """
        Add an obseration with respones set.

        Args:
            obs: Observation
        """
        if obs._Y is None:
            raise ValueError("Must set the objective value first with .set_response()")
        if self.observations is None:
            self.observations = Observation(**tree_map(lambda x: jnp.asarray(x)[None], asdict(obs)))
        else:
            self.observations = Observation(**tree_map(lambda x, y: jnp.concatenate([x, jnp.asarray(y)[None]], axis=0),
                                                       asdict(self.observations), asdict(obs)))

    def choose_next_sample_location(self) -> Observation:
        self.key, key1, key2 = random.split(self.key, 3)
        choose_I1 = random.uniform(key1) < self.beta
        t0 = default_timer()
        if choose_I1:
            print("Top-one selected")
            U_test = self.choose_next_U(key=key2,
                                        U=self.U,
                                        Y=self.Y)
        else:
            print("Top-two selected")
            U_test = self.choose_next_U_top_two(key=key2,
                                                U=self.U,
                                                Y=self.Y)
        print(f"Choice of next point took: {default_timer() - t0} seconds")
        X_test = self.prior_chain.filter_sample(self.prior_chain(U_test))
        return Observation(U_test, X_test)
