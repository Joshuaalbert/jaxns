from jaxns.gaussian_process.kernels import RBF, M12
from jaxns.gaussian_process.tomographic_kernel import TomographicKernel
from jaxns.examples.frozen_flow.generate_data import rbf_dtec
from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior, GaussianProcessKernelPrior, MVNPrior, \
    DeterministicTransformPrior, MVNDiagPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics
from jax import random, jit, make_jaxpr
from jax.scipy.linalg import solve_triangular
from jax import numpy as jnp, disable_jit
import pylab as plt
from timeit import default_timer


def main(kernel):
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean

        dx = solve_triangular(L, dx, lower=True)
        # maha = dx @ jnp.linalg.solve(cov, dx)
        maha = dx @ dx
        # logdet = jnp.log(jnp.linalg.det(cov))
        logdet = jnp.sum(jnp.diag(L))
        log_prob = -0.5 * x.size * jnp.log(2. * jnp.pi) - logdet - 0.5 * maha
        return log_prob

    true_height, true_width, true_sigma, true_l, true_uncert, true_v = 200., 100., 1., 10., 2.5, jnp.array([0.3,0.,0.])
    nant = 2
    ndir = 1
    ntime = 20
    X, Y, Y_obs = rbf_dtec(nant, ndir, ntime, true_height, true_width, true_sigma, true_l, true_uncert,true_v)
    a = X[:, 0:3]
    k = X[:, 3:6]
    t = X[:,6:7]
    x0 = a[0, :]

    def log_likelihood(dtec, uncert, **kwargs):
        """
        P(Y|sigma, half_width) = N[Y, mu, K]
        Args:
            sigma:
            l:

        Returns:

        """
        data_cov = jnp.square(uncert) * jnp.eye(X.shape[0])
        return log_normal(Y_obs, dtec, data_cov)

    def predict_f(dtec, **kwargs):
        return dtec

    def predict_fvar(dtec, **kwargs):
        return dtec ** 2

    def tec_to_dtec(tec):
        tec = tec.reshape((nant, ndir, ntime))
        dtec = jnp.reshape(tec - tec[0, :, :], (-1,))
        return dtec

    v_dir = DeterministicTransformPrior('v_dir', lambda n: n/jnp.linalg.norm(n),(3,),
                                              MVNDiagPrior('n', jnp.zeros(3), jnp.ones(3),
                                                           tracked=False), tracked=False)
    v_mag = UniformPrior('v_mag',0., 0.5, tracked=False)
    v = DeterministicTransformPrior('v', lambda v_dir, v_mag: v_mag*v_dir,
                                    (3,),v_dir, v_mag, tracked=True)

    X_frozen_flow = DeterministicTransformPrior('X', lambda v: X[:,0:6] - jnp.concatenate([v,jnp.zeros(3)])*X[:,6:7],X[:,0:6].shape, v, tracked=False)

    K = GaussianProcessKernelPrior('K',
                                   TomographicKernel(x0, kernel, S=20),
                                   X_frozen_flow,
                                   UniformPrior('height', 100., 300.),
                                   UniformPrior('width', 50., 150.),
                                   UniformPrior('l', 0., 20.),
                                  UniformPrior('sigma', 0., 2.), tracked=False)
    tec = MVNPrior('tec', jnp.zeros((X.shape[0],)), K, ill_cond=True, tracked=False)
    dtec = DeterministicTransformPrior('dtec', tec_to_dtec, tec.to_shape, tec, tracked=False)
    prior_chain = PriorChain() \
        .push(dtec) \
        .push(UniformPrior('uncert', 0., 5.))

    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='multi_ellipsoid', predict_f=predict_f,
                       predict_fvar=predict_fvar)

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      stoachastic_uncertainty=False,
                      sampler_kwargs=dict(depth=4))

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print("Efficiency", results.efficiency)
        print("Time to run (including compile)", default_timer() - t0)
        # t0 = default_timer()
        # results = run(random.PRNGKey(1))
        # print(results.efficiency)
        # print("Time to run (no compile)", default_timer() - t0)
        return results

    for n in [10]:
        results = run_with_n(n)
        plt.scatter(n, results.logZ)
        plt.errorbar(n, results.logZ, yerr=results.logZerr)
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.ylabel('log Z')
    plt.show()

    fstd = jnp.sqrt(results.marginalised['predict_fvar'] - results.marginalised['predict_f'] ** 2)
    plt.scatter(jnp.arange(Y.size),Y_obs, marker='+', label='data')
    plt.scatter(jnp.arange(Y.size),Y, marker="o", label='underlying')
    plt.scatter(jnp.arange(Y.size), results.marginalised['predict_f'], marker=".", label='underlying')
    plt.errorbar(jnp.arange(Y.size), results.marginalised['predict_f'], yerr=fstd, label='marginalised')
    plt.title("Kernel: {}".format(kernel.__class__.__name__))
    plt.legend()
    plt.show()

    plot_diagnostics(results)
    plot_cornerplot(results)
    return results.logZ, results.logZerr


if __name__ == '__main__':
    logZ_rbf, logZerr_rbf = main(RBF())
    logZ_m12, logZerr_m12 = main(M12())
    plt.errorbar(['rbf', 'm12'],
                 [logZ_rbf, logZ_m12],
                 [logZerr_rbf, logZerr_m12])
    plt.ylabel("log Z")
    plt.show()
