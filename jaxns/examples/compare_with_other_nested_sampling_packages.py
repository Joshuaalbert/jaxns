from numpy import pi, log
import pylab as plt
from timeit import default_timer



def main():

    ndims = 30
    sigma = 0.1
    num_live_points = 1000
    do_dynesty = True
    do_polychord = True
    do_multinest = True

    def run_polychord(num_live_points):
        try:
            import pypolychord
            from pypolychord.settings import PolyChordSettings
            from pypolychord.priors import UniformPrior
        except:
            raise ImportError("Polychord not installed.\nRun `git clone https://github.com/PolyChord/PolyChordLite.git \ncd PolyChordLite\npython setup.py install`.")

        def likelihood(theta):
            """ Simple Gaussian Likelihood"""
            nDims = len(theta)
            r2 = sum(theta**2)
            logL = -log(2*pi*sigma*sigma)*nDims/2.0
            logL += -r2/2/sigma/sigma
            return logL, [r2]


        def prior(hypercube):
            """ Uniform prior from [-1,1]^D. """
            return UniformPrior(-1, 1)(hypercube)


        def dumper(live, dead, logweights, logZ, logZerr):
            return
            # print("Last dead point:", dead[-1])

        settings = PolyChordSettings(ndims, 1)
        settings.file_root = 'gaussian'
        settings.nlive = num_live_points
        settings.do_clustering = True
        settings.read_resume = False

        t0 = default_timer()
        output = pypolychord.run_polychord(likelihood, ndims, 1, settings, prior, dumper)
        run_time = default_timer() - t0
        print("polychord log(Z):", output.logZ)
        return run_time

    def run_dynest(num_live_points):
        try:
            import dynesty
        except:
            raise ImportError("Dynesty not installed. Run `pip install dynesty`.")


        def loglikelihood(theta):
            """Multivariate normal log-likelihood."""
            nDims = len(theta)
            r2 = sum(theta**2)
            logL = -log(2*pi*sigma*sigma)*nDims/2.0
            logL += -r2/2/sigma/sigma
            return logL

        # prior transform (iid standard normal prior)
        def prior_transform(u):
            """Transforms our unit cube samples `u` to a standard normal prior."""
            return u*2.-1.

        sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndims,
                                        nlive=num_live_points,
                                        bound='single', sample='slice',
                                        slices=25)
        t0 = default_timer()
        sampler.run_nested(dlogz=0.01)
        res = sampler.results
        print("Dynesty log(Z):",res['logz'])
        run_time = default_timer()-t0
        return run_time


    def run_multinest(num_live_points):
        ### multinest
        try:
            from pymultinest.solve import solve
        except:
            raise ImportError("Multinest is not installed.\nFollow directions on http://johannesbuchner.github.io/PyMultiNest/install.html.")
        import os
        os.makedirs('chains', exist_ok=True)
        prefix = "chains/3-"
        parameters = ['w', "x", "y", 'z']

        def loglikelihood(theta):
            """Multivariate normal log-likelihood."""
            nDims = len(theta)
            r2 = sum(theta**2)
            logL = -log(2*pi*sigma*sigma)*nDims/2.0
            logL += -r2/2/sigma/sigma
            return logL

        # prior transform (iid standard normal prior)
        def prior_transform(u):
            """Transforms our unit cube samples `u` to a standard normal prior."""
            return u*2.-1.

        # run MultiNest
        t0 = default_timer()
        result = solve(LogLikelihood=loglikelihood, Prior=prior_transform,
            n_dims=ndims, outputfiles_basename=prefix, verbose=False,
                       n_live_points=num_live_points)
        run_time = default_timer() - t0
        print("Multinest results:", result)
        return run_time

    def run_jaxns(num_live_points):
        try:
            from jaxns.nested_sampling import NestedSampler
            from jaxns.prior_transforms import PriorChain, UniformPrior
        except:
            raise ImportError("Install JaxNS!")
        from timeit import default_timer
        from jax import random, jit
        import jax.numpy as jnp

        def log_likelihood(theta, **kwargs):
            r2 = jnp.sum(theta ** 2)
            logL = -0.5 * jnp.log(2. * jnp.pi * sigma ** 2) * ndims
            logL += -0.5 * r2 / sigma ** 2
            return logL

        prior_transform = PriorChain().push(UniformPrior('theta', -jnp.ones(ndims), jnp.ones(ndims)))
        ns = NestedSampler(log_likelihood, prior_transform, sampler_name='slice')

        def run_with_n(n):
            @jit
            def run(key):
                return ns(key=key,
                          num_live_points=n,
                          max_samples=1e6,
                          collect_samples=False,
                          termination_frac=0.01,
                          stoachastic_uncertainty=False,
                          sampler_kwargs=dict(depth=3, num_slices=2))

            results = run(random.PRNGKey(0))
            results.logZ.block_until_ready()
            t0 = default_timer()
            results = run(random.PRNGKey(1))
            print("Efficiency and logZ", results.efficiency, results.logZ)
            run_time = (default_timer() - t0)
            return run_time

        return run_with_n(num_live_points)

    names = []
    run_time = []
    if do_dynesty:
        names.append("Dynesty")
        run_time.append(run_dynest(num_live_points))
    if do_polychord:
        names.append("PolyChord")
        run_time.append(run_polychord(num_live_points))
    if do_multinest:
        names.append("MultiNest")
        run_time.append(run_multinest(num_live_points))
    names.append('JaxNS')
    run_time.append(run_jaxns(num_live_points))

    plt.bar(names, run_time, fc="none", ec='black', lw=3.)
    plt.xlabel("Nested sampling package")
    plt.ylabel("Execution time (s)")
    plt.yscale('log')
    plt.savefig("./speed_test.png")
    plt.show()

if __name__ == '__main__':
    main()