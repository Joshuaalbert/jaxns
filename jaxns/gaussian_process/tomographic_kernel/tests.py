from jax import numpy as jnp, random, vmap

from jaxns.gaussian_process.tomographic_kernel.tomographic_kernel_utils import tomographic_weight_function_stochastic, \
    tomographic_weight_function_outer, tomographic_weight_function, log_tomographic_weight_dimensionless_function, \
    get_polynomial_form, cumulative_tomographic_weight_dimensionless_function, \
    cumulative_tomographic_weight_dimensionless_polynomial


def test_tomographic_weight_function_stochastic():
    import pylab as plt
    u = jnp.linspace(0., 4., 200)**2
    x1 = random.normal(random.PRNGKey(0), shape=(1,2,))
    p1 = random.normal(random.PRNGKey(1), shape=(1,2,))
    x2 = random.normal(random.PRNGKey(2), shape=(1,2,))
    p2 = random.normal(random.PRNGKey(3), shape=(1,2,))
    w = tomographic_weight_function_stochastic(random.PRNGKey(4),
                                               u,
                                               x1, p1, x2, p2)
    plt.plot(u, w[0,0,:])
    plt.show()


def test_tomographic_weight_function_outer():
    keys = random.split(random.PRNGKey(0),6)
    x1 = random.normal(keys[0], shape=(5,2,))
    p1 = random.normal(keys[1], shape=(5, 2,))
    x2 = random.normal(keys[2], shape=(6,2,))
    p2 = random.normal(keys[3], shape=(6,2,))

    gamma = jnp.linspace(0., 20., 100)
    assert tomographic_weight_function_outer(gamma,x1,x2,p1,p2,S=15).shape == (x1.shape[0],x2.shape[0],gamma.size-1)


def test_tomographic_weight():
    import pylab as plt
    from jax import jit

    # @jit
    def tomo_weight(gamma, x1,x2,p1,p2):
        return tomographic_weight_function(gamma, x1, x2,p1,p2,S=15)

    @jit
    def tomo_weight_ref(gamma, x1, x2, p1, p2):
        return tomographic_weight_function(gamma, x1, x2, p1, p2, S=150)

    @jit
    def tomo_weight_dimensionless_ref(gamma, x1, x2, p1, p2):
        x12 = x1-x2
        h = jnp.linalg.norm(x12)
        n = x12/h
        w1=p1/h
        w2=p2/h
        gamma_prime = gamma/h**2
        return jnp.exp(log_tomographic_weight_dimensionless_function(gamma_prime, n,w1, w2, S=150))/h**2



    for i in range(10):
        keys = random.split(random.PRNGKey(i),6)
        x1 = jnp.concatenate([10. * random.uniform(keys[0], shape=( 2,)), jnp.zeros((1,))], axis=-1)
        p1 = jnp.concatenate(
            [4. * jnp.pi / 180. * random.uniform(keys[1], shape=(2,), minval=-1, maxval=1),
             jnp.ones((1,))], axis=-1)
        p1 = 4*p1 / jnp.linalg.norm(p1, axis=-1, keepdims=True)

        x2 = jnp.concatenate([4. * random.uniform(keys[2], shape=(2,)), jnp.zeros((1,))], axis=-1)
        p2 = jnp.concatenate(
            [4. * jnp.pi / 180. * random.uniform(keys[3], shape=(2,), minval=-1, maxval=1),
             jnp.ones((1,))], axis=-1)
        p2 = 4*p2 / jnp.linalg.norm(p2, axis=-1, keepdims=True)

        t1 = random.uniform(keys[4], shape=(10000,))
        t2 = random.uniform(keys[5], shape=(10000,))
        u1 = x1 + t1[:, None]*p1
        u2 = x2 + t2[:, None]*p2
        gamma = jnp.linalg.norm(u1-u2,axis=1)**2
        plt.hist(gamma.flatten(),bins=100, density=True, label='histogram')
        hist, bins = jnp.histogram(gamma.flatten(), density=True, bins=100)
        gamma = 0.5*(bins[:-1]+bins[1:])
        w = tomo_weight(bins, x1, x2, p1, p2)
        plt.plot(gamma, w, label='analytic')
        w_ref = tomo_weight_ref(bins, x1, x2, p1, p2)
        # w_ref = tomo_weight_dimensionless_ref(bins, x1,x2,p1,p2)
        plt.plot(gamma, w_ref, label='analytic ref')
        plt.legend()
        plt.show()
        plt.plot()

        # test same as dimensionless
        # assert jnp.all(jnp.isclose(tomo_weight_dimensionless_ref(bins, x1, x2, p1, p2), w_ref))


def test_tomographic_weight_rel_err():
    import pylab as plt
    from jax import jit

    for S in range(5,30,5):
        @jit
        def tomo_weight(gamma, x1,x2,p1,p2):
            return tomographic_weight_function(gamma, x1, x2,p1,p2,S=S)

        @jit
        def tomo_weight_ref(gamma, x1, x2, p1, p2):
            return tomographic_weight_function(gamma, x1, x2, p1, p2, S=150)

        rel_error = []
        for i in range(400):
            keys = random.split(random.PRNGKey(i),6)
            x1 = jnp.concatenate([4. * random.uniform(keys[0], shape=( 2,)), jnp.zeros((1,))], axis=-1)
            p1 = jnp.concatenate(
                [4. * jnp.pi / 180. * random.uniform(keys[1], shape=(2,), minval=-1, maxval=1),
                 jnp.ones((1,))], axis=-1)
            p1 = 4*p1 / jnp.linalg.norm(p1, axis=-1, keepdims=True)

            x2 = jnp.concatenate([4. * random.uniform(keys[2], shape=(2,)), jnp.zeros((1,))], axis=-1)
            p2 = jnp.concatenate(
                [4. * jnp.pi / 180. * random.uniform(keys[3], shape=(2,), minval=-1, maxval=1),
                 jnp.ones((1,))], axis=-1)
            p2 = 4*p2 / jnp.linalg.norm(p2, axis=-1, keepdims=True)

            # x1 = random.normal(keys[0], shape_dict=(2,))
            # p1 = random.normal(keys[1], shape_dict=(2,))
            # x2 = random.normal(keys[2], shape_dict=(2,))
            # p2 = random.normal(keys[3], shape_dict=(2,))

            t1 = random.uniform(keys[4], shape=(10000,))
            t2 = random.uniform(keys[5], shape=(10000,))
            u1 = x1 + t1[:, None]*p1
            u2 = x2 + t2[:, None]*p2
            gamma = jnp.linalg.norm(u1-u2,axis=1)**2
            hist, bins = jnp.histogram(gamma.flatten(), density=True, bins=100)
            w = tomo_weight(bins, x1, x2, p1, p2)
            w_ref = tomo_weight_ref(bins, x1, x2, p1, p2)
            rel_error.append(jnp.max(jnp.abs(w - w_ref)) / jnp.max(w_ref))
        rel_error = jnp.array(rel_error)
        plt.hist(rel_error,bins ='auto')
        plt.title("{} : {:.2f}|{:.2f}|{:.2f}".format(S, *jnp.percentile(rel_error, [5,50,95])))
        plt.show()


def test_get_Q():
    Q = get_polynomial_form()
    import pylab as plt
    from jax import jit

    @jit
    def tomo_weight_ref(gamma, x1, x2, p1, p2):
        return tomographic_weight_function(gamma, x1, x2, p1, p2, S=150)

    @jit
    def cumulative_tomo_weight_function_dimensionless(gamma, x1, x2, p1, p2):
        x12 = x1 - x2
        h = jnp.linalg.norm(x12)
        n = x12 / h
        w1 = p1 / h
        w2 = p2 / h
        gamma_prime = gamma / h ** 2
        return cumulative_tomographic_weight_dimensionless_function(gamma_prime, n, w1, w2, S=150)

    @jit
    def cumulative_tomo_weight_polynomial_dimensionless(gamma, x1, x2, p1, p2):
        x12 = x1 - x2
        h = jnp.linalg.norm(x12)
        n = x12 / h
        w1 = p1 / h
        w2 = p2 / h
        gamma_prime = gamma / h ** 2
        return vmap(lambda gamma_prime: cumulative_tomographic_weight_dimensionless_polynomial(Q, gamma_prime, n, w1, w2))(gamma_prime)
        # return jnp.exp(log_tomographic_weight_dimensionless_function(gamma_prime, n, w1, w2, S=150)) / h ** 2

    for i in range(10):
        keys = random.split(random.PRNGKey(i), 6)
        x1 = jnp.concatenate([10. * random.uniform(keys[0], shape=(2,)), jnp.zeros((1,))], axis=-1)
        p1 = jnp.concatenate(
            [4. * jnp.pi / 180. * random.uniform(keys[1], shape=(2,), minval=-1, maxval=1),
             jnp.ones((1,))], axis=-1)
        p1 = 4 * p1 / jnp.linalg.norm(p1, axis=-1, keepdims=True)

        x2 = jnp.concatenate([4. * random.uniform(keys[2], shape=(2,)), jnp.zeros((1,))], axis=-1)
        p2 = jnp.concatenate(
            [4. * jnp.pi / 180. * random.uniform(keys[3], shape=(2,), minval=-1, maxval=1),
             jnp.ones((1,))], axis=-1)
        p2 = 4 * p2 / jnp.linalg.norm(p2, axis=-1, keepdims=True)

        t1 = random.uniform(keys[4], shape=(10000,))
        t2 = random.uniform(keys[5], shape=(10000,))
        u1 = x1 + t1[:, None] * p1
        u2 = x2 + t2[:, None] * p2
        gamma = jnp.linalg.norm(u1 - u2, axis=1) ** 2
        plt.hist(gamma.flatten(), bins=100, density=True, label='histogram')
        hist, bins = jnp.histogram(gamma.flatten(), density=True, bins=100)
        gamma = 0.5 * (bins[:-1] + bins[1:])
        w_ref = tomo_weight_ref(bins, x1, x2, p1, p2)
        plt.plot(gamma, w_ref, label='analytic ref')
        plt.legend()
        plt.show()
        cdf_ref = cumulative_tomo_weight_function_dimensionless(gamma, x1, x2, p1, p2)
        cdf_poly = cumulative_tomo_weight_polynomial_dimensionless(gamma, x1, x2, p1, p2)
        gamma_prime = gamma/jnp.linalg.norm(x1-x2)
        plt.plot(gamma_prime,cdf_ref,label='ref')
        plt.plot(gamma_prime,cdf_poly, label='poly')
        plt.legend()
        plt.show()


def test_clip_algebra():
    z = jnp.linspace(-1., 2., 10)
    def max0(z):
        return 0.5*(z + jnp.abs(z))
    assert jnp.all(max0(z)==jnp.maximum(z,0.))

    def min1(z):
        #min(z,1) = min(z-1,0) + 1 = -max(1-z,0) + 1
        # return jnp.minimum(z-1,0) + 1.# -max0(1.-z) + 1.
        return  -max0(1.-z) + 1.

    assert jnp.all(jnp.isclose(min1(z), jnp.minimum(z, 1.)))

    def clip01(z):
        # 1/2 + 1/4 * (z+abs(z))  - abs(1/2 - 1/4 * (z+abs(z)))
        z = 0.25*z
        z = z + jnp.abs(z)
        return 0.5 + z - jnp.abs(0.5 - z)
        # return min1(max0(z))

    assert jnp.all(jnp.isclose(clip01(z) , jnp.clip(z, 0., 1.)))