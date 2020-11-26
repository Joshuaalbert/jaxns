from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as jnp, vmap

from jaxns.gaussian_process.utils import squared_norm, product_log


def test_squared_norm():
    x = jnp.linspace(0., 1., 100)[:, None]
    y = jnp.linspace(1., 2., 50)[:, None]
    assert jnp.all(jnp.isclose(squared_norm(x, x), jnp.sum(jnp.square(x[:, None, :] - x[None, :, :]), axis=-1)))
    assert jnp.all(jnp.isclose(squared_norm(x, y), jnp.sum(jnp.square(x[:, None, :] - y[None, :, :]), axis=-1)))


def test_integrate_sqrt_quadratic():
    a = 1.
    b = 2.
    c = 3.
    ans = jnp.sqrt(a + b + c) - ((b + c) * jnp.log(jnp.sqrt(a) * jnp.sqrt(b + c))) / jnp.sqrt(a) + (
                (b + c) * jnp.log(a + jnp.sqrt(a) * jnp.sqrt(a + b + c))) / jnp.sqrt(a)

    u = jnp.linspace(0., 1., 1000)
    ref = jnp.sum(jnp.sqrt(a * u**2 + b * u + c))/(u.size-1)
    print(ref,ans)


def test_product_log():
    from scipy.special import lambertw

    # import pylab as plt
    w = jnp.linspace(-1./jnp.exp(1)+0.001, 0., 100)
    # plt.plot(w, lambertw(w, 0))
    # plt.plot(w, vmap(product_log)(w))
    # plt.show()

    assert jnp.all(jnp.isclose(vmap(product_log)(w), lambertw(w, 0), atol=1e-2))


def test_fourier_tomographic_kernel():
    from jaxns.gaussian_process.kernels import RBF, RationalQuadratic, M12
    from jaxns.gaussian_process.fourier import fft_freqs, fourier, inv_fourier
    from jax import vmap,jit,random
    x = jnp.linspace(-60., 60., 220)
    X,Y = jnp.meshgrid(x,x,indexing='ij')
    X = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    kernel = RationalQuadratic()
    K = kernel(X, jnp.zeros((1,2)), 10., 1., 11.).reshape([x.size]*2)
    sx,sy = fft_freqs(x,x)
    Sx,Sy = jnp.meshgrid(sx,sy,indexing='ij')
    S = jnp.stack([Sx,Sy], axis=-1)
    Sk = fourier(K, x, x)

    a = 100.
    b = 300.
    def tomographic_kernel(n1, n2):
        Sn1 = jnp.sum(S * n1, axis=-1)
        Sn2 = jnp.sum(S * n2, axis=-1)
        # print(n1[2])
        efac_f1 = jnp.exp(2j * jnp.pi * Sn1*b/n1[2])
        efac_f2 = jnp.exp(-2j * jnp.pi * Sn2*b/n2[2])
        efac_i1 = jnp.exp(2j * jnp.pi * Sn1 *a/n1[2])
        efac_i2 = jnp.exp(-2j * jnp.pi * Sn2 *a/n2[2])
        # print((efac_f1 - efac_i1), (efac_f2 - efac_i2))
        signal = Sk * (efac_f1 - efac_i1) * (efac_f2 - efac_i2) / (4. * jnp.pi ** 2 * Sn1 * Sn2)
        signal = jnp.where(jnp.isnan(signal), 0., signal.real)
        return jnp.sum(signal)*(sx[1]-sx[0])*(sy[1]-sy[0])

    T = tomographic_kernel(jnp.array([0.,1.]), jnp.array([0.1,0.9]))

    theta = jnp.linspace(-10.*jnp.pi/180., 10.*jnp.pi/180., 100)
    def to_cartesian(theta):
        return jnp.asarray([jnp.sin(theta), jnp.cos(theta)])

    n = vmap(to_cartesian)(theta)

    K_dir = jit(vmap(lambda n1: vmap(lambda n2:tomographic_kernel(n1,n2))(n)))(n)
    import pylab as plt
    plt.imshow(K_dir,origin='lower', extent=(-10., 10., -10., 10.), cmap='jet')
    plt.xlabel("Zenith [deg]")
    plt.ylabel("Zenith [deg]")
    plt.title("TEC covariance: {}, layer {} to {} km".format(kernel.__class__.__name__, a, b))
    plt.colorbar()
    plt.show()

    from jaxns.utils import msqrt
    L = msqrt(K_dir)
    Z = L@random.normal(random.PRNGKey(3245), shape=(L.shape[0],1))
    Z = Z.reshape((100,))
    plt.plot(theta, Z)
    plt.xlabel('Zenith angle [deg]')
    plt.ylabel('TEC [mTECU]')
    plt.title("Example realisation of TEC")
    plt.show()


def test_frozenflow_tomographic_kernel():
    from jaxns.gaussian_process.kernels import RBF, RationalQuadratic, M12
    from jaxns.gaussian_process.fourier import fft_freqs, fourier, inv_fourier
    from jax import vmap,jit,random
    x = jnp.linspace(-60., 60., 220)
    X,Y = jnp.meshgrid(x,x,indexing='ij')
    X = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    kernel = RBF()
    K = kernel(X, jnp.zeros((1,2)), 10., 1.).reshape([x.size]*2)
    sx,sy = fft_freqs(x,x)
    Sx,Sy = jnp.meshgrid(sx,sy,indexing='ij')
    S = jnp.stack([Sx,Sy], axis=-1)
    Sk = fourier(K, x, x)

    a = 100.
    b = 300.
    def tomographic_kernel(x, n1, n2):
        Sn1 = jnp.sum(S * n1, axis=-1)
        Sn2 = jnp.sum(S * n2, axis=-1)
        # print(n1[2])
        efac_f1 = jnp.exp(2j * jnp.pi * Sn1*b/n1[2])
        efac_f2 = jnp.exp(-2j * jnp.pi * Sn2*b/n2[2])
        efac_i1 = jnp.exp(2j * jnp.pi * Sn1 *a/n1[2])
        efac_i2 = jnp.exp(-2j * jnp.pi * Sn2 *a/n2[2])
        # print((efac_f1 - efac_i1), (efac_f2 - efac_i2))
        signal = Sk * (efac_f1 - efac_i1) * (efac_f2 - efac_i2) / (4. * jnp.pi ** 2 * Sn1 * Sn2)
        signal = jnp.where(jnp.isnan(signal), 0., signal)
        return jnp.sum(signal*jnp.exp(2j*jnp.pi*(x[0]*Sx + x[1]*Sy))).real*(sx[1]-sx[0])*(sy[1]-sy[0])

    T = tomographic_kernel(jnp.array([0.,1.]), jnp.array([0.,1.]), jnp.array([0.1,0.9]))

    theta = jnp.linspace(-10.*jnp.pi/180., 10.*jnp.pi/180., 100)
    def to_cartesian(theta):
        return jnp.asarray([jnp.sin(theta), jnp.cos(theta)])

    n = vmap(to_cartesian)(theta)

    n = jnp.array([0.,1.])

    v = 0.2

    time = jnp.arange(200)*4.
    vt = v*time
    K_dir = []
    _K_dir = jit(lambda y: vmap(lambda x: tomographic_kernel((x - y) * jnp.array([0., 1.]), n, n))(vt))
    for u in vt:
        K_dir.append(_K_dir(u))
    K_dir = jnp.stack(K_dir,axis=0) #* kernel(vt[:,None],vt[:,None],2000.,1.)

    import pylab as plt
    plt.imshow(K_dir,origin='lower', extent=(0,time.max(),0,time.max()), cmap='jet')
    plt.xlabel("Time [s]")
    plt.ylabel("Time [s]")
    plt.title("Frozen-flow TEC covariance: {}, layer {} to {} km".format(kernel.__class__.__name__, a, b))
    plt.colorbar()
    plt.show()

    from jaxns.utils import msqrt
    L = msqrt(K_dir)
    Z = L@random.normal(random.PRNGKey(3245), shape=(L.shape[0],1))
    Z = Z.reshape((time.size,))
    plt.plot(time, Z)
    plt.xlabel('Time [s]')
    plt.ylabel('TEC [mTECU]')
    plt.title("Example realisation of TEC")
    plt.show()

