import jax.numpy as jnp


def fourier(a, *coords):
    """
    Evaluates

    F[a](s) = int a(x) e^{-2pi i s x} dx
    A(k ds) = sum_m a(x_m) e^{-2pi i k ds (x0 + corner_indices dx)} dx
            = e^{-2pi i k ds x0} dx sum_m a(x_m) e^{-2pi i k ds corner_indices dx}
    dx ds = 1/n => ds = 1/(dx n)
    ds x0 = k ds x0 = k/n * x0/dx
    """

    factor = fft_factor(*coords)
    return jnp.fft.fftshift(jnp.fft.fftn(a) * factor)


def inv_fourier(A, *coords):
    factor = ifft_factor(*coords)
    return jnp.fft.ifftn(jnp.fft.ifftshift(A) * factor)


def fft_freqs(*coords):
    s = []
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        s.append(jnp.fft.fftshift(jnp.fft.fftfreq(c.size, dx)))
    return tuple(s)


def ifft_freqs(*coords):
    x = []
    for i, c in enumerate(coords):
        ds = c[1] - c[0]
        x.append(jnp.fft.fftshift(jnp.fft.fftfreq(c.size, ds)))
    return tuple(x)


def test_coords_transformations():
    import numpy as np
    assert jnp.allclose(np.fft.fftfreq(10, 1.), jnp.fft.fftfreq(10, 1.))
    assert jnp.allclose(np.fft.fftfreq(11, 1.), jnp.fft.fftfreq(11, 1.))
    #odd
    x = jnp.linspace(-10, 10, 101)
    (s,) = fft_freqs(x)
    print(s)
    _x = ifft_freqs(s)
    print(x)
    print(_x)
    assert jnp.isclose(_x, x).all()
    # #even
    # x = jnp.linspace(-10, 10, 100)
    # (s,) = fft_freqs(x)
    # _x = ifft_freqs(s)
    # print(x)
    # print(_x)
    # assert jnp.isclose(_x, x).all()


def fft_factor(*coords):
    def _add_dims(t, i):
        dims = list(range(len(coords)))
        del dims[i]
        return jnp.expand_dims(t, tuple(dims))

    log_factors = 0.
    dx_factor = 1.
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        x0 = c[0]
        s = jnp.fft.fftfreq(c.size, dx)
        log_factor = - 2j * jnp.pi * s * x0
        dx_factor *= dx
        log_factors = log_factors + _add_dims(log_factor, i)
    factor = jnp.exp(log_factors) * dx_factor
    return factor


def ifft_factor(*coords):
    def _add_dims(t, i):
        dims = list(range(len(coords)))
        del dims[i]
        return jnp.expand_dims(t, tuple(dims))

    log_factors = 0.
    dx_factor = 1.
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        x0 = c[0]
        s = jnp.fft.fftfreq(c.size, dx)
        dx_factor /= dx
        log_factor = 2j * jnp.pi * s * x0
        log_factors = log_factors + _add_dims(log_factor, i)
    factor = jnp.exp(log_factors) * dx_factor
    return factor


def test_fourier():
    def f(x):
        return jnp.exp(-jnp.pi * x ** 2)

    import pylab as plt
    x = jnp.linspace(-10., 10., 101)
    a = f(x)

    F = fourier(a, x)
    (s,) = fft_freqs(x)
    _a = inv_fourier(F, x)

    plt.plot(s, f(s), label='A true')
    plt.plot(s, jnp.real(F), label='A numeric')
    plt.legend()
    plt.show()

    plt.plot(x, a, label='a')
    plt.plot(x, _a, label='a rec')
    plt.legend()
    # plt.ylim(-10,3)
    plt.show()