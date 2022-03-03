from jax import numpy as jnp

from jaxns.modules.experimental.fourier.fourier import fft_freqs, ifft_freqs, fourier, inv_fourier


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