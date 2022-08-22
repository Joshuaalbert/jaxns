import jax.numpy as jnp


def fourier(a, *coords):
    """
    Evaluates

    .. math::
    
        F[a](s) = \int a(x) \exp(-2\pi i s x) dx,
    
    and,
    
    .. math::
    
        A(k ds) = \sum_m a(x_m) \exp(-2\pi i k ds (x_0 + corner_indices dx)) dx
                = \exp(-2\pi i k ds x_0) dx \sum_m a(x_m) \exp(-2\pi i k ds corner_indices dx)
        
    where,
    
    .. math::
        
        dx ds = 1/N \\rightarrow ds = 1/(dx N)
        
    and,
    
    .. math::
    
        ds x_0 = k ds x_0 = k/N x_0/dx
    
    where :math:`N` is the number of options, and...
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
