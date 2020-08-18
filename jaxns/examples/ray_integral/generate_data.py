from jaxns.examples.ray_integral.fed_kernels import rbf_act
from jaxns.examples.ray_integral.tomographic_kernel import dtec_tomographic_kernel
import jax.numpy as jnp
from jax import random

from jaxns.examples.ray_integral.utils import msqrt, make_coord_array


def rbf_dtec(nant, ndir, height, width, sigma, l, uncert=1.):
    import pylab as plt
    a = jnp.concatenate([10.*random.uniform(random.PRNGKey(0), shape=(nant,2)),jnp.zeros((nant, 1))], axis=1)
    k = jnp.concatenate([4.*jnp.pi/180.*random.uniform(random.PRNGKey(0), shape=(ndir,2), minval=-1, maxval=1),jnp.ones((ndir, 1))], axis=1)
    k = k / jnp.linalg.norm(k,axis=1, keepdims=True)
    X = make_coord_array(a, k)
    a = X[:,0:3]
    k = X[:,3:6]
    K = dtec_tomographic_kernel(random.PRNGKey(1), a[0,:], a, a, k, k, jnp.zeros(3), rbf_act, height, width, l, S=100, sigma=sigma)
    plt.imshow(K)
    plt.colorbar()
    plt.show()
    plt.plot(jnp.sqrt(jnp.diag(K)))
    plt.show()


    L = msqrt(K)#jnp.linalg.cholesky(K + jnp.eye(K.shape[0])*1e-3)

    dtec = L @ random.normal(random.PRNGKey(2), shape=(L.shape[0],))
    plt.plot(dtec)
    plt.show()
    return X, dtec, dtec + uncert*random.normal(random.PRNGKey(3),shape=dtec.shape)

def main():
    rbf_dtec(10,10,200., 100., 0.3, 10., 1.)

if __name__=='__main__':
    main()
