from jax.config import config
config.update("jax_enable_x64", True)
from jaxns.likelihood_samplers.ellipsoid_utils import minimum_volume_enclosing_ellipsoid, sample_ellipsoid
from jax import numpy as jnp, random
from jax.lax import while_loop, dynamic_update_slice, scan

def debug_triangular_solve():
    from jax.lax_linalg import triangular_solve
    from scipy.linalg import solve_triangular
    from jax.scipy.linalg import solve_triangular as solve_triangular_jax
    import jax.numpy as jnp
    ndims = 2
    A = jnp.diag(jnp.ones(ndims))
    A = jnp.where(A == 0., 0.95, A)
    b = jnp.ones(ndims)
    L = jnp.linalg.cholesky(A)
    assert jnp.all(L @ L.T == A)

    x = jnp.linalg.solve(L, b)
    print("Solving L.x  = b with scipy")
    print("x should be {}".format(x))
    scipy_x = solve_triangular(L, b, lower=True)
    assert jnp.all(scipy_x == x)
    print("Works as expected!")

    print("Now note JAX's solution to L^T.x = b corresponds to scipy's L.x = b")
    jax_x = triangular_solve(L, b, lower=True, transpose_a=True)
    assert jnp.all(jax_x==scipy_x)
    print("Likewise, JAX's solution to L.x = b corresponds to scipy's L^T.x = b")
    assert jnp.all(triangular_solve(L, b, lower=True) == solve_triangular(L, b, lower=True, trans=1))

    print("Note, I have not tested for the L^H.x=b case.")

    jax_x = solve_triangular_jax(L, b, lower=True)
    assert jnp.all(scipy_x == jax_x)


def debug_mvee():
    import pylab as plt

    n = random.normal(random.PRNGKey(0), (10000,2))
    n = n /jnp.linalg.norm(n, axis=1, keepdims=True)
    angle = jnp.arctan2(n[:,1], n[:,0])
    plt.hist(angle, bins=100)
    plt.show()
    N = 120
    D = 2
    points = random.uniform(random.PRNGKey(0), (N, D))

    from jax import disable_jit
    with disable_jit():
        center, radii, rotation = minimum_volume_enclosing_ellipsoid(points, 0.01)

    plt.hist(jnp.linalg.norm((rotation.T @ (points.T - center[:, None])) / radii[:, None], axis=0))
    plt.show()
    print(center, radii, rotation)
    plt.scatter(points[:, 0], points[:, 1])
    theta = jnp.linspace(0., jnp.pi*2, 100)
    ellipsis = center[:, None] + rotation @ jnp.stack([radii[0]*jnp.cos(theta), radii[1]*jnp.sin(theta)], axis=0)
    plt.plot(ellipsis[0,:], ellipsis[1,:])

    for i in range(1000):
        y = sample_ellipsoid(random.PRNGKey(i), center, radii, rotation)
        plt.scatter(y[0], y[1])



    C = jnp.linalg.pinv(jnp.cov(points, rowvar=False, bias=True))
    p = (N - D - 1)/N
    def q(p):
        return p + p**2/(4.*(D-1))
    C = C / q(p)
    c = jnp.mean(points, axis=0)
    W, Q, Vh = jnp.linalg.svd(C)
    radii = jnp.reciprocal(jnp.sqrt(Q))
    rotation = Vh.conj().T
    ellipsis = c[:, None] + rotation @ jnp.stack([radii[0] * jnp.cos(theta), radii[1] * jnp.sin(theta)], axis=0)
    plt.plot(ellipsis[0, :], ellipsis[1, :])

    plt.show()

def debug_riddle():
    import numpy as np
    import pylab as plt

    t = np.linspace(0., 12., 10000)
    m = (t - np.floor(t))*2*np.pi
    h = t/12.*2.*np.pi




    plt.plot(t, h)
    plt.show()
    plt.plot(t, m)
    plt.show()

    def time(m, h):
        return h/(2.*np.pi)*12.

    plt.plot(t, time(m,h))
    plt.plot(t, time(h,m))
    plt.show()

def debug_vmap_bfgs():
    import jax.numpy as jnp
    from jax import jit, config
    from jax.scipy.optimize import minimize
    import os
    config.enable_omnistaging()
    ncpu=2
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"

    def cost_fn(x):
        return -jnp.sum(x**2)

    x = random.uniform(random.PRNGKey(0), (3,), minval=-1, maxval=1)
    result = jit(lambda x: minimize(cost_fn, x, method='BFGS'))(x)
    print(result)

def debug_random_angles():
    from jax import random
    import jax.numpy as jnp
    import pylab as plt
    S = 10000
    p1 = random.uniform(random.PRNGKey(0),shape=(S,2))
    p1 /= jnp.linalg.norm(p1, axis=-1, keepdims=True)
    plt.hist(jnp.arctan2(p1[:,1],p1[:,0]), bins='auto', alpha=0.5)
    p2 = random.normal(random.PRNGKey(0), shape=(S, 2))
    p2 /= jnp.linalg.norm(p2, axis=-1, keepdims=True)
    plt.hist(jnp.arctan2(p2[:,1],p2[:,0]), bins='auto', alpha=0.5)
    plt.show()

def debug_simplex_sampler():
    from jax import random
    import jax.numpy as jnp
    import pylab as plt
    from jaxns.gaussian_process.utils import squared_norm
    S = 10
    D = 2
    u = random.uniform(random.PRNGKey(0),shape=(S,2))
    plt.scatter(u[:,0],u[:,1])
    inter_point_distance = squared_norm(u, u)
    inter_point_distance = jnp.where(inter_point_distance < 1e-20, jnp.inf, inter_point_distance)
    knn_indices = jnp.argsort(inter_point_distance, axis=-1)[:, :D + 1]
    i = 2
    plt.scatter(u[i,0],u[i,1],c='red')
    for j in range(S):
        for k in range(D+1):
            plt.plot([u[i,0], u[knn_indices[i,k],0]],[u[i,1], u[knn_indices[i,k],1]])
    plt.show()

def debug_fft():
    from jax.config import config

    config.update("jax_enable_x64", True)

    import time
    import numpy as np

    import jax
    from jax import numpy as jnp

    np.random.seed(0)

    signal = np.random.randn(2 ** 20)
    signal_jax = jnp.array(signal)

    jfft = jax.jit(jnp.fft.fft)

    import tensorflow as tf
    signal_tf = tf.constant(signal, dtype=tf.complex128)

    def tffft(x):
        return tf.signal.fft(x).numpy()

    X_np = np.fft.fft(signal)
    X_jax = jfft(signal_jax)
    X_tf = tffft(signal_tf)


    print(np.mean(np.abs(X_np)))
    print("With JAX:")
    print('max:\t', jnp.max(jnp.abs(X_np - X_jax)))
    print('mean:\t', jnp.mean(jnp.abs(X_np - X_jax)))
    print('min:\t', jnp.min(jnp.abs(X_np - X_jax)))

    print("With Tensorflow:")
    print('max:\t', jnp.max(jnp.abs(X_np - X_tf)))
    print('mean:\t', jnp.mean(jnp.abs(X_np - X_tf)))
    print('min:\t', jnp.min(jnp.abs(X_np - X_tf)))

    ### CPU
    # 907.3490574884647
    # max:	 2.8773885332210747
    # mean:	 0.3903197564919141
    # min:	 2.4697454729898156e-05

    ### GPU
    # 907.3490574884647
    # max:	 0.001166179716824765
    # mean:	 0.00020841654559267488
    # min:	 2.741492442122853e-07

    R = 100
    ts = time.time()
    for i in range(R):
        _ = np.fft.fft(signal)
    print('numpy fft execution time [ms]:\t', (time.time() - ts) / R * 1000)

    # Compile
    _ = jfft(signal_jax).block_until_ready()

    ts = time.time()
    for i in range(R):
        _ = jfft(signal_jax).block_until_ready()
    print('jax fft execution time [ms]:\t', (time.time() - ts) / R * 1000)

    ts = time.time()
    for i in range(R):
        _ = tffft(signal_tf)
    print('tensorflow fft execution time [ms]:\t', (time.time() - ts) / R * 1000)

def delay_accuracy():
    import astropy.units as au
    from jax import vmap
    V = 30.*au.km/au.s
    c = 299792458.*au.m/au.s
    def delay_perfect(b):
        return -b/c * (1. - 0.5*V**2/c**2) - b*V/c**2 * (1. + 0.5*V/c)/(1. + V/c)
    def delay_nonperfect(b):
        return -b/c
    def delay(b,s, V):
        return (-(b@s)/c * (1. - 0.5*V@V/c**2) - (b @ V)/c**2 * (1. + 0.5*(s@V)/c))/(1.+(s@V)/c)

    for _b in 10**jnp.linspace(0., 4, 10):
        delay_diff = []
        for i in range(1000):
            keys = random.split(random.PRNGKey(i), 3)
            b = random.normal(keys[0], shape=(3,))
            b /= jnp.linalg.norm(b)
            b = b * _b * au.km
            s = random.normal(keys[1], shape=(3,))
            s /= jnp.linalg.norm(s)
            V = random.normal(keys[2], shape=(3,))
            V /= jnp.linalg.norm(V)
            V = V * 30.*au.km/au.s
            delay_diff.append(jnp.abs(delay(b, s, V).to(au.ns).value - delay(b, s, 0*V).to(au.ns).value))
        delay_diff = jnp.array(delay_diff)
        print("Baseline={:.2f}, max diff={:.2f}ns, mean diff={:.2f}ns".format(
            _b*au.km, jnp.max(delay_diff), jnp.mean(delay_diff)
        ))
        # print('For baseline of ',b,'magnitude of relativistic corrections', -delay_perfect(b*au.km).to(au.ns)+delay_nonperfect(b*au.km).to(au.ns))

def quick_select(k, a):
    """
    Performs iterative quick select.
    Args:
        k: the kth smallest element to select
        a: the 1D array to select from

    Returns: the kth smallest element in `a`
    """

    def partition(arr, low, high):
        """
        Lomuto partition function.
        """
        if len(arr.shape) > 1:
            raise ValueError("Partition works on 1D arrays. Use vmap.")
        pivot = arr[high]
        def body(state):
            (j, i,arr) = state
            do_swap = arr[j] <= pivot
            i = jnp.where(do_swap, i+1, i)
            ai = arr[i, None]
            aj = arr[j, None]
            arr_swapped = dynamic_update_slice(arr, aj, [i])
            arr_swapped = dynamic_update_slice(arr_swapped, ai, [j])
            arr_swapped = jnp.where(do_swap, arr_swapped, arr)
            return (j + 1, i, arr_swapped)

        (j, i, arr) = while_loop(lambda state: state[0] < high,
                              body,(low, low - 1, arr))

        ai = arr[i+1, None]
        aj = arr[high, None]
        arr_swapped = dynamic_update_slice(arr, aj, [i+1])
        arr_swapped = dynamic_update_slice(arr_swapped, ai, [high])
        return (i + 1, arr_swapped)

    def body(state):
        (left, right, _, _, arr) = state
        pivot_index, arr = partition(arr, left, right)
        done = pivot_index == k-1
        kth_smallest = arr[pivot_index]
        right = jnp.where(pivot_index > k-1, pivot_index - 1, right)
        left = jnp.where(pivot_index < k-1, pivot_index + 1, left)
        return (left, right, done, kth_smallest, arr)

    (left, right, _, kth_smallest, arr) = while_loop(lambda state: (state[0]<=state[1]) & (~state[2]),
                               body,
                               (jnp.asarray(0), jnp.asarray(len(a) - 1), jnp.array(False), a[0], a))
    return kth_smallest

def quick_sort_median(a):
    """
    Performs a quickselect median which is equivalent to,
        jnp.percentile(a, 50, interpolation='higher')
    Thus, for odd sized arrays it's the same as jnp.median.
    Args:
        a: 1D array. To do ND arrays use vmap.

    Returns: median of array choosing the 'higher' of the two middle values
    when `a` has an even number of points.
    """
    if len(a.shape) > 1:
        raise ValueError("Only works on 1D arrays. Use vmap.")
    k = a.size//2 + 1
    median = quick_select(k, a)
    return median

def test_quick_select():

    arr = jnp.asarray([10, 4, 5, 8, 11, 6, 26, 7]) # even number of points
    assert quick_sort_median(arr) == jnp.percentile(arr, 50, interpolation='higher')
    arr = jnp.asarray([10, 4, 5, 8, 11, 6, 26]) # odd number of points
    assert quick_sort_median(arr) == jnp.percentile(arr, 50, interpolation='higher')
    arr = jnp.asarray([10, 4, 5, 8, 11, 6, 26])  # odd number of points
    assert quick_sort_median(arr) == jnp.median(arr)

    arr = jnp.asarray([10, 4, 5, 8, 11, 6, 26, 7])  # even number of points
    try:
        assert quick_sort_median(arr) == jnp.median(arr)
    except AssertionError:
        print("Not equivalent to median when array has an even size.")

def test_median_speed():
    from jax import jit, random
    import numpy as np
    from timeit import default_timer
    import pylab as plt

    jax_median = jit(jnp.median)
    quick_median = jit(quick_sort_median)

    n = []
    jax_time = []
    quick_time = []
    numpy_time = []
    for b in range(1,15):
        random_array = random.uniform(random.PRNGKey(b), shape=(2**b + 1,))
        jax_median(random_array)
        quick_median(random_array)
        for i in range(10):
            n.append(2**b + 1)
            # random_array = random.uniform(random.PRNGKey(b+1), shape=(2 ** b + 1,))
            t0 = default_timer()
            print(jax_median(random_array))
            jax_time.append(default_timer() - t0)
            t0 = default_timer()
            print(quick_median(random_array))
            quick_time.append(default_timer() - t0)
            random_array = np.array(random_array)
            t0 = default_timer()
            print(np.median(random_array))
            numpy_time.append(default_timer() - t0)
    plt.scatter(n, jax_time, label='jax median')
    plt.scatter(n, quick_time, label='quick select median')
    plt.scatter(n, numpy_time, label='numpy median')
    plt.legend()
    plt.yscale('log')
    plt.ylabel('time (s)')
    plt.show()


def test_speed_jax():

    import numpy as np
    from jax import jit,value_and_grad, random
    from jaxns.optimize import minimize as minimize_jax
    from scipy.optimize import minimize as minimize_np
    import pylab as plt
    from timeit import default_timer

    S = 3
    t_scipy_halfjax,t_scipy_jax,t_jax,t_numpy = [],[],[],[]
    N_array = [2,10,50, 100, 200, 400]
    for N in N_array:
        print("Working on N={}".format(N))

        A = random.normal(random.PRNGKey(0), shape=(N, N))
        u = jnp.ones(N)
        x0 = -2. * jnp.ones(N)

        def f_prescale(x, u):
            y = A @ x
            dx = u - y
            return jnp.sum(dx**2) + 0.1*jnp.sum(jnp.abs(x))

        # Due to https://github.com/google/jax/issues/4594 we scale the loss
        # so that scipy and jax linesearch perform similarly.
        jac_norm = jnp.linalg.norm(value_and_grad(f_prescale)(x0, u)[1])
        jac_norm_np = np.array(jac_norm)

        def f(x, u):
            y = A @ x
            dx = u - y
            return (jnp.sum(dx**2) + 0.1*jnp.sum(jnp.abs(x)))/jac_norm

        def f_np(x, u):
            y = A @ x[:,None]
            dx = u - y
            return (np.sum(dx**2) + 0.1*np.sum(np.abs(x)))/jac_norm_np

        print("Testing scipy+numpy")
        t0 = default_timer()
        args= (np.array(x0), (np.array(u),))
        results_np = minimize_np(f_np, *args, method='BFGS')
        for _ in range(S):
            results_np = minimize_np(f_np, *args, method='BFGS')
        t_numpy.append((default_timer() - t0) / S)
        print("nfev",results_np.nfev, "njev", results_np.njev)
        print("Time for scipy + numpy", t_numpy[-1])


        print("Testing scipy + jitted function and numeric grad")
        @jit
        def _f(x0, u):
            return f(x0, u)

        _f(x0, u).block_until_ready()
        t0 = default_timer()
        for _ in range(S):
            results_np = minimize_np(_f, x0, (u,), method='BFGS')
        t_scipy_halfjax.append((default_timer() - t0) / S)
        print("nfev",results_np.nfev, "njev", results_np.njev)
        print("Time for scipy + jitted function and numeric grad", t_scipy_halfjax[-1])

        print("Testing scipy + jitted function and grad")
        @jit
        def _f(x0, u):
            v, g = value_and_grad(f)(x0, u)
            return v, g
        _f(x0, u)[1].block_until_ready()
        t0 = default_timer()
        for _ in range(S):
            results_np = minimize_np(_f, x0, (u,), method='BFGS', jac=True)
        t_scipy_jax.append((default_timer() - t0) / S)
        print("nfev",results_np.nfev, "njev", results_np.njev)
        print("Time for scipy + jitted function and grad", t_scipy_jax[-1])


        print("Testing pure JAX implementation")
        @jit
        def do_minimize_jax(x0, u):
            results = minimize_jax(f, x0, args=(u,),method='BFGS')
            return results.x

        results_jax = minimize_jax(f, x0, args=(u,),method='BFGS')
        do_minimize_jax(x0, u).block_until_ready()

        t0 = default_timer()
        for _ in range(S):
            do_minimize_jax(x0, u).block_until_ready()
        t_jax.append((default_timer() - t0)/S)
        print("nfev", results_jax.nfev, "njev", results_jax.njev)
        print("Time for pure JAX implementation", t_jax[-1])

    plt.figure(figsize=(8,5))
    plt.plot(N_array,t_scipy_jax,label='scipy+jitted(func and grad)')
    plt.plot(N_array,t_scipy_halfjax,label='scipy+jitted(func)')
    plt.plot(N_array,t_jax,label='pure JAX')
    plt.plot(N_array,t_numpy,label='scipy+numpy')
    plt.yscale('log')
    plt.legend()
    plt.title("Run time of BFGS on N-D Least squares + L1 regularisation.")
    plt.ylabel('Time [s]')
    plt.xlabel("N")
    plt.show()


def test_speed_MM_jax():
    from jax import jit,vmap
    import numpy as np
    import pylab as plt
    from timeit import default_timer
    S = 10
    plt.figure(figsize=(8, 5))
    t1,t2 = [],[]
    for N in [100,200,400,800,1600,3200]:

        M1 = jnp.ones((N,N))
        M2 = jnp.ones((N,N))
        M3 = jnp.ones((N,N))

        def mm(M1,M2,M3):
            return (M1**2 @ M2**2 + M3**2).sum()**2

        _M1 = np.array(M1)
        _M2 = np.array(M2)
        _M3 = np.array(M3)
        t0 = default_timer()
        for _ in range(S):
            results_jax = mm(_M1, _M2, _M3)
        t1.append((default_timer() - t0) / S)
        print("Time for numpy", t1)

        _mm = jit(mm)
        _mm(M1,M2,M3).block_until_ready()
        t0 = default_timer()
        for _ in range(S):
            results_jax = _mm(M1,M2,M3).block_until_ready()
        t2.append((default_timer() - t0) / S)
        print("Time for jax", t2)

    plt.plot([100,200,400,800,1600,3200], t1, label='numpy')
    plt.plot([100,200,400,800,1600,3200], t2, label='jax')
    plt.legend()
    plt.yscale('log')
    plt.xlabel("N")
    plt.title("GEMM: M1 @ M2 + M3")
    plt.ylabel('Time [s]')
    plt.show()





def test_example_omni():
    import jax.numpy as jnp
    from jax import jit

    @jit
    def select_tril(x):
        mask = jnp.arange(x.shape[0])[:, None] > jnp.arange(x.shape[1])
        return jnp.where(mask, x, jnp.zeros_like(x))

    import numpy as np
    x = np.arange(12).reshape((3, 4)) # mix in numpy as desired
    select_tril(x)




if __name__ == '__main__':
    # test_example_omni()
    test_speed_jax()
    # delay_accuracy()
    # debug_vmap_bfgs()
    # debug_random_angles()
    # debug_simplex_sampler()
    # debug_triangular_solve()
    # debug_fft()
    # debug_nestest_sampler()
    # debug_tec_clock_inference()
    # debug_riddle()
    # debug_mvee()