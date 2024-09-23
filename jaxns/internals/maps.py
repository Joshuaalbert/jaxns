import inspect
import warnings
from typing import TypeVar, Callable, Optional, Tuple, List, Union, Any

import jax
import numpy as np
from jax import pmap, numpy as jnp, lax
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from jax.experimental.mesh_utils import create_device_mesh
from jaxlib.xla_extension import NamedSharding

from jaxns.internals.mixed_precision import int_type, mp_policy


def replace_index(operand, update, start_index):
    """
    Replaces an index or slice with an update.
    If update is too big to respect start_index then start_index is shifted, which will give non-intuitive results.
    """
    if len(np.shape(operand)) != len(np.shape(update)):
        raise ValueError(
            f"Operand and update must have the same number of dimensions, got {len(np.shape(operand))} and {len(np.shape(update))}")
    start_index = jnp.asarray(start_index, mp_policy.index_dtype)
    start_indices = [start_index] + [jnp.asarray(0, start_index.dtype)] * (len(update.shape) - 1)
    return lax.dynamic_update_slice(operand, update.astype(operand.dtype), start_indices)


def get_index(operand, start_index, length):
    return lax.dynamic_slice(operand,
                             [start_index] + [jnp.asarray(0, int_type)] * (len(operand.shape) - 1),
                             (length,) + operand.shape[1:])


def prepare_func_args(f):
    """
    Takes a callable(a,b,...,z=Z) and prepares it into ``callable(**kwargs)``, such that only
    a,b,...,z are taken from ``**kwargs`` and the rest ignored.

    This allows ``f(**kwarg)`` to work even if ``f()`` is missing some keys from kwargs.

    Args:
        f: ``callable(a,b,...,z=Z)``

    Returns:
        ``callable(**kwargs)`` where ``**kwargs`` are the filtered for args of the original function.
    """
    if hasattr(f, "__old_name__"):
        raise ValueError(f"function {f.__old_name__} has already done prepare_func_args.")

    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations) = \
        inspect.getfullargspec(f)

    # TODO: this gets displayed each time we prepare a function. Using a cache would be cleaner for user.
    if varargs is not None:
        warnings.warn(f"Function {f.__name__} has *varargs parameter ({varargs}), and is being dropped.")
    if varkw is not None:
        warnings.warn(f"Function {f.__name__} has **varkw parameter ({varkw}), and is being dropped.")

    expected_keys = set(args + kwonlyargs)

    if defaults is None:  # no defaults
        num_defaults = 0
        defaults = ()
    else:
        num_defaults = len(defaults)

    num_non_default = len(args) - num_defaults

    def _f(**kwargs):
        # Replace any defaults with kwargs
        args_with_values = dict(zip(args[num_non_default:], defaults))
        if kwonlydefaults is not None:
            args_with_values.update(kwonlydefaults)
        args_with_values.update(kwargs)
        args_with_values = dict(filter(lambda item: item[0] in expected_keys, args_with_values.items()))
        for key in expected_keys:
            if key not in args_with_values:
                raise KeyError(f"{f.__name__} is missing argument {key}")

        return f(**args_with_values)

    _f.__doc__ = f.__doc__
    _f.__old_name__ = f.__name__
    return _f


F = TypeVar('F')
FV = TypeVar('FV')


def chunked_pmap(f: Callable[..., FV], chunk_size: Optional[int] = None, unroll: int = 1) -> Callable[..., FV]:
    """
    A version of pmap which chunks the input into smaller pieces to avoid memory issues.

    Args:
        f: callable
        chunk_size: the size of the chunks. Default is len(devices())
        unroll: the number of times to unroll the computation

    Returns:
        a chunked version of f
    """
    if chunk_size is None:
        chunk_size = len(jax.devices())

    def _f(*args, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = lax.scan(body, (), (args, kwargs), unroll=unroll)
            return result

        if chunk_size > 1:
            # Get from first leaf
            if len(args) > 0:
                batch_size = jax.tree.leaves(args)[0].shape[0]
            else:
                batch_size = jax.tree.leaves(kwargs)[0].shape[0]
            remainder = batch_size % chunk_size
            extra = (chunk_size - remainder) % chunk_size
            if extra > 0:
                (args, kwargs) = jax.tree.map(lambda x: _pad_extra(x, chunk_size), (args, kwargs))
            (args, kwargs) = jax.tree.map(
                lambda x: jnp.reshape(x, (chunk_size, x.shape[0] // chunk_size) + x.shape[1:]),
                (args, kwargs)
            )
            result = pmap(queue)(*args, **kwargs)  # [chunksize, batch_size // chunksize, ...]
            result = jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), result)
            if extra > 0:
                result = jax.tree.map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f


def _pad_extra(arg, chunksize):
    N = arg.shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = (chunksize - remainder) % chunksize
        arg = jnp.concatenate([arg] + [arg[0:1]] * extra, axis=0)
        N = N + extra
    else:
        extra = 0
    T = N // chunksize
    # arg = jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:])
    return arg


def prepad(a, chunksize: int):
    return jax.tree.map(lambda arg: _pad_extra(arg, chunksize), a)


T = TypeVar('T')


def remove_chunk_dim(py_tree: T) -> T:
    """
    Remove the chunk dimension from a pytree

    Args:
        py_tree: pytree to remove chunk dimension from

    Returns:
        pytree with chunk dimension removed
    """
    leaves = jax.tree.leaves(py_tree)

    # Check consistency
    for leaf in leaves:
        if len(leaf.shape) < 1:
            raise ValueError(f"Expected all leaves to have at least one dimension, got {leaf.shape}")
        if leaf.shape[0] != leaves[0].shape[0]:
            raise ValueError(
                f"Expected all leaves to have the same batch dimension, got {leaf.shape[0]} != {leaves[0].shape[0]}"
            )

    def _remove_chunk_dim(a):
        shape = list(a.shape)
        if len(shape) == 1:
            return a[0]
        shape = [shape[0] * shape[1]] + shape[2:]
        return jnp.reshape(a, shape)

    return jax.tree.map(_remove_chunk_dim, py_tree)


def add_chunk_dim(py_tree: T, chunk_size: int) -> T:
    """
    Add a chunk dimension to a pytree

    Args:
        py_tree: pytree to add chunk dimension to
        chunk_size: size of chunk dimension

    Returns:
        pytree with chunk dimension added
    """

    def _add_chunk_dim(a):
        shape = list(a.shape)
        shape = [chunk_size, shape[0] // chunk_size] + shape[1:]
        return jnp.reshape(a, shape)

    return jax.tree.map(_add_chunk_dim, py_tree)


def chunked_vmap(f, chunk_size: Optional[int] = None, unroll: int = 1):
    """
    A version of vmap which chunks the input into smaller pieces to avoid memory issues.

    Args:
        f: the function to be mapped
        chunk_size: the size of the chunks. Default is len(devices())
        unroll: the number of times to unroll the computation

    Returns:

    """
    if chunk_size is None:
        chunk_size = len(jax.devices())

    def _f(*args, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = lax.scan(f=body, init=(), xs=(args, kwargs), unroll=unroll)
            return result

        if chunk_size > 1:
            # Get from first leaf
            if len(args) > 0:
                batch_size = jax.tree.leaves(args)[0].shape[0]
            else:
                batch_size = jax.tree.leaves(kwargs)[0].shape[0]
            remainder = batch_size % chunk_size
            extra = (chunk_size - remainder) % chunk_size
            if extra > 0:
                (args, kwargs) = jax.tree.map(lambda x: _pad_extra(x, chunk_size), (args, kwargs))
            (args, kwargs) = jax.tree.map(
                lambda x: jnp.reshape(x, (chunk_size, x.shape[0] // chunk_size) + x.shape[1:]),
                (args, kwargs)
            )
            result = jax.vmap(queue)(*args, **kwargs)  # [chunksize, batch_size // chunksize, ...]
            result = jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), result)
            if extra > 0:
                result = jax.tree.map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f


PT = TypeVar('PT')


def pytree_unravel(example_tree: PT) -> Tuple[Callable[[PT], jax.Array], Callable[[jax.Array], PT]]:
    """
    Returns functions to ravel and unravel a pytree.

    Args:
        example_tree: a pytree to be unravelled

    Returns:
        ravel_fun: a function to ravel the pytree
        unravel_fun: a function to unravel
    """
    leaf_list, tree_def = jax.tree.flatten(example_tree)

    sizes = [np.size(leaf) for leaf in leaf_list]
    shapes = [np.shape(leaf) for leaf in leaf_list]
    dtypes = [leaf.dtype for leaf in leaf_list]

    def ravel_fun(pytree: PT) -> jax.Array:
        leaf_list, tree_def = jax.tree.flatten(pytree)
        # promote types to common one
        common_dtype = jnp.result_type(*dtypes)
        leaf_list = [leaf.astype(common_dtype) for leaf in leaf_list]
        return jnp.concatenate([lax.reshape(leaf, (size,)) for leaf, size in zip(leaf_list, sizes)])

    def unravel_fun(flat_array: jax.Array) -> PT:
        leaf_list = []
        start = 0
        for size, shape, dtype in zip(sizes, shapes, dtypes):
            leaf_list.append(lax.reshape(flat_array[start:start + size], shape).astype(dtype))
            start += size
        return jax.tree.unflatten(tree_def, leaf_list)

    return ravel_fun, unravel_fun


def pytree_unpack(example_tree: PT) -> Tuple[Callable[[PT], List[jax.Array]], Callable[[List[jax.Array]], PT]]:
    """
    Returns functions to ravel and unravel a pytree.
    """
    leaf_list, tree_def = jax.tree.flatten(example_tree)

    def pack_fun(pytree: PT) -> List[jax.Array]:
        leaf_list, tree_def = jax.tree.flatten(pytree)
        return leaf_list

    def unpack_fun(leaf_list: List[jax.Array]) -> PT:
        return jax.tree.unflatten(tree_def, leaf_list)

    return pack_fun, unpack_fun


PV = TypeVar('PV')


class PyTree:
    """
    For acting on W space.
    """

    def __init__(self, tree: PV):
        self.tree = tree

    def __add__(self, other: PV) -> PV:
        return jax.tree.map(lambda x, y: x + y, self.tree, other)

    def __sub__(self, other: PV) -> PV:
        return jax.tree.map(lambda x, y: x - y, self.tree, other)

    def __mul__(self, other: PV) -> PV:
        return jax.tree.map(lambda x, y: x * y, self.tree, other)

    def __truediv__(self, other: PV) -> PV:
        return jax.tree.map(lambda x, y: x / y, self.tree, other)

    def __pow__(self, other: PV) -> PV:
        return jax.tree.map(lambda x, y: x ** y, self.tree, other)

    def __neg__(self):
        return jax.tree.map(lambda x: -x, self.tree)


def create_mesh(shape, axis_names, devices=None):
    """
    Create a mesh from a shape and axis names.

    Args:
        shape: the shape of the mesh, total size must evenly divide number of devices.
        axis_names: the axis names of the mesh.
        devices: the devices to use, if None, uses all devices.

    Returns:
        the mesh
    """
    if len(shape) != len(axis_names):
        raise ValueError(f"Shape {shape} and axis names {axis_names} must have the same length.")
    mesh_size = int(np.prod(shape))
    if devices is None:
        devices = jax.devices()
        if mesh_size < len(devices):
            devices = devices[:mesh_size]
    if mesh_size % len(devices) != 0:
        raise ValueError(f"Mesh size {mesh_size} must evenly divide number of devices {len(devices)}.")
    mesh_devices = create_device_mesh(mesh_shape=shape, devices=devices)
    mesh = Mesh(mesh_devices, axis_names=axis_names)
    return mesh


SPT = TypeVar('SPT')


def tree_device_put(tree: SPT, mesh: Mesh, axis_names: Tuple[Union[str, None], ...]) -> SPT:
    """
    Put a pytree on a device.

    Args:
        tree: the pytree to put on a device.
        mesh: the mesh to put the pytree on.
        axis_names: the axis names of the mesh.

    Returns:
        the pytree on the device.
    """
    sharding = NamedSharding(mesh, PartitionSpec(*axis_names))
    return jax.tree.map(lambda x: jax.device_put(x, sharding), tree)


BUX = TypeVar('BUX', bound=Union[jax.Array, Any])


def block_until_ready(x: BUX) -> BUX:
    return jax.block_until_ready(x)
