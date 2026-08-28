import dataclasses
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar, Union

import jax
import numpy as np
from jax import numpy as jnp


def save_pytree(pytree: Any, filename: str) -> None:
    """Persist an arbitrary pytree using Python's pickle protocol."""
    path = Path(filename)
    with path.open('wb') as file:
        pickle.dump(pytree, file)


def load_pytree(filename: str):
    """Load a pytree written by :func:`save_pytree`."""
    path = Path(filename)
    with path.open('rb') as file:
        return pickle.load(file)


class Pytree(ABC):

    def size_info(self):
        def format_size_info(leaf):
            try:
                nbytes = leaf.nbytes
                if nbytes > 1024 ** 3:
                    memory = nbytes // 1024 ** 3
                    unit = "GB"
                elif nbytes > 1024 ** 2:
                    memory = nbytes // 1024 ** 2
                    unit = "MB"
                elif nbytes > 1024:
                    memory = nbytes // 1024
                    unit = "KB"
                else:
                    memory = nbytes
                    unit = "B"
                return f"{np.result_type(leaf)}{np.shape(leaf)}[size={np.size(leaf)}, bytes={memory:.1f} {unit}]"
            except AttributeError:
                return f"{np.result_type(leaf)}{np.shape(leaf)}[size={np.size(leaf)}]"

        return repr(jax.tree.map(format_size_info, self))

    def save(self, filename: str):
        """
        Serialise the model to file.

        Args:
            filename: the filename
        """
        if not filename.endswith('.pkl'):
            warnings.warn(f"Filename {filename} does not end with .pkl")
        with open(filename, 'wb') as f:
            try:
                pickle.dump(self, f)
            except AttributeError as e:
                if "Can't pickle local object" in str(e):
                    warnings.warn(
                        f"Failed to pickle {self.__class__.__name__}. "
                        f"It's possibly locally defined. Make sure it is globally defined."
                    )
                    raise

    @staticmethod
    def load(filename: str):
        """
        Load the model from file.

        Args:
            filename: the filename

        Returns:
            the model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def build_flatten(cls, this, aux_names: list[str]):
        """
        Helper function to facilitate dataclass Pytrees.
        """
        contents = this.__dict__
        if hasattr(this, '__dataclass_fields__'):
            # Only the fields, ignore set attrs
            fields = this.__dataclass_fields__
            contents = {f: getattr(this, f) for f in fields}

        children_dict = dict(item for item in contents.items() if item[0] not in aux_names)
        aux_data_dict = dict(item for item in contents.items() if item[0] in aux_names)
        return [children_dict], (aux_data_dict,)

    @classmethod
    def build_unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        """
        Helper function to facilitate dataclass Pytrees.
        """
        [children_dict] = children
        (aux_data_dict,) = aux_data
        return cls(**children_dict, **aux_data_dict)

    def __reduce__(self):
        # Return the class method for deserialization and the actor as an argument
        children, aux_data = self.flatten(self)
        serialised = (aux_data, children)
        return (self._deserialise, (serialised,))

    @classmethod
    def _deserialise(cls, serialised):
        # Create a new instance, bypassing __init__ and setting the actor directly
        (aux_data, children) = serialised
        return cls.unflatten(aux_data, children)

    @classmethod
    def register_pytree(cls):
        jax.tree_util.register_pytree_node(cls, cls.flatten, cls.unflatten)

    # an abstract classmethod

    @classmethod
    @abstractmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        ...

    @classmethod
    @abstractmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        ...

    def to_json(self) -> dict:
        """
        Convert the Pytree to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the Pytree.
        """
        children, aux_data = self.flatten(self)
        json_dict = {}

        # two keys: 'children' and 'aux_data'
        # First transform children (a list of arrays) to array structures {shape: ..., dtype: ..., data: [...]}
        # data is flat list of bytes
        def _array_to_dict(arr):
            arr_np = np.asarray(arr)
            return {
                'shape': np.shape(arr_np),
                'dtype': str(np.result_type(arr_np)),
                'data': arr_np.tobytes().decode('latin1')  # use latin1 to preserve byte values
            }

        json_dict['children'] = jax.tree.map(_array_to_dict, children)
        # Now aux_data (a tuple of auxiliary data)
        json_dict['aux_data'] = aux_data  # assuming aux_data is JSON-serial
        return json_dict

    @classmethod
    def from_json(cls, json_dict: dict):
        """
        Load the Pytree from a JSON-serializable dictionary.

        Args:
            json_dict: A dictionary representation of the Pytree.
        """

        # inverse of to_json
        def _dict_to_array(d):
            return np.frombuffer(d['data'].encode('latin1'), dtype=d['dtype']).reshape(d['shape'])

        children = jax.tree.map(_dict_to_array, json_dict['children'])
        aux_data = json_dict['aux_data']
        return cls.unflatten(aux_data, children)


class PureDataclassPytree(Pytree):
    """
    A Pytree that is a pure dataclass, i.e. all fields are data fields. Still need to call register_pytree.
    """

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, [])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)


def _tree_dot(x, y):
    dots = jax.tree.leaves(jax.tree.map(jnp.vdot, x, y))
    return sum(dots[1:], start=dots[0])


def _tree_norm(x):
    norm2 = _tree_dot(x, x)
    if jnp.issubdtype(norm2.dtype, jnp.complexfloating):
        return jnp.sqrt(norm2.real)
    return jnp.sqrt(norm2)


PV = TypeVar('PV')


@dataclasses.dataclass(slots=True, frozen=True)
class TreeField(PureDataclassPytree, Generic[PV]):
    tree: Union['TreeField[PV]', Any]  # [...] arbitrary aligned pytree leaves

    def batch_dim(self) -> int:
        leaves = jax.tree.leaves(self.tree)
        if not leaves:
            return 0
        assert all(leaf.shape[0] == leaves[0].shape[0] for leaf in leaves), "All leaves must have the same batch dimension"
        return leaves[0].shape[0]

    def ndim(self) -> int:
        return sum(jax.tree.map(np.size, jax.tree.leaves(self)))

    def from_flat_matrix(self, flat) -> 'TreeField[PV]':
        """
        Turn a matrix into pytree of pytrees so that matmul contract (: PyTree x PyTree -> PyTree) holds.

        Args:
            flat: [N, N]

        Returns:
            A TreeField with the same structure as self, but with the leaves replaced by the corresponding slices of the flat matrix.
        """
        _, unravel_fn = pytree_ravel(self)
        return jax.vmap(unravel_fn)(flat)

    def __matmul__(self, other: 'TreeField[PV]') -> jax.Array:
        if not isinstance(other, TreeField):
            raise TypeError("Only works on other same matching trees.")
        return jax.vmap(lambda x: _tree_dot(x, other))(self)

    def __rsub__(self, other: Union['TreeField[PV]', Any]) -> 'TreeField[PV]':
        if isinstance(other, TreeField):
            return jax.tree.map(lambda x, y: y - x, self, other)
        return jax.tree.map(lambda x: other - x, self)

    def __add__(self, other: Union['TreeField[PV]', Any]) -> 'TreeField[PV]':
        if isinstance(other, TreeField):
            return jax.tree.map(lambda x, y: x + y, self, other)
        return jax.tree.map(lambda x: x + other, self)

    def __sub__(self, other: Union['TreeField[PV]', Any]) -> 'TreeField[PV]':
        if isinstance(other, TreeField):
            return jax.tree.map(lambda x, y: x - y, self, other)
        return jax.tree.map(lambda x: x - other, self)

    def __mul__(self, other: Union['TreeField[PV]', Any]) -> 'TreeField[PV]':
        if isinstance(other, TreeField):
            return jax.tree.map(lambda x, y: x * y, self, other)
        return jax.tree.map(lambda x: x * other, self)

    def __truediv__(self, other: Union['TreeField[PV]', Any]) -> 'TreeField[PV]':
        if isinstance(other, TreeField):
            return jax.tree.map(lambda x, y: x / y, self, other)
        return jax.tree.map(lambda x: x / other, self)

    def __pow__(self, other: Union['TreeField[PV]', Any]) -> 'TreeField[PV]':
        if isinstance(other, TreeField):
            return jax.tree.map(lambda x, y: x ** y, self, other)
        return jax.tree.map(lambda x: x ** other, self)

    def __neg__(self) -> 'TreeField[PV]':
        return jax.tree.map(lambda x: -x, self)

    def norm(self):
        return _tree_norm(self)

    def min(self):
        return jax.tree.reduce(lambda x, y: jnp.minimum(jnp.min(x), jnp.min(y)), self)

    def max(self):
        return jax.tree.reduce(lambda x, y: jnp.maximum(jnp.max(x), jnp.max(y)), self)

    def ones_like(self) -> 'TreeField[PV]':
        return jax.tree.map(lambda x: jnp.ones_like(x), self)

    def zeros_like(self) -> 'TreeField[PV]':
        return jax.tree.map(lambda x: jnp.zeros_like(x), self)

    def random_normal_like(self, key) -> 'TreeField[PV]':
        leaves = jax.tree.leaves(self)
        if not leaves:
            return self
        keys = jax.random.split(key, len(leaves))
        z = [jax.random.normal(k, shape=x.shape, dtype=x.dtype) for k, x in zip(keys, leaves)]
        return jax.tree.unflatten(jax.tree.structure(self), z)

    def random_uniform_like(self, key) -> 'TreeField[PV]':
        leaves = jax.tree.leaves(self)
        if not leaves:
            return self
        keys = jax.random.split(key, len(leaves))
        z = [jax.random.uniform(k, shape=x.shape, dtype=x.dtype) for k, x in zip(keys, leaves)]
        return jax.tree.unflatten(jax.tree.structure(self), z)


TreeField.register_pytree()


def pytree_ravel(pytree):
    leaves, tree_def = jax.tree.flatten(pytree)
    # concatenate all leaves into a single 1D array
    flat = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    leaves_sizes = [np.size(leaf) for leaf in leaves]
    leaves_shapes = [np.shape(leaf) for leaf in leaves]

    def _unravel_fn(_flat):
        split_indices = np.cumsum(leaves_sizes[:-1])
        split_leaves = jnp.split(_flat, split_indices)
        reshaped_leaves = [jnp.reshape(leaf, shape) for leaf, shape in zip(split_leaves, leaves_shapes)]
        return jax.tree.unflatten(tree_def, reshaped_leaves)

    return flat, _unravel_fn
