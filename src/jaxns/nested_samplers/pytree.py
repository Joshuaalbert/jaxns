import pickle
import warnings
from abc import abstractmethod, ABC
from typing import Tuple, List, Any

import jax


class Pytree(ABC):

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
                    raise e

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
            contents = dict((f, getattr(this, f)) for f in fields)

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
