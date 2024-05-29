from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from jaxns.internals.namedtuple_utils import serialise_namedtuple, \
    deserialise_namedtuple, issubclass_namedtuple, serialise_ndarray


class TestAge(NamedTuple):
    years: int
    months: int


class TestPerson(NamedTuple):
    name: str
    age: TestAge


def test_isinstance_namedtuple():
    # Example NamedTuple
    data = TestPerson('Alice', TestAge(25, 6))
    assert isinstance(data, TestPerson)

    data = ()
    assert not isinstance(data, TestPerson)


def test_issubclass_namedtuple():
    class TestSubclass(NamedTuple):
        pass

    assert issubclass_namedtuple(TestSubclass)

    class TestSubclass(tuple):
        pass

    assert not issubclass_namedtuple(TestSubclass)

    class C:
        pass

    assert not issubclass_namedtuple(C)


def test_serialise_namedtuple():
    # Example NamedTuple
    data = TestPerson('Alice', TestAge(25, 6))
    # Serialise
    serialized_data = serialise_namedtuple(data)
    print(serialized_data)

    # Deserialize
    restored_data = deserialise_namedtuple(serialized_data)
    print(restored_data)

    assert data == restored_data


def test_serialise_ndarray():
    array = np.array([1, 2, 3])
    serialized_data = serialise_ndarray(array)
    print(serialized_data)
    restored_data = deserialise_namedtuple(serialized_data)
    print(restored_data)

    np.testing.assert_allclose(array, restored_data)


def test_serialise_jax_ndarray():
    array = jnp.array([1, 2, 3])
    serialized_data = serialise_ndarray(array)
    print(serialized_data)
    restored_data = deserialise_namedtuple(serialized_data)
    print(restored_data)

    np.testing.assert_allclose(array, restored_data)
