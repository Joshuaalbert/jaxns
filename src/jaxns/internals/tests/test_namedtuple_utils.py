from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxns.internals.namedtuple_utils import isinstance_namedtuple
from jaxns.internals.namedtuple_utils import serialise_namedtuple, \
    deserialise_namedtuple, issubclass_namedtuple, serialise_ndarray


class MockNested(NamedTuple):
    name: str


class MockModel(NamedTuple):
    string: str
    integer: int
    float: float
    complex: complex
    ndarray: np.ndarray
    jaxndarray: jax.Array
    nested: MockNested


def test_isinstance_namedtuple():
    # Example NamedTuple
    data = MockModel(
        string='Alice',
        integer=25,
        float=3.14,
        complex=1 + 2j,
        ndarray=np.array([1, 2, 3]),
        jaxndarray=jnp.array([1, 2, 3]),
        nested=MockNested('Bob')
    )

    assert isinstance_namedtuple(data)
    assert not isinstance_namedtuple(('Bob',))

    assert not isinstance(('Bob',), MockNested)


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
    data = MockModel(
        string='Alice',
        integer=25,
        float=3.14,
        complex=1 + 2j,
        ndarray=np.array([1, 2, 3]),
        jaxndarray=jnp.array([1, 2, 3]),
        nested=MockNested('Bob')
    )

    # Serialise
    serialized_data = serialise_namedtuple(data)
    print(serialized_data)

    # Deserialize
    restored_data = deserialise_namedtuple(serialized_data)
    print(restored_data)

    assert data.string == restored_data.string
    assert data.integer == restored_data.integer
    assert data.float == restored_data.float
    assert data.complex == restored_data.complex
    np.testing.assert_allclose(data.ndarray, restored_data.ndarray)
    np.testing.assert_allclose(data.jaxndarray, restored_data.jaxndarray)
    assert data.nested.name == restored_data.nested.name


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
