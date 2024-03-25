import json
from typing import NamedTuple

import numpy as np

from jaxns.internals.namedtuple_utils import issubclass_namedtuple, serialise_namedtuple, deserialise_namedtuple


# Example NamedTuple
class TestAge(NamedTuple):
    years: int
    months: np.ndarray


class TestPerson(NamedTuple):
    name: str
    age: TestAge


def test_isinstance_namedtuple():
    # Example NamedTuple
    data = TestPerson('Alice', TestAge(25, np.asarray(6)))
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
    data = TestPerson('Alice', TestAge(25, np.array(6)))
    # Serialise
    serialized_data = serialise_namedtuple(data)
    print(serialized_data)

    # Deserialize
    restored_data = deserialise_namedtuple(serialized_data)
    print(restored_data)

    assert data == restored_data

def test_to_json():
    data = TestPerson('Alice', TestAge(25, np.array(6)))
    s = json.dumps(serialise_namedtuple(data), indent=2)
    print(s)
    restored_data = deserialise_namedtuple(json.loads(s))
    print(restored_data)
    assert data == restored_data