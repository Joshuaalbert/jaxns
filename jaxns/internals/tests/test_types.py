from jaxns.internals.types import TerminationConditionDisjunction, TerminationConditionConjunction, \
    TerminationCondition, isinstance_namedtuple


def test_termination_condition():
    assert isinstance(TerminationCondition() | TerminationCondition(), TerminationConditionDisjunction)
    assert isinstance(TerminationCondition() & TerminationCondition(), TerminationConditionConjunction)


def test_isinstance_namedtuple():
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    assert isinstance_namedtuple(p)

    # Test that it works with a custom namedtuple
    class Point2(namedtuple('Point2', ['x', 'y'])):
        pass

    p2 = Point2(1, 2)

    assert isinstance_namedtuple(p2)

    assert not isinstance_namedtuple((1, 2))
