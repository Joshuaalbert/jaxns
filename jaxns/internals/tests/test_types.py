from jaxns.internals.types import TerminationConditionDisjunction, TerminationConditionConjunction, TerminationCondition


def test_termination_condition():
    assert isinstance(TerminationCondition() | TerminationCondition(), TerminationConditionDisjunction)
    assert isinstance(TerminationCondition() & TerminationCondition(), TerminationConditionConjunction)
