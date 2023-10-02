from jaxns import TerminationCondition
from jaxns.types import TerminationConditionDisjunction, TerminationConditionConjunction


def test_termination_condition():
    assert isinstance(TerminationCondition() | TerminationCondition(), TerminationConditionDisjunction)
    assert isinstance(TerminationCondition() & TerminationCondition(), TerminationConditionConjunction)
