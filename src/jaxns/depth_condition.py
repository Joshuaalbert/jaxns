"""Expected-volume contour limits for one compiled depth iteration."""

import dataclasses

from jaxns.pytree import PureDataclassPytree
from jaxns.types import FloatArray


@dataclasses.dataclass(slots=True, frozen=True)
class DepthCondition(PureDataclassPytree):
    """Choose how far a frozen allocation schedule needs to extend.

    Both fields are evaluated from the expected classic shrinkage path at a
    planning-round boundary. Final evidence uncertainty, ESS, likelihood
    budgets, and other scientific goals belong to the user-provided Python
    goal over ``State`` or results.
    """

    dlogZ: FloatArray | None = None  # []
    cummax_XL_frac: FloatArray | None = None  # []


DepthCondition.register_pytree()
