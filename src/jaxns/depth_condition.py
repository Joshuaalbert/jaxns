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

    Args:
        dlogZ: Remaining-evidence fraction threshold, normally in (0, 1).
            At contour g this compares L_g X_g / (Z_through_g + L_g X_g)
            with the threshold. Despite its historical name, it is not a
            target log-evidence uncertainty. None disables this cutoff.
        cummax_XL_frac: Threshold, normally in (0, 1), for L_g X_g divided
            by its largest value up to contour g. None disables this cutoff.
            With both cutoffs set, either can end the depth traversal. With
            neither set, an allocation target completes without a tail cutoff.
    """

    dlogZ: FloatArray | None = None  # []
    cummax_XL_frac: FloatArray | None = None  # []


DepthCondition.register_pytree()
