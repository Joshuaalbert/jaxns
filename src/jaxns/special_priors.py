"""Special-prior classes supported through the public JAXNS namespace.

These are exact aliases of the JAXCTX implementations. Keeping an explicit
list makes additions to this public surface a deliberate compatibility choice
instead of silently exposing every future JAXCTX implementation detail.
"""

from jaxctx.priors.special_priors import (
    Bernoulli,
    Beta,
    Categorical,
    Empirical,
    ExplicitDensityPrior,
    ForcedIdentifiability,
    Poisson,
    TruncationWrapper,
    UnnormalisedDirichlet,
)

__all__ = [
    "Bernoulli",
    "Beta",
    "Categorical",
    "Empirical",
    "ExplicitDensityPrior",
    "ForcedIdentifiability",
    "Poisson",
    "TruncationWrapper",
    "UnnormalisedDirichlet",
]
