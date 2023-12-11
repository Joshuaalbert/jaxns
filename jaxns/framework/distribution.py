from typing import Optional, List, Union, Tuple

import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from jaxns.framework.bases import BaseAbstractDistribution
from jaxns.internals.types import FloatArray, IntArray, BoolArray, float_type

__all__ = [
    "Distribution",
    "SingularDistribution",
    "InvalidDistribution"
]

tfpd = tfp.distributions


class InvalidDistribution(Exception):
    """
    Raised when a distribution does not have a quantile.
    """

    def __init__(self, dist: Optional[tfpd.Distribution] = None):
        super(InvalidDistribution, self).__init__(
            f'Distribution {dist} is missing a quantile. '
            f'Try checking if your desired prior exists in `jaxns.special_priors`.')


def distribution_chain(dist: tfpd.Distribution) -> List[
    Union[tfpd.TransformedDistribution, tfpd.Sample, tfpd.Distribution]]:
    """
    Returns a list of distributions that make up the chain of distributions.

    Args:
        dist: A TFP distribution, transformed distribution or sample.

    Returns:
        A list of distributions.
    """
    chain = []
    while True:
        chain.append(dist)
        if isinstance(dist, tfpd.TransformedDistribution):
            dist = dist.distribution
            continue
        break
    # Must reverse the chain because the first distribution is the last in the chain.
    return chain[::-1]


class Distribution(BaseAbstractDistribution):
    """
    Represents a distribution, which must have defined forward and inverse transformations, and a log_prob.
    """

    def __init__(self, dist: tfpd.Distribution):
        self.dist_chain = distribution_chain(dist)
        check_dist = self.dist_chain[0]
        if isinstance(self.dist_chain[0], tfpd.Sample):
            check_dist = self.dist_chain[0].distribution
        if '_quantile' not in check_dist.__class__.__dict__:
            # TODO(Joshuaalbert): we could numerically approximate it. This requires knowing the support of dist.
            # Repartitioning the prior also requires knowing the support and choosing a replacement, which is not
            # always easy from stats. E.g. StudentT variance doesn't exist but a numerial quantile can be made.
            raise InvalidDistribution(dist=dist)

    def __repr__(self):
        return " -> ".join(map(repr, self.dist_chain))

    def _dtype(self):
        return self.dist_chain[-1].dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return tuple(self.dist_chain[0].batch_shape_tensor()) + tuple(self.dist_chain[0].event_shape_tensor())

    def _shape(self) -> Tuple[int, ...]:
        return tuple(self.dist_chain[-1].batch_shape_tensor()) + tuple(self.dist_chain[-1].event_shape_tensor())

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        dist = self.dist_chain[0]
        if isinstance(dist, tfpd.Sample):
            dist = dist.distribution
        X = dist.quantile(U)
        for dist in self.dist_chain[1:]:
            X = dist.bijector.forward(X)
        return X

    def _inverse(self, X) -> FloatArray:
        for dist in reversed(self.dist_chain[1:]):
            X = dist.bijector.inverse(X)
        dist = self.dist_chain[0]
        if isinstance(dist, tfpd.Sample):
            dist = dist.distribution
        X = dist.cdf(X)
        return X

    def _log_prob(self, X):
        return self.dist_chain[-1].log_prob(X)


class SingularDistribution(BaseAbstractDistribution):
    """
    Represents a singular distribution, which has no inverse transformation, but does have a log_prob
        (at the singular value).
    """

    # TODO(Joshuaalbert): Perhaps this should become a SingularPrior because it is not a distribution.
    #  See note in bases.py
    def __init__(self, value: jnp.ndarray, dist: Distribution):
        self.value = value
        self.dist = dist

    def __repr__(self):
        return f"{self.value} -> {self.dist}"

    def _dtype(self):
        return self.dist.dtype

    def _base_shape(self) -> Tuple[int, ...]:
        return ()

    def _shape(self) -> Tuple[int, ...]:
        return self.dist.shape

    def _forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        return self.value

    def _inverse(self, X) -> FloatArray:
        return jnp.asarray([], float_type)

    def _log_prob(self, X):
        return self.dist.log_prob(X)
