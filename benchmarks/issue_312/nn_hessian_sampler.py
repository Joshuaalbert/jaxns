"""Frozen NN predictive-mean Hessian directions with the existing MH chain."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
from jax import numpy as jnp

from benchmarks.issue_312.hessian_chain import (
    ChainState,
    Geometry,
    sample_hessian_chain,
)
from benchmarks.issue_312.nn_geometry import NN, negative_predictive_hessian
from jaxns.constrained_sampler import AbstractSampler
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree, pytree_ravel
from jaxns.samples import PhantomSamples, SeedPoint


def local_geometry(network: NN, point: jax.Array, scale: jax.Array) -> Geometry:
    """Apply the same contour scale and prior-variance cap as the GP study."""
    eigenvalues, rotation = jnp.linalg.eigh(negative_predictive_hessian(network, point))
    return Geometry(rotation, jnp.maximum(eigenvalues / scale, 12.0))


@dataclass(frozen=True, slots=True)
class NNHessianSliceSampler(AbstractSampler, PureDataclassPytree):
    """Use a frozen network's local Hessian, retaining rejected self-transitions.

    The known peak and quadratic dimension retain the previous toy oracle
    scale. Network fits belong to host boundaries, never to this chain.
    """

    network: NN
    log_likelihood_peak: float
    quadratic_dimension: int = 15
    num_slices: int = 90
    num_phantom_samples: int = 0
    correct_mh: bool = True

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [
            "log_likelihood_peak", "quadratic_dimension", "num_slices",
            "num_phantom_samples", "correct_mh",
        ])

    def __post_init__(self):
        if self.num_slices < 1 or self.quadratic_dimension < 1:
            raise ValueError("Slices and quadratic dimension must be positive.")
        if not 0 <= self.num_phantom_samples < self.num_slices:
            raise ValueError("Phantoms must exclude the final transition.")

    def num_phantom(self) -> int:
        return self.num_phantom_samples

    def max_num_phantom(self) -> int:
        return self.num_slices - 1

    def _with_phantom_count(self, num_phantom_samples: int):
        return replace(self, num_phantom_samples=num_phantom_samples)

    def validate_core(self, dimension: int) -> None:
        if dimension != self.network.input_mean.shape[0]:
            raise ValueError("NN and model dimensions differ.")

    def run_chain(
        self, key: jax.Array, point: jax.Array, log_likelihood: jax.Array,
        contour: jax.Array, log_likelihood_fn: Callable[[jax.Array], jax.Array],
    ) -> tuple[ChainState, tuple[jax.Array, jax.Array]]:
        scale = 2 * jnp.maximum(self.log_likelihood_peak - contour, 1e-12) / (
            self.quadratic_dimension + 2
        )
        return sample_hessian_chain(
            key, point, log_likelihood, contour, log_likelihood_fn,
            lambda value: local_geometry(self.network, value, scale),
            self.num_slices, self.correct_mh,
        )

    def get_sample(
        self, key, log_L_constraint, seed_point: SeedPoint,
        *, model: Model, args=(), params=None,
    ):
        point, unravel = pytree_ravel(seed_point.U0)

        def likelihood(value):
            return model.log_likelihood(unravel(value), args=args, params=params)

        final, (points, log_likelihoods) = self.run_chain(
            key, point, seed_point.log_L0, log_L_constraint, likelihood,
        )
        phantom = PhantomSamples(
            U_samples=jax.vmap(unravel)(points[:self.num_phantom_samples]),
            log_L=log_likelihoods[:self.num_phantom_samples],
            valid_mask=jnp.asarray(self.num_phantom_samples > 0),
        )
        return unravel(final.point), final.log_likelihood, final.evaluations, phantom


NNHessianSliceSampler.register_pytree()
