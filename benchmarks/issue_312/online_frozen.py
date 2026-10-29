"""An independently trained likelihood lookup, with no evaluation-tree input."""

from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp

from benchmarks.issue_312.online_policy import COMPRESSION_KNOTS, choose_log_parameters
from jaxns.pytree import PureDataclassPytree


@dataclass(frozen=True, slots=True)
class FrozenLikelihoodPolicy(PureDataclassPytree):
    """Fixed-size interpolation operands derived exclusively from training trees."""

    contours: jax.Array  # [268], sorted finite prefix then +inf padding
    log_parameters: jax.Array  # [268, 2], log desired h and nominal duration
    num_knots: jax.Array  # []

    @classmethod
    def flatten(cls, this):
        return cls.build_flatten(this, [])

    def choose(self, key: jax.Array, contour: jax.Array, *, explore: bool = True):
        if self.contours.shape != (268,) or self.log_parameters.shape != (268, 2):
            raise ValueError('Frozen likelihood policy requires fixed [268] / [268, 2] arrays.')
        lower, upper = self.contours[0], self.contours[self.num_knots - 1]
        location = jnp.clip(contour, lower, upper)
        logs = jax.vmap(lambda values: jnp.interp(location, self.contours, values), in_axes=1)(
            self.log_parameters,
        )
        tail = (contour < lower) | (contour > upper)
        return choose_log_parameters(logs, key, explore=explore,
                                     extrapolated=tail, extrapolation_clamped=tail)


FrozenLikelihoodPolicy.register_pytree()


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    contours: np.ndarray  # [blocks], finite sorted classic likelihoods
    compression: np.ndarray  # [blocks], final classic -log E[X]
    log_parameters: np.ndarray  # [65, 2], final online policy grid


def freeze_training_profiles(profiles: tuple[TrainingProfile, ...]) -> FrozenLikelihoodPolicy:
    """Prespecified equal-run geometric average, never selecting a best training run."""
    if len(profiles) != 4:
        raise ValueError('Each allocation requires exactly four independent training profiles.')
    knots = []
    for profile in profiles:
        if (len(profile.contours) < 2 or profile.contours.shape != profile.compression.shape
                or not np.isfinite(profile.contours).all()
                or not np.isfinite(profile.compression).all()
                or np.any(np.diff(profile.contours) <= 0)
                or np.any(np.diff(profile.compression) < 0)
                or profile.log_parameters.shape != (65, 2)):
            raise ValueError('Training profile must contain a finite ordered classic volume path.')
        knots.extend(np.interp(COMPRESSION_KNOTS, profile.compression, profile.contours))
        knots.extend((profile.contours[0], profile.contours[-1]))
    grid = np.unique(knots)
    if len(grid) > 268:
        raise ValueError('Training likelihood grid exceeds its prespecified capacity.')
    predictions = []
    for profile in profiles:
        indices = np.searchsorted(profile.contours, grid, side='right') - 1
        c = np.where(indices >= 0, profile.compression[np.maximum(indices, 0)], 0.)
        predictions.append(np.column_stack([
            np.interp(c, COMPRESSION_KNOTS, profile.log_parameters[:, dimension])
            for dimension in range(2)
        ]))
    average = np.mean(predictions, axis=0)  # [num_knots, 2], equal independent-run weights
    contours = np.full(268, np.inf)
    contours[:len(grid)] = grid
    parameters = np.tile(average[-1], (268, 1))
    parameters[:len(grid)] = average
    return FrozenLikelihoodPolicy(jnp.asarray(contours), jnp.asarray(parameters),
                                  jnp.asarray(len(grid), jnp.int64))
