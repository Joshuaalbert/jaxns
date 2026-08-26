from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp

from jaxns.pytree import PureDataclassPytree


class OffsetNode:
    def __init__(self, offset: float = 1.0, delay_s: float = 0.0):
        self.offset = float(offset)
        self.delay_s = float(delay_s)

    def evaluate(self, u):
        if self.delay_s > 0.0:
            time.sleep(self.delay_s)
        return float(jnp.asarray(u)) + self.offset


class QuadraticNode:
    def __init__(self, centre: float = 0.25):
        self.centre = float(centre)

    def evaluate(self, u):
        return float(quadratic_log_likelihood(u, centre=self.centre))


class QuadraticEvaluator:
    def __init__(self, centre: float = 0.25):
        self.centre = float(centre)

    def evaluate(self, u):
        return quadratic_log_likelihood(u, centre=self.centre)


def make_offset_node():
    return OffsetNode(offset=1.0)


def make_slow_identity_node():
    return OffsetNode(offset=0.0, delay_s=0.5)


def make_quadratic_node():
    return QuadraticNode(centre=0.25)


def quadratic_log_likelihood(u, centre: float = 0.25):
    u = jnp.asarray(u)
    return -jnp.square(u - centre)


@dataclasses.dataclass(frozen=True, slots=True)
class ToyModel(PureDataclassPytree):
    centre: float = 0.25

    def U_ndims(self, args=(), params=None) -> int:
        del args, params
        return 1

    def sample_U(self, key, args=(), params=None):
        del args, params
        return jax.random.uniform(key, minval=0.0, maxval=1.0)

    def transform_to_X(self, U, args=(), params=None):
        del args, params
        return U

    def log_likelihood(self, U, args=(), params=None, *, allow_nan: bool = True):
        del args, params, allow_nan
        return quadratic_log_likelihood(U, centre=self.centre)

    def log_prior(self, U, args=(), params=None):
        del args, params
        inside = jnp.logical_and(U >= 0.0, U <= 1.0)
        return jnp.where(inside, 0.0, -jnp.inf)

    def log_joint(self, U, args=(), params=None, *, allow_nan: bool = True):
        return self.log_prior(U, args=args, params=params) + self.log_likelihood(
            U,
            args=args,
            params=params,
            allow_nan=allow_nan,
        )


def make_toy_model() -> ToyModel:
    return ToyModel()


ToyModel.register_pytree()
