"""Immutable scientific registration shared by distributed transports."""

import dataclasses

from jaxctx import CtxParams

from jaxns.constrained_sampler import AbstractSampler
from jaxns.model import Model
from jaxns.pytree import PureDataclassPytree


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerSession(PureDataclassPytree):
    """Model and sampler data registered once in every worker process.

    The runtime transports this immutable scientific contract as an opaque
    registration payload. Owning it here prevents transport clients from
    depending on the distributed runner that happens to create the session.
    """

    model: Model
    sampler: AbstractSampler
    args: tuple  # [...] arbitrary model argument pytrees
    params: CtxParams | None  # [...] arbitrary parameter pytree leaves


WorkerSession.register_pytree()
