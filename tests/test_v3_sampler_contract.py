from typing import Callable, NamedTuple

import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.constrained_sampler_distributed import DistributedUniDimSliceSampler
from jaxns.samples import SeedPoint
from tests.distributed_support import QuadraticEvaluator, make_toy_model


class SamplerCase(NamedTuple):
    name: str
    build: Callable[..., object]


class InvalidSeedCase(NamedTuple):
    name: str
    seed_u: float
    log_l0: float
    log_l_constraint: float


def _build_local_sampler(**kwargs):
    return UniDimSliceSampler(model=make_toy_model(), **kwargs)


def _build_distributed_sampler(**kwargs):
    return DistributedUniDimSliceSampler(
        model=make_toy_model(),
        evaluator=QuadraticEvaluator(),
        **kwargs,
    )


SAMPLER_CASES = (
    SamplerCase(name="local", build=_build_local_sampler),
    SamplerCase(name="distributed", build=_build_distributed_sampler),
)

INVALID_SEED_CASES = (
    InvalidSeedCase(
        name="below_constraint",
        seed_u=1.0,
        log_l0=-0.5625,
        log_l_constraint=-0.05,
    ),
    InvalidSeedCase(
        name="equal_to_constraint",
        seed_u=0.0,
        log_l0=-0.0625,
        log_l_constraint=-0.0625,
    ),
)


@pytest.mark.parametrize(
    "sampler_case",
    SAMPLER_CASES,
    ids=lambda case: case.name,
)
@pytest.mark.parametrize("phantom_burn_in", [-1, 4, 1.5, "1", True])
def test_sampler_construction_rejects_invalid_phantom_burn_in(
    sampler_case,
    phantom_burn_in,
):
    with pytest.raises(ValueError, match="phantom_burn_in"):
        sampler_case.build(
            num_slices=4,
            no_step_out=True,
            collect_phantom_samples=True,
            phantom_burn_in=phantom_burn_in,
        )


@pytest.mark.parametrize(
    "sampler_case",
    SAMPLER_CASES,
    ids=lambda case: case.name,
)
def test_num_phantom_is_non_negative_at_maximum_valid_burn_in(sampler_case):
    sampler = sampler_case.build(
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=3,
    )

    assert sampler.num_phantom() == 0


@pytest.mark.parametrize(
    "sampler_case",
    SAMPLER_CASES,
    ids=lambda case: case.name,
)
@pytest.mark.parametrize(
    "seed_case",
    INVALID_SEED_CASES,
    ids=lambda case: case.name,
)
def test_sampler_rejects_seed_that_does_not_satisfy_strict_constraint(
    sampler_case,
    seed_case,
):
    sampler = sampler_case.build(
        num_slices=2,
        no_step_out=True,
        collect_phantom_samples=False,
    )
    seed_point = SeedPoint(
        U0=jnp.asarray(seed_case.seed_u),
        log_L0=jnp.asarray(seed_case.log_l0),
    )

    with pytest.raises(ValueError, match="seed|strict|log_L0|constraint"):
        sampler.get_sample(
            key=random.PRNGKey(0),
            log_L_constraint=jnp.asarray(seed_case.log_l_constraint),
            seed_point=seed_point,
        )


@pytest.mark.parametrize(
    "sampler_case",
    SAMPLER_CASES,
    ids=lambda case: case.name,
)
def test_sampler_returns_likelihood_only_phantom_clusters_after_burn_in(
    sampler_case,
):
    sampler = sampler_case.build(
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )

    _, log_likelihood, _, phantom_samples = sampler.get_sample(
        key=random.PRNGKey(1),
        log_L_constraint=jnp.asarray(-0.1),
        seed_point=SeedPoint(
            U0=jnp.asarray(0.25),
            log_L0=jnp.asarray(0.0),
        ),
    )

    assert sampler.num_phantom() == 2
    assert float(log_likelihood) > -0.1
    assert phantom_samples.U_samples is None
    assert phantom_samples.log_L.shape == (2,)
    assert phantom_samples.valid_mask.shape == (2,)
    assert np.all(np.asarray(phantom_samples.valid_mask))
    assert np.all(np.asarray(phantom_samples.log_L) > -0.1)
