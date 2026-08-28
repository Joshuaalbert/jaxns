"""Scientific contracts for continuous periodic base coordinates."""

import pickle

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import (
    ConstrainedSampleRequest,
    sample_request,
)
from jaxns.core import NestedSampler
from jaxns.distributed_core import DistributedNestedSampler
from jaxns.model import Model
from jaxns.runtime.session import WorkerSession
from jaxns.samples import SeedPoint

tfpd = tfp.distributions
CONCENTRATION = 12.0


def _angular_prior_model(centre):
    angle = Prior(
        tfpd.Uniform(low=-jnp.pi, high=jnp.pi),
        name="angle",
    ).realise(periodic=True)
    return CONCENTRATION * jnp.cos(angle - centre)


def test_periodic_evidence_and_moment_are_rotation_invariant():
    """Moving one physical mode across the chart seam changes no inference."""
    model = Model(prior_model=_angular_prior_model)
    log_evidence = float(
        jnp.log(jax.scipy.special.i0e(CONCENTRATION))
        + CONCENTRATION
    )
    resultant_length = float(
        jax.scipy.special.i1e(CONCENTRATION)
        / jax.scipy.special.i0e(CONCENTRATION)
    )
    centres = (-jnp.pi + 0.03, 0.0, 1.2)
    estimates = []
    uncertainties = []

    for index, centre in enumerate(centres):
        sampler = NestedSampler(
            model=model,
            args=(jnp.asarray(centre),),
            root_allocation_degree=120,
        )
        assert sampler.sampler._periodic == (True,)
        result = sampler.run(
            jax.random.PRNGKey(400 + index)
        ).to_result().trim()

        estimate = float(result.log_Z_mean)
        uncertainty = float(result.log_Z_uncert)
        estimates.append(estimate)
        uncertainties.append(uncertainty)
        np.testing.assert_allclose(
            estimate,
            log_evidence,
            atol=3.0 * uncertainty,
            rtol=0.0,
        )

        # [N] normalised posterior weights for the classic samples.
        weight = np.exp(np.asarray(result.log_dp))
        # [N] inferred physical angles in radians.
        angle = np.asarray(result.X_samples["angle"])
        moment = np.sum(weight * np.exp(1j * angle))
        centred_moment = moment * np.exp(-1j * float(centre))
        np.testing.assert_allclose(
            np.abs(moment),
            resultant_length,
            atol=0.04,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.angle(centred_moment),
            0.0,
            atol=0.05,
            rtol=0.0,
        )

    # This catches chart-origin sensitivity even if every individual run
    # happens to remain inside its comparatively broad one-run uncertainty.
    assert max(estimates) - min(estimates) < 3.0 * max(uncertainties)


def test_distributed_worker_replays_the_configured_periodic_sampler():
    """The serialised worker session owns the same chart/key behaviour."""
    centre = jnp.asarray(-jnp.pi + 0.03)
    model = Model(prior_model=_angular_prior_model)
    runner = DistributedNestedSampler(
        model=model,
        args=(centre,),
        coordinator_port=5555,
        root_allocation_degree=4,
    )
    assert runner.sampler._periodic == (True,)
    session = WorkerSession(
        model=model,
        sampler=runner.sampler,
        args=(centre,),
        params=None,
    )
    restored = pickle.loads(pickle.dumps(session))

    seed_structure = restored.model.sample_U(
        jax.random.PRNGKey(278),
        args=restored.args,
        params=restored.params,
    )
    # [S, ...] U leaves straddle the seam while preserving CtxParams scopes.
    seeds = jax.tree.map(
        lambda value: jnp.asarray([0.99, 0.01], dtype=value.dtype),
        seed_structure,
    )
    log_likelihoods = jax.vmap(
        lambda value: restored.model.log_likelihood(
            value,
            args=restored.args,
            params=restored.params,
            allow_nan=False,
        )
    )(seeds)
    request = ConstrainedSampleRequest(
        keys=jax.random.split(jax.random.PRNGKey(279), 2),
        valid=jnp.ones((2,), dtype=jnp.bool_),
        log_L_constraints=jnp.full((2,), -CONCENTRATION),
        seed_points=SeedPoint(
            U0=seeds,
            log_L0=log_likelihoods,
        ),
        sampler_data=None,
    )

    first = sample_request(
        restored.sampler,
        request,
        args=restored.args,
        params=restored.params,
    )
    second = sample_request(
        restored.sampler,
        request,
        args=restored.args,
        params=restored.params,
    )

    for left, right in zip(
        jax.tree.leaves(first),
        jax.tree.leaves(second),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    for value in (
        *jax.tree.leaves(first.U_samples),
        *jax.tree.leaves(first.phantom_samples.U_samples),
    ):
        assert np.all(np.asarray(value) >= 0.0)
        assert np.all(np.asarray(value) < 1.0)
