"""System test specified by local_run_round_trip.md."""

import jax
import numpy as np
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import jaxns
from jaxns.core import NestedSampler
from jaxns.model import Model

tfpd = tfp.distributions


def _build_standard_model() -> Model:
    """Build the model inside the designed system boundary."""

    def prior_model():
        x = jaxns.Prior(
            tfpd.Uniform(low=-1.0, high=1.0),
            name="x",
        ).realise()
        return -0.5 * jnp.square((x - 0.2) / 0.25)

    return Model(prior_model=prior_model)


def test_local_run_reaches_both_final_evidence_modes() -> None:
    model = _build_standard_model()
    sampler = NestedSampler(
        model=model,
        root_allocation_degree=8,
        shell_size=4,
        max_samples=32,
        collect_phantom_samples=True,
    )
    state = sampler.run(key=jax.random.PRNGKey(17))
    results = state.to_result().trim()

    classic = results.sample_evidence_mc(
        num_samples=8,
        conditioning="classic",
        key=jax.random.PRNGKey(19),
    )
    phantom = results.sample_evidence_mc(
        num_samples=8,
        conditioning="phantom",
        key=jax.random.PRNGKey(23),
    )

    assert int(results.total_num_samples) == results.log_L.shape[0]
    assert int(results.total_phantom_samples) > 0
    assert classic.log_Z_samples.shape == (8,)
    assert phantom.log_Z_samples.shape == (8,)
    assert np.all(np.isfinite(np.asarray(classic.log_Z_samples)))
    assert np.all(np.isfinite(np.asarray(phantom.log_Z_samples)))
