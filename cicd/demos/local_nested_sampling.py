"""Run the supported local state-to-results workflow."""

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model

tfpd = tfp.distributions


def build_standard_model() -> Model:
    """Build the complete demo model without repository-local imports."""

    def prior_model():
        x = Prior(tfpd.Uniform(low=-1.0, high=1.0), name="x").realise()
        return -0.5 * jnp.square((x - 0.2) / 0.25)

    return Model(prior_model=prior_model)


def main() -> None:
    model = build_standard_model()
    sampler = NestedSampler(
        model=model,
        root_allocation_degree=8,
        shell_size=4,
        max_samples=32,
        collect_phantom_samples=True,
    )
    state = sampler.run(key=jax.random.PRNGKey(7))
    results = state.to_result().trim()

    classic = results.sample_evidence_mc(
        num_samples=16,
        conditioning="classic",
        key=jax.random.PRNGKey(11),
    )
    phantom = results.sample_evidence_mc(
        num_samples=16,
        conditioning="phantom",
        key=jax.random.PRNGKey(13),
    )

    assert int(results.total_num_samples) > 0
    assert int(results.total_phantom_samples) > 0
    assert np.all(np.isfinite(np.asarray(classic.log_Z_samples)))
    assert np.all(np.isfinite(np.asarray(phantom.log_Z_samples)))
    print(
        "local nested sampling:",
        f"samples={int(results.total_num_samples)}",
        f"phantoms={int(results.total_phantom_samples)}",
        f"log_Z={float(results.log_Z_mean):.3f}",
    )


if __name__ == "__main__":
    main()
