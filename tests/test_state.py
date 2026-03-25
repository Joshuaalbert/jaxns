import numpy as np
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model


tfpd = tfp.distributions


def _make_basic_model() -> Model:
    def prior_model():
        x = Prior(tfpd.Uniform(low=0.0, high=1.0), name='x').realise()
        return -x ** 2

    return Model(prior_model=prior_model)


def test_to_result_marks_no_phantoms_invalid():
    model = _make_basic_model()
    results = NestedSampler(model=model).run().to_result().trim()

    assert results.log_L_phantom.shape[1] == 0
    assert int(results.total_phantom_samples) == 0
    assert not np.any(np.asarray(results.valid_phantom))

    evidence_samples = results.sample_mc_shrinkage(num_samples=16)
    np.testing.assert_allclose(np.asarray(evidence_samples.rho_samples), 1.0)
    np.testing.assert_allclose(np.asarray(evidence_samples.eta_samples), 0.0)
    np.testing.assert_allclose(np.asarray(evidence_samples.rho_eta_samples), 0.0)
