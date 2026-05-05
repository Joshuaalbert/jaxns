import numpy as np
import jax.numpy as jnp
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.termination_condition import TerminationCondition
from jaxns.samples import PhantomSamples, Samples


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


def test_run_records_termination_reason():
    """NestedSampler.run should record the final termination bitmask in state."""

    model = _make_basic_model()
    state = NestedSampler(
        model=model,
        target_num_live_points=10,
        max_samples=10,
        shell_size=1,
        termination_condition=TerminationCondition(max_samples=jnp.asarray(10)),
    ).run()

    assert int(state.termination_reason) & 1 == 1


def test_samples_resize_preserves_constraints_and_shapes():
    """Samples.resize should preserve data and extend phantom buffers consistently."""

    samples = Samples(
        log_L_constraints=jnp.asarray([0.0, 1.0]),
        log_likelihoods=jnp.asarray([0.5, 1.5]),
        sample_ids=jnp.asarray([0, 1]),
        U_samples={'x': jnp.asarray([[1.0], [2.0]])},
        out_degree=jnp.asarray([1, 0]),
        num_likelihood_evaluations=jnp.asarray([3, 4]),
        phantom_samples=PhantomSamples(
            U_samples={'x': jnp.asarray([[[10.0]], [[20.0]]])},
            valid_mask=jnp.asarray([[True], [False]]),
            log_L=jnp.asarray([[0.25], [0.75]]),
        ),
    )

    resized = samples.resize(4)

    assert resized.log_L_constraints.shape == (4,)
    np.testing.assert_allclose(np.asarray(resized.log_L_constraints[:2]), np.asarray(samples.log_L_constraints))
    assert resized.log_likelihoods.shape == (4,)
    np.testing.assert_array_equal(np.asarray(resized.sample_ids), np.asarray([0, 1, 2, 3]))
    assert resized.U_samples['x'].shape == (4, 1)
    assert resized.phantom_samples.U_samples['x'].shape == (4, 1, 1)


def test_samples_sort_breaks_equal_likelihood_ties_with_sample_id():
    """Samples.sort should use sample_ids as the deterministic tie-breaker."""

    samples = Samples(
        log_L_constraints=jnp.asarray([0.0, 0.0, 0.0]),
        log_likelihoods=jnp.asarray([1.0, 1.0, 1.0]),
        sample_ids=jnp.asarray([2, 0, 1]),
        U_samples={'x': jnp.asarray([[2.0], [0.0], [1.0]])},
        out_degree=jnp.asarray([0, 0, 0]),
        num_likelihood_evaluations=jnp.asarray([1, 1, 1]),
        phantom_samples=PhantomSamples(
            U_samples=None,
            valid_mask=jnp.zeros((3, 0), dtype=bool),
            log_L=jnp.zeros((3, 0)),
        ),
    )

    sorted_samples = samples.sort()

    np.testing.assert_array_equal(np.asarray(sorted_samples.sample_ids), np.asarray([0, 1, 2]))
    np.testing.assert_allclose(np.asarray(sorted_samples.U_samples['x']).reshape((-1,)), np.asarray([0.0, 1.0, 2.0]))

