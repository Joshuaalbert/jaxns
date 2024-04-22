import time

import jax
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from jaxns import Prior, Model
from jaxns.experimental import EvidenceMaximisation

tfpd = tfp.distributions


@pytest.mark.parametrize("solver", ['adam', 'armijo'])
def test_basic(solver):
    def prior_model():
        x = yield Prior(tfpd.Uniform(0., 1.))
        y = yield Prior(tfpd.Normal(x, 1.), name='y').parametrised()
        sigma = yield Prior(tfpd.Exponential(1.))
        return y, sigma

    def log_likelihood(y, sigma):
        return tfpd.Normal(y, sigma).log_prob(0.)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    em = EvidenceMaximisation(model=model, ns_kwargs=dict(max_samples=1e5), verbose=False, solver=solver)
    t0 = time.time()
    ns_results, params = em.train(num_steps=3)
    print(f"Time taken ({solver}): {time.time() - t0}")

    def change(x, y):
        if np.size(x) > 0:
            return np.any(np.abs(x - y) > 1e-2)
        return True

    assert all(
        change(p, p_) for p, p_ in zip(jax.tree.leaves(model.params), jax.tree.leaves(params)))


def test_basic_zero_size_param():
    def prior_model():
        x = yield Prior(tfpd.Uniform(0., 1.))
        y = yield Prior(tfpd.Normal(x, 1.), name='y').parametrised()
        z = yield Prior(0., name='z').parametrised()  # This is a zero size parameter
        sigma = yield Prior(tfpd.Exponential(1.))
        return y, z, sigma

    def log_likelihood(y, z, sigma):
        return tfpd.Normal(y, sigma).log_prob(0.) + z

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    em = EvidenceMaximisation(model=model, ns_kwargs=dict(max_samples=1e5))
    assert any(np.size(p) == 0 for p in jax.tree.leaves(model.params))

    ns_results, params = em.train(num_steps=1)

    def change(x, y):
        if np.size(x) > 0:
            return np.any(np.abs(x - y) > 1e-2)
        return True

    assert all(
        change(p, p_) for p, p_ in zip(jax.tree.leaves(model.params), jax.tree.leaves(params)))
