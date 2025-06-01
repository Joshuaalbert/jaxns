import time

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from jaxns import Prior, Model
from jaxns.experimental import EvidenceMaximisation

tfpd = tfp.distributions


def test_basic():
    def prior_model():
        x = yield Prior(tfpd.Uniform(0., 1.))
        y = yield Prior(tfpd.Normal(x, 1.), name='y').parametrised()
        sigma = yield Prior(tfpd.Exponential(1.))
        return y, sigma

    def log_likelihood(y, sigma):
        return tfpd.Normal(y, sigma).log_prob(0.)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    em = EvidenceMaximisation(model=model, verbose=False)
    t0 = time.time()
    ns_results, params = em.train(num_steps=3)
    print(f"Time taken: {time.time() - t0}")


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

    em = EvidenceMaximisation(model=model)
    assert any(np.size(p) == 0 for p in jax.tree.leaves(model.params))

    ns_results, params = em.train(num_steps=1)
