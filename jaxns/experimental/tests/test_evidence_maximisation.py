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

    em = EvidenceMaximisation(model=model, ns_kwargs=dict(max_samples=1e5))
    params = em.train(num_steps=10)
