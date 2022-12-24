from jaxns.new_code.initial_state import init_sample_collection
from jaxns.new_code.model import Model
import tensorflow_probability.substrates.jax as tfp

from jaxns.new_code.prior import PriorModelGen, Prior

tfpd = tfp.distributions

def test_init_sample_collection():
    def prior_model() -> PriorModelGen:
        x = yield Prior(tfpd.Uniform(low=0, high=2))
        y = yield Prior(tfpd.Normal(loc=2, scale=x))
        z = x + y
        return z

    def log_likelihood(z):
        return -z ** 2

    model = Model(prior_model=prior_model,
                  log_likelihood=log_likelihood)

    sample_collection = init_sample_collection(size=10, model=model)
    print(sample_collection)
