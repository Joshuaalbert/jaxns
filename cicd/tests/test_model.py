import jax
from jax import numpy as jnp
from jaxctx import scope
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.model import Model

tfpd = tfp.distributions


def test_init_params_forwards_model_and_explicit_args() -> None:
    """Model data and parameters remain explicit rather than closure-bound."""

    def prior_model(observations):
        location = Prior(
            tfpd.Normal(loc=0.0, scale=1.0),
            name="location",
        ).realise()
        uncertainty = Prior(
            tfpd.Exponential(rate=1.0),
            name="uncertainty",
        ).parameter()
        likelihood = tfpd.Normal(location, uncertainty)
        return jnp.sum(likelihood.log_prob(observations))

    model = Model(prior_model=prior_model)
    args = (jnp.asarray([-0.1, 0.2]),)
    params = model.init_params(
        key=jax.random.PRNGKey(0),
        args=args,
    )
    sample = model.sample_U(
        key=jax.random.PRNGKey(1),
        args=args,
        params=params,
    )
    log_likelihood = model.log_likelihood(
        sample,
        args=args,
        params=params,
    )
    transformed = model.transform_to_X(
        sample,
        args=args,
        params=params,
    )
    log_prior = model.log_prior(
        sample,
        args=args,
        params=params,
    )
    log_joint = model.log_joint(
        sample,
        args=args,
        params=params,
    )

    assert "uncertainty" in params
    assert "location" in transformed
    assert model.U_ndims(args=args, params=params) > 0
    assert jnp.all(jnp.isfinite(jnp.asarray([
        log_likelihood,
        log_prior,
        log_joint,
    ])))
    assert jnp.allclose(log_joint, log_prior + log_likelihood)


def test_periodic_coordinates_expand_in_sampler_order() -> None:
    """Whole-prior declarations become scalar flags without changing U."""

    def prior_model():
        with scope("calibration"):
            angles = Prior(
                tfpd.Uniform(
                    low=jnp.zeros((2,)),
                    high=jnp.ones((2,)),
                ),
                name="angles",
            ).realise(periodic=True)
            radius = Prior(
                tfpd.Uniform(low=0.0, high=1.0),
                name="radius",
            ).realise()
        return -jnp.sum(jnp.square(angles)) - jnp.square(radius)

    model = Model(prior_model=prior_model)
    sample = model.sample_U(jax.random.PRNGKey(2))

    assert model._periodic_coordinates() == (True, True, False)
    assert jax.tree.structure(sample) == jax.tree.structure(
        model.sample_U(jax.random.PRNGKey(3))
    )


def test_periodic_coordinates_report_all_false_topology() -> None:
    """Ordinary models retain an aligned all-false topology."""

    def prior_model():
        value = Prior(
            tfpd.Uniform(low=0.0, high=1.0),
            name="value",
        ).realise()
        return -jnp.square(value)

    assert Model(prior_model=prior_model)._periodic_coordinates() == (False,)
