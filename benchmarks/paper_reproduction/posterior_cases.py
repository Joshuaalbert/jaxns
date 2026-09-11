"""Ten-dimensional adapters of the pinned jaxns-cosmology test functions.

Source: Joshuaalbert/jaxns-cosmology@67e45dac6d9b273c4f4d957433f53228bf9ea7e0.
The source returns log likelihoods, including Eggbox's positive fifth power.
SS10 variants explicitly separate covariance diagonals from standard deviations.
"""

import numpy as np
from jax import numpy as jnp
from jaxctx.priors.prior import Prior
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr
from tensorflow_probability.substrates import jax as tfp

from jaxns.model import Model

DIMENSION = 10
LEGACY_COMMIT = "67e45dac6d9b273c4f4d957433f53228bf9ea7e0"
ESS_TARGETS = {"eggbox10": 5000, "rosenbrock10": 3000, "rastrigin10": 15000}
PRIOR_BOUNDS = {
    "eggbox10": (0., 10 * np.pi), "rosenbrock10": (-5., 5.),
    "rastrigin10": (-5.12, 5.12), "ss10": (-4., 8.),
}
SS10_DEFINITIONS = ("repository", "sd_first_two", "sd_all_ten")


def eggbox_log_likelihood(x):
    """Legacy Eggbox log likelihood; this expression is already log L."""
    return (2. + jnp.prod(jnp.cos(x / 2.))) ** 5


def rosenbrock_log_likelihood(x):
    """Legacy curved Rosenbrock log likelihood on ten coordinates."""
    return -jnp.sum(100. * (x[1:] - x[:-1] ** 2) ** 2 + (1. - x[:-1]) ** 2)


def rastrigin_log_likelihood(x):
    """Legacy separable Rastrigin log likelihood."""
    return -(10. * x.size + jnp.sum(x ** 2 - 10. * jnp.cos(2. * jnp.pi * x)))


def ss10_parameters(definition: str) -> tuple[np.ndarray, np.ndarray]:
    """Return explicitly defined component means [2,D] and SDs [2]."""
    if definition not in SS10_DEFINITIONS:
        raise ValueError(f"Unknown SS10 definition: {definition}")
    means = np.zeros((2, DIMENSION))  # [2, D]
    means[:, :2] = np.array([6., 2.5])[:, None]
    if definition == "sd_all_ten":
        means[:] = np.array([6., 2.5])[:, None]
    scales = np.array([.08, .8])  # [2]
    if definition == "repository":
        scales = np.sqrt(scales)
    return means, scales


def ss10_log_likelihood(x, definition: str):
    """Sum of two normalised 10D densities, without a factor of one half."""
    means, scales = ss10_parameters(definition)
    displacement = (x[None, :] - jnp.asarray(means)) / jnp.asarray(scales[:, None])
    component_log_L = (
        -.5 * DIMENSION * jnp.log(2. * jnp.pi)
        - DIMENSION * jnp.log(jnp.asarray(scales))
        - .5 * jnp.sum(displacement ** 2, axis=1)
    )  # [2]
    return jnp.logaddexp(component_log_L[0], component_log_L[1])


def prior_model(case: str, ss10_definition: str | None = None):
    """Use the historical independent uniform prior and current Prior API."""
    low, high = PRIOR_BOUNDS[case]
    x = Prior(tfp.distributions.Uniform(
        low=jnp.full((DIMENSION,), low, dtype=jnp.float64),
        high=jnp.full((DIMENSION,), high, dtype=jnp.float64),
    ), name="x").realise()
    if case == "eggbox10":
        return eggbox_log_likelihood(x)
    if case == "rosenbrock10":
        return rosenbrock_log_likelihood(x)
    if case == "rastrigin10":
        return rastrigin_log_likelihood(x)
    if case == "ss10":
        return ss10_log_likelihood(x, ss10_definition)
    raise ValueError(f"Unknown case: {case}")


def eggbox_prior():
    """Module-level entry point for durable model/state pickles."""
    return prior_model("eggbox10")


def rosenbrock_prior():
    """Module-level entry point for durable model/state pickles."""
    return prior_model("rosenbrock10")


def rastrigin_prior():
    """Module-level entry point for durable model/state pickles."""
    return prior_model("rastrigin10")


def ss10_repository_prior():
    """Module-level entry point for the historical code definition."""
    return prior_model("ss10", "repository")


def ss10_sd_first_two_prior():
    """Module-level entry point for SD scales and two shifted coordinates."""
    return prior_model("ss10", "sd_first_two")


def ss10_sd_all_ten_prior():
    """Module-level entry point for SD scales and ten shifted coordinates."""
    return prior_model("ss10", "sd_all_ten")


def build_model(case: str, ss10_definition: str | None = None) -> Model:
    """Build a pickle-safe model with an explicit SS10 interpretation."""
    if case not in PRIOR_BOUNDS:
        raise ValueError(f"Unknown case: {case}")
    if case == "ss10":
        ss10_parameters(ss10_definition)
    elif ss10_definition is not None:
        raise ValueError("SS10 definition supplied for a different problem")
    functions = {
        "eggbox10": eggbox_prior, "rosenbrock10": rosenbrock_prior,
        "rastrigin10": rastrigin_prior,
    }
    if case == "ss10":
        functions[case] = {
            "repository": ss10_repository_prior,
            "sd_first_two": ss10_sd_first_two_prior,
            "sd_all_ten": ss10_sd_all_ten_prior,
        }[ss10_definition]
    return Model(prior_model=functions[case])


def reference_values(case: str, ss10_definition: str | None = None) -> dict:
    """Independent quadrature or exact finite-box Gaussian reference values."""
    if case == "ss10":
        means, scales = ss10_parameters(ss10_definition)
        mass = ndtr((8. - means) / scales[:, None]) - ndtr(
            (-4. - means) / scales[:, None],
        )  # [2, D]
        log_component_Z = np.log(mass).sum(axis=1) - DIMENSION * np.log(12.)
        log_Z = float(logsumexp(log_component_Z))
        return {
            "method": "product of Gaussian CDF differences over the finite prior box",
            "log_Z": log_Z,
            "component_log_Z": log_component_Z.tolist(),
            "spike_mass": float(np.exp(log_component_Z[0] - log_Z)),
            "means": means.tolist(), "standard_deviations": scales.tolist(),
        }
    if case == "rastrigin10":
        def likelihood(x):
            return np.exp(-10. - x * x + 10. * np.cos(2. * np.pi * x))
        mass, error = quad(likelihood, -5.12, 5.12, epsabs=1e-12,
                           epsrel=1e-12, points=np.arange(-5., 6.), limit=300)
        variance = quad(lambda x: x * x * likelihood(x), -5.12, 5.12,
                        epsabs=1e-12, epsrel=1e-12,
                        points=np.arange(-5., 6.), limit=300)[0] / mass
        return {
            "method": "factorised one-dimensional adaptive quadrature",
            "log_Z": float(DIMENSION * np.log(mass / 10.24)),
            "one_dimensional_mass": mass, "quadrature_absolute_error": error,
            "mean": 0., "standard_deviation": float(np.sqrt(variance)),
        }
    return {"method": "no reference evidence used"}
