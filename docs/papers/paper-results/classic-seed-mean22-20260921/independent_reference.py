"""Independent mean22 reference check, executed on dorrie."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp, roots_hermitenorm
from scipy.stats import multivariate_normal, norm

mean = np.array([2., -2.] + [0.] * 8)
covariance = .01 * np.eye(10) + .99
g_logz = float(multivariate_normal.logpdf(mean, cov=covariance + np.eye(10)))
# This mean is orthogonal to the common-correlation direction. The combined
# covariance has eigenvalues 1.01 (nine times) and 10.91 (once).
g_eigenvalue_logz = float(-.5 * (
    10 * np.log(2 * np.pi) + 9 * np.log(1.01) + np.log(10.91) + 8 / 1.01
))
conditional = covariance[1:, 1:] - np.outer(
    covariance[1:, 0], covariance[0, 1:],
)
combined = conditional + np.eye(9)
precision = np.linalg.inv(combined)
normalizer = -.5 * (9 * np.log(2 * np.pi) + np.linalg.slogdet(combined)[1])


def log_rest_density(first):
    conditional_mean = mean[1:] + covariance[1:, 0] * (first - mean[0])
    conditional_mean[0] += .4 * ((first - mean[0])**2 - 1)
    return normalizer - .5 * conditional_mean @ precision @ conditional_mean


# Integrate the original product of first-coordinate densities directly,
# rather than the posterior-factorized integrand used by the archived code.
value, error = quad(
    lambda first: np.exp(
        norm.logpdf(first) + norm.logpdf(first, loc=2)
        + log_rest_density(first)
    ),
    -np.inf, np.inf, epsabs=1e-20, epsrel=1e-11, limit=400,
)
cg_logz = float(np.log(value))
hermite = {}
for order in (128, 256):
    nodes, weights = roots_hermitenorm(order)
    # The integration measure here is the original N(2,1) likelihood marginal.
    first = 2 + nodes
    terms = np.array([norm.logpdf(t) + log_rest_density(t) for t in first])
    hermite[str(order)] = float(
        logsumexp(np.log(weights) + terms) - .5 * np.log(2 * np.pi)
    )
np.testing.assert_allclose(g_logz, g_eigenvalue_logz, rtol=0, atol=1e-12)
np.testing.assert_allclose(list(hermite.values()), cg_logz, rtol=0, atol=1e-10)
result = {
    "hostname": "dorrie",
    "mean": mean.tolist(),
    "g10_conjugate_log_Z": g_logz,
    "g10_eigenvalue_log_Z": g_eigenvalue_logz,
    "cg10_direct_quad_log_Z": cg_logz,
    "cg10_direct_quad_relative_error": float(error / value),
    "cg10_original_marginal_hermite_log_Z": hermite,
}
Path('/tmp/jaxns-mean22-independent-reference.json').write_text(
    json.dumps(result, indent=2) + '\n',
)
print(json.dumps(result, indent=2))
