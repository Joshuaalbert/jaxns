import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.special import roots_legendre
from scipy.special import ndtr

parser = argparse.ArgumentParser(
    description="Independent numerical posterior references"
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
out = args.output
out.mkdir(parents=True, exist_ok=True)


def eggbox_series(terms):
    # exp((2+t)^5 - 243) has nonnegative power coefficients summing to one.
    # The recurrence follows by differentiating the exponential polynomial.
    a = np.zeros(terms)
    a[0] = np.exp(-211.0)
    factors = (80.0, 160.0, 120.0, 40.0, 5.0)
    for n in range(1, terms):
        a[n] = sum(factors[k - 1] * a[n - k] for k in range(1, min(n, 5) + 1)) / n
    k = np.arange(0, terms, 2) // 2
    moment = np.exp(gammaln(2 * k + 1) - 2 * gammaln(k + 1) - k * np.log(4.0))
    return a[::2], moment


records = {"eggbox10": [], "rosenbrock10": []}
for terms in (600, 800, 1200):
    a, moment = eggbox_series(terms)
    scaled_evidence = np.sum(a * moment**10)
    logz = 243.0 + np.log(scaled_evidence)
    records["eggbox10"].append(dict(terms=terms, log_Z=float(logz)))
    if terms == 1200:
        coefficients = a * moment**9 / (scaled_evidence * 10 * np.pi)
        grid = np.linspace(0.0, 10 * np.pi, 16385)
        density = np.polynomial.polynomial.polyval(
            np.cos(grid / 2.0) ** 2, coefficients
        )

        def marginal(x):
            return np.polynomial.polynomial.polyval(np.cos(x / 2.0) ** 2, coefficients)

        mass = quad(
            marginal,
            0.0,
            10 * np.pi,
            epsabs=1e-11,
            points=np.arange(11) * np.pi,
            limit=300,
        )[0]
        variance = quad(
            lambda x: (x - 5 * np.pi) ** 2 * marginal(x),
            0.0,
            10 * np.pi,
            epsabs=1e-9,
            points=np.arange(11) * np.pi,
            limit=300,
        )[0]
        # Validate the recurrence independently by 1D quadrature and 2D
        # tensor Chebyshev quadrature (the cosine arcsine measure).
        q1 = (
            quad(
                lambda x: np.exp((2 + np.cos(x)) ** 5 - 243),
                0,
                np.pi,
                epsabs=1e-13,
                limit=300,
            )[0]
            / np.pi
        )
        np.testing.assert_allclose(np.sum(a * moment), q1, rtol=1e-12)
        z = np.cos((np.arange(512) + 0.5) * np.pi / 512)
        q2 = np.mean(np.exp((2 + z[:, None] * z[None, :]) ** 5 - 243))
        np.testing.assert_allclose(np.sum(a * moment**2), q2, rtol=1e-12)
        np.testing.assert_allclose(mass, 1.0, rtol=1e-11)
        np.savez_compressed(
            out / "eggbox10.npz", x=grid, pdf=density, even_coefficients=coefficients
        )
        records["eggbox10_reference"] = dict(
            log_Z=float(logz),
            mean=float(5 * np.pi),
            standard_deviation=float(np.sqrt(variance)),
            marginal_integral=mass,
            validation_1d_relative_error=float(np.sum(a * moment) / q1 - 1),
            validation_2d_relative_error=float(np.sum(a * moment**2) / q2 - 1),
        )
    print("eggbox", terms, logz, flush=True)

for n in (512, 1024, 2048):
    nodes, weights = roots_legendre(n)
    x, w = 5 * nodes, 5 * weights  # [Q], integrate over [-5,5]
    kernel = np.exp(
        -100 * (x[None, :] - x[:, None] ** 2) ** 2 - (1 - x[:, None]) ** 2
    )  # [Q,Q]
    forward = [np.ones(n)]
    log_integral = 0.0
    for coordinate in range(9):
        following = (forward[-1] * w) @ kernel  # [Q]
        scale = np.max(following)
        forward.append(following / scale)
        log_integral += np.log(scale)
    log_integral += np.log(forward[-1] @ w)
    backward = [np.ones(n)]
    for coordinate in range(9):
        preceding = kernel @ (backward[-1] * w)  # [Q]
        backward.append(preceding / np.max(preceding))
    backward.reverse()
    pdf = np.asarray(forward) * np.asarray(backward)  # [D,Q]
    pdf /= (pdf @ w)[:, None]
    means = pdf @ (w * x)
    std = np.sqrt(pdf @ (w * x * x) - means**2)
    logz = log_integral - 10 * np.log(10.0)
    records["rosenbrock10"].append(
        dict(
            order=n,
            log_Z=float(logz),
            mean=means.tolist(),
            standard_deviation=std.tolist(),
        )
    )
    print("rosenbrock", n, logz, means, std, flush=True)
    if n == 2048:
        np.savez_compressed(out / "rosenbrock10.npz", x=x, weights=w, pdf=pdf)
np.testing.assert_allclose(
    [r["log_Z"] for r in records["eggbox10"][-2:]],
    records["eggbox10"][-1]["log_Z"],
    atol=1e-10,
    rtol=0,
)
np.testing.assert_allclose(
    records["rosenbrock10"][-2]["log_Z"],
    records["rosenbrock10"][-1]["log_Z"],
    atol=1e-8,
    rtol=0,
)
np.testing.assert_allclose(
    records["rosenbrock10"][-2]["mean"],
    records["rosenbrock10"][-1]["mean"],
    atol=1e-8,
    rtol=0,
)
np.testing.assert_allclose(
    records["rosenbrock10"][-2]["standard_deviation"],
    records["rosenbrock10"][-1]["standard_deviation"],
    atol=1e-8,
    rtol=0,
)


# Independently integrate the two-dimensional Rosenbrock kernel using the
# analytic Gaussian integral in the second coordinate and adaptive quadrature
# in the first. This checks the transfer-kernel normalisation and orientation.
def integrated_kernel(x):
    gaussian_mass = ndtr(np.sqrt(200.) * (5. - x*x)) - ndtr(
        np.sqrt(200.) * (-5. - x*x),
    )
    return np.exp(-(1. - x)**2) * np.sqrt(np.pi) / 10. * gaussian_mass


adaptive_2d = quad(integrated_kernel, -5., 5., epsabs=1e-12, epsrel=1e-12)[0]
transfer_2d = (w @ kernel) @ w
np.testing.assert_allclose(transfer_2d, adaptive_2d, rtol=1e-11)
records["rosenbrock_kernel_2d_relative_error"] = float(transfer_2d / adaptive_2d - 1.)
(out / "REFERENCE.json").write_text(json.dumps(records, indent=2))
