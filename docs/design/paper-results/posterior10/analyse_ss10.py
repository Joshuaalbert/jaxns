"""Reduce the repository SS10 run and validate finite-box posterior references."""

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import jax
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.special import ndtr
from scipy.stats import norm
from scipy.stats import truncnorm

from benchmarks.paper_reproduction.posterior_cases import reference_values
from benchmarks.paper_reproduction.posterior_cases import ss10_parameters
from benchmarks.paper_reproduction.prefix_sweep import sample_phantom_prefix_sweep_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mc-draws", type=int, default=2048)
    args = parser.parse_args()
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Pin SS10 analysis to exactly one CPU")
    if args.mc_draws < 2:
        raise ValueError("At least two shrinkage draws are required")
    cell = args.root / "ss10/seed-00"
    core = json.loads((cell / "CORE.json").read_text())
    manifest = json.loads((cell / "MANIFEST.json").read_text())
    if manifest["ss10_definition"] != "repository":
        raise ValueError("This report requires the repository SS10 definition")
    if manifest["allocation_target"] != "evidence_improving":
        raise ValueError("SS10 must use A+C")
    started = time.perf_counter()
    means, scales = ss10_parameters("repository")  # [2,D], [2]
    reference = reference_values("ss10", "repository")
    component_mass = np.exp(np.asarray(reference["component_log_Z"])
                            - reference["log_Z"])  # [2]
    a = (-4. - means) / scales[:, None]  # [2,D]
    b = (8. - means) / scales[:, None]  # [2,D]
    box_mass = ndtr(b) - ndtr(a)  # [2,D]
    bounded_mean, bounded_var = truncnorm.stats(
        a, b, loc=means, scale=scales[:, None], moments="mv",
    )  # [2,D] each
    mean = component_mass @ bounded_mean  # [D]
    var = component_mass @ (bounded_var + bounded_mean ** 2) - mean ** 2
    grid = np.linspace(-4., 8., 16385)  # [M]
    component_pdf = norm.pdf(
        grid[None, None, :], loc=means[:, :, None],
        scale=scales[:, None, None],
    ) / box_mass[:, :, None]  # [2,D,M], bounded component marginals
    marginal_pdf = np.einsum("k,kdm->dm", component_mass, component_pdf)  # [D,M]
    # Adaptive quadrature independently checks the closed-form marginal
    # normalisation and first two moments for both distinct coordinate types.
    quadrature = {}
    for dim in (0, 2):
        def marginal(x):
            return np.sum(component_mass * norm.pdf(
                x, loc=means[:, dim], scale=scales,
            ) / box_mass[:, dim])
        integral = quad(marginal, -4., 8., epsabs=1e-12)[0]
        first = quad(lambda x: x * marginal(x), -4., 8., epsabs=1e-12)[0]
        second = quad(lambda x: x*x * marginal(x), -4., 8., epsabs=1e-12)[0]
        np.testing.assert_allclose(integral, 1., atol=1e-11, rtol=0)
        np.testing.assert_allclose(first, mean[dim], atol=1e-11, rtol=0)
        np.testing.assert_allclose(second-first**2, var[dim], atol=1e-10, rtol=0)
        quadrature[str(dim)] = {"mass": integral, "mean": first,
                                "variance": second-first**2}
    reference.update(posterior_mean=mean.tolist(),
                     posterior_standard_deviation=np.sqrt(var).tolist(),
                     marginal_quadrature=quadrature)
    args.root.joinpath("references").mkdir(exist_ok=True)
    np.savez_compressed(args.root / "references/ss10.npz", x=grid, pdf=marginal_pdf)
    (args.root / "references/SS10_REFERENCE.json").write_text(
        json.dumps(reference, indent=2),
    )
    posterior = np.load(cell / "classic_posterior.npz")
    x = posterior["x"]  # [N,D], classic coordinates
    log_weights = posterior["log_dp"]  # [N], classic expected weights
    weights = np.exp(log_weights-logsumexp(log_weights))
    component_log_L = np.stack([
        norm.logpdf(x, loc=mu, scale=scale).sum(axis=1)
        for mu, scale in zip(means, scales, strict=True)
    ], axis=1)  # [N,2]
    responsibilities = np.exp(component_log_L[:, 0]
                              - logsumexp(component_log_L, axis=1))  # [N]
    spike_mass = float(weights @ responsibilities)
    classified_spike_mass = float(weights[responsibilities > .5].sum())
    record = {
        "case": "ss10", "seed": 0, "ss10_definition": "repository",
        "source_commit": manifest["source_commit"], "reference": reference,
        "classic_expected_log_Z": core["classic_log_Z"],
        "classic_expected_log_Z_error": core["classic_log_Z"]-reference["log_Z"],
        "classic_expected_log_Z_uncert": core["classic_log_Z_uncert"],
        "classic_spike_responsibility_mass": spike_mass,
        "classic_classified_spike_mass": classified_spike_mass,
        "classic_spike_mass_error": spike_mass-reference["spike_mass"],
        "classic_kish_ess": float(1./np.sum(weights**2)),
        "analysis_affinity": sorted(os.sched_getaffinity(0)),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(record), flush=True)
    with (cell / "state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    results = state.to_result().trim()
    np.testing.assert_allclose(float(results.ess), record["classic_kish_ess"],
                               rtol=1e-10)
    draws, gates = sample_phantom_prefix_sweep_reference(
        key=jax.random.fold_in(jax.random.PRNGKey(0), 1),
        log_L_constraints=results.log_L_constraints,
        K_classic=results.num_live_points_per_sample,
        valid_phantom=results.valid_phantom,
        log_L_phantom=results.log_L_phantom[:, :90],
        num_samples=results.total_num_samples,
        block_state=results.block_data.to_block_state(),
        dimension=10, num_draws=args.mc_draws, batch_size=1,
        num_workers=1, num_groups=9, C_min=20.,
    )
    np.savez_compressed(cell / "evidence_draws.npz", log_Z=draws)
    record.update(
        prefix_sizes=list(range(0, 91, 10)), mc_draws=args.mc_draws,
        log_Z_mean=draws.mean(axis=0).tolist(),
        log_Z_uncert=draws.std(axis=0, ddof=1).tolist(),
        log_Z_error=(draws.mean(axis=0)-reference["log_Z"]).tolist(),
        gate_fraction=gates.mean(axis=1).tolist(),
        analysis_seconds=time.perf_counter()-started,
    )
    with (cell / "ANALYSIS.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
