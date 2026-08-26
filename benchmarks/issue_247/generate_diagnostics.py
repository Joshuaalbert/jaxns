"""Generate representative Spike-Slab review plots from standard problems."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np
from matplotlib import pyplot as plt

from cicd.tests.test_ns_standard_problems import STANDARD_PROBLEM_CASES_BY_NAME
from jaxns.constrained_sampler import UniDimSliceSampler
from jaxns.core import NestedSampler

plt.switch_backend("Agg")


REPRESENTATIVE_SEEDS = {
    "spike_slab": 16,
    "spike_slab10": 2,
}


def _run_case(case_name: str, seed: int):
    """Run one case with the same configuration as the final matrix."""
    case = STANDARD_PROBLEM_CASES_BY_NAME[case_name]
    model, truth = case.build_case()
    ndims = int(model.U_ndims())
    num_slices = 5 * ndims
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=num_slices,
        no_step_out=True,
        gradient_guided=False,
        collect_phantom_samples=True,
        phantom_burn_in=num_slices - 1 - ndims,
    )
    nested_sampler = NestedSampler(
        model=model,
        root_allocation_degree=30 * ndims,
        shell_size=10 * ndims,
        max_samples=100 * 30 * ndims,
        collect_phantom_samples=True,
        sampler=sampler,
    )
    result = nested_sampler.run(jax.random.PRNGKey(seed)).to_result().trim()
    return result, truth


def _plot_phantom_conditioning(
        result,
        key: jax.Array,
        save_path: Path,
) -> None:
    """Plot the block probabilities and Kish gate used by conditioning."""
    evidence = result.sample_evidence_mc(
        num_samples=1_000,
        conditioning="phantom",
        key=key,
    )
    valid = np.isfinite(np.asarray(evidence.log_L_blocks))
    blocks = np.flatnonzero(valid)
    alpha_total = (
        np.asarray(evidence.classic_alpha_gt)
        + np.asarray(evidence.classic_alpha_eq)
        + np.asarray(evidence.classic_alpha_lt)
    )
    classic = np.divide(
        np.asarray(evidence.classic_alpha_gt),
        alpha_total,
        out=np.zeros_like(alpha_total),
        where=alpha_total > 0,
    )
    phantom_A = np.asarray(evidence.phantom_A)
    observed = np.divide(
        np.asarray(evidence.phantom_B),
        phantom_A,
        out=np.zeros_like(phantom_A),
        where=phantom_A > 0,
    )
    conditioned = np.asarray(evidence.p_gt_mean)
    kish = np.asarray(evidence.kish_participating_cluster_counts)
    gate = np.asarray(evidence.phantom_gate_active, dtype=bool)

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    axes[0].plot(blocks, classic[valid], label="classic mean", linewidth=1)
    axes[0].plot(
        blocks,
        conditioned[valid],
        label="phantom-conditioned mean",
        linewidth=1,
    )
    axes[0].scatter(
        blocks[gate[valid]],
        observed[valid][gate[valid]],
        label="phantom B/A (gated)",
        s=5,
        alpha=0.5,
    )
    axes[0].set_ylabel(r"$p_{>g}$")
    axes[0].legend()

    axes[1].plot(blocks, kish[valid], linewidth=1)
    axes[1].axhline(20, color="black", linestyle="--", label=r"$C_{min}=20$")
    axes[1].fill_between(
        blocks,
        0,
        kish[valid],
        where=gate[valid],
        alpha=0.2,
        label="conditioning active",
    )
    axes[1].set_xlabel("likelihood block index")
    axes[1].set_ylabel("Kish participating clusters")
    axes[1].legend()
    figure.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/issue_247/diagnostics"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for case_name, seed in REPRESENTATIVE_SEEDS.items():
        result, truth = _run_case(case_name, seed)
        prefix = args.output / f"{case_name}_seed_{seed}"
        evidence_key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
        result.plot_cornerplot(save_name=f"{prefix}_cornerplot.png")
        result.plot_diagnostics(save_file=f"{prefix}_diagnostics.png")
        _plot_phantom_conditioning(
            result,
            evidence_key,
            Path(f"{prefix}_phantom_conditioning.png"),
        )
        evidence = result.sample_evidence_mc(
            num_samples=1_000,
            conditioning="phantom",
            key=evidence_key,
        )
        print(
            f"{case_name} seed={seed} truth={truth:.6f} "
            f"classic={float(result.log_Z_mean):.6f} "
            f"phantom={float(evidence.log_Z_mean):.6f} "
            f"phantom_std={float(evidence.log_Z_uncert):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
