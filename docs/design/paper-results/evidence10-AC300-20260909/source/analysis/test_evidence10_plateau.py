"""Independent plateau-law checks for the sparse evidence-only reducer."""

import jax
from jax import numpy as jnp
import numpy as np
import pytest

from benchmarks.paper_reproduction.prefix_sweep import sample_phantom_prefix_sweep_reference
from jaxns.algorithm.race_tree import BlockState
from jaxns.shrinkage.classic import gamma_weighted_phantom_probabilities_from_draws


@pytest.mark.parametrize('minimum_clusters', [1., 5.])
def test_mixed_blocks_match_dense_three_category_core(minimum_clusters):
    levels = np.array([-3., -1., 0., 1.])  # [G]
    sizes = np.array([1, 2, 1, 3])  # [G], seven classic samples
    incoming = np.array([8, 7, 6, 4])
    constraints = np.array([-np.inf, -np.inf, -3., -3., -1., -1., 0.])  # [C]
    phantom = np.random.default_rng(17).uniform(-4., 3., (7, 4))  # [C,P]
    phantom[:, 1] = -1.  # Exact equality observations at a plateau.
    phantom[:, 3] = 1.
    blocks = BlockState(
        log_L_blocks=jnp.asarray(levels), block_first_idx=jnp.asarray([0, 1, 3, 4]),
        block_size=jnp.asarray(sizes), incoming_K=jnp.asarray(incoming),
        block_out_degree=jnp.zeros(4, dtype=int), valid=jnp.ones(4, dtype=bool),
    )
    key = jax.random.PRNGKey(19)
    actual, gates = sample_phantom_prefix_sweep_reference(
        key=key, log_L_constraints=jnp.asarray(constraints),
        K_classic=jnp.ones(7, dtype=int), valid_phantom=jnp.ones(7, dtype=bool),
        log_L_phantom=jnp.asarray(phantom), num_samples=jnp.int64(7),
        block_state=blocks, dimension=2, num_groups=2,
        num_draws=8, batch_size=1, C_min=minimum_clusters,
    )
    previous = np.r_[-np.inf, levels[:-1]]
    counts = []
    expected_gates = []
    for prefix in (0, 2, 4):
        values = phantom[:, :prefix, None]  # [C,prefix,G]
        eligible = constraints[:, None] <= previous[None, :]
        a = np.sum(values > previous, axis=1) * eligible
        b = np.sum(values > levels, axis=1) * eligible
        e = np.sum(values == levels, axis=1) * eligible
        counts.append((a, b, e))
        if prefix:
            denominator = np.sum(a**2, axis=0)
            kish = np.divide(a.sum(axis=0)**2, denominator,
                             out=np.zeros(4), where=denominator > 0)
            expected_gates.append((denominator > 0) & (kish >= minimum_clusters))
    np.testing.assert_array_equal(gates, expected_gates)
    for draw in range(8):
        subkeys = jax.random.split(jax.random.fold_in(key, draw), 4)
        rng = [np.random.default_rng(np.random.SeedSequence(np.asarray(k))) for k in subkeys]
        gt = rng[0].gamma(incoming - sizes + 1.)
        complement = rng[2].exponential(size=4)
        atom = sizes > 1
        complement[atom] = rng[1].gamma(sizes[atom] + 1.)
        weights = rng[3].exponential(size=7)
        # The deterministic core accepts explicit draws. Any positive split
        # of the complement must give the same p_>; E and R cancel in its
        # denominator even though their separate posterior masses differ.
        eq = np.where(atom, .37 * complement, 0.)
        lt = complement - eq
        expected = []
        for a, b, e in counts:
            probability = gamma_weighted_phantom_probabilities_from_draws(
                block_state=blocks, A_cg=jnp.asarray(a, dtype=float),
                B_cg=jnp.asarray(b, dtype=float), E_cg=jnp.asarray(e, dtype=float),
                race_gamma_gt=jnp.asarray(gt), race_gamma_eq=jnp.asarray(eq),
                race_gamma_lt=jnp.asarray(lt), cluster_weights=jnp.asarray(weights),
                C_min=minimum_clusters,
            )
            p_gt = np.asarray(probability.p_gt)
            prior_volume = np.r_[1., np.cumprod(p_gt)[:-1]]
            expected.append(np.log(np.sum(prior_volume * (1-p_gt) * np.exp(levels))))
        np.testing.assert_allclose(actual[draw], expected, atol=1e-12, rtol=0)


def test_sevenfold_plateau_matches_analytic_beta_moments():
    # K=10 and m=7 imply p_> ~ Beta(4,8), so Z=2*(1-p_>) on this one block.
    blocks = BlockState(
        log_L_blocks=jnp.log(jnp.asarray([2.])), block_first_idx=jnp.asarray([0]),
        block_size=jnp.asarray([7]), incoming_K=jnp.asarray([10]),
        block_out_degree=jnp.asarray([0]), valid=jnp.asarray([True]),
    )
    draws, gates = sample_phantom_prefix_sweep_reference(
        key=jax.random.PRNGKey(41), log_L_constraints=jnp.full(7, -jnp.inf),
        K_classic=jnp.full(7, 10), valid_phantom=jnp.zeros(7, dtype=bool),
        log_L_phantom=jnp.zeros((7, 4)), num_samples=jnp.int64(7),
        block_state=blocks, dimension=2, num_groups=2,
        num_draws=32768, batch_size=512,
    )
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    np.testing.assert_array_equal(draws[:, 0], draws[:, 2])
    assert not gates.any()
    evidence = np.exp(draws[:, 0])
    mean = 2.*8./12.
    variance = 4.*4.*8./(12.**2*13.)
    assert abs(evidence.mean()-mean) < 6.*np.sqrt(variance/len(evidence))
    np.testing.assert_allclose(evidence.var(ddof=1), variance, rtol=.04)
