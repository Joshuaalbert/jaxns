import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from jaxctx.priors.prior import Prior
from tensorflow_probability.substrates import jax as tfp

from jaxns.constrained_sampler import UniDimSliceSampler, _new_proposal
from jaxns.model import Model
from jaxns.pytree import TreeField
from jaxns.samples import SeedPoint

tfpd = tfp.distributions


def _log_likelihood_1d(U):
    x = U[0]
    return -(x - 0.5) ** 2


def _build_monotonic_model() -> Model:
    def prior_model():
        x = Prior(tfpd.Uniform(0.0, 1.0), name="x").realise()
        return x

    return Model(prior_model=prior_model)


def _stationary_seed_batch(
        model: Model,
        key,
        *,
        num_samples: int,
        lower: float,
) -> SeedPoint:
    template = model.sample_U(random.PRNGKey(0))
    values = random.uniform(
        key,
        shape=(num_samples,),
        minval=lower,
        maxval=1.0,
    )
    U0 = jax.tree.map(
        lambda leaf: values.astype(leaf.dtype),
        template,
    )
    return SeedPoint(U0=U0, log_L0=values)


def test_new_proposal_nonperfect_first_uses_full_slice_width():
    U0 = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    point_U, log_L, num_evals, _, next_slice_width = _new_proposal(
        key=random.PRNGKey(0),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    np.testing.assert_allclose(next_slice_width, 2.0)
    assert 0.0 <= float(point_U.tree[0]) <= 1.0
    assert float(log_L) > -1.0
    assert int(num_evals) >= 1


def test_new_proposal_nonperfect_finite_width_clips_and_steps_out():
    U0 = TreeField(jnp.asarray([0.95]))
    direction = TreeField(jnp.asarray([1.0]))

    _, _, num_evals, _, next_slice_width = _new_proposal(
        key=random.PRNGKey(1),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(0.05),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    assert int(num_evals) > 1
    assert float(next_slice_width) > 0.0
    np.testing.assert_allclose(next_slice_width, 2.0)


def test_new_proposal_nonperfect_reuses_previous_width():
    U0 = TreeField(jnp.asarray([0.5]))
    direction = TreeField(jnp.asarray([1.0]))

    point_U, _, _, direction_1, slice_width_1 = _new_proposal(
        key=random.PRNGKey(2),
        U0=U0,
        direction=direction,
        slice_width=jnp.asarray(jnp.inf),
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    _, _, num_evals_2, _, slice_width_2 = _new_proposal(
        key=random.PRNGKey(3),
        U0=point_U,
        direction=direction_1,
        slice_width=slice_width_1,
        no_step_out=False,
        gradient_guided=False,
        log_L_constraint=jnp.asarray(-1.0),
        log_likelihood_fn=_log_likelihood_1d,
    )

    assert isinstance(direction_1, TreeField)
    assert not isinstance(direction_1.tree, TreeField)
    np.testing.assert_allclose(direction_1.tree, jnp.asarray([1.0]))
    assert jnp.isfinite(slice_width_1)
    np.testing.assert_allclose(slice_width_1, 2.0)
    assert int(num_evals_2) > 1
    assert float(slice_width_2) > 0.0


def test_slice_sampler_preserves_stationary_constrained_uniform_distribution():
    """A stationary seed ensemble must remain uniform above the contour."""
    model = _build_monotonic_model()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=3,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=0,
    )
    num_samples = 4096
    lower = 0.2
    seed_points = _stationary_seed_batch(
        model,
        random.PRNGKey(11),
        num_samples=num_samples,
        lower=lower,
    )
    keys = random.split(random.PRNGKey(12), num_samples)

    U_samples, log_L, _, phantom_samples = jax.jit(jax.vmap(
        sampler.get_sample,
        in_axes=(0, None, 0),
    ))(
        keys,
        jnp.asarray(lower),
        seed_points,
    )
    samples = np.asarray(jax.tree.leaves(U_samples)[0])

    assert np.all(np.asarray(log_L) > lower)
    assert np.all((samples > lower) & (samples < 1.0))

    # The Dvoretzky-Kiefer-Wolfowitz bound gives a distribution-wide,
    # predeclared tolerance rather than checking only a convenient moment.
    uniform_samples = np.sort((samples - lower) / (1.0 - lower))
    empirical_upper = np.arange(1, num_samples + 1) / num_samples
    empirical_lower = np.arange(num_samples) / num_samples
    cdf_error = max(
        np.max(empirical_upper - uniform_samples),
        np.max(uniform_samples - empirical_lower),
    )
    alpha = 1e-6
    dkw_bound = np.sqrt(np.log(2.0 / alpha) / (2.0 * num_samples))
    assert cdf_error < dkw_bound

    # Every retained chain state is another application of the same
    # stationary kernel under the same contour. Check each transition rather
    # than treating the correlated states as one artificially large sample.
    retained = np.asarray(jax.tree.leaves(phantom_samples.U_samples)[0])
    assert retained.shape == (num_samples, sampler.num_phantom())
    assert np.all(np.asarray(phantom_samples.log_L) > lower)
    for phantom_idx in range(sampler.num_phantom()):
        uniform_phantom = np.sort(
            (retained[:, phantom_idx] - lower) / (1.0 - lower)
        )
        phantom_cdf_error = max(
            np.max(empirical_upper - uniform_phantom),
            np.max(uniform_phantom - empirical_lower),
        )
        assert phantom_cdf_error < dkw_bound


def test_slice_sampler_outputs_are_invariant_to_vmap_batch_partitioning():
    """Static batch width must not change a lane's random transition."""
    model = _build_monotonic_model()
    sampler = UniDimSliceSampler(
        model=model,
        num_slices=4,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    num_samples = 12
    lower = 0.25
    seed_points = _stationary_seed_batch(
        model,
        random.PRNGKey(21),
        num_samples=num_samples,
        lower=lower,
    )
    keys = random.split(random.PRNGKey(22), num_samples)
    batched_get_sample = jax.jit(jax.vmap(
        sampler.get_sample,
        in_axes=(0, None, 0),
    ))

    full_batch = batched_get_sample(
        keys,
        jnp.asarray(lower),
        seed_points,
    )
    chunked_batches = []
    for start, stop in ((0, 5), (5, 9), (9, 12)):
        chunked_batches.append(batched_get_sample(
            keys[start:stop],
            jnp.asarray(lower),
            jax.tree.map(
                lambda leaf, start=start, stop=stop: leaf[start:stop],
                seed_points,
            ),
        ))
    partitioned_batch = jax.tree.map(
        lambda *leaves: jnp.concatenate(leaves, axis=0),
        *chunked_batches,
    )

    for full_leaf, partitioned_leaf in zip(
            jax.tree.leaves(full_batch),
            jax.tree.leaves(partitioned_batch),
            strict=True,
    ):
        np.testing.assert_array_equal(full_leaf, partitioned_leaf)


def test_enabling_phantom_retention_does_not_change_classic_transition():
    """Phantoms are a retained prefix, not extra sampler transitions."""
    model = _build_monotonic_model()
    classic_sampler = UniDimSliceSampler(
        model=model,
        num_slices=5,
        no_step_out=True,
        collect_phantom_samples=False,
    )
    phantom_sampler = UniDimSliceSampler(
        model=model,
        num_slices=5,
        no_step_out=True,
        collect_phantom_samples=True,
        phantom_burn_in=1,
    )
    seed = _stationary_seed_batch(
        model,
        random.PRNGKey(31),
        num_samples=1,
        lower=0.3,
    )
    seed = jax.tree.map(lambda leaf: leaf[0], seed)
    key = random.PRNGKey(32)

    classic = classic_sampler.get_sample(key, jnp.asarray(0.3), seed)
    with_phantoms = phantom_sampler.get_sample(key, jnp.asarray(0.3), seed)

    for classic_leaf, phantom_leaf in zip(
            jax.tree.leaves(classic[:3]),
            jax.tree.leaves(with_phantoms[:3]),
            strict=True,
    ):
        np.testing.assert_array_equal(classic_leaf, phantom_leaf)
    assert with_phantoms[3].log_L.shape == (3,)
