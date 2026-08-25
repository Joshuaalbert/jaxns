"""Check whether the released ParameterPack preserves Bayesian U sampling."""

import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import tensorflow_probability.substrates.jax as tfp
from jaxctx.priors import ParameterPack, Prior

from jaxns.model import Model

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")
tfpd = tfp.distributions
JAXNS_SOURCE = Path(inspect.getsourcefile(Model)).resolve()
if not JAXNS_SOURCE.is_relative_to(REPO_ROOT):
    raise RuntimeError(
        f"Expected JAXNS from {REPO_ROOT}, imported {JAXNS_SOURCE}."
    )


def _make_priors():
    return (
        Prior(tfpd.Uniform(0.0, 1.0), name="x"),
        Prior(tfpd.Uniform(0.0, 1.0), name="y"),
    )


def _make_current_model():
    priors = _make_priors()

    def prior_model():
        return sum(prior.realise() for prior in priors)

    return Model(prior_model)


def _make_parameter_pack_model(*, random_init):
    pack = ParameterPack(_make_priors(), name="packed")

    def prior_model():
        return sum(pack.parameter(random_init=random_init))

    return Model(prior_model)


def _sample_summary(model, keys):
    samples = jax.vmap(model.sample_U)(keys)
    flattened = np.concatenate(
        [np.asarray(leaf).reshape(-1) for leaf in jax.tree.leaves(samples)]
    )
    return {
        "leaf_count": len(jax.tree.leaves(samples)),
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
        "uniform_ks_statistic": float(
            scipy.stats.kstest(flattened, "uniform").statistic
        ),
    }


def main():
    keys = jax.random.split(jax.random.PRNGKey(0), 10_000)
    current = _make_current_model()
    fixed_pack = _make_parameter_pack_model(random_init=False)
    random_pack = _make_parameter_pack_model(random_init=True)

    supplied_u = fixed_pack.sample_U(jax.random.PRNGKey(1))
    supplied_u = jax.tree.map(
        lambda value: jnp.linspace(0.1, 0.9, value.size).reshape(value.shape),
        supplied_u,
    )
    transformed = fixed_pack.transform_to_X(supplied_u)
    transformed_values = np.concatenate(
        [np.asarray(leaf).reshape(-1) for leaf in jax.tree.leaves(transformed)]
    )

    output = {
        "jax": jax.__version__,
        "jaxctx": importlib.metadata.version("jaxctx"),
        "jaxns_source": str(JAXNS_SOURCE),
        "sample_count": len(keys),
        "uniform_target_std": float(1.0 / np.sqrt(12.0)),
        "parameter_pack_has_realise": hasattr(ParameterPack, "realise"),
        "current_prior_realise": _sample_summary(current, keys),
        "parameter_pack_default": _sample_summary(fixed_pack, keys),
        "parameter_pack_random_init": _sample_summary(random_pack, keys),
        "supplied_packed_u": [0.1, 0.9],
        "physical_values_after_supplied_u": transformed_values.tolist(),
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
