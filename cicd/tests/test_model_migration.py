"""The migration helper preserves the model and rejects guesses."""

import runpy
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from docs.examples.migrate_v2_model import convert_prior_model

V2_SOURCE = """\
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns.priors import Prior

tfpd = tfp.distributions


def prior_model_v2():
    # Keep the location prior and its units: μ.
    x = yield Prior(tfpd.Normal(0.0, 1.0), name="x")
    y = yield Prior(tfpd.HalfNormal(jnp.abs(x) + 1.0), name="y")
    return (
        x,  # The likelihood consumes location first.
        y,
    )  # Retain both model outputs.


def log_likelihood_v2(x, y):
    return tfpd.Normal(x, y).log_prob(0.5)
"""


def test_migration_cli_preserves_conditional_model_and_comments(
    tmp_path,
) -> None:
    source_path = tmp_path / "old_model.py"
    source_path.write_text(V2_SOURCE, encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[2]
        / "docs/examples/migrate_v2_model.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(source_path),
            "--prior",
            "prior_model_v2",
            "--likelihood",
            "log_likelihood_v2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert source_path.read_text(encoding="utf-8") == V2_SOURCE
    assert "# Keep the location prior and its units: μ." in completed.stdout
    assert "# The likelihood consumes location first." in completed.stdout
    assert "# Retain both model outputs." in completed.stdout
    ported_path = tmp_path / "ported_model.py"
    ported_path.write_text(
        V2_SOURCE + "\n" + completed.stdout, encoding="utf-8"
    )
    namespace = runpy.run_path(str(ported_path))
    model = namespace["model_v3"]
    u = model.sample_U(jax.random.PRNGKey(0))
    physical = model.transform_to_X(u)
    expected_x = namespace["tfpd"].Normal(0.0, 1.0).quantile(u["x"])
    expected_y = (
        namespace["tfpd"]
        .HalfNormal(jnp.abs(expected_x) + 1.0)
        .quantile(u["y"])
    )
    assert jnp.allclose(physical["x"], expected_x)
    assert jnp.allclose(physical["y"], expected_y)
    assert jnp.allclose(
        model.log_likelihood(u),
        namespace["log_likelihood_v2"](expected_x, expected_y),
    )
    model.sanity_check(jax.random.PRNGKey(1), num_samples=4)


@pytest.mark.parametrize(
    "body",
    [
        "    x = yield from other_prior()\n    return (x,)\n",
        "    x = yield\n    return (x,)\n",
        (
            "    def nested():\n        return 1\n"
            "    x = yield Prior(...)\n    return (x,)\n"
        ),
        "    x = yield Prior(...)\n    return x\n",
        "    x = yield Prior(...)\n    return ((yield Prior(...)),)\n",
    ],
)
def test_migration_rejects_unsupported_generator_patterns(body) -> None:
    with pytest.raises(ValueError):
        convert_prior_model(
            "def prior_model():\n" + body, "prior_model", "log_likelihood"
        )
