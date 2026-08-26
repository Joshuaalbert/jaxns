import subprocess
import sys

from jaxctx.priors import special_priors as jaxctx_special_priors
from jaxctx.priors.prior import Prior as JaxctxPrior

import jaxns


def test_root_prior_is_exact_jaxctx_class() -> None:
    """The convenience surface adds no wrapper or behavioral fork."""
    assert jaxns.Prior is JaxctxPrior
    assert jaxns.Prior is jaxns.Prior
    assert "Prior" in jaxns.__all__
    assert "Prior" in dir(jaxns)


def test_special_prior_facade_is_exact_and_complete() -> None:
    """The deliberate special-prior surface mirrors JAXCTX by identity."""
    assert set(jaxns.special_priors.__all__) == set(
        jaxctx_special_priors.__all__
    )
    for name in jaxns.special_priors.__all__:
        assert getattr(jaxns.special_priors, name) is getattr(
            jaxctx_special_priors,
            name,
        )


def test_root_import_keeps_cli_dependency_boundary() -> None:
    """Merely importing JAXNS must not load scientific or distributed stacks."""
    script = """
import sys
import jaxns

assert "Prior" in dir(jaxns)
for module in ("jax", "tensorflow_probability", "zmq", "cloudpickle"):
    assert module not in sys.modules, module
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
