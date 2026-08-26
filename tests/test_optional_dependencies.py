import builtins
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test environment.
    import tomli as tomllib

from jaxns.optional import import_matplotlib


def test_missing_plotting_dependency_has_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def block_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_matplotlib)

    with pytest.raises(ImportError, match=r"pip install jaxns"):
        import_matplotlib()


def test_dependency_metadata_matches_feature_boundaries():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    base = set(project["dependencies"])
    extras = project["optional-dependencies"]
    assert "jax>=0.6.0" in base
    assert "jaxctx>=1.1.5" in base
    assert "matplotlib" in base
    assert "numpy" in base
    assert "scipy" in base
    assert "tfp-nightly" in base
    assert "jaxlib" not in base
    assert "zmq" not in base
    assert "plotting" not in extras
    assert "matplotlib" not in extras["examples"]
    assert "matplotlib" not in extras["tests"]
    assert "tomli; python_version < '3.11'" in extras["tests"]
    assert "scripts" not in project
