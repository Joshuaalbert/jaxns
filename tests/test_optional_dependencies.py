import builtins
from pathlib import Path

import pytest
import tomllib

from jaxns.optional import import_matplotlib


def test_missing_plotting_dependency_has_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def block_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_matplotlib)

    with pytest.raises(ImportError, match=r"jaxns\[plotting\]"):
        import_matplotlib()


def test_dependency_metadata_matches_feature_boundaries():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    base = set(project["dependencies"])
    extras = project["optional-dependencies"]
    assert "jax>=0.6.0" in base
    assert "jaxctx>=1.1.5" in base
    assert "numpy" in base
    assert "scipy" in base
    assert "tfp-nightly" in base
    assert "jaxlib" not in base
    assert "matplotlib" not in base
    assert "zmq" not in base
    assert extras["plotting"] == ["matplotlib"]
    assert "matplotlib" in extras["examples"]
    assert "matplotlib" in extras["tests"]
    assert "scripts" not in project
