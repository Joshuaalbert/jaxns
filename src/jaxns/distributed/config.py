# config.py
import functools
import os
from pathlib import Path

try:  # Python ≥3.11
    import tomllib  # std-lib TOML reader
except ModuleNotFoundError:  # Python ≤3.10 → pip install tomli
    import tomli as tomllib

ENV_VAR = "JAXNS_CONFIG"  # change if you prefer


class ConfigError(RuntimeError):
    """Raised when the config file or a key is missing."""


@functools.lru_cache(maxsize=1)  # lazy, cached after first call
def _load() -> dict:
    path = os.getenv(ENV_VAR, 'config.toml')
    if not path:
        raise ConfigError(f"{ENV_VAR} is not set")
    file = Path(path).expanduser()
    if not file.is_file():
        raise ConfigError(f"Config file {file} not found")
    with file.open("rb") as fh:  # tomllib needs *binary* mode
        return tomllib.load(fh)


def get(*keys: str, default=...):
    """
    Retrieve a value via a chain of keys, e.g.:

        get("database", "url")

    If 'default' is given it is returned instead of raising when the
    path is missing.
    """
    node = _load()
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            if default is ...:
                joined = ".".join(keys)
                raise ConfigError(f"Key '{joined}' not found")
            return default
    return node


cfg = get  # alias for convenience


def reload():  # call this after you edit the TOML file at runtime
    _load.cache_clear()
