"""Fail fast when a prepared benchmark imports the wrong release line."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import jax

import jaxns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("v2", "v3"), required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    version = importlib.metadata.version("jaxns")
    expected_prefix = "2." if args.implementation == "v2" else "3."
    if not version.startswith(expected_prefix):
        raise RuntimeError(
            f"Expected {args.implementation}, imported jaxns {version} from "
            f"{jaxns.__file__}."
        )
    if len(args.source_id) != 40:
        raise ValueError("source-id must be a full commit SHA.")

    record = {
        "implementation": args.implementation,
        "source_id": args.source_id,
        "jaxns_version": version,
        "jaxns_module": os.path.realpath(jaxns.__file__),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
