"""Validated TOML configuration for one trusted local worker stack."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 dependency
    import tomli as tomllib


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerConfig:
    """One process/device specialization owned by the supervisor."""

    name: str
    platform: str
    device: str
    batch_size: int


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved lifecycle, transport, and worker configuration."""

    source: Path
    fingerprint: str
    stack_id: str
    runtime_dir: Path
    log_dir: Path
    endpoint: str
    manifest: Path
    lock: Path
    startup_timeout_s: float
    shutdown_timeout_s: float
    task_timeout_s: float
    program_cache_size: int
    workers: tuple[WorkerConfig, ...]


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Read and validate one complete local runtime configuration."""
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read runtime configuration {source}: {exc}") from exc
    document = tomllib.loads(raw.decode("utf-8"))
    _reject_unknown(document, {"runtime", "workers"}, "configuration")
    runtime = document.get("runtime")
    if type(runtime) is not dict:
        raise ValueError("Configuration requires a [runtime] table.")
    _reject_unknown(
        runtime,
        {
            "stack_id",
            "runtime_dir",
            "log_dir",
            "startup_timeout_s",
            "shutdown_timeout_s",
            "task_timeout_s",
            "program_cache_size",
        },
        "runtime",
    )

    stack_id = runtime.get("stack_id")
    if type(stack_id) is not str or not stack_id:
        raise ValueError("runtime.stack_id must be a non-empty string.")
    if not _valid_identifier(stack_id):
        raise ValueError(
            "runtime.stack_id may contain only letters, digits, '.', '_', "
            "or '-'."
        )

    runtime_root = _configured_path(
        source.parent,
        runtime.get("runtime_dir"),
        Path(tempfile.gettempdir()) / f"jaxns-{os.getuid()}",
        "runtime.runtime_dir",
    )
    runtime_dir = runtime_root / stack_id
    log_root = _configured_path(
        source.parent,
        runtime.get("log_dir"),
        source.parent / "logs",
        "runtime.log_dir",
    )
    log_dir = log_root / stack_id
    startup_timeout_s = _positive_float(
        runtime.get("startup_timeout_s", 120.0),
        "runtime.startup_timeout_s",
    )
    shutdown_timeout_s = _positive_float(
        runtime.get("shutdown_timeout_s", 20.0),
        "runtime.shutdown_timeout_s",
    )
    task_timeout_s = _positive_float(
        runtime.get("task_timeout_s", 300.0),
        "runtime.task_timeout_s",
    )
    program_cache_size = runtime.get("program_cache_size", 4)
    if type(program_cache_size) is not int or program_cache_size < 1:
        raise ValueError("runtime.program_cache_size must be positive.")

    raw_workers = document.get("workers")
    if type(raw_workers) is not list or not raw_workers:
        raise ValueError("Configuration requires at least one [[workers]] table.")
    workers: list[WorkerConfig] = []
    names: set[str] = set()
    for worker_idx, raw_worker in enumerate(raw_workers):
        if type(raw_worker) is not dict:
            raise ValueError(f"workers[{worker_idx}] must be a table.")
        _reject_unknown(
            raw_worker,
            {"name", "platform", "device", "batch_size", "count"},
            f"workers[{worker_idx}]",
        )
        base_name = raw_worker.get("name")
        platform = raw_worker.get("platform")
        device = raw_worker.get("device", "0")
        batch_size = raw_worker.get("batch_size", 1)
        count = raw_worker.get("count", 1)
        if type(base_name) is not str or not base_name:
            raise ValueError(f"workers[{worker_idx}].name must be a string.")
        if not _valid_identifier(base_name):
            raise ValueError(
                f"workers[{worker_idx}].name may contain only letters, "
                "digits, '.', '_', or '-'."
            )
        if type(platform) is not str or platform not in ("cpu", "gpu", "tpu"):
            raise ValueError(
                f"workers[{worker_idx}].platform must be cpu, gpu, or tpu."
            )
        if type(device) not in (str, int):
            raise ValueError(f"workers[{worker_idx}].device must be a string or integer.")
        if platform in ("cpu", "tpu"):
            try:
                device_index = int(device)
            except ValueError as exc:
                raise ValueError(
                    f"workers[{worker_idx}].device must be a non-negative "
                    f"integer for {platform}."
                ) from exc
            if device_index < 0 or str(device_index) != str(device):
                raise ValueError(
                    f"workers[{worker_idx}].device must be a non-negative "
                    f"integer for {platform}."
                )
        elif not str(device):
            raise ValueError(f"workers[{worker_idx}].device cannot be empty.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError(f"workers[{worker_idx}].batch_size must be positive.")
        if type(count) is not int or count < 1:
            raise ValueError(f"workers[{worker_idx}].count must be positive.")
        for instance in range(count):
            name = base_name if count == 1 else f"{base_name}-{instance}"
            if name in names:
                raise ValueError(f"Worker name {name!r} is duplicated.")
            names.add(name)
            workers.append(WorkerConfig(
                name=name,
                platform=platform,
                device=str(device),
                batch_size=batch_size,
            ))

    # Fingerprint resolved behavior rather than TOML formatting. Relative
    # paths are already resolved above, so moving a config that changes its
    # effective directories is correctly treated as configuration drift.
    fingerprint_document = {
        "stack_id": stack_id,
        "runtime_dir": str(runtime_dir),
        "log_dir": str(log_dir),
        "startup_timeout_s": startup_timeout_s,
        "shutdown_timeout_s": shutdown_timeout_s,
        "task_timeout_s": task_timeout_s,
        "program_cache_size": program_cache_size,
        "workers": [dataclasses.asdict(worker) for worker in workers],
    }
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    ownership = Path(tempfile.gettempdir()) / f"jaxns-{os.getuid()}" / "ownership"
    return RuntimeConfig(
        source=source,
        fingerprint=fingerprint.hexdigest(),
        stack_id=stack_id,
        runtime_dir=runtime_dir,
        log_dir=log_dir,
        endpoint=f"ipc://{runtime_dir / 'pool.ipc'}",
        manifest=ownership / f"{stack_id}.json",
        lock=ownership / f"{stack_id}.lock",
        startup_timeout_s=startup_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
        task_timeout_s=task_timeout_s,
        program_cache_size=program_cache_size,
        workers=tuple(workers),
    )


def _configured_path(
        parent: Path,
        value: object,
        default: Path,
        name: str,
) -> Path:
    if value is None:
        return default.resolve()
    if type(value) is not str:
        raise ValueError(f"{name} must be a path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def _positive_float(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _reject_unknown(
        values: dict[str, object],
        allowed: set[str],
        name: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}.")


def _valid_identifier(value: str) -> bool:
    return all(
        character.isalnum() or character in "._-"
        for character in value
    )
