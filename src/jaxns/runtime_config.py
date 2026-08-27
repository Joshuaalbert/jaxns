"""Validated topology and lifecycle configuration for a JAXNS worker node."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import socket
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 dependency
    import tomli as tomllib


@dataclasses.dataclass(frozen=True, slots=True)
class WorkerConfig:
    """One process/device specialization owned by this node."""

    platform: str
    device: str
    batch_size: int


@dataclasses.dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Coordinator port or reachable remote-coordinator endpoint."""

    port: int | None
    coordinator: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resolved lifecycle, transport, and worker configuration."""

    source: Path
    fingerprint: str
    stack_id: str
    node_id: str
    role: str
    runtime_dir: Path
    log_dir: Path
    endpoint: str
    worker_endpoint: str
    manifest: Path
    lock: Path
    startup_timeout_s: float
    shutdown_timeout_s: float
    task_timeout_s: float
    heartbeat_interval_s: float
    missed_heartbeats: int
    batch_wait_s: float
    program_cache_size: int
    network: NetworkConfig
    workers: tuple[WorkerConfig, ...]


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Read and validate one coordinator or worker-node configuration."""
    source = Path(path).expanduser().resolve()
    try:
        document = tomllib.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read runtime configuration {source}: {exc}") from exc
    _reject_unknown(document, {"runtime", "network", "workers"}, "configuration")
    runtime = document.get("runtime")
    if type(runtime) is not dict:
        raise ValueError("Configuration requires a [runtime] table.")
    _reject_unknown(
        runtime,
        {
            "stack_id",
            "node_id",
            "runtime_dir",
            "log_dir",
            "startup_timeout_s",
            "shutdown_timeout_s",
            "task_timeout_s",
            "heartbeat_interval_s",
            "missed_heartbeats",
            "batch_wait_s",
            "program_cache_size",
        },
        "runtime",
    )

    stack_id = _identifier(runtime.get("stack_id"), "runtime.stack_id")
    network = _network_config(document.get("network"))
    role = "node" if network.coordinator is not None else "coordinator"
    default_node = stack_id if role == "coordinator" else socket.gethostname()
    node_id = _identifier(runtime.get("node_id", default_node), "runtime.node_id")

    runtime_root = _configured_path(
        source.parent,
        runtime.get("runtime_dir"),
        Path(tempfile.gettempdir()) / f"jaxns-{os.getuid()}",
        "runtime.runtime_dir",
    )
    log_root = _configured_path(
        source.parent,
        runtime.get("log_dir"),
        source.parent / "logs",
        "runtime.log_dir",
    )
    # Nodes may share a filesystem. Node-qualified directories and ownership
    # records prevent one machine from claiming another's process.
    runtime_dir = runtime_root / stack_id / node_id
    log_dir = log_root / stack_id / node_id
    endpoint = (
        coordinator_endpoint(network.port)
        if role == "coordinator"
        else f"ipc://{runtime_dir / 'node.ipc'}"
    )
    worker_endpoint = (
        endpoint
        if role == "coordinator"
        else network.coordinator
    )
    if worker_endpoint is None:  # pragma: no cover - guarded by role
        raise ValueError("A worker node requires network.coordinator.")

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
    heartbeat_interval_s = _positive_float(
        runtime.get("heartbeat_interval_s", 2.0),
        "runtime.heartbeat_interval_s",
    )
    missed_heartbeats = runtime.get("missed_heartbeats", 2)
    if type(missed_heartbeats) is not int or missed_heartbeats < 2:
        raise ValueError("runtime.missed_heartbeats must be at least two.")
    batch_wait_s = _non_negative_float(
        runtime.get("batch_wait_s", 0.002),
        "runtime.batch_wait_s",
    )
    program_cache_size = runtime.get("program_cache_size", 4)
    if type(program_cache_size) is not int or program_cache_size < 1:
        raise ValueError("runtime.program_cache_size must be positive.")
    workers = _worker_configs(
        document.get("workers"),
        allow_empty=role == "coordinator",
    )

    behavior = {
        "stack_id": stack_id,
        "node_id": node_id,
        "role": role,
        "runtime_dir": str(runtime_dir),
        "log_dir": str(log_dir),
        "startup_timeout_s": startup_timeout_s,
        "shutdown_timeout_s": shutdown_timeout_s,
        "task_timeout_s": task_timeout_s,
        "heartbeat_interval_s": heartbeat_interval_s,
        "missed_heartbeats": missed_heartbeats,
        "batch_wait_s": batch_wait_s,
        "program_cache_size": program_cache_size,
        "network": dataclasses.asdict(network),
        "workers": [dataclasses.asdict(worker) for worker in workers],
    }
    fingerprint = hashlib.sha256(json.dumps(
        behavior,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    ownership = Path(tempfile.gettempdir()) / f"jaxns-{os.getuid()}" / "ownership"
    owner = f"{stack_id}-{node_id}"
    return RuntimeConfig(
        source=source,
        fingerprint=fingerprint,
        stack_id=stack_id,
        node_id=node_id,
        role=role,
        runtime_dir=runtime_dir,
        log_dir=log_dir,
        endpoint=endpoint,
        worker_endpoint=worker_endpoint,
        manifest=ownership / f"{owner}.json",
        lock=ownership / f"{owner}.lock",
        startup_timeout_s=startup_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
        task_timeout_s=task_timeout_s,
        heartbeat_interval_s=heartbeat_interval_s,
        missed_heartbeats=missed_heartbeats,
        batch_wait_s=batch_wait_s,
        program_cache_size=program_cache_size,
        network=network,
        workers=workers,
    )


def _network_config(value: object) -> NetworkConfig:
    if type(value) is not dict:
        raise ValueError("Configuration requires a [network] table.")
    _reject_unknown(value, {"port", "coordinator"}, "network")
    port = value.get("port")
    if port is not None and (
        type(port) is not int or not 1 <= port <= 65_535
    ):
        raise ValueError("network.port must be an integer from 1 to 65,535.")
    coordinator = _optional_tcp(
        value.get("coordinator"),
        "network.coordinator",
        wildcard=False,
    )
    if (port is None) == (coordinator is None):
        raise ValueError(
            "network must define exactly one of port or coordinator."
        )
    return NetworkConfig(
        port=port,
        coordinator=coordinator,
    )


def coordinator_endpoint(port: int | None) -> str:
    """Derive the same-user scientific IPC endpoint from the TCP port."""
    if port is None:
        raise ValueError("A coordinator configuration requires network.port.")
    path = (
        Path(tempfile.gettempdir())
        / f"jaxns-{os.getuid()}"
        / "coordinators"
        / f"{port}.ipc"
    )
    return f"ipc://{path}"


def _worker_configs(
        value: object,
        *,
        allow_empty: bool,
) -> tuple[WorkerConfig, ...]:
    if value is None and allow_empty:
        return ()
    if type(value) is not list or not value:
        raise ValueError("Configuration requires at least one [[workers]] table.")
    workers: list[WorkerConfig] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if type(item) is not dict:
            raise ValueError(f"workers[{index}] must be a table.")
        _reject_unknown(
            item,
            {"platform", "device", "batch_size"},
            f"workers[{index}]",
        )
        platform = item.get("platform")
        device = item.get("device", "0")
        batch_size = item.get("batch_size", 1)
        if type(platform) is not str or platform not in ("cpu", "gpu", "tpu"):
            raise ValueError(f"workers[{index}].platform must be cpu, gpu, or tpu.")
        if type(device) not in (str, int):
            raise ValueError(f"workers[{index}].device must be a string or integer.")
        if platform in ("cpu", "tpu"):
            try:
                device_index = int(device)
            except ValueError as exc:
                raise ValueError(
                    f"workers[{index}].device must be a non-negative integer for {platform}."
                ) from exc
            if device_index < 0 or str(device_index) != str(device):
                raise ValueError(
                    f"workers[{index}].device must be a non-negative integer for {platform}."
                )
        elif not str(device):
            raise ValueError(f"workers[{index}].device cannot be empty.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError(f"workers[{index}].batch_size must be positive.")
        worker = WorkerConfig(
            platform=platform,
            device=str(device),
            batch_size=batch_size,
        )
        name = worker_name(worker)
        if name in names:
            raise ValueError(f"Worker {name!r} is duplicated.")
        names.add(name)
        workers.append(worker)
    return tuple(workers)


def worker_name(worker: WorkerConfig) -> str:
    """Return the fixed public name for one configured device process."""
    return f"{worker.platform}-{worker.device}"


def _configured_path(parent: Path, value: object, default: Path, name: str) -> Path:
    if value is None:
        return default.resolve()
    if type(value) is not str:
        raise ValueError(f"{name} must be a path string.")
    path = Path(value).expanduser()
    return (parent / path if not path.is_absolute() else path).resolve()


def _optional_tcp(value: object, name: str, *, wildcard: bool) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.startswith("tcp://"):
        raise ValueError(f"{name} must be a tcp:// endpoint.")
    if not wildcard and ("*" in value or "0.0.0.0" in value):
        raise ValueError(f"{name} must name a reachable host, not a wildcard.")
    return value


def _positive_float(value: object, name: str) -> float:
    resolved = _number(value, name)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _non_negative_float(value: object, name: str) -> float:
    resolved = _number(value, name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or not _valid_identifier(value):
        raise ValueError(
            f"{name} may contain only letters, digits, '.', '_', or '-'."
        )
    return value


def _reject_unknown(values: dict[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(unknown)}.")


def _valid_identifier(value: str) -> bool:
    return all(character.isalnum() or character in "._-" for character in value)
