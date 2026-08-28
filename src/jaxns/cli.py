"""Command-line lifecycle control for JAXNS coordinator and worker nodes."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from jaxns.runtime.client import RuntimeUnavailableError
from jaxns.runtime.config import RuntimeConfig, load_runtime_config


def _print(document: dict[str, object]) -> None:
    print(json.dumps(document, indent=2, sort_keys=True))


def _ownership_is_held(config: RuntimeConfig) -> bool:
    config.lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_file = config.lock.open("a+b")
    try:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            return True
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        lock_file.close()


def _connect(config: RuntimeConfig):
    from jaxns.runtime.client import SupervisorClient

    return SupervisorClient.from_config(config.source, use_manifest=True)


def _try_ping(
        config: RuntimeConfig,
        timeout_s: float,
) -> dict[str, object] | None:
    try:
        with _connect(config) as client:
            return client.ping(timeout_s=timeout_s)
    except RuntimeUnavailableError:
        return None


def _validate(config: RuntimeConfig) -> int:
    network = config.network
    _print({
        "config": str(config.source),
        "config_fingerprint": config.fingerprint,
        "endpoint": config.endpoint,
        "node_id": config.node_id,
        "role": config.role,
        "worker_endpoint": config.worker_endpoint,
        "network": {
            "coordinator": network.coordinator,
            "port": network.port,
        },
        "log_dir": str(config.log_dir),
        "program_cache_size": config.program_cache_size,
        "runtime_dir": str(config.runtime_dir),
        "stack_id": config.stack_id,
        "startup_timeout_s": config.startup_timeout_s,
        "shutdown_timeout_s": config.shutdown_timeout_s,
        "task_timeout_s": config.task_timeout_s,
        "workers": [
            {
                "batch_size": worker.batch_size,
                "device": worker.device,
                "platform": worker.platform,
            }
            for worker in config.workers
        ],
    })
    return 0


def _stop_started_process(
        config: RuntimeConfig,
        process: subprocess.Popen,
) -> None:
    """Bound cleanup after this CLI invocation fails to start its stack."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=config.shutdown_timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _up(config: RuntimeConfig) -> int:
    observed = _try_ping(config, timeout_s=0.25)
    if observed is not None:
        if observed.get("config_fingerprint") != config.fingerprint:
            raise RuntimeError(
                f"Stack {config.stack_id!r} is already owned by a different "
                "configuration. Run `jaxns-cli --config ... down` first."
            )
        with _connect(config) as client:
            status = client.status(timeout_s=1.0)
        status["idempotent"] = True
        _print(status)
        return 0
    if _ownership_is_held(config):
        raise RuntimeError(
            f"Stack {config.stack_id!r} has an owner that is not responding. "
            "Inspect its supervisor log before retrying."
        )

    # One durable owner creates every local device process. On the main node it
    # also coordinates scientific work; remote nodes connect their workers to
    # that coordinator over TCP on the trusted scientific network.
    module = (
        "jaxns.runtime.coordinator"
        if config.role == "coordinator"
        else "jaxns.runtime.node"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            module,
            "--config",
            str(config.source),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + config.startup_timeout_s
    last_status = None
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Supervisor exited during startup with code {exit_code}. "
                f"Inspect {config.log_dir}."
            )
        try:
            with _connect(config) as client:
                last_status = client.status(timeout_s=0.5)
        except RuntimeUnavailableError:
            time.sleep(0.05)
            continue
        workers = last_status.get("workers", [])
        if type(workers) is list:
            exited = [
                worker for worker in workers
                if type(worker) is dict
                and worker.get("exit_code") is not None
            ]
            if exited:
                try:
                    with _connect(config) as client:
                        client.shutdown(timeout_s=1.0)
                except RuntimeError:
                    # Preserve the original worker-startup error; the still
                    # held ownership lock and logs make cleanup failure visible.
                    pass
                _stop_started_process(config, process)
                raise RuntimeError(
                    "A configured worker exited during startup. Inspect "
                    f"{config.log_dir}: {exited}"
                )
            local_workers = [
                worker for worker in workers
                if type(worker) is dict
                and worker.get("node_id", config.node_id) == config.node_id
            ]
            workers_started = len(local_workers) == len(config.workers)
            # A remote node is a durable desired-state owner. Its workers may
            # remain connecting through a long partition and register when
            # the coordinator becomes reachable, so `up` must not kill the
            # node supervisor merely because no lease exists yet.
            workers_ready = config.role == "node" or all(
                worker.get("ready") is True for worker in local_workers
            )
            if workers_started and workers_ready:
                last_status["idempotent"] = False
                _print(last_status)
                return 0
        time.sleep(0.05)
    _stop_started_process(config, process)
    raise RuntimeError(
        f"Stack {config.stack_id!r} did not become ready within "
        f"{config.startup_timeout_s:g}s. Last status: {last_status}. "
        f"Inspect {config.log_dir}."
    )


def _status(config: RuntimeConfig) -> int:
    observed = _try_ping(config, timeout_s=0.5)
    if observed is None:
        state = "unknown" if _ownership_is_held(config) else "stopped"
        _print({"stack_id": config.stack_id, "state": state})
        return 1 if state == "unknown" else 0
    with _connect(config) as client:
        _print(client.status(timeout_s=1.0))
    return 0


def _down(config: RuntimeConfig) -> int:
    observed = _try_ping(config, timeout_s=0.5)
    if observed is None:
        if _ownership_is_held(config):
            raise RuntimeError(
                f"Stack {config.stack_id!r} has an unresponsive owner."
            )
        _print({
            "idempotent": True,
            "stack_id": config.stack_id,
            "state": "stopped",
        })
        return 0
    with _connect(config) as client:
        client.shutdown(timeout_s=1.0)

    deadline = time.monotonic() + config.shutdown_timeout_s
    while time.monotonic() < deadline:
        if not _ownership_is_held(config):
            _print({
                "idempotent": False,
                "stack_id": config.stack_id,
                "state": "stopped",
            })
            return 0
        time.sleep(0.05)
    raise RuntimeError(
        f"Stack {config.stack_id!r} did not stop within "
        f"{config.shutdown_timeout_s:g}s."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaxns-cli")
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config")
    config_commands = config.add_subparsers(
        dest="config_command",
        required=True,
    )
    config_commands.add_parser("validate")
    commands.add_parser("up")
    commands.add_parser("status")
    commands.add_parser("down")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.config is None:
            raise ValueError("--config is required for this command.")
        config = load_runtime_config(args.config)
        if args.command == "config":
            return _validate(config)
        # Lifecycle commands need ZeroMQ, whereas configuration validation is
        # intentionally useful from the default installation alone.
        import zmq
        del zmq
        if args.command == "up":
            return _up(config)
        if args.command == "status":
            return _status(config)
        if args.command == "down":
            return _down(config)
        raise RuntimeError(f"Unsupported command {args.command!r}.")
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"jaxns-cli: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
