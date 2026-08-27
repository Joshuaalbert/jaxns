"""One-process, one-device constrained-sampling worker entry point."""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import os
import pickle
import resource
import signal
import sys
import threading
import time
import traceback
from uuid import uuid4


def _run(
        config_path: str,
        endpoint: str,
        platform: str,
        configured_device: str,
        batch_size: int,
        device_index: int,
        program_cache_size: int,
        instance_id: str,
) -> None:
    # Device visibility is set by the owning node process before this module
    # imports JAX. The coordinator therefore never creates a JAX backend.
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import zmq

    from jaxns.constrained_sampler import (
        ConstrainedSampleRequest,
        LikelihoodRequest,
        evaluate_request,
        sample_request,
    )
    from jaxns.runtime_config import load_runtime_config
    from jaxns.runtime_protocol import (
        ERROR,
        EVALUATE,
        LEASED,
        READY,
        REGISTER,
        REGISTERED,
        RELEASE,
        RESULT,
        SAMPLE,
        STOP,
        STOPPED,
        TASK,
        decode_header,
        encode_header,
    )
    from jaxns.samples import SeedPoint

    config = load_runtime_config(config_path)
    name = f"{platform}-{configured_device}"
    worker_id = f"{config.node_id}/{name}"

    def make_sample_program(registered):
        def sample_program(request):
            return sample_request(
                registered.sampler,
                request,
                args=registered.args,
                params=registered.params,
            )

        return sample_program

    def make_likelihood_program(registered):
        def likelihood_program(request):
            return evaluate_request(
                registered.model,
                request,
                args=registered.args,
                params=registered.params,
            )

        return likelihood_program

    class Program:
        __slots__ = ("compiled", "programs")

        def __init__(self, registered):
            self.programs = {
                SAMPLE: jax.jit(make_sample_program(registered)),
                EVALUATE: jax.jit(make_likelihood_program(registered)),
            }
            self.compiled: dict[str, dict[int, object]] = {
                SAMPLE: {},
                EVALUATE: {},
            }

    devices = jax.devices()
    if not 0 <= device_index < len(devices):
        raise ValueError(
            f"Worker {worker_id!r} requested device index {device_index}, but "
            f"only {len(devices)} {jax.default_backend()} device(s) are visible."
        )
    device = devices[device_index]
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    identity = f"worker:{worker_id}:{instance_id}"
    socket.setsockopt(zmq.IDENTITY, identity.encode())
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    socket.send_multipart([
        encode_header(
            READY,
            role="worker",
            worker_id=worker_id,
            node_id=config.node_id,
            configured_platform=platform,
            configured_device=configured_device,
            instance_id=instance_id,
            batch_size=batch_size,
            platform=jax.default_backend(),
            device=str(device),
            process_id=os.getpid(),
            capabilities={
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "jaxns": importlib.metadata.version("jaxns"),
                "jax": jax.__version__,
                "jaxlib": jax.lib.__version__,
                "x64": bool(jax.config.x64_enabled),
                "measure_dtype": str(jnp.dtype(jnp.float64)),
            },
        ),
        b"",
    ])
    frames = socket.recv_multipart()
    if len(frames) != 2:
        raise RuntimeError("Worker lease response had an invalid frame count.")
    lease_header = decode_header(frames[0])
    if lease_header["command"] != LEASED:
        raise RuntimeError(f"Worker expected a lease, received {lease_header}.")
    lease_id = _string_field(lease_header, "lease_id")
    lease_generation = _integer_field(lease_header, "lease_generation")
    heartbeat_interval_s = float(lease_header["heartbeat_interval_s"])
    missed_heartbeats = _integer_field(lease_header, "missed_heartbeats")

    alive = threading.Event()
    alive.set()
    stopping = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(
            context,
            endpoint,
            worker_id,
            instance_id,
            lease_id,
            lease_generation,
            heartbeat_interval_s,
            missed_heartbeats,
            alive,
            stopping,
        ),
        daemon=True,
        name=f"jaxns-heartbeat-{name}",
    )
    heartbeat.start()

    programs: dict[str, Program] = {}
    cache: collections.OrderedDict[str, Program] = collections.OrderedDict()
    try:
        while alive.is_set():
            if not socket.poll(timeout=100):
                continue
            frames = socket.recv_multipart()
            if len(frames) < 2:
                raise RuntimeError("Worker received an invalid frame count.")
            header = decode_header(frames[0])
            payloads = frames[1:]
            command = header["command"]
            if command == STOP:
                socket.send_multipart([
                    encode_header(
                        STOPPED,
                        worker_id=worker_id,
                        lease_id=lease_id,
                        lease_generation=lease_generation,
                    ),
                    b"",
                ])
                return
            if command == REGISTER:
                session_id = _string_field(header, "session_id")
                fingerprint = _string_field(header, "fingerprint")
                try:
                    record = cache.get(fingerprint)
                    if record is None:
                        session = cloudpickle.loads(payloads[0])
                        record = Program(session)
                        cache[fingerprint] = record
                        while len(cache) > program_cache_size:
                            cache.popitem(last=False)
                    else:
                        cache.move_to_end(fingerprint)
                    programs[session_id] = record
                    socket.send_multipart([
                        encode_header(
                            REGISTERED,
                            session_id=session_id,
                            worker_id=worker_id,
                            lease_id=lease_id,
                            lease_generation=lease_generation,
                        ),
                        b"",
                    ])
                except Exception as exc:  # noqa: BLE001
                    socket.send_multipart([
                        encode_header(
                            ERROR,
                            session_id=session_id,
                            worker_id=worker_id,
                            lease_id=lease_id,
                            lease_generation=lease_generation,
                            error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc(),
                        ),
                        b"",
                    ])
                continue
            if command == RELEASE:
                programs.pop(_string_field(header, "session_id"), None)
                continue
            if command != TASK:
                raise RuntimeError(f"Worker received unsupported {command!r}.")

            session_id = _string_field(header, "session_id")
            assignment_id = _string_field(header, "assignment_id")
            operation = _string_field(header, "operation")
            task_ids = _integer_list(header, "task_ids")
            compile_s = 0.0
            started = time.perf_counter()
            try:
                if session_id not in programs:
                    raise RuntimeError(f"Session {session_id!r} is not registered.")
                if not 0 < len(task_ids) <= batch_size:
                    raise ValueError(
                        f"Worker capacity is {batch_size}, received {len(task_ids)} tasks."
                    )
                if len(payloads) != len(task_ids):
                    raise ValueError("Worker task IDs and payloads have different lengths.")
                requests = [pickle.loads(payload) for payload in payloads]
                if operation == SAMPLE:
                    request = _combine_requests(
                        requests,
                        ConstrainedSampleRequest,
                        SeedPoint,
                        jnp,
                        jax,
                    )
                elif operation == EVALUATE:
                    request = _combine_likelihood_requests(
                        requests,
                        LikelihoodRequest,
                        jnp,
                        jax,
                    )
                else:
                    raise ValueError(
                        f"Worker received unsupported operation {operation!r}."
                    )
                record = programs[session_id]
                width = len(task_ids)
                with jax.default_device(device):
                    compiled = record.compiled[operation].get(width)
                    if compiled is None:
                        compile_started = time.perf_counter()
                        compiled = record.programs[operation].lower(
                            request
                        ).compile()
                        compile_s = time.perf_counter() - compile_started
                        record.compiled[operation][width] = compiled
                    batch = jax.device_get(compiled(request))
                if not alive.is_set():
                    return
                results = [
                    pickle.dumps(
                        jax.tree.map(
                            lambda value, index=index: value[index:index + 1],
                            batch,
                        ),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    for index in range(width)
                ]
                socket.send_multipart([
                    encode_header(
                        RESULT,
                        session_id=session_id,
                        task_ids=task_ids,
                        assignment_id=assignment_id,
                        worker_id=worker_id,
                        lease_id=lease_id,
                        lease_generation=lease_generation,
                        elapsed_s=time.perf_counter() - started,
                        compile_s=compile_s,
                        peak_rss_kib=resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    ),
                    *results,
                ])
            except Exception as exc:  # noqa: BLE001
                socket.send_multipart([
                    encode_header(
                        ERROR,
                        session_id=session_id,
                        task_ids=task_ids,
                        assignment_id=assignment_id,
                        worker_id=worker_id,
                        lease_id=lease_id,
                        lease_generation=lease_generation,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    ),
                    b"",
                ])
    finally:
        stopping.set()
        heartbeat.join(timeout=heartbeat_interval_s * 2.0)
        socket.close(linger=0)
        context.term()


def _heartbeat_loop(
        context,
        endpoint: str,
        worker_id: str,
        instance_id: str,
        lease_id: str,
        lease_generation: int,
        interval_s: float,
        missed_limit: int,
        alive: threading.Event,
        stopping: threading.Event,
) -> None:
    """Keep liveness independent from data-dependent JAX sampler latency."""
    import zmq

    from jaxns.runtime_protocol import (
        HEARTBEAT,
        HEARTBEAT_ACK,
        decode_header,
        encode_header,
    )
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(
        zmq.IDENTITY,
        f"heartbeat:{worker_id}:{instance_id}:{uuid4().hex}".encode(),
    )
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    missed = 0
    try:
        while not stopping.is_set() and alive.is_set():
            heartbeat_started = time.monotonic()
            socket.send_multipart([
                encode_header(
                    HEARTBEAT,
                    worker_id=worker_id,
                    instance_id=instance_id,
                    lease_id=lease_id,
                    lease_generation=lease_generation,
                ),
                b"",
            ])
            if socket.poll(timeout=max(1, int(interval_s * 1000))):
                frames = socket.recv_multipart()
                header = decode_header(frames[0])
                if (
                    header["command"] == HEARTBEAT_ACK
                    and header.get("accepted") is True
                ):
                    missed = 0
                else:
                    _fence_process(alive, stopping)
                    return
            else:
                missed += 1
                if missed >= missed_limit:
                    _fence_process(alive, stopping)
                    return
            # An immediate acknowledgement must not turn this control thread
            # into a tight network loop. Preserve one heartbeat per configured
            # interval while allowing shutdown to interrupt the wait.
            elapsed_s = time.monotonic() - heartbeat_started
            stopping.wait(max(0.0, interval_s - elapsed_s))
    finally:
        socket.close(linger=0)


def _fence_process(
        alive: threading.Event,
        stopping: threading.Event,
) -> None:
    """Terminate a fenced worker even while its JAX call is still blocked."""
    alive.clear()
    if stopping.is_set():
        return
    # A worker owns no scientific state. Abrupt exit is intentional here: the
    # coordinator has already retained/requeued the immutable task, while the
    # node supervisor must observe process death before it can create a fresh
    # lease. Waiting for a blocked device call could starve this device forever.
    os.kill(os.getpid(), signal.SIGTERM)


def _combine_requests(requests, request_type, seed_type, jnp, jax):
    """Stack compatible scalar logical threads into one device call."""
    first = requests[0]
    # The local scientific client fingerprints direction state and the
    # JAX-free coordinator groups only matching fingerprints. Re-serializing
    # potentially large ellipsoid state here would put host work on every
    # worker hot path. Fit bookkeeping and completed-chain counters may differ
    # within a group, but the sampler never reads them; every field that can
    # affect a direction is identical, so using the first execution state is
    # intentional.

    def concatenate(*values):
        return jnp.concatenate(values, axis=0)

    return request_type(
        keys=concatenate(*(request.keys for request in requests)),
        valid=concatenate(*(request.valid for request in requests)),
        log_L_constraints=concatenate(*(
            request.log_L_constraints for request in requests
        )),
        seed_points=seed_type(
            U0=jax.tree.map(
                concatenate,
                *(request.seed_points.U0 for request in requests),
            ),
            log_L0=concatenate(*(
                request.seed_points.log_L0 for request in requests
            )),
        ),
        sampler_data=first.sampler_data,
    )


def _combine_likelihood_requests(requests, request_type, jnp, jax):
    """Stack scalar prior-space points for one worker likelihood call."""

    def concatenate(*values):
        return jnp.concatenate(values, axis=0)

    return request_type(
        U_samples=jax.tree.map(
            concatenate,
            *(request.U_samples for request in requests),
        )
    )


def _string_field(header: dict[str, object], name: str) -> str:
    value = header.get(name)
    if type(value) is not str:
        raise ValueError(f"Protocol field {name!r} must be a string.")
    return value


def _integer_field(header: dict[str, object], name: str) -> int:
    value = header.get(name)
    if type(value) is not int:
        raise ValueError(f"Protocol field {name!r} must be an integer.")
    return value


def _integer_list(header: dict[str, object], name: str) -> list[int]:
    value = header.get(name)
    if type(value) is not list or not value or not all(
        type(item) is int for item in value
    ):
        raise ValueError(f"Protocol field {name!r} must be integer list.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--configured-device", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device-index", required=True, type=int)
    parser.add_argument("--program-cache-size", required=True, type=int)
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args(argv)
    _run(
        args.config,
        args.endpoint,
        args.platform,
        args.configured_device,
        args.batch_size,
        args.device_index,
        args.program_cache_size,
        args.instance_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
