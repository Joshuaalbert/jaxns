"""One-process, one-device constrained-sampling worker entry point."""

from __future__ import annotations

import argparse
import collections
import os
import pickle
import resource
import time
import traceback


def _run(
        endpoint: str,
        name: str,
        batch_size: int,
        device_index: int,
        program_cache_size: int,
) -> None:
    # The supervisor sets platform visibility before this module imports JAX.
    # Importing here prevents the JAX-free supervisor from creating a backend
    # or inheriting accelerator state into child processes.
    import cloudpickle
    import jax
    import zmq

    from jaxns.constrained_sampler import sample_request
    from jaxns.runtime_protocol import (
        ERROR,
        READY,
        REGISTER,
        REGISTERED,
        RELEASE,
        RESULT,
        STOP,
        STOPPED,
        TASK,
        decode_header,
        encode_header,
    )

    def make_program(registered):
        def program(request):
            return sample_request(
                registered.sampler,
                request,
                args=registered.args,
                params=registered.params,
            )

        return program

    class Program:
        __slots__ = ("compiled", "program")

        def __init__(self, program):
            self.program = jax.jit(program)
            self.compiled = None

    devices = jax.devices()
    if not 0 <= device_index < len(devices):
        raise ValueError(
            f"Worker {name!r} requested device index {device_index}, but "
            f"only {len(devices)} {jax.default_backend()} device(s) are visible."
        )
    device = devices[device_index]
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, f"worker:{name}".encode())
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    programs: dict[str, Program] = {}
    program_cache: collections.OrderedDict[str, Program] = (
        collections.OrderedDict()
    )
    socket.send_multipart([
        encode_header(
            READY,
            role="worker",
            name=name,
            batch_size=batch_size,
            platform=jax.default_backend(),
            device=str(device),
            process_id=os.getpid(),
        ),
        b"",
    ])
    try:
        while True:
            frames = socket.recv_multipart()
            if len(frames) != 2:
                raise RuntimeError("Worker received an invalid frame count.")
            header = decode_header(frames[0])
            payload = frames[1]
            command = header["command"]
            if command == STOP:
                socket.send_multipart([
                    encode_header(STOPPED, name=name),
                    b"",
                ])
                return
            if command == REGISTER:
                session_id = _string_field(header, "session_id")
                fingerprint = _string_field(header, "fingerprint")
                try:
                    record = program_cache.get(fingerprint)
                    if record is None:
                        session = cloudpickle.loads(payload)
                        # Capture registration data in a closure so the
                        # compiled program has exactly one dynamic task input.
                        record = Program(make_program(session))
                        program_cache[fingerprint] = record
                        while len(program_cache) > program_cache_size:
                            program_cache.popitem(last=False)
                    else:
                        program_cache.move_to_end(fingerprint)
                    programs[session_id] = record
                    socket.send_multipart([
                        encode_header(
                            REGISTERED,
                            session_id=session_id,
                            name=name,
                        ),
                        b"",
                    ])
                except Exception as exc:  # noqa: BLE001
                    # Model deserialization and tracing are user-code fault
                    # boundaries; report them to the owning session verbatim.
                    socket.send_multipart([
                        encode_header(
                            ERROR,
                            session_id=session_id,
                            name=name,
                            error=f"{type(exc).__name__}: {exc}",
                            traceback=traceback.format_exc(),
                        ),
                        b"",
                    ])
                continue
            if command == RELEASE:
                session_id = _string_field(header, "session_id")
                programs.pop(session_id, None)
                continue
            if command != TASK:
                raise RuntimeError(f"Worker received unsupported {command!r}.")

            session_id = _string_field(header, "session_id")
            task_id = _integer_field(header, "task_id")
            compile_s = 0.0
            started = time.perf_counter()
            try:
                if session_id not in programs:
                    raise RuntimeError(
                        f"Session {session_id!r} is not registered."
                    )
                request = pickle.loads(payload)
                width = request.log_L_constraints.shape[0]
                if width != batch_size:
                    raise ValueError(
                        f"Worker batch size is {batch_size}, received {width}."
                    )
                record = programs[session_id]
                with jax.default_device(device):
                    if record.compiled is None:
                        compile_started = time.perf_counter()
                        record.compiled = record.program.lower(request).compile()
                        compile_s = time.perf_counter() - compile_started
                    batch = jax.device_get(record.compiled(request))
                result = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
                elapsed_s = time.perf_counter() - started
                socket.send_multipart([
                    encode_header(
                        RESULT,
                        session_id=session_id,
                        task_id=task_id,
                        name=name,
                        elapsed_s=elapsed_s,
                        compile_s=compile_s,
                        peak_rss_kib=(
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        ),
                    ),
                    result,
                ])
            except Exception as exc:  # noqa: BLE001
                # A failed scientific task is isolated to this worker message
                # and must be visible to the client with its traceback.
                socket.send_multipart([
                    encode_header(
                        ERROR,
                        session_id=session_id,
                        task_id=task_id,
                        name=name,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    ),
                    b"",
                ])
    finally:
        socket.close(linger=0)
        context.term()


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device-index", required=True, type=int)
    parser.add_argument("--program-cache-size", required=True, type=int)
    args = parser.parse_args(argv)
    _run(
        args.endpoint,
        args.name,
        args.batch_size,
        args.device_index,
        args.program_cache_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
