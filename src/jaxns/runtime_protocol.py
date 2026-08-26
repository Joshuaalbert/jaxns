"""JAX-free framing constants for trusted local supervisor IPC."""

from __future__ import annotations

import json

PROTOCOL_VERSION = 1
ROLE = "role"
PING = "ping"
STATUS = "status"
SHUTDOWN = "shutdown"
READY = "ready"
REGISTER = "register"
REGISTERED = "registered"
RELEASE = "release"
RELEASED = "released"
TASK = "task"
RESULT = "result"
ACK = "ack"
ERROR = "error"
STOP = "stop"
STOPPED = "stopped"


def encode_header(command: str, **fields) -> bytes:
    """Encode one small routing header without importing scientific code."""
    return json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "command": command,
            **fields,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_header(value: bytes) -> dict[str, object]:
    """Decode and validate one routing header."""
    try:
        header = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid runtime protocol header.") from exc
    if type(header) is not dict:
        raise ValueError("Runtime protocol header must be an object.")
    if header.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("Unsupported runtime protocol version.")
    if type(header.get("command")) is not str:
        raise ValueError("Runtime protocol command is missing.")
    return header
