"""JAX-free framing constants for local IPC and authenticated cluster TCP."""

from __future__ import annotations

import json

PROTOCOL_VERSION = 3
MAX_HEADER_BYTES = 65_536
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
CAPACITY = "capacity"
ERROR = "error"
STOP = "stop"
STOPPED = "stopped"
LEASED = "leased"
HEARTBEAT = "heartbeat"
HEARTBEAT_ACK = "heartbeat_ack"
DRAIN = "drain"
DRAINED = "drained"
NODE_STOPPED = "node_stopped"
NODE_STATUS = "node_status"


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
    if len(value) > MAX_HEADER_BYTES:
        raise ValueError("Runtime protocol header exceeds 65,536 bytes.")
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
