"""Durable, corruption-detecting persistence for complete run states."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Generic, TypeVar, cast
from uuid import uuid4

from jaxns.logging import jaxns_logger
from jaxns.pytree import Pytree

CHECKPOINT_CADENCE_SECONDS = 3600.0
CHECKPOINT_SCHEMA_VERSION = 1
CHECKSUM_ALGORITHM = "sha256"
MANIFEST_NAME = "CHECKPOINT"
LOCK_NAME = "CHECKPOINT.lock"
STATE_PREFIX = "state-"
STATE_SUFFIX = ".pkl"
RETAINED_GENERATIONS = 2


class CheckpointError(RuntimeError):
    """Base error for checkpoint persistence and recovery."""

    __slots__ = ()


class CheckpointCorruptionError(CheckpointError):
    """A published checkpoint is incomplete, malformed, or corrupted."""

    __slots__ = ()


class CheckpointInUseError(CheckpointError):
    """Another process already owns the checkpoint directory."""

    __slots__ = ()


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Commit record that makes one durable state generation visible."""

    schema_version: int
    generation: int
    state_file: str
    checksum_algorithm: str
    checksum: str

    def to_json_bytes(self) -> bytes:
        """Return the canonical on-disk representation."""
        payload = {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "state_file": self.state_file,
            "checksum_algorithm": self.checksum_algorithm,
            "checksum": self.checksum,
        }
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> CheckpointManifest:
        """Parse and strictly validate the persisted manifest schema."""
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptionError(
                "CHECKPOINT is not valid UTF-8 JSON. Restore a valid "
                "checkpoint directory or remove it to begin a new run."
            ) from exc
        if type(payload) is not dict:
            raise CheckpointCorruptionError(
                "CHECKPOINT must contain one JSON object."
            )
        values = cast(dict[str, object], payload)
        expected = {
            "schema_version",
            "generation",
            "state_file",
            "checksum_algorithm",
            "checksum",
        }
        if set(values) != expected:
            raise CheckpointCorruptionError(
                "CHECKPOINT has an unsupported field schema."
            )

        schema_version = values["schema_version"]
        generation = values["generation"]
        state_file = values["state_file"]
        checksum_algorithm = values["checksum_algorithm"]
        checksum = values["checksum"]
        if type(schema_version) is not int:
            raise CheckpointCorruptionError(
                "CHECKPOINT schema_version must be an integer."
            )
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptionError(
                "Unsupported checkpoint schema version "
                f"{schema_version}; this JAXNS build supports "
                f"{CHECKPOINT_SCHEMA_VERSION}."
            )
        if type(generation) is not int or generation < 1:
            raise CheckpointCorruptionError(
                "CHECKPOINT generation must be a positive integer."
            )
        if type(state_file) is not str:
            raise CheckpointCorruptionError(
                "CHECKPOINT state_file must be a string."
            )
        expected_state_file = _state_name(generation)
        if state_file != expected_state_file:
            raise CheckpointCorruptionError(
                "CHECKPOINT state_file does not match its generation."
            )
        if checksum_algorithm != CHECKSUM_ALGORITHM:
            raise CheckpointCorruptionError(
                "Unsupported checkpoint checksum algorithm "
                f"{checksum_algorithm!r}."
            )
        if type(checksum) is not str or len(checksum) != 64:
            raise CheckpointCorruptionError(
                "CHECKPOINT checksum must be a SHA-256 hexadecimal digest."
            )
        if any(character not in "0123456789abcdef" for character in checksum):
            raise CheckpointCorruptionError(
                "CHECKPOINT checksum must be lowercase hexadecimal."
            )
        return cls(
            schema_version=schema_version,
            generation=generation,
            state_file=state_file,
            checksum_algorithm=checksum_algorithm,
            checksum=checksum,
        )


StateType = TypeVar("StateType", bound=Pytree)


class CheckpointManager(Generic[StateType]):
    """Own one checkpoint directory for the lifetime of a run.

    The manager intentionally does not validate model, arguments, sampler, or
    runner compatibility. A caller that opts into automatic resume owns that
    semantic contract; JAXNS verifies only storage integrity and schema.
    """

    __slots__ = (
        "_cadence_seconds",
        "_checkpoint_dir",
        "_clock",
        "_last_checkpoint_time",
        "_lock_file",
        "_manifest",
        "_saved_state",
    )

    def __init__(
            self,
            checkpoint_dir: str | Path,
            cadence_seconds: float = CHECKPOINT_CADENCE_SECONDS,
            clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(cadence_seconds) or cadence_seconds < 0.0:
            raise ValueError(
                "checkpoint_cadence must be finite and non-negative."
            )
        self._checkpoint_dir = Path(checkpoint_dir)
        self._cadence_seconds = float(cadence_seconds)
        self._clock = clock
        self._last_checkpoint_time = self._clock()
        self._lock_file = None
        self._manifest: CheckpointManifest | None = None
        self._saved_state: StateType | None = None

    def __enter__(self) -> CheckpointManager[StateType]:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._checkpoint_dir / LOCK_NAME
        lock_file = lock_path.open("a+b")
        try:
            _acquire_lock(lock_file)
        except OSError as exc:
            lock_file.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise CheckpointInUseError(
                    f"Checkpoint directory {self._checkpoint_dir} is already "
                    "owned by another process."
                ) from exc
            raise CheckpointError(
                f"Could not lock checkpoint directory {self._checkpoint_dir}: "
                f"{exc}"
            ) from exc
        self._lock_file = lock_file
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        lock_file = self._lock_file
        if lock_file is None:
            return
        _release_lock(lock_file)
        lock_file.close()
        self._lock_file = None

    def load(self) -> StateType | None:
        """Load the latest committed generation after checksum verification."""
        self._require_lock()
        manifest_path = self._checkpoint_dir / MANIFEST_NAME
        if not manifest_path.exists():
            incomplete = tuple(self._checkpoint_dir.glob(f"{STATE_PREFIX}*"))
            incomplete += tuple(self._checkpoint_dir.glob(".*.tmp.pkl"))
            if incomplete:
                raise CheckpointCorruptionError(
                    "Checkpoint state files exist without a committed "
                    "CHECKPOINT manifest. Restore or clear the incomplete "
                    "checkpoint directory."
                )
            return None

        try:
            manifest_data = manifest_path.read_bytes()
        except OSError as exc:
            raise CheckpointCorruptionError(
                f"Could not read checkpoint manifest {manifest_path}."
            ) from exc
        manifest = CheckpointManifest.from_json_bytes(manifest_data)
        state_path = self._checkpoint_dir / manifest.state_file
        if not state_path.is_file():
            raise CheckpointCorruptionError(
                f"Checkpoint state file {manifest.state_file} is missing."
            )
        checksum = _sha256(state_path)
        if checksum != manifest.checksum:
            raise CheckpointCorruptionError(
                f"Checkpoint checksum mismatch for {manifest.state_file}. "
                "The state was not deserialized."
            )
        try:
            state = cast(StateType, Pytree.load(str(state_path)))
        except Exception as exc:
            raise CheckpointCorruptionError(
                f"Checkpoint {manifest.state_file} passed its checksum but "
                "could not be deserialized in this Python environment."
            ) from exc
        self._manifest = manifest
        self._saved_state = state
        self._last_checkpoint_time = self._clock()
        jaxns_logger.info(
            "Loaded checkpoint generation %d from %s.",
            manifest.generation,
            self._checkpoint_dir,
        )
        return state

    def maybe_save(self, state: StateType) -> bool:
        """Persist ``state`` when the configured monotonic cadence is due."""
        now = self._clock()
        if now - self._last_checkpoint_time < self._cadence_seconds:
            return False
        self.save(state, checkpoint_time=now)
        return True

    def save_if_changed(self, state: StateType) -> bool:
        """Persist a final or exceptional state unless it was just saved."""
        if state is self._saved_state:
            return False
        self.save(state)
        return True

    def save(
            self,
            state: StateType,
            checkpoint_time: float | None = None,
    ) -> None:
        """Durably publish the complete Pytree as one new generation."""
        self._require_lock()
        generation = 1
        if self._manifest is not None:
            generation = self._manifest.generation + 1
        state_name = _state_name(generation)
        state_path = self._checkpoint_dir / state_name
        state_temporary = self._checkpoint_dir / (
            f".{state_name}.{uuid4().hex}.tmp.pkl"
        )
        try:
            # Pytree.save is the single serialization surface. Durability and
            # publication remain filesystem concerns owned by this manager.
            state.save(str(state_temporary))
            _fsync_file(state_temporary)
            checksum = _sha256(state_temporary)
            os.replace(state_temporary, state_path)
            _fsync_directory(self._checkpoint_dir)
        except Exception:
            state_temporary.unlink(missing_ok=True)
            raise

        manifest = CheckpointManifest(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            generation=generation,
            state_file=state_name,
            checksum_algorithm=CHECKSUM_ALGORITHM,
            checksum=checksum,
        )
        manifest_temporary = self._checkpoint_dir / (
            f".{MANIFEST_NAME}.{uuid4().hex}.tmp"
        )
        try:
            with manifest_temporary.open("wb") as file:
                file.write(manifest.to_json_bytes())
                file.flush()
                os.fsync(file.fileno())
            # The manifest is the commit record. Publishing it last ensures a
            # reader sees either the prior complete generation or this one.
            os.replace(
                manifest_temporary,
                self._checkpoint_dir / MANIFEST_NAME,
            )
            _fsync_directory(self._checkpoint_dir)
        except Exception:
            manifest_temporary.unlink(missing_ok=True)
            raise

        self._manifest = manifest
        self._saved_state = state
        if checkpoint_time is None:
            checkpoint_time = self._clock()
        self._last_checkpoint_time = checkpoint_time
        try:
            self._prune_old_generations()
        except OSError as exc:
            # Publication is already durable. A cleanup failure may consume
            # extra disk space, but reporting the committed save as failed
            # would give the caller a false recovery picture.
            jaxns_logger.warning(
                "Checkpoint generation %d is valid, but old generations "
                "could not be pruned from %s: %s",
                generation,
                self._checkpoint_dir,
                exc,
            )
        jaxns_logger.info(
            "Saved checkpoint generation %d to %s.",
            generation,
            self._checkpoint_dir,
        )

    def _prune_old_generations(self) -> None:
        state_paths = sorted(
            path
            for path in self._checkpoint_dir.glob(
                f"{STATE_PREFIX}*{STATE_SUFFIX}"
            )
            if _is_state_name(path.name)
        )
        for state_path in state_paths[:-RETAINED_GENERATIONS]:
            state_path.unlink()
        if len(state_paths) > RETAINED_GENERATIONS:
            _fsync_directory(self._checkpoint_dir)

    def _require_lock(self) -> None:
        if self._lock_file is None:
            raise RuntimeError(
                "CheckpointManager must be entered before loading or saving."
            )


def _state_name(generation: int) -> str:
    return f"{STATE_PREFIX}{generation:020d}{STATE_SUFFIX}"


def _is_state_name(name: str) -> bool:
    if not name.startswith(STATE_PREFIX) or not name.endswith(STATE_SUFFIX):
        return False
    generation = name[len(STATE_PREFIX):-len(STATE_SUFFIX)]
    return len(generation) == 20 and generation.isdecimal()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb+") as file:
        os.fsync(file.fileno())


if os.name == "nt":
    import msvcrt

    def _acquire_lock(file) -> None:
        file.seek(0)
        if file.read(1) == b"":
            file.write(b"\0")
            file.flush()
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)

    def _release_lock(file) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)

    def _fsync_directory(path: Path) -> None:
        # Windows has no directory file descriptor compatible with fsync.
        # os.replace still provides atomic publication on one volume.
        del path

else:
    import fcntl

    def _acquire_lock(file) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release_lock(file) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)

    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
