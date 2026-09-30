"""Python-boundary interruption control for checkpointed sampling."""

import signal
import threading
from types import FrameType, TracebackType

# Return to Python periodically without callbacks inside the compiled sampler.
# Each replacement batch completes its chains before exposing resumable state.
INTERRUPT_BATCHES = 32


class DeferredSIGINT:
    """Remember Ctrl-C while finishing an atomic operation or saving state.

    Only the main thread owns process signal handlers. Background callers still
    propagate explicitly raised KeyboardInterrupt through the run's cleanup.
    """

    __slots__ = ("enabled", "previous", "requested")

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = (
            enabled and threading.current_thread() is threading.main_thread()
        )
        self.requested = False
        self.previous = None

    def __enter__(self) -> "DeferredSIGINT":  # noqa: PYI034
        if self.enabled:
            self.previous = signal.signal(signal.SIGINT, self._request)
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
    ) -> None:
        if self.enabled:
            signal.signal(signal.SIGINT, self.previous)

    def _request(self, signum: int, frame: FrameType | None) -> None:
        del signum, frame
        self.requested = True

    def raise_if_requested(self) -> None:
        if self.requested:
            raise KeyboardInterrupt
