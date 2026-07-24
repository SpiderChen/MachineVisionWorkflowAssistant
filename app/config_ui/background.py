"""Small Qt-safe background task helper for slow OCR/vision operations."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..qt_compat import QObject, Signal

logger = logging.getLogger(__name__)


class BackgroundTask(QObject):
    """Run one Python callable off the Qt event thread and deliver its result safely."""

    finished = Signal(object, object)  # (result, exception)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._callback: Callable | None = None
        self._thread: threading.Thread | None = None
        self.finished.connect(self._deliver)

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, name: str, fn: Callable, callback: Callable) -> bool:
        """Start ``fn`` unless another task is active; callback(result, error) runs in Qt."""
        if self._busy:
            return False
        self._busy = True
        self._callback = callback

        def _work() -> None:
            result = None
            error = None
            try:
                result = fn()
            except Exception as exc:  # pass the original error back to the UI
                error = exc
                logger.exception("后台任务失败: %s", name)
            self.finished.emit(result, error)

        self._thread = threading.Thread(
            target=_work, name=f"ui-{name}", daemon=True)
        self._thread.start()
        return True

    def _deliver(self, result, error) -> None:
        callback = self._callback
        self._callback = None
        self._thread = None
        self._busy = False
        if callback is not None:
            callback(result, error)
