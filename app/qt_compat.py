"""Qt5 / Qt6 binding compatibility.

Windows prefers PyQt5 because Qt 5 still runs on older systems such as
Windows Server 2016.  Other platforms keep using PySide6 by default.  Set
``MVWA_QT_API=pyqt5`` or ``MVWA_QT_API=pyside6`` to force a binding.
"""
from __future__ import annotations

import os
import sys


def _binding_order() -> tuple[str, str]:
    requested = os.environ.get("MVWA_QT_API", "").strip().lower()
    if requested in {"pyqt5", "pyside6"}:
        other = "pyside6" if requested == "pyqt5" else "pyqt5"
        return requested, other
    if sys.platform == "win32":
        return "pyqt5", "pyside6"
    return "pyside6", "pyqt5"


_errors: list[str] = []
for _candidate in _binding_order():
    try:
        if _candidate == "pyqt5":
            from PyQt5 import QtCore, QtGui, QtWidgets

            QT_API = "PyQt5"
        else:
            from PySide6 import QtCore, QtGui, QtWidgets

            QT_API = "PySide6"
        break
    except (ImportError, OSError) as exc:
        _errors.append(f"{_candidate}: {exc}")
else:
    raise ImportError("Qt GUI 依赖不可用；" + "；".join(_errors))


Signal = getattr(QtCore, "Signal", None) or getattr(QtCore, "pyqtSignal")


def event_pos(event):
    """Return an integer QPoint for Qt5 and Qt6 mouse events."""
    position = getattr(event, "position", None)
    if position is not None:
        return position().toPoint()
    return event.pos()


def qt_exec(obj, *args):
    """Run QApplication/QMenu using the binding-specific exec method."""
    method = getattr(obj, "exec", None) or getattr(obj, "exec_", None)
    if method is None:
        raise AttributeError(f"{type(obj).__name__} 没有 exec/exec_ 方法")
    return method(*args)


def __getattr__(name: str):
    """Expose Qt classes so callers can keep concise binding-neutral imports."""
    for module in (QtCore, QtGui, QtWidgets):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)

