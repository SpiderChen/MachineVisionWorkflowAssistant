from __future__ import annotations

import signal
import unittest
from unittest.mock import patch

import run
from app.main import _install_qt_interrupt_handler


class _FakeApp:
    def __init__(self) -> None:
        self.quit_calls = 0

    def quit(self) -> None:
        self.quit_calls += 1


class InterruptHandlingTests(unittest.TestCase):
    def test_sigint_requests_qt_quit(self) -> None:
        app = _FakeApp()
        with patch("signal.signal") as install:
            _install_qt_interrupt_handler(app)

        self.assertEqual(install.call_args.args[0], signal.SIGINT)
        handler = install.call_args.args[1]
        handler(signal.SIGINT, None)
        self.assertEqual(app.quit_calls, 1)

    def test_keyboard_interrupt_is_not_reported_as_crash(self) -> None:
        with patch("traceback.format_exception") as format_exception:
            run._report_crash(KeyboardInterrupt, KeyboardInterrupt(), None)

        format_exception.assert_not_called()


if __name__ == "__main__":
    unittest.main()
