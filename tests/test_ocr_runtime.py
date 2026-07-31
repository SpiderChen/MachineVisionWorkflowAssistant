from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.vision import VisionError, _load_rapidocr_class, _ocr_load_error
from run import _finish_frozen_ocr_self_test


class OcrRuntimeTests(unittest.TestCase):
    def test_frozen_windows_self_test_forces_process_exit(self):
        fake_exit = Mock(side_effect=RuntimeError("os._exit called"))
        with patch("run.sys.platform", "win32"), \
                patch("run.sys.frozen", True, create=True), \
                patch("run.sys.argv", ["app.exe", "--ocr-self-test"]), \
                patch("run.os._exit", fake_exit), \
                patch("logging.shutdown"):
            with self.assertRaisesRegex(RuntimeError, "os._exit called"):
                _finish_frozen_ocr_self_test(0)
        fake_exit.assert_called_once_with(0)

    def test_normal_runtime_does_not_force_process_exit(self):
        with patch("run.os._exit") as fake_exit:
            self.assertEqual(_finish_frozen_ocr_self_test(2), 2)
        fake_exit.assert_not_called()

    def test_frozen_self_test_environment_forces_process_exit(self):
        fake_exit = Mock(side_effect=RuntimeError("os._exit called"))
        with patch("run.sys.platform", "win32"), \
                patch("run.sys.frozen", True, create=True), \
                patch("run.sys.argv", ["app.exe"]), \
                patch.dict("run.os.environ", {"MVWA_OCR_SELF_TEST": "1"}), \
                patch("run.os._exit", fake_exit), \
                patch("logging.shutdown"):
            with self.assertRaisesRegex(RuntimeError, "os._exit called"):
                _finish_frozen_ocr_self_test(0)
        fake_exit.assert_called_once_with(0)

    def test_packaged_trigger_file_requests_self_test(self):
        with patch("run.sys.argv", ["app.exe"]), \
                patch.dict("run.os.environ", {}, clear=True), \
                patch("run.os.path.isfile", return_value=True):
            from run import _ocr_self_test_requested

            self.assertTrue(_ocr_self_test_requested())

    def test_self_test_executable_name_requests_self_test(self):
        with patch("run.sys.argv", ["app.exe"]), \
                patch("run.sys.executable", "MachineVisionWorkflowAssistant-selftest.exe"), \
                patch.dict("run.os.environ", {}, clear=True), \
                patch("run.os.path.isfile", return_value=False):
            from run import _ocr_self_test_requested

            self.assertTrue(_ocr_self_test_requested())

    def test_falls_back_to_rapidocr_v2_only_when_legacy_package_is_absent(self):
        marker = object()
        missing = ModuleNotFoundError(
            "No module named 'rapidocr_onnxruntime'",
            name="rapidocr_onnxruntime",
        )
        with patch("app.vision.importlib.import_module", side_effect=[
                missing, SimpleNamespace(RapidOCR=marker)]):
            self.assertIs(_load_rapidocr_class(), marker)

    def test_internal_dll_failure_is_not_misreported_as_package_missing(self):
        failure = ImportError(
            "DLL load failed while importing onnxruntime_pybind11_state")
        with patch("app.vision.sys.platform", "win32"):
            with patch("app.vision.logger.exception"), patch(
                    "app.vision.importlib.import_module", side_effect=failure):
                with self.assertRaises(VisionError) as caught:
                    _load_rapidocr_class()

        message = str(caught.exception)
        self.assertIn("ONNX Runtime DLL 加载失败", message)
        self.assertIn("Visual C++", message)
        self.assertNotIn("RapidOCR 未安装", message)

    def test_missing_internal_dependency_keeps_its_real_name(self):
        failure = ModuleNotFoundError(
            "No module named 'onnxruntime'", name="onnxruntime")

        message = str(_ocr_load_error("rapidocr-onnxruntime", failure))

        self.assertIn("onnxruntime", message)
        self.assertIn("依赖", message)


if __name__ == "__main__":
    unittest.main()
