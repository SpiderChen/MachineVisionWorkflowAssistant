from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.vision import VisionError, _load_rapidocr_class, _ocr_load_error


class OcrRuntimeTests(unittest.TestCase):
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
