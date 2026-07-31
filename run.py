"""PyInstaller / 命令行统一入口。

打包后运行的是本文件（而非 `python -m app.main`），所以这里加一层顶层异常兜底：
无控制台窗口的 exe 万一启动即崩，也能在 exe 旁边留下 startup_error.log 并弹窗提示，
而不是静默消失。命令行参数与 `python -m app.main` 完全一致（见 app/main.py 文档）。
"""
import os
import sys


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _report_crash(exc_type, exc, tb) -> None:
    import traceback

    # Ctrl+C 是用户主动停止，不是启动崩溃。Qt 回调中的 SIGINT 会走 sys.excepthook，
    # 因此这里也必须过滤，否则会弹出误导性的“启动失败”窗口。
    if issubclass(exc_type, KeyboardInterrupt):
        return

    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        with open(os.path.join(_base_dir(), "startup_error.log"), "a", encoding="utf-8") as f:
            f.write(text + "\n" + "=" * 60 + "\n")
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, text[-1500:], "启动失败 / Startup Error", 0x10)
        except Exception:
            pass


def _ensure_std_streams() -> None:
    """无控制台窗口的 exe 里 sys.stdout / sys.stderr 为 None，任何库一旦 stdout.isatty()
    或 write() 就崩（uvicorn 彩色日志、click 等）。把缺失的标准流指到 devnull（真实文件
    对象：isatty() 返回 False、write 丢弃），避免整类 'NoneType has no attribute ...' 崩溃。"""
    for name, mode in (("stdin", "r"), ("stdout", "w"), ("stderr", "w")):
        if getattr(sys, name, None) is None:
            try:
                setattr(sys, name, open(os.devnull, mode))
            except Exception:
                pass


def _finish_frozen_ocr_self_test(exit_code: int) -> int:
    """Windows onefile 自检完成后直接结束子进程。

    ONNX Runtime 的原生线程在 PyInstaller onefile 子进程解释器清理阶段偶尔不会退出，
    即使推理和日志都已完成，Actions 的 Start-Process 仍会一直等待。这个特殊退出只用于
    ``--ocr-self-test`` 验收命令；正常托盘/服务模式仍走完整的 Python 清理流程。
    """
    self_test = ("--ocr-self-test" in sys.argv
                 or os.environ.get("MVWA_OCR_SELF_TEST") == "1")
    if sys.platform == "win32" and getattr(sys, "frozen", False) and self_test:
        try:
            import logging

            logging.shutdown()
        finally:
            os._exit(int(exit_code or 0))
    return int(exit_code or 0)


def main() -> int:
    _ensure_std_streams()
    sys.excepthook = _report_crash
    # macOS 从 Finder 启动 .app 时，旧系统会塞入 -psn_xxx 参数，argparse 不认会直接退出
    sys.argv = [a for a in sys.argv if not a.startswith("-psn_")]
    try:
        # GitHub Windows Runner 的冻结包验收走专用直达入口，完全绕开普通 GUI/托盘
        # 启动分支。环境变量只由 workflow 临时设置，不影响用户正常启动。
        if os.environ.get("MVWA_OCR_SELF_TEST") == "1":
            from app.logger import init_logging
            from app.main import _cli_ocr_self_test
            from app.settings import CONFIG_PATH, Settings

            settings = Settings.load(CONFIG_PATH)
            init_logging(settings.log.level)
            return _finish_frozen_ocr_self_test(_cli_ocr_self_test(settings))

        from app.main import main as app_main

        return _finish_frozen_ocr_self_test(app_main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # 没有进入 Qt 事件循环或平台不支持 SIGINT handler 时的最后一道兜底。
        return 130
    except BaseException:  # 顶层兜底：记录后退出，不让 exe 静默消失
        _report_crash(*sys.exc_info())
        return 1


if __name__ == "__main__":
    sys.exit(main())
