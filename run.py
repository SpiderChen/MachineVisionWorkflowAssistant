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


def main() -> int:
    sys.excepthook = _report_crash
    try:
        from app.main import main as app_main

        return app_main()
    except SystemExit:
        raise
    except BaseException:  # 顶层兜底：记录后退出，不让 exe 静默消失
        _report_crash(*sys.exc_info())
        return 1


if __name__ == "__main__":
    sys.exit(main())
