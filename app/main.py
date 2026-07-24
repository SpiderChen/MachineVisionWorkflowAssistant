"""入口：拉起托盘 + 各线程（README §7）。

用法：
  python -m app.main                     # 常驻服务：托盘 + 配置窗口 + DB 轮询 + HTTP API
  python -m app.main --headless          # 无 UI 运行（仅触发源 + 引擎）
  python -m app.main --print A0231       # M1 验证：命令行一单，同步执行“定位→输入→点打印”
  python -m app.main --locate all        # 定位诊断：命令行版「测试定位」，标注图存 logs/
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading

logger = logging.getLogger(__name__)


def _build_core(settings):
    from .engine import Engine
    from .input_ctl import InputController, make_dpi_aware
    from .vision import Vision

    make_dpi_aware()
    vision = Vision(settings)
    inputs = InputController()
    engine = Engine(settings, vision, inputs)
    return vision, inputs, engine


def _cli_print(engine, vehicle_no: str) -> int:
    from .engine import Task

    result = engine.execute_sync(Task(vehicle_no=vehicle_no, source="cli"))
    print(f"{'✔ 成功' if result.ok else '✘ 失败'}: {result.vehicle_no} — {result.message}")
    if result.snapshot:
        print(f"失败快照: {result.snapshot}")
    return 0 if result.ok else 1


def _cli_locate(vision, which: str) -> int:
    import cv2

    from . import locators
    from .settings import LOGS_DIR
    from .vision import draw_result

    names = list(locators.ALL.keys()) if which == "all" else [which]
    unknown = [n for n in names if n not in locators.ALL]
    if unknown:
        print(f"未知元素: {unknown}，可用: {list(locators.ALL.keys())}")
        return 1
    shot = vision.capture()
    annotated = shot.img.copy()
    for name in names:
        res = vision.locate(locators.ALL[name], shot)
        draw_result(annotated, res)
        print(res.describe())
        print()
    out = LOGS_DIR / f"diag_{which}.png"
    cv2.imwrite(str(out), annotated)
    print(f"标注截图: {out}")
    return 0


def _run_service(settings, vision, inputs, engine, headless: bool) -> int:
    from .triggers.api_server import ApiServer
    from .triggers.db_poller import DbPoller

    engine.start()
    poller = DbPoller(settings, engine)
    poller.start()
    api = ApiServer(settings, engine)
    if settings.api.enabled:
        api.start()

    stop_all = threading.Event()

    def shutdown():
        if not stop_all.is_set():
            stop_all.set()
            engine.stop()
            poller.stop()
            api.stop()

    if headless:
        logger.info("headless 模式运行中，Ctrl+C 退出")
        try:
            while not stop_all.wait(1.0):
                pass
        except KeyboardInterrupt:
            pass
        shutdown()
        return 0

    # ---- UI 模式：QApplication(主线程) + pystray(独立线程)，经 Qt 信号桥接 ----
    try:
        from .config_ui import ConfigWindow, window as cw
        from .qt_compat import QApplication, QObject, QT_API, Signal, qt_exec
        from .tray import Tray
    except (ImportError, OSError) as e:
        logger.exception("UI 依赖加载失败")
        shutdown()
        raise RuntimeError(f"UI 依赖加载失败：{e}") from e

    class Bridge(QObject):
        open_config = Signal()
        test_print = Signal()
        view_logs = Signal()
        quit_app = Signal()

    app = QApplication(sys.argv)
    win = ConfigWindow(settings, vision, engine)
    if sys.platform == "darwin":
        # macOS 不用 pystray 菜单栏（AppKit 要求主线程、依赖 pyobjc），直接把配置窗口
        # 作为主界面常驻；引擎 / DB 轮询 / HTTP API 已在本函数前面启动。
        app.setQuitOnLastWindowClosed(True)
        win.closeEvent = lambda e: e.accept()
        win.showNormal()
        win.raise_()
        win.activateWindow()
        logger.info("macOS：以配置窗口模式运行（无菜单栏托盘）")
        rc = qt_exec(app)
        shutdown()
        return rc
    app.setQuitOnLastWindowClosed(False)
    bridge = Bridge()
    bridge.open_config.connect(lambda: (win.showNormal(), win.raise_(), win.activateWindow()))
    bridge.test_print.connect(win.prompt_test_print)
    bridge.view_logs.connect(lambda: cw.open_folder(_logs_dir()))
    bridge.quit_app.connect(app.quit)

    tray = Tray(
        engine,
        on_open_config=bridge.open_config.emit,
        on_test_print=bridge.test_print.emit,
        on_view_logs=bridge.view_logs.emit,
        on_quit=bridge.quit_app.emit,
    )
    engine.notifier = tray.notify
    tray.run_detached()
    logger.info("托盘已就绪（%s；双击图标打开配置）", QT_API)

    rc = qt_exec(app)
    shutdown()
    tray.stop()
    return rc


def _logs_dir():
    from .settings import LOGS_DIR

    return LOGS_DIR


def _run_config_window(settings, vision, engine) -> int:
    """直接打开配置窗口（不依赖托盘）。WSL 下配合 QT_QPA_PLATFORM=wayland 使用。"""
    import os

    from .config_ui import ConfigWindow
    from .qt_compat import QApplication, QFont, QFontDatabase, QT_API, qt_exec

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)      # 无托盘：关窗即退出
    # WSL 无内置中文字体时借用 Windows 微软雅黑，避免界面中文显示为方框
    win_font = "/mnt/c/Windows/Fonts/msyh.ttc"
    if os.path.exists(win_font):
        app.setStyle("Fusion")
        fid = QFontDatabase.addApplicationFont(win_font)
        fams = QFontDatabase.applicationFontFamilies(fid)
        if fams:
            app.setFont(QFont(fams[0], 10))
    engine.start()
    win = ConfigWindow(settings, vision, engine)
    win.closeEvent = lambda e: e.accept()    # 覆盖“关窗仅隐藏”的托盘行为
    win.showNormal()
    win.raise_()
    win.activateWindow()
    logger.info("配置窗口已打开（%s；--ui 模式，无托盘）", QT_API)
    rc = qt_exec(app)
    engine.stop()
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="出货单自动打印助手")
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    parser.add_argument("--print", dest="print_no", metavar="VEHICLE_NO",
                        help="命令行执行一单后退出（M1 验证）")
    parser.add_argument("--locate", metavar="NAME|all",
                        help="命令行测试元素定位后退出")
    parser.add_argument("--headless", action="store_true", help="无托盘/无配置窗口运行")
    parser.add_argument("--ui", action="store_true",
                        help="直接打开配置窗口（不依赖托盘；WSL 下配合 wayland 调试用）")
    args = parser.parse_args()

    from .logger import init_logging
    from .settings import CONFIG_PATH, Settings

    settings = Settings.load(args.config or CONFIG_PATH)
    init_logging(settings.log.level)
    logger.info("AutoPrintDeliveryOrder 启动（配置: %s）", settings.path)

    vision, inputs, engine = _build_core(settings)

    if args.print_no:
        return _cli_print(engine, args.print_no.strip())
    if args.locate:
        return _cli_locate(vision, args.locate.strip())
    if args.ui:
        return _run_config_window(settings, vision, engine)
    return _run_service(settings, vision, inputs, engine, args.headless)


if __name__ == "__main__":
    sys.exit(main())
