"""pystray 托盘（README §6）：一键启停 / 打开配置 / 测试打印 / 查看日志 / 退出。"""
from __future__ import annotations

import logging

import pystray
from PIL import Image, ImageDraw
from pystray import Menu, MenuItem

logger = logging.getLogger(__name__)


def _make_icon_image() -> Image.Image:
    """画一个简易打印机图标（免外置素材）。"""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([18, 6, 46, 24], fill=(235, 235, 235, 255))     # 进纸
    d.rounded_rectangle([8, 20, 56, 46], radius=6, fill=(30, 90, 180, 255))  # 机身
    d.rectangle([18, 38, 46, 60], fill=(255, 255, 255, 255))    # 出纸
    d.rectangle([44, 26, 50, 32], fill=(80, 220, 120, 255))     # 指示灯
    return img


class Tray:
    def __init__(self, engine, on_open_config, on_test_print, on_view_logs, on_quit):
        self.engine = engine
        menu = Menu(
            MenuItem(self._state_text, self._toggle),
            Menu.SEPARATOR,
            MenuItem("打开配置...", lambda: on_open_config(), default=True),
            MenuItem("立即测试打印", lambda: on_test_print()),
            MenuItem("查看日志", lambda: on_view_logs()),
            Menu.SEPARATOR,
            MenuItem("退出", lambda: on_quit()),
        )
        self.icon = pystray.Icon(
            "AutoPrintDeliveryOrder", _make_icon_image(), "出货单自动打印助手", menu)

    def _state_text(self, item) -> str:
        return "⏸ 已暂停（点击恢复）" if self.engine.paused else "● 运行中（点击暂停）"

    def _toggle(self) -> None:
        self.engine.toggle()
        self.icon.update_menu()

    def notify(self, title: str, msg: str) -> None:
        """托盘气泡告警。"""
        try:
            self.icon.notify(msg, title)
        except Exception:
            logger.debug("托盘气泡失败", exc_info=True)

    def run_detached(self) -> None:
        self.icon.run_detached()

    def stop(self) -> None:
        try:
            self.icon.stop()
        except Exception:
            pass
