"""系统级鼠标/键盘输入（README §2 技术选型）。

Windows 上直接走 Win32 SendInput：
- 鼠标：绝对坐标移动 + 左键点击（任何程序都认的真实事件）
- 文本：KEYEVENTF_UNICODE 直注字符，不经输入法、不占剪贴板
- 剪贴板粘贴仅作个别程序不认注入时的后备（粘贴前保存、粘贴后恢复）

保险丝（README §8）：移动后校验光标位置，被外力移动则中止本任务。
非 Windows 环境降级为 pynput（开发调试用），再不行用 Dummy 只记日志。
"""
from __future__ import annotations

import logging
import sys
import time

logger = logging.getLogger(__name__)

IS_WIN = sys.platform == "win32"


class InputError(Exception):
    pass


class MouseInterference(InputError):
    """鼠标位置与预期不符——被外力移动，中止本任务重试。"""


def make_dpi_aware() -> None:
    """声明 DPI-aware，让截屏坐标与物理坐标一致（README §8）。须在首次截屏前调用。"""
    if not IS_WIN:
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            logger.warning("设置 DPI-aware 失败，坐标换算可能有偏差")


# =====================================================================
# Win32 SendInput 后端
# =====================================================================

if IS_WIN:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    ULONG_PTR = wintypes.WPARAM

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG), ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
        )

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = (
            ("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        )

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = (
            ("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        )

    class _INPUTUNION(ctypes.Union):
        _fields_ = (("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT))

    class _INPUT(ctypes.Structure):
        _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))

    _INPUT_MOUSE, _INPUT_KEYBOARD = 0, 1
    _KEYEVENTF_KEYUP, _KEYEVENTF_UNICODE = 0x0002, 0x0004
    _MOUSEEVENTF_MOVE, _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP = 0x0001, 0x0002, 0x0004
    _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
    _MOUSEEVENTF_ABSOLUTE, _MOUSEEVENTF_VIRTUALDESK = 0x8000, 0x4000
    # GetSystemMetrics 虚拟桌面
    _SM_XVIRTUALSCREEN, _SM_YVIRTUALSCREEN = 76, 77
    _SM_CXVIRTUALSCREEN, _SM_CYVIRTUALSCREEN = 78, 79


VK = {
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10,
    "enter": 0x0D, "esc": 0x1B, "tab": 0x09,
    "delete": 0x2E, "backspace": 0x08,
    "home": 0x24, "end": 0x23,
    "f5": 0x74,
    "a": 0x41, "v": 0x56, "0": 0x30,
}


class _Win32Backend:
    def _send(self, *inputs) -> None:
        n = len(inputs)
        arr = (_INPUT * n)(*inputs)
        sent = _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        if sent != n:
            raise InputError(f"SendInput 只发送了 {sent}/{n} 个事件")

    def _mouse_input(self, dx=0, dy=0, flags=0) -> "_INPUT":
        inp = _INPUT(type=_INPUT_MOUSE)
        inp.u.mi = _MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
        return inp

    def _key_input(self, vk=0, scan=0, flags=0) -> "_INPUT":
        inp = _INPUT(type=_INPUT_KEYBOARD)
        inp.u.ki = _KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
        return inp

    def move_to(self, x: int, y: int) -> None:
        vx = _user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
        vy = _user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
        vw = _user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
        vh = _user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)
        nx = int((x - vx) * 65535 / max(vw - 1, 1))
        ny = int((y - vy) * 65535 / max(vh - 1, 1))
        self._send(self._mouse_input(
            nx, ny, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK))

    def click_down_up(self) -> None:
        self._send(self._mouse_input(flags=_MOUSEEVENTF_LEFTDOWN))
        time.sleep(0.03)
        self._send(self._mouse_input(flags=_MOUSEEVENTF_LEFTUP))

    def right_down_up(self) -> None:
        self._send(self._mouse_input(flags=_MOUSEEVENTF_RIGHTDOWN))
        time.sleep(0.03)
        self._send(self._mouse_input(flags=_MOUSEEVENTF_RIGHTUP))

    def cursor_pos(self) -> tuple[int, int]:
        pt = wintypes.POINT()
        _user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def key_down(self, vk: int) -> None:
        self._send(self._key_input(vk=vk))

    def key_up(self, vk: int) -> None:
        self._send(self._key_input(vk=vk, flags=_KEYEVENTF_KEYUP))

    def type_text(self, text: str) -> None:
        """KEYEVENTF_UNICODE 逐字符注入（含中文；代理对按 UTF-16 单元发送）。"""
        for ch in text:
            units = ch.encode("utf-16-le")
            for i in range(0, len(units), 2):
                scan = int.from_bytes(units[i:i + 2], "little")
                self._send(self._key_input(scan=scan, flags=_KEYEVENTF_UNICODE))
                self._send(self._key_input(
                    scan=scan, flags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP))
            time.sleep(0.01)

    def get_clipboard(self) -> str | None:
        import win32clipboard
        import win32con

        try:
            win32clipboard.OpenClipboard()
            try:
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                return None
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            logger.warning("读取剪贴板失败", exc_info=True)
            return None

    def set_clipboard(self, text: str) -> None:
        import win32clipboard
        import win32con

        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
        finally:
            win32clipboard.CloseClipboard()


class _PynputBackend:
    """非 Windows 的开发降级（X11 等）。"""

    def __init__(self):
        from pynput.keyboard import Controller as Kb, Key
        from pynput.mouse import Button, Controller as Mouse

        self._mouse = Mouse()
        self._kb = Kb()
        self._Button = Button
        self._Key = Key
        self._keymap = {
            "ctrl": Key.ctrl, "alt": Key.alt, "shift": Key.shift,
            "enter": Key.enter, "esc": Key.esc, "tab": Key.tab,
            "delete": Key.delete, "backspace": Key.backspace,
            "home": Key.home, "end": Key.end, "f5": Key.f5,
        }

    def _key(self, name: str):
        return self._keymap.get(name, name)

    def move_to(self, x, y):
        self._mouse.position = (x, y)

    def click_down_up(self):
        self._mouse.click(self._Button.left)

    def right_down_up(self):
        self._mouse.click(self._Button.right)

    def cursor_pos(self):
        return tuple(int(v) for v in self._mouse.position)

    def key_down(self, vk):
        self._kb.press(self._vk_to_key(vk))

    def key_up(self, vk):
        self._kb.release(self._vk_to_key(vk))

    def _vk_to_key(self, vk):
        for name, code in VK.items():
            if code == vk:
                return self._key(name)
        return chr(vk).lower()

    def type_text(self, text):
        self._kb.type(text)

    def get_clipboard(self):
        return None

    def set_clipboard(self, text):
        logger.info("[pynput] 忽略剪贴板写入")


class _DummyBackend:
    """无显示环境（CI/WSL 无 X）：只记日志，便于走通流程调试。"""

    def __init__(self):
        self._pos = (0, 0)

    def move_to(self, x, y):
        self._pos = (x, y)
        logger.info("[dummy] move_to(%s, %s)", x, y)

    def click_down_up(self):
        logger.info("[dummy] click at %s", (self._pos,))

    def right_down_up(self):
        logger.info("[dummy] right click at %s", (self._pos,))

    def cursor_pos(self):
        return self._pos

    def key_down(self, vk):
        logger.info("[dummy] key_down 0x%02X", vk)

    def key_up(self, vk):
        logger.info("[dummy] key_up 0x%02X", vk)

    def type_text(self, text):
        logger.info("[dummy] type_text(%r)", text)

    def get_clipboard(self):
        return None

    def set_clipboard(self, text):
        logger.info("[dummy] set_clipboard(%r)", text)


class InputController:
    """对执行引擎暴露的原子输入操作。所有点击带鼠标位置保险丝。"""

    MOUSE_TOLERANCE = 5  # 像素

    def __init__(self):
        if IS_WIN:
            self.backend = _Win32Backend()
        else:
            try:
                self.backend = _PynputBackend()
                logger.warning("非 Windows 环境，使用 pynput 降级后端（仅供开发调试）")
            except Exception:
                self.backend = _DummyBackend()
                logger.warning("无可用输入后端，使用 Dummy（只记日志）")

    def click(self, x: int, y: int) -> None:
        """移动 → 校验位置（保险丝）→ 点击。"""
        self._move_checked(x, y)
        self.backend.click_down_up()

    def double_click(self, x: int, y: int) -> None:
        self._move_checked(x, y)
        self.backend.click_down_up()
        time.sleep(0.08)
        self.backend.click_down_up()

    def right_click(self, x: int, y: int) -> None:
        self._move_checked(x, y)
        self.backend.right_down_up()

    def _move_checked(self, x: int, y: int) -> None:
        self.backend.move_to(int(x), int(y))
        time.sleep(0.05)
        cx, cy = self.backend.cursor_pos()
        if abs(cx - x) > self.MOUSE_TOLERANCE or abs(cy - y) > self.MOUSE_TOLERANCE:
            raise MouseInterference(
                f"鼠标位置异常：预期 ({x},{y})，实际 ({cx},{cy})——疑似被外力移动")

    def key_tap(self, name: str) -> None:
        vk = VK.get(name.lower())
        if vk is None:
            raise InputError(f"未知按键: {name}")
        self.backend.key_down(vk)
        time.sleep(0.02)
        self.backend.key_up(vk)

    def hotkey(self, *names: str) -> None:
        """如 hotkey('ctrl', 'a')：按下修饰键 → 主键 → 逆序释放。"""
        vks = []
        for n in names:
            vk = VK.get(n.lower())
            if vk is None:
                raise InputError(f"未知按键: {n}")
            vks.append(vk)
        for vk in vks:
            self.backend.key_down(vk)
            time.sleep(0.02)
        for vk in reversed(vks):
            self.backend.key_up(vk)
            time.sleep(0.02)

    def select_all_clear(self) -> None:
        """Ctrl+A 全选 → Delete 清空（流程②）。"""
        self.hotkey("ctrl", "a")
        time.sleep(0.08)
        self.key_tap("delete")
        time.sleep(0.08)

    def type_text(self, text: str) -> None:
        self.backend.type_text(text)

    def paste_text(self, text: str) -> None:
        """剪贴板粘贴后备：保存 → 写入 → Ctrl+V → 恢复。"""
        saved = self.backend.get_clipboard()
        self.backend.set_clipboard(text)
        time.sleep(0.1)
        self.hotkey("ctrl", "v")
        time.sleep(0.25)
        if saved is not None:
            try:
                self.backend.set_clipboard(saved)
            except Exception:
                logger.warning("恢复剪贴板失败", exc_info=True)


def activate_window(title_substring: str) -> bool:
    """按窗口标题把浏览器窗口置顶并最大化（README §8 唯一一处窗口级 API）。"""
    if not title_substring:
        return False
    if not IS_WIN:
        logger.info("[dev] 跳过窗口激活: %s", title_substring)
        return False
    try:
        import win32con
        import win32gui
    except ImportError:
        logger.warning("pywin32 未安装，无法激活窗口")
        return False

    matches: list[int] = []

    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and title_substring in title:
                matches.append(hwnd)
        return True

    win32gui.EnumWindows(_cb, None)
    if not matches:
        logger.warning("未找到标题包含 %r 的窗口", title_substring)
        return False
    hwnd = matches[0]
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            # 前台限制：先发一个 ALT 事件再试一次
            ctl = _Win32Backend()
            ctl.key_down(VK["alt"])
            ctl.key_up(VK["alt"])
            win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        logger.warning("激活窗口失败", exc_info=True)
        return False
