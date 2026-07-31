"""自动登录流程（README §5，带开关）。

登录的 UI 交互（输用户名 → 输密码 → 识别图片验证码 → 点登录）定义在内置流程
"login"（flows.yaml，流程编排页可视化编辑/重新框选）；验证码识别在
app/captcha.py（算术题或英文字母/数字 OCR），作为登录流程里的一个输入步骤执行。
本类只保留登录特有的包装：凭据检查、登录后等待主页面出现、失败重试与告警。
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class LoginFlow:
    MAIN_PAGE_TIMEOUT = 15.0   # 登录后等待主页面特征出现
    RETRIES = 1                # 失败重试次数（README：失败重试 1 次后告警）

    def __init__(self, settings, vision, inputs, notify=None, run_flow=None):
        self.settings = settings
        self.vision = vision
        self.inputs = inputs
        self.notify = notify or (lambda title, msg: None)
        self._run_flow = run_flow          # Engine.run_flow，执行内置流程 "login"

    def login(self) -> bool:
        """执行自动登录，成功（主页面出现）返回 True。"""
        try:
            creds = self.settings.get_credentials()
        except RuntimeError as e:
            logger.error("%s", e)
            creds = None
        if not creds:
            self.notify("自动登录失败", "未配置登录凭据，请在配置页「登录凭据」中设置")
            return False

        for attempt in range(1, self.RETRIES + 2):
            logger.info("自动登录：第 %d 次尝试", attempt)
            if self._do_login() and self._wait_main_page():
                logger.info("自动登录成功")
                return True
            logger.warning("自动登录第 %d 次尝试失败", attempt)
        self.notify("自动登录失败", "已重试仍未进入主页面，请人工检查")
        return False

    def _do_login(self) -> bool:
        try:
            self._run_flow("login")
        except Exception as e:
            logger.warning("登录流程执行失败: %s", e)
            if "验证码" in str(e):
                self.notify("需要人工处理验证码", str(e))
            return False
        return True

    def _wait_main_page(self) -> bool:
        deadline = time.monotonic() + self.MAIN_PAGE_TIMEOUT
        while time.monotonic() < deadline:
            if self.vision.page_state() == "main":
                return True
            time.sleep(1.0)
        return False
