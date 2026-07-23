"""反应式派发规则（截屏 → 匹配命中谁 → 执行谁）。

替代原先「先判页面状态、再走固定顺序」的硬编码分支。派发器每轮只截一次屏，
按优先级从高到低匹配规则，命中第一条即执行其处理器，处理完重新截屏再派发。

分层，兼顾灵活与安全：
- **反应式的是「现在屏幕是什么，就做对应的事」这一层**：登录页、验证码、报错弹窗、
  会话过期都成为一条独立规则——新增一种情况 = 加一条规则，不动派发骨架。
- **主线动作（登录、打印）仍是有序流程**（在规则的处理器里调 engine.run_flow），
  保证「先输车号再点打印」不会被打乱、不乱点。

安全上沿用 locators.py 的「宁可失败不点错」：多规则同屏命中时靠**优先级显式消歧**
（不猜）；没有任何规则命中时 F5 刷新一轮，仍无命中则失败并快照，绝不盲目操作。

优先级数字越小越先匹配。规则是程序逻辑（同 locators.py），不进配置文件；
装机时按目标系统在此增删规则即可。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from . import locators as L

if TYPE_CHECKING:
    from .engine import Engine, Task
    from .vision import Screenshot


class Outcome(Enum):
    """规则处理器的结果。未命中不用本枚举表示——detect 返回 False 即跳过该规则。"""

    DONE = "done"        # 任务完成，结束派发
    HANDLED = "handled"  # 处理了一步（有进展），重新截屏继续派发


@dataclass
class Rule:
    """一条反应式规则：detect 命中屏幕则执行 act。

    priority：越小越先匹配（同屏多命中时用于确定性消歧）。
    about：给「运行实况」展示 & 当文档的一句话说明。
    """

    name: str
    title: str
    priority: int
    detect: Callable[["Engine", "Screenshot"], bool]
    act: Callable[["Engine", "Task", dict], Outcome]
    about: str = ""


def _has_any(features):
    """屏幕上出现 features 中任一特征即命中（宽松存在性，同 page_state）。"""
    def detect(engine: "Engine", shot: "Screenshot") -> bool:
        return any(engine.vision.exists(f, shot) for f in features)
    return detect


def default_rules() -> list[Rule]:
    """默认规则集（对应 README §3 的页面状态分支）。

    顺序沿用原 page_state 语义：主页面优先于登录页（先判 main 再判 login），
    避免同时具备两类特征时的歧义。
    """

    def act_main(engine: "Engine", task: "Task", ctx: dict) -> Outcome:
        engine._run_main_task(task, ctx)   # 执行任务流程（有序），完成即终态
        return Outcome.DONE

    def act_login(engine: "Engine", task: "Task", ctx: dict) -> Outcome:
        engine._handle_login()             # 自动登录或等待人工登录，返回即已登录
        return Outcome.HANDLED             # 重新截屏，下一轮应命中主页面规则

    return [
        Rule("main_page", "主页面", 10, _has_any(L.MAIN_PAGE_FEATURES), act_main,
             about="出现主页面特征（自编号输入框 / 主页模板）→ 执行任务流程"),
        Rule("login_page", "登录页", 20, _has_any(L.LOGIN_PAGE_FEATURES), act_login,
             about="出现登录页特征（密码框 / 登录页模板）→ 自动或人工登录"),
    ]
