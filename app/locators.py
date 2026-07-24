"""元素定位定义（README §2.1，声明式，随 Git 版本管理）。

定位是程序逻辑，不进配置文件；config.yaml 只留运行参数。
每个元素独立指定定位方式：T1 OCR 文字锚点（默认，跨分辨率），T2 模板匹配（无文字元素
或 T1 失败时的降级 fallback）。同名文字用「消歧漏斗」收窄：
长锚词 → ROI 搜索域 → 空间关系约束 → 多候选裁决（默认 abort，宁可失败不点错）。

装机时按目标系统的实际文案/布局调整本文件，然后在配置页「运行实况」逐步执行动作验证。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .settings import TEMPLATES_DIR


class Match(Enum):
    EXACT = "exact"        # 归一化后全匹配（去空白、去尾部冒号）
    CONTAINS = "contains"  # 锚点词包含于识别文本


class Direction(Enum):
    RIGHT_OF = "right-of"
    LEFT_OF = "left-of"
    ABOVE = "above"
    BELOW = "below"
    SAME_ROW = "same-row"


class Pick(Enum):
    NEAREST = "nearest"    # 距参照锚点（无参照则 ROI/屏幕中心）最近
    FIRST = "first"        # OCR 阅读顺序第一个
    TOPMOST = "topmost"    # 最靠上者


class OnAmbiguous(Enum):
    ABORT = "abort"        # 多候选且无裁决策略：记快照、报告警，不误点


@dataclass
class Relation:
    """空间关系约束：相对另一个可唯一定位的锚点。"""

    ref: "TextLocator | TemplateLocator"
    direction: Direction
    max_distance: int | None = None    # 候选中心与参照中心的最大像素距离


@dataclass
class TemplateLocator:
    """T2：多尺度模板匹配。模板图存 templates/，现场可「重新截取」更新而无需发版。"""

    image: str                                     # templates/ 下的文件名
    threshold: float = 0.85
    roi: tuple[float, float, float, float] | None = None   # (l, t, r, b) 屏幕百分比
    click_offset: tuple[int, int] = (0, 0)
    click_offset_ratio: tuple[float, float] | None = None  # 相对命中框宽/高的比例偏移
    scales: tuple[float, float, float] = (0.5, 1.5, 0.05)  # (起, 止, 步长)
    name: str = ""

    @property
    def path(self) -> Path:
        return TEMPLATES_DIR / self.image


@dataclass
class TextLocator:
    """T1：OCR 文字锚点。定位到锚点文字框，加 click_offset 得到点击点。"""

    anchor: str
    match: Match = Match.EXACT
    click_offset: tuple[int, int] = (0, 0)
    click_offset_ratio: tuple[float, float] | None = None  # 相对锚框宽/高的比例偏移
    roi: tuple[float, float, float, float] | None = None   # (l, t, r, b) 屏幕百分比
    relation: Relation | None = None
    pick: Pick | None = None                       # 约束后仍多候选时的裁决策略
    on_ambiguous: OnAmbiguous = OnAmbiguous.ABORT  # 无裁决策略时的默认行为
    fallback: TemplateLocator | None = None        # T1 失败自动降级尝试的 T2 模板
    name: str = ""


# =====================================================================
# 主页面元素
# =====================================================================

# 车辆自编号输入框：定位「自编号」标签文字，点击其右侧输入框。
# 实际页面若标签为「车辆自编号：」，CONTAINS 亦命中；装机时按实际文案调整。
VEHICLE_INPUT = TextLocator(
    anchor="自编号",
    match=Match.CONTAINS,
    click_offset=(120, 0),
    fallback=TemplateLocator("vehicle_input.png"),
)

# 打印按钮：表头/行内/菜单可能都有「打印」——用空间关系约束到
# 「与自编号输入框同一行的那个」，约束后仍多候选取最近者。
PRINT_BUTTON = TextLocator(
    anchor="打印",
    match=Match.CONTAINS,
    relation=Relation(ref=VEHICLE_INPUT, direction=Direction.SAME_ROW),
    pick=Pick.NEAREST,
    on_ambiguous=OnAmbiguous.ABORT,
    fallback=TemplateLocator("print_button.png"),
)

# 打印成功特征（可选，engine.verify_success 开启且模板图存在时才校验）
PRINT_SUCCESS = TemplateLocator("print_success.png", threshold=0.85)

# =====================================================================
# 登录页元素（README §5）
# =====================================================================

USERNAME_INPUT = TextLocator(
    anchor="用户名",
    match=Match.CONTAINS,
    click_offset=(140, 0),
    fallback=TemplateLocator("username_input.png"),
)

PASSWORD_INPUT = TextLocator(
    anchor="密码",
    match=Match.EXACT,               # EXACT 避免命中「确认密码」「修改密码」等
    click_offset=(140, 0),
    fallback=TemplateLocator("password_input.png"),
)

CAPTCHA_INPUT = TextLocator(
    anchor="验证码",
    match=Match.CONTAINS,
    click_offset=(140, 0),
    fallback=TemplateLocator("captcha_input.png"),
)

LOGIN_BUTTON = TextLocator(
    anchor="登录",
    match=Match.EXACT,               # EXACT 排除「用户登录」标题等
    fallback=TemplateLocator("login_button.png"),
)

# =====================================================================
# 页面状态特征（README §3 状态判定；exists 命中任一即判定为该状态）
# =====================================================================

LOGIN_PAGE_TMPL = TemplateLocator("login_page.png", threshold=0.80)
MAIN_PAGE_TMPL = TemplateLocator("main_page.png", threshold=0.80)

MAIN_PAGE_FEATURES = (VEHICLE_INPUT, MAIN_PAGE_TMPL)
LOGIN_PAGE_FEATURES = (PASSWORD_INPUT, LOGIN_PAGE_TMPL)

# =====================================================================
# 注册表：变量名即元素名（配置页「运行实况」的动作测试下拉中列出）
# =====================================================================

ALL: dict[str, TextLocator | TemplateLocator] = {}
for _name, _obj in list(globals().items()):
    if isinstance(_obj, (TextLocator, TemplateLocator)):
        _obj.name = _name
        ALL[_name] = _obj
for _name, _obj in ALL.items():
    if isinstance(_obj, TextLocator) and _obj.fallback is not None:
        _obj.fallback.name = f"{_name}.fallback"
del _name, _obj
