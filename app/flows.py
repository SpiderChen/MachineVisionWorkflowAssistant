"""工作流编排：流程/步骤数据模型、flows.yaml 存取、框选→定位器自动推导。

设计目标是「用户零手写配置」：在流程编排页截屏 → 框选目标 → 下拉选动作即可。
锚点词、ROI 收窄、旁侧标签借用、模板降级、点击点比例偏移全部由 derive_locator()
从框选区域自动推导，并在标注图上跑一次真实 vision.locate() 自检；推导失败自动
降级为图像模板（框选区域裁剪即模板，天然唯一）。

存储：flows.yaml（内部格式，用户不直接编辑）+ flows_assets/ 标注底图
（活文档 + 回归验证素材）+ templates/step_*.png 自动裁剪的模板降级素材。
内置流程 print_order / login 不可删除，步骤引用 locators.py 注册元素；
现场重新框选某一步后即转为本地标注定位（无需发版）。
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

from . import locators as L
from .locators import Match, Pick, TemplateLocator, TextLocator
from .settings import ROOT_DIR, TEMPLATES_DIR

logger = logging.getLogger(__name__)

FLOWS_PATH = ROOT_DIR / "flows.yaml"
SHOTS_DIR = ROOT_DIR / "flows_assets"

ACTION_LABELS = {
    "click": "点击",
    "double_click": "双击",
    "right_click": "右键点击",
    "input": "输入",
    "key": "按键",
    "wait": "等待出现",
    "sleep": "等待",
}
# 需要框选目标的动作（key/sleep 无目标）
LOCATOR_ACTIONS = ("click", "double_click", "right_click", "input", "wait")
VALUE_SOURCES = {"vehicle_no": "车辆自编号", "username": "用户名",
                 "password": "密码", "captcha": "图片验证码", "fixed": "固定文本"}
CAPTCHA_MODES = {"arith": "算术题", "alnum": "英文字母/数字"}
KEY_CHOICES = ("enter", "tab", "esc", "f5", "delete", "backspace", "home", "end")


class FlowError(Exception):
    pass


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


# =====================================================================
# 数据模型
# =====================================================================

@dataclass
class StepLocator:
    """框选自动推导出的定位规格（步骤私有；builtin 步骤用 ref 引用注册元素）。"""

    kind: str = "text"                      # text / template
    anchor: str = ""                        # kind=text 的 OCR 锚点词
    match: str = "exact"                    # exact / contains
    template: str = ""                      # templates/ 下文件名（模板主定位或 T1 降级）
    roi: list | None = None                 # (l,t,r,b) 屏幕百分比，自动收窄用
    pick: str = ""                          # "" / nearest / first / topmost
    offset_ratio: list = field(default_factory=lambda: [0.0, 0.0])       # 相对锚框
    tmpl_offset_ratio: list = field(default_factory=lambda: [0.0, 0.0])  # 相对模板框

    def to_locator(self, name: str = "") -> TextLocator | TemplateLocator:
        roi = tuple(self.roi) if self.roi else None
        tmpl = None
        if self.template:
            tmpl = TemplateLocator(
                image=self.template, roi=roi,
                click_offset_ratio=tuple(self.tmpl_offset_ratio),
                name=f"{name}.tmpl")
        if self.kind == "template":
            if tmpl is None:
                raise FlowError(f"步骤定位缺少模板图: {name}")
            return tmpl
        match = {
            "exact": Match.EXACT,
            "contains": Match.CONTAINS,
            "arith": Match.ARITH,
        }.get(self.match, Match.EXACT)
        return TextLocator(
            anchor=self.anchor,
            match=match,
            roi=roi,
            pick=Pick(self.pick) if self.pick else None,
            click_offset_ratio=tuple(self.offset_ratio),
            fallback=tmpl,
            name=name,
        )

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "offset_ratio": list(self.offset_ratio),
             "tmpl_offset_ratio": list(self.tmpl_offset_ratio)}
        if self.kind == "text":
            d["anchor"] = self.anchor
            d["match"] = self.match
        if self.template:
            d["template"] = self.template
        if self.roi:
            d["roi"] = [round(float(v), 4) for v in self.roi]
        if self.pick:
            d["pick"] = self.pick
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StepLocator":
        return cls(
            kind=d.get("kind", "text"), anchor=d.get("anchor", ""),
            match=d.get("match", "exact"), template=d.get("template", ""),
            roi=list(d["roi"]) if d.get("roi") else None, pick=d.get("pick", ""),
            offset_ratio=list(d.get("offset_ratio", [0.0, 0.0])),
            tmpl_offset_ratio=list(d.get("tmpl_offset_ratio", [0.0, 0.0])),
        )


@dataclass
class Step:
    action: str = "click"
    id: str = field(default_factory=_new_id)
    name: str = ""                          # 空则由 label() 自动生成
    ref: str = ""                           # 引用 locators.ALL 注册元素（内置流程）
    locator: StepLocator | None = None      # 框选推导的定位（与 ref 二选一）
    captcha_locator: StepLocator | None = None  # 算术验证码图片上的动态算式锚点
    # input 专用
    value_source: str = "fixed"             # fixed / vehicle_no / username / password
    text: str = ""                          # value_source=fixed 时的文本
    clear_first: bool = True                # 输入前 Ctrl+A + Delete 清空
    verify: bool = True                     # OCR 回读校验（password 强制跳过）
    captcha_mode: str = "arith"             # value_source=captcha: arith / alnum
    # key / sleep / wait 专用
    key: str = "enter"
    seconds: float = 1.0
    timeout: float = 0.0                    # 0 = 用全局 step_timeout
    optional: bool = False                  # 定位失败跳过而非中止任务
    # 验证码（value_source=captcha）：图片区域，相对定位命中框的比例表示
    captcha_box: list | None = None         # 标注底图上的像素区域（编辑回显用）
    captcha_ratio: list | None = None       # (l,t,r,b) 相对命中框中心，单位=命中框宽/高
    # 标注素材（活文档 + 回归验证）
    shot: str = ""                          # flows_assets/ 下标注底图文件名
    box: list | None = None                 # 框选区域（底图像素坐标 l,t,r,b）
    click: list | None = None               # 点击点（底图像素坐标）

    # ---- 展示 ----

    def target_text(self) -> str:
        if self.locator is not None:
            if self.locator.kind == "text" and self.locator.anchor:
                return f"「{self.locator.anchor}」"
            return "图像模板"
        if self.ref:
            return self.ref
        return "当前焦点"

    def label(self) -> str:
        if self.name:
            return self.name
        a = self.action
        if a == "input":
            src = VALUE_SOURCES.get(self.value_source, self.value_source)
            what = f"“{self.text}”" if self.value_source == "fixed" else src
            return f"输入{what} → {self.target_text()}"
        if a == "key":
            return f"按键 {self.key.upper()}"
        if a == "sleep":
            return f"等待 {self.seconds:g} 秒"
        if a == "wait":
            return f"等待 {self.target_text()} 出现"
        return f"{ACTION_LABELS.get(a, a)}{self.target_text()}"

    # ---- 定位 ----

    def build_locator(self):
        """返回可交给 vision.locate 的定位器；无目标动作返回 None。"""
        if self.action not in LOCATOR_ACTIONS:
            return None
        if self.locator is not None:
            return self.locator.to_locator(f"step:{self.label()}")
        if self.ref:
            loc = L.ALL.get(self.ref)
            if loc is None:
                raise FlowError(f"步骤引用的注册元素不存在: {self.ref}")
            return loc
        if self.action == "input":
            return None                     # 无定位的输入 = 打给当前焦点
        raise FlowError(f"步骤「{self.label()}」缺少定位（未框选目标）")

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "action": self.action}
        if self.name:
            d["name"] = self.name
        if self.ref:
            d["ref"] = self.ref
        if self.locator is not None:
            d["locator"] = self.locator.to_dict()
        if (self.captcha_locator is not None and self.value_source == "captcha"
                and self.captcha_mode == "arith"):
            d["captcha_locator"] = self.captcha_locator.to_dict()
        if self.action == "input":
            d.update(value_source=self.value_source, clear_first=self.clear_first,
                     verify=self.verify)
            if self.value_source == "fixed":
                d["text"] = self.text
            if self.value_source == "captcha":
                d["captcha_mode"] = self.captcha_mode
            if self.captcha_box:
                d["captcha_box"] = [int(v) for v in self.captcha_box]
            if self.captcha_ratio:
                d["captcha_ratio"] = [round(float(v), 3) for v in self.captcha_ratio]
        if self.action == "key":
            d["key"] = self.key
        if self.action == "sleep":
            d["seconds"] = self.seconds
        if self.timeout:
            d["timeout"] = self.timeout
        if self.optional:
            d["optional"] = True
        if self.shot:
            d["shot"] = self.shot
        if self.box:
            d["box"] = [int(v) for v in self.box]
        if self.click:
            d["click"] = [int(v) for v in self.click]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        step = cls(
            action=d.get("action", "click"), id=d.get("id") or _new_id(),
            name=d.get("name", ""), ref=d.get("ref", ""),
            locator=StepLocator.from_dict(d["locator"]) if d.get("locator") else None,
            captcha_locator=(StepLocator.from_dict(d["captcha_locator"])
                             if d.get("captcha_locator") else None),
            value_source=d.get("value_source", "fixed"), text=d.get("text", ""),
            clear_first=bool(d.get("clear_first", True)),
            verify=bool(d.get("verify", True)),
            captcha_mode=(d.get("captcha_mode", "arith")
                          if d.get("captcha_mode", "arith") in CAPTCHA_MODES else "arith"),
            key=d.get("key", "enter"), seconds=float(d.get("seconds", 1.0)),
            timeout=float(d.get("timeout", 0.0)),
            optional=bool(d.get("optional", False)),
            captcha_box=list(d["captcha_box"]) if d.get("captcha_box") else None,
            captcha_ratio=list(d["captcha_ratio"]) if d.get("captcha_ratio") else None,
            shot=d.get("shot", ""),
            box=list(d["box"]) if d.get("box") else None,
            click=list(d["click"]) if d.get("click") else None,
        )
        # 旧版会把标注当时的具体题目（如 0*9=?）保存为 EXACT。
        # 加载旧 flows.yaml 时自动升级为动态算式匹配，无需用户删除重配。
        if (step.action == "input" and step.value_source == "captcha"
                and step.captcha_mode == "arith"
                and step.locator is not None and step.locator.kind == "text"
                and step.locator.match == "exact"):
            from .captcha import parse_arith
            if parse_arith(step.locator.anchor) is not None:
                step.locator.match = "arith"
        if step.captcha_mode != "arith":
            step.captcha_locator = None
        return step


@dataclass
class Flow:
    key: str
    title: str
    steps: list[Step] = field(default_factory=list)
    builtin: bool = False                   # 内置流程不可删除

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "builtin": self.builtin,
                "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Flow":
        return cls(key=d["key"], title=d.get("title", d["key"]),
                   builtin=bool(d.get("builtin", False)),
                   steps=[Step.from_dict(s) for s in d.get("steps", [])])


# =====================================================================
# 内置默认流程（与 README §3/§5 的手写流程一一对应）
# =====================================================================

def default_flows() -> dict[str, Flow]:
    print_order = Flow(key="print_order", title="出货单打印", builtin=True, steps=[
        Step(action="input", ref="VEHICLE_INPUT", value_source="vehicle_no",
             name="输入车辆自编号（自动先点击输入框）"),
        Step(action="click", ref="PRINT_BUTTON", name="点击打印按钮"),
    ])
    login = Flow(key="login", title="自动登录", builtin=True, steps=[
        Step(action="input", ref="USERNAME_INPUT", value_source="username",
             verify=False, name="输入用户名"),
        Step(action="input", ref="PASSWORD_INPUT", value_source="password",
             verify=False, name="输入密码"),
        # 图片验证码：默认算术模式，也可在流程编排页改为英文字母/数字；
        # optional——没有验证码的站点自动跳过；
        # 图片区域需在流程编排页框选一次（相对输入框比例，跨分辨率）
        Step(action="input", ref="CAPTCHA_INPUT", value_source="captcha",
             verify=False, optional=True, timeout=3.0,
             name="识别并输入图片验证码（如无验证码自动跳过）"),
        Step(action="click", ref="LOGIN_BUTTON", name="点击登录按钮"),
    ])
    return {f.key: f for f in (print_order, login)}


class FlowStore:
    """flows.yaml 读写。内置流程缺失时自动补齐默认定义。"""

    def __init__(self, path: Path = FLOWS_PATH):
        self.path = Path(path)
        self.flows: dict[str, Flow] = {}

    @classmethod
    def load(cls, path: Path = FLOWS_PATH) -> "FlowStore":
        store = cls(path)
        SHOTS_DIR.mkdir(parents=True, exist_ok=True)
        raw = {}
        if store.path.exists():
            try:
                raw = yaml.safe_load(store.path.read_text(encoding="utf-8")) or {}
            except Exception:
                logger.exception("读取 %s 失败，使用内置默认流程", store.path)
        for d in raw.get("flows", []):
            try:
                f = Flow.from_dict(d)
                store.flows[f.key] = f
            except Exception:
                logger.exception("流程定义解析失败，已跳过: %r", d)
        for key, f in default_flows().items():
            if key not in store.flows:
                store.flows[key] = f
            else:
                store.flows[key].builtin = True
        return store

    def save(self) -> None:
        data = {"flows": [f.to_dict() for f in self.flows.values()]}
        self.path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        logger.info("流程定义已保存到 %s", self.path)

    def get(self, key: str) -> Flow:
        f = self.flows.get(key)
        if f is None:
            raise FlowError(f"流程不存在: {key}")
        return f

    def create(self, title: str) -> Flow:
        key = f"flow_{_new_id()}"
        f = Flow(key=key, title=title or "未命名流程")
        self.flows[key] = f
        return f

    def delete(self, key: str) -> None:
        f = self.get(key)
        if f.builtin:
            raise FlowError("内置流程不可删除（可重新框选各步骤覆盖定位）")
        del self.flows[key]


# =====================================================================
# 框选 → 定位器自动推导
# =====================================================================

def _intersect_area(a, b) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _overlap_ratio(inner, outer) -> float:
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1)
    return _intersect_area(inner, outer) / area


def _nearest_label(lines, box, W, H):
    """框内无文字时，找左侧/上方最近的文字标签作锚点（输入框的典型形态）。"""
    l, t, r, b = box
    best = None       # (加权距离, Candidate)
    for c in lines:
        if not (c.text or "").strip():
            continue
        v_overlap = min(b, c.box[3]) - max(t, c.box[1])
        h_overlap = min(r, c.box[2]) - max(l, c.box[0])
        if c.box[2] <= l + 8 and v_overlap > 0:            # 左侧且垂直有交叠
            gap = l - c.box[2]
            if gap <= 0.25 * W:
                score = gap
                if best is None or score < best[0]:
                    best = (score, c)
        elif c.box[3] <= t + 8 and h_overlap > 0:          # 上方且水平有交叠
            gap = t - c.box[3]
            if gap <= 0.12 * H:
                score = gap * 1.5                          # 左侧标签优先于上方
                if best is None or score < best[0]:
                    best = (score, c)
    return best[1] if best else None


def _expand_roi(boxes, W, H, margin: float = 0.6) -> list:
    """取若干框的并集，按并集尺寸外扩 margin 倍，转屏幕百分比 ROI。"""
    l = min(b[0] for b in boxes); t = min(b[1] for b in boxes)
    r = max(b[2] for b in boxes); btm = max(b[3] for b in boxes)
    mw = (r - l) * margin
    mh = (btm - t) * margin
    return [round(max(0.0, (l - mw) / W), 4), round(max(0.0, (t - mh) / H), 4),
            round(min(1.0, (r + mw) / W), 4), round(min(1.0, (btm + mh) / H), 4)]


def relative_box_ratio(box, reference) -> list[float]:
    """把区域换算成相对定位候选框中心、宽高的比例。"""
    l, t, r, b = (float(v) for v in box)
    return [round((l - reference.cx) / max(reference.w, 1), 3),
            round((t - reference.cy) / max(reference.h, 1), 3),
            round((r - reference.cx) / max(reference.w, 1), 3),
            round((b - reference.cy) / max(reference.h, 1), 3)]


def derive_dynamic_arith_locator(vision, shot, captcha_box, click_point,
                                 base: StepLocator | None = None):
    """用用户框选的蓝色区域生成「动态算式锚点」。

    anchor 只保留当次题目作为界面说明；运行时 Match.ARITH 会匹配
    任意可解析的「数字 运算符 数字」，不要求题目文字相同。
    返回 (StepLocator, OCR Candidate)；蓝框内无完整算式时返回 None。
    """
    from .captcha import parse_arith

    l, t, r, b = (int(v) for v in captcha_box)
    lines = vision.ocr_lines(shot)
    cands = [c for c in lines
             if parse_arith(c.text or "") is not None
             and _overlap_ratio(c.box, (l, t, r, b)) >= 0.35]
    if not cands:
        return None
    chosen = max(cands, key=lambda c: _intersect_area(c.box, (l, t, r, b)))
    cx, cy = (int(click_point[0]), int(click_point[1]))
    H, W = shot.img.shape[:2]
    sl = StepLocator(
        kind="text", anchor=(chosen.text or "").strip(), match="arith",
        roi=_expand_roi([(l, t, r, b)], W, H, margin=1.5),
        pick="nearest",
        offset_ratio=[round((cx - chosen.cx) / max(chosen.w, 1), 3),
                      round((cy - chosen.cy) / max(chosen.h, 1), 3)],
        template=base.template if base is not None else "",
        tmpl_offset_ratio=(list(base.tmpl_offset_ratio)
                           if base is not None else [0.0, 0.0]),
    )
    return sl, chosen


def derive_locator(vision, shot, box, click_point, step_id) -> tuple[StepLocator, list[str]]:
    """从框选区域自动推导定位器，并在标注图上自检。

    box/click_point 为截屏原图坐标；模板裁剪图自动存 templates/step_<id>.png。
    返回 (StepLocator, 推导说明列表)——说明直接展示给用户，讲清系统替他做了什么。
    """
    from .vision import _norm

    report: list[str] = []
    H, W = shot.img.shape[:2]
    l, t, r, b = (int(v) for v in box)
    cx, cy = (int(click_point[0]), int(click_point[1]))

    tmpl_name = f"step_{step_id}.png"
    cv2.imwrite(str(TEMPLATES_DIR / tmpl_name), shot.img[t:b, l:r])
    tmpl_ratio = [round((cx - (l + r) / 2) / max(r - l, 1), 3),
                  round((cy - (t + b) / 2) / max(b - t, 1), 3)]

    def template_only(reason: str) -> StepLocator:
        report.append(reason)
        report.append("已裁剪框选区域作为图像模板（T2 匹配）")
        return StepLocator(kind="template", template=tmpl_name,
                           tmpl_offset_ratio=tmpl_ratio)

    lines = None
    try:
        lines = vision.ocr_lines(shot)
    except Exception as e:
        return _self_check(vision, shot, box, click_point, report,
                           template_only(f"OCR 不可用（{e}）"))

    # ① 框内文字优先作锚点；② 否则借用左侧/上方最近标签（输入框典型形态）
    in_box = [c for c in lines if (c.text or "").strip()
              and _overlap_ratio(c.box, (l, t, r, b)) >= 0.5]
    anchor_c = max(in_box, key=lambda c: _intersect_area(c.box, (l, t, r, b))) \
        if in_box else _nearest_label(lines, (l, t, r, b), W, H)
    if anchor_c is None or not _norm(anchor_c.text or ""):
        return _self_check(vision, shot, box, click_point, report,
                           template_only("框选区域内及附近未识别到可用文字"))

    anchor_text = (anchor_c.text or "").strip()
    sl = StepLocator(
        kind="text", anchor=anchor_text, match="exact", template=tmpl_name,
        offset_ratio=[round((cx - anchor_c.cx) / max(anchor_c.w, 1), 3),
                      round((cy - anchor_c.cy) / max(anchor_c.h, 1), 3)],
        tmpl_offset_ratio=tmpl_ratio,
    )
    if in_box:
        report.append(f"框内识别到文字「{anchor_text}」→ 文字锚点定位（T1，跨分辨率）")
    else:
        report.append(f"框内无文字，借用旁边标签「{anchor_text}」定位，"
                      f"点击点按相对位置自动换算")

    # ③ 同名文字消歧：出现多处 → 自动限定搜索区域，仍多则取最近者
    norm_anchor = _norm(anchor_text)
    dup = [c for c in lines if _norm(c.text or "") == norm_anchor]
    if len(dup) > 1:
        sl.roi = _expand_roi([(l, t, r, b), anchor_c.box], W, H)
        in_roi = [c for c in dup
                  if sl.roi[0] * W <= c.cx <= sl.roi[2] * W
                  and sl.roi[1] * H <= c.cy <= sl.roi[3] * H]
        if len(in_roi) > 1:
            sl.pick = "nearest"
        report.append(f"「{anchor_text}」在页面出现 {len(dup)} 处 → 已自动限定搜索区域"
                      + ("并取区域中心最近者" if len(in_roi) > 1 else ""))
    report.append("已同时保存图像模板作为文字定位失败时的自动降级")

    return _self_check(vision, shot, box, click_point, report, sl)


def _self_check(vision, shot, box, click_point, report,
                sl: StepLocator) -> tuple[StepLocator, list[str]]:
    """在标注底图上真跑一次定位，验证命中点落在用户框选处；文字锚点不准则降级模板。"""
    l, t, r, b = box
    tol_x = max(12, (r - l) // 2 + 6)
    tol_y = max(12, (b - t) // 2 + 6)

    def hit(res) -> bool:
        return bool(res.ok and res.click_point
                    and abs(res.click_point[0] - click_point[0]) <= tol_x
                    and abs(res.click_point[1] - click_point[1]) <= tol_y)

    try:
        res = vision.locate(sl.to_locator("标注自检"), shot)
    except Exception as e:
        report.append(f"⚠ 自检异常: {e}")
        return sl, report

    if hit(res):
        report.append("自检 ✔：在标注图上定位命中，与框选位置一致")
        return sl, report

    if sl.kind == "text" and sl.template:
        degraded = StepLocator(kind="template", template=sl.template,
                               tmpl_offset_ratio=sl.tmpl_offset_ratio)
        try:
            res2 = vision.locate(degraded.to_locator("标注自检(模板)"), shot)
        except Exception:
            res2 = None
        if res2 is not None and hit(res2):
            report.append("⚠ 文字锚点自检未命中框选位置，已自动改用图像模板定位")
            return degraded, report

    reason = res.reason if not res.ok else "命中位置偏离框选区域"
    report.append(f"⚠ 自检未通过（{reason}），请重新框选或先「重新截屏」后再试")
    return sl, report
