"""截屏 + 元素定位（README §2.1 三级策略的 T1/T2）+ 等待出现。

- 截屏：mss，限定配置指定的显示器
- T1：RapidOCR 文字锚点 + 消歧漏斗（锚词 → ROI → 空间关系 → 裁决）
- T2：多尺度模板匹配（0.5x~1.5x），T1 失败按元素配置自动降级
- 坐标：全部在截屏原图坐标系计算，to_screen() 换算为物理屏幕坐标
"""
from __future__ import annotations

import importlib
import logging
import math
import os
import re
import sys
import threading
import time
from hashlib import blake2b
from dataclasses import dataclass, field

from typing import Callable

import cv2
import mss
import numpy as np

from . import locators as L
from .locators import (
    Direction, Match, OnAmbiguous, Pick, TemplateLocator, TextLocator,
)

logger = logging.getLogger(__name__)


class VisionError(Exception):
    pass


_DLL_DIR_HANDLES: list[object] = []


def _prepare_onnx_dll_search_path() -> None:
    """让冻结后的 Windows 进程能从 ONNX Runtime 自己的目录加载依赖 DLL。"""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    root = getattr(sys, "_MEIPASS", "")
    capi_dir = os.path.join(root, "onnxruntime", "capi") if root else ""
    if not capi_dir or not os.path.isdir(capi_dir):
        return
    try:
        # 返回的句柄必须保活，否则目录会立刻从进程 DLL 搜索路径移除。
        _DLL_DIR_HANDLES.append(os.add_dll_directory(capi_dir))
    except (AttributeError, FileNotFoundError, OSError):
        logger.debug("无法追加 ONNX Runtime DLL 搜索目录: %s", capi_dir,
                     exc_info=True)


def _exception_detail(exc: BaseException) -> str:
    detail = " ".join(str(exc).split()) or type(exc).__name__
    return detail[:300]


def _ocr_load_error(package: str, exc: BaseException) -> VisionError:
    """把真实依赖/DLL 错误保留下来，避免一律误报为 RapidOCR 未安装。"""
    detail = _exception_detail(exc)
    lowered = detail.lower()
    if sys.platform == "win32" and (
            "dll" in lowered or "dynamic link library" in lowered):
        hint = (
            "ONNX Runtime DLL 加载失败；请安装 Microsoft Visual C++ "
            "2015-2022 Redistributable (x64)，或使用通过 OCR 自检的 Windows 完整包"
        )
    elif isinstance(exc, ModuleNotFoundError) and exc.name:
        hint = f"{package} 的依赖 {exc.name} 缺失"
        if not getattr(sys, "frozen", False):
            hint += "；请重新执行 python -m pip install -r requirements.txt"
    else:
        hint = f"{package} 加载失败"
    return VisionError(f"{hint}（{type(exc).__name__}: {detail}）")


def _missing_ocr_error() -> VisionError:
    if getattr(sys, "frozen", False):
        return VisionError("Windows 完整包缺少 RapidOCR 组件，请重新下载通过 OCR 自检的构建")
    return VisionError(
        "RapidOCR 未安装（python -m pip install "
        "rapidocr-onnxruntime==1.4.4 onnxruntime==1.20.1）")


def _load_rapidocr_class():
    """优先载入兼容版；仅当整个包不存在时才尝试 RapidOCR v2。"""
    _prepare_onnx_dll_search_path()
    try:
        return importlib.import_module("rapidocr_onnxruntime").RapidOCR
    except ModuleNotFoundError as exc:
        if exc.name != "rapidocr_onnxruntime":
            logger.exception("rapidocr-onnxruntime 的内部依赖加载失败")
            raise _ocr_load_error("rapidocr-onnxruntime", exc) from exc
    except (ImportError, OSError, AttributeError) as exc:
        logger.exception("rapidocr-onnxruntime 加载失败")
        raise _ocr_load_error("rapidocr-onnxruntime", exc) from exc

    try:
        return importlib.import_module("rapidocr").RapidOCR
    except ModuleNotFoundError as exc:
        if exc.name == "rapidocr":
            raise _missing_ocr_error() from exc
        logger.exception("rapidocr 的内部依赖加载失败")
        raise _ocr_load_error("rapidocr", exc) from exc
    except (ImportError, OSError, AttributeError) as exc:
        logger.exception("rapidocr 加载失败")
        raise _ocr_load_error("rapidocr", exc) from exc


def _norm(s: str) -> str:
    """文本归一化：去空白（含全角空格）、去尾部冒号，用于锚点词匹配。"""
    s = re.sub(r"[\s　]+", "", s or "")
    return s.rstrip(":：")


def _text_matches(loc: TextLocator, text: str) -> bool:
    """判定 OCR 文本是否命中定位器。

    ARITH 不绑定标注时的具体题目，而是动态匹配任意可解析算式。
    """
    if loc.match == Match.ARITH:
        from .captcha import parse_arith
        return parse_arith(text or "") is not None
    t = _norm(text or "")
    anchor = _norm(loc.anchor)
    if loc.match == Match.EXACT:
        return t == anchor
    return anchor in t


@dataclass
class Candidate:
    box: tuple[int, int, int, int]      # (l, t, r, b) 截屏原图坐标
    score: float = 0.0
    text: str | None = None

    @property
    def cx(self) -> int:
        return (self.box[0] + self.box[2]) // 2

    @property
    def cy(self) -> int:
        return (self.box[1] + self.box[3]) // 2

    @property
    def w(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> int:
        return self.box[3] - self.box[1]


@dataclass
class Screenshot:
    img: np.ndarray                     # BGR
    left: int = 0                       # 所在显示器的物理偏移
    top: int = 0
    ts: float = 0.0
    ocr: list[Candidate] | None = None  # OCR 结果缓存（同一张截图只跑一次 OCR）
    ocr_regions: dict[tuple[int, int, int, int], list[Candidate]] = field(
        default_factory=dict)            # ROI OCR 缓存（候选框已转回全屏坐标）


@dataclass
class LocateResult:
    locator: object
    ok: bool = False
    chosen: Candidate | None = None
    click_point: tuple[int, int] | None = None   # 截屏原图坐标
    matches: list[Candidate] = field(default_factory=list)    # 初筛（锚词/模板）候选
    survivors: list[Candidate] = field(default_factory=list)  # 漏斗约束后的候选
    reason: str = ""
    used_fallback: bool = False
    shot: Screenshot | None = None

    def describe(self) -> str:
        name = getattr(self.locator, "name", "?") or "?"
        lines = [f"[{name}] {'✔ 命中' if self.ok else '✘ 失败'}"]
        lines.append(f"  初筛候选 {len(self.matches)} 个，约束后 {len(self.survivors)} 个")
        if self.chosen:
            lines.append(
                f"  选中 box={self.chosen.box} click={self.click_point} "
                f"score={self.chosen.score:.2f}"
                + (f" text={self.chosen.text!r}" if self.chosen.text else "")
            )
        if self.used_fallback:
            lines.append("  （T1 未命中，已降级使用 T2 模板）")
        if not self.ok and self.reason:
            lines.append(f"  原因: {self.reason}")
        if self.ok and len(self.survivors) > 1:
            lines.append("  ⚠ 多候选靠裁决策略选中——建议补 ROI 或空间关系约束使其唯一")
        return "\n".join(lines)


def draw_result(img: np.ndarray, res: LocateResult) -> np.ndarray:
    """在截图上标注定位结果：黄=初筛候选，橙=约束后候选，绿=最终选中，红点=点击点。"""
    for c in res.matches:
        cv2.rectangle(img, res_pt(c.box, 0), res_pt(c.box, 1), (0, 255, 255), 1)
    for c in res.survivors:
        cv2.rectangle(img, res_pt(c.box, 0), res_pt(c.box, 1), (0, 165, 255), 2)
    if res.chosen:
        c = res.chosen
        cv2.rectangle(img, res_pt(c.box, 0), res_pt(c.box, 1), (0, 200, 0), 2)
        name = getattr(res.locator, "name", "") or ""
        cv2.putText(img, name, (c.box[0], max(c.box[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    if res.click_point:
        cv2.circle(img, tuple(int(v) for v in res.click_point), 5, (0, 0, 255), -1)
    return img


def res_pt(box, which: int):
    return (int(box[0]), int(box[1])) if which == 0 else (int(box[2]), int(box[3]))


def click_point_of(loc, chosen: Candidate) -> tuple[int, int]:
    """命中框中心 + 偏移。比例偏移（相对命中框宽/高）优先于绝对像素偏移叠加。"""
    dx, dy = loc.click_offset
    ratio = getattr(loc, "click_offset_ratio", None)
    if ratio:
        dx += int(round(ratio[0] * chosen.w))
        dy += int(round(ratio[1] * chosen.h))
    return chosen.cx + dx, chosen.cy + dy


class Vision:
    def __init__(self, settings):
        self.settings = settings
        self._ocr_engine = None
        self._ocr_lock = threading.Lock()
        self._last_ocr_key: tuple | None = None
        self._last_ocr_result: list[Candidate] | None = None
        self._warmup_thread: threading.Thread | None = None
        self._tmpl_cache: dict[str, tuple[float, np.ndarray]] = {}

    def warm_up_async(self) -> None:
        """后台预热 OCR，避免第一个业务任务才支付模型初始化成本。"""
        if self._warmup_thread is not None and self._warmup_thread.is_alive():
            return

        def run() -> None:
            try:
                blank = np.full((64, 256, 3), 255, dtype=np.uint8)
                self._run_ocr(blank)
                logger.info("RapidOCR 后台预热完成")
            except Exception:
                # 预热失败不应阻止程序启动；真正识别时仍会返回明确错误。
                logger.warning("RapidOCR 后台预热失败", exc_info=True)

        self._warmup_thread = threading.Thread(
            target=run, name="ocr-warmup", daemon=True)
        self._warmup_thread.start()

    # ------------------------------------------------------------------
    # 截屏
    # ------------------------------------------------------------------

    def capture(self) -> Screenshot:
        idx = self.settings.engine.monitor
        with mss.mss() as sct:
            if idx >= len(sct.monitors):
                logger.warning("显示器编号 %s 不存在，回退主屏", idx)
                idx = 1
            mon = sct.monitors[idx]
            raw = sct.grab(mon)
            img = np.asarray(raw)[:, :, :3].copy()  # BGRA → BGR
        return Screenshot(img=img, left=mon["left"], top=mon["top"], ts=time.time())

    def to_screen(self, shot: Screenshot, point: tuple[int, int]) -> tuple[int, int]:
        """截屏原图坐标 → 物理屏幕坐标。"""
        return shot.left + int(point[0]), shot.top + int(point[1])

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _get_ocr(self):
        if self._ocr_engine is None:
            RapidOCR = _load_rapidocr_class()
            cfg = self.settings.engine
            cpu_count = os.cpu_count() or 1
            threads = int(getattr(cfg, "ocr_threads", 0) or 0)
            if threads <= 0:
                threads = min(cpu_count, 4)
            use_cls = bool(getattr(cfg, "ocr_use_angle_cls", False))
            kwargs = {
                "use_cls": use_cls,
                "intra_op_num_threads": max(1, threads),
                "inter_op_num_threads": 1,
            }
            logger.info("RapidOCR 初始化（CPU 线程=%d，方向分类=%s）...",
                        threads, "开" if use_cls else "关")
            t0 = time.time()
            try:
                try:
                    self._ocr_engine = RapidOCR(**kwargs)
                except (TypeError, ValueError, KeyError):
                    # 兼容不接受这些参数的新/旧 RapidOCR 变体。
                    logger.warning("RapidOCR 当前版本不支持性能参数，回退默认初始化",
                                   exc_info=True)
                    self._ocr_engine = RapidOCR()
            except Exception as exc:
                logger.exception("RapidOCR 初始化失败")
                raise VisionError(
                    f"RapidOCR 初始化失败（{type(exc).__name__}: "
                    f"{_exception_detail(exc)}）") from exc
            logger.info("RapidOCR 初始化完成，耗时 %.2fs", time.time() - t0)
        return self._ocr_engine

    @staticmethod
    def _image_key(img: np.ndarray) -> tuple:
        """给画面生成安全的精确指纹；只有像素完全相同才会复用 OCR。"""
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
        digest = blake2b(memoryview(img), digest_size=16).digest()
        return img.shape, img.dtype.str, digest

    def _run_ocr(self, img: np.ndarray, *, reuse_identical: bool = False) -> list[Candidate]:
        key = self._image_key(img) if reuse_identical else None
        with self._ocr_lock:
            if (key is not None and key == self._last_ocr_key
                    and self._last_ocr_result is not None):
                logger.debug("OCR 复用相同画面结果（%d 行）", len(self._last_ocr_result))
                return self._last_ocr_result
            out = self._get_ocr()(img)
        items: list[tuple] = []
        if isinstance(out, tuple):                     # rapidocr_onnxruntime: (result, elapse)
            items = list(out[0] or [])
        elif out is not None and hasattr(out, "boxes"):  # rapidocr v2: RapidOCROutput
            if out.boxes is not None:
                items = list(zip(out.boxes, out.txts, out.scores))
        cands = []
        for box4, text, score in items:
            xs = [p[0] for p in box4]
            ys = [p[1] for p in box4]
            cands.append(Candidate(
                box=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                score=float(score), text=str(text)))
        if key is not None:
            self._last_ocr_key = key
            self._last_ocr_result = cands
        return cands

    def ocr_lines(self, shot: Screenshot,
                  roi: tuple[float, float, float, float] | None = None) -> list[Candidate]:
        """读取全屏或 ROI 文字；ROI 候选框仍使用全屏坐标。"""
        if roi is None or shot.ocr is not None:
            if shot.ocr is None:
                t0 = time.time()
                shot.ocr = self._run_ocr(shot.img, reuse_identical=True)
                logger.debug("OCR 全屏 %d 行，耗时 %.2fs", len(shot.ocr), time.time() - t0)
            return shot.ocr

        H, W = shot.img.shape[:2]
        l, t, r, b = roi
        px = (max(0, int(l * W)), max(0, int(t * H)),
              min(W, int(r * W)), min(H, int(b * H)))
        if px in shot.ocr_regions:
            return shot.ocr_regions[px]
        x1, y1, x2, y2 = px
        if x2 - x1 < 4 or y2 - y1 < 4:
            shot.ocr_regions[px] = []
            return []
        t0 = time.time()
        local = self._run_ocr(shot.img[y1:y2, x1:x2], reuse_identical=True)
        translated = [Candidate(
            box=(c.box[0] + x1, c.box[1] + y1,
                 c.box[2] + x1, c.box[3] + y1),
            score=c.score, text=c.text) for c in local]
        shot.ocr_regions[px] = translated
        logger.debug("OCR 局部 %s %d 行，耗时 %.2fs", px, len(translated), time.time() - t0)
        return translated

    def ocr_image(self, img: np.ndarray) -> list[Candidate]:
        """对任意图片跑 OCR（验证码识别等独立小图场景）。"""
        return self._run_ocr(img)

    def read_region(self, shot: Screenshot, box: tuple[int, int, int, int]) -> str:
        """OCR 读取指定区域文本（流程④输入回读校验 / 读弹窗提示）。"""
        h, w = shot.img.shape[:2]
        l = max(0, int(box[0])); t = max(0, int(box[1]))
        r = min(w, int(box[2])); b = min(h, int(box[3]))
        if r - l < 4 or b - t < 4:
            return ""
        crop = shot.img[t:b, l:r]
        return " ".join(c.text or "" for c in self._run_ocr(crop, reuse_identical=True))

    # ------------------------------------------------------------------
    # 定位（T1 消歧漏斗 / T2 模板）
    # ------------------------------------------------------------------

    def locate(self, locator, shot: Screenshot | None = None) -> LocateResult:
        if shot is None:
            shot = self.capture()
        if isinstance(locator, TemplateLocator):
            return self._locate_template(locator, shot)
        return self._locate_text(locator, shot)

    def _locate_text(self, loc: TextLocator, shot: Screenshot) -> LocateResult:
        res = LocateResult(locator=loc, shot=shot)
        try:
            lines = self.ocr_lines(shot, loc.roi)
        except VisionError as e:
            res.reason = str(e)
            return self._try_fallback(loc, shot, res)

        matches = []
        for c in lines:
            if _text_matches(loc, c.text or ""):
                matches.append(c)
        res.matches = matches
        cands = list(matches)
        if not cands:
            res.reason = f"锚点词 {loc.anchor!r} 未在屏幕上命中"
            return self._try_fallback(loc, shot, res)

        # ② ROI 搜索域（屏幕百分比，跨分辨率成立）
        if loc.roi:
            cands = [c for c in cands if self._in_roi(c, loc.roi, shot)]
            if not cands:
                res.reason = f"ROI {loc.roi} 内无候选"
                return self._try_fallback(loc, shot, res)

        # ③ 空间关系约束
        ref: Candidate | None = None
        if loc.relation:
            ref_res = self.locate(loc.relation.ref, shot)
            if not ref_res.ok or ref_res.chosen is None:
                res.reason = (f"参照锚点 {getattr(loc.relation.ref, 'name', '?')} "
                              f"未定位到: {ref_res.reason}")
                return self._try_fallback(loc, shot, res)
            ref = ref_res.chosen
            cands = [c for c in cands if self._relation_ok(c, ref, loc.relation)]
            if not cands:
                res.reason = "空间关系约束后无候选"
                return self._try_fallback(loc, shot, res)

        res.survivors = list(cands)

        # ④ 多候选裁决
        if len(cands) == 1:
            chosen = cands[0]
        elif loc.pick is not None:
            chosen = self._pick(cands, loc.pick, ref, loc.roi, shot)
        elif loc.on_ambiguous == OnAmbiguous.ABORT:
            res.reason = f"{len(cands)} 个候选无法唯一化（on_ambiguous=ABORT，不误点）"
            return self._try_fallback(loc, shot, res)
        else:
            chosen = cands[0]

        res.chosen = chosen
        res.click_point = click_point_of(loc, chosen)
        res.ok = True
        return res

    def _try_fallback(self, loc: TextLocator, shot: Screenshot,
                      res: LocateResult) -> LocateResult:
        """T1 失败 → 按元素配置自动降级 T2 模板。"""
        if loc.fallback is not None and loc.fallback.path.exists():
            fres = self._locate_template(loc.fallback, shot)
            if fres.ok:
                fres.locator = loc
                fres.used_fallback = True
                fres.matches = res.matches + fres.matches
                logger.info("[%s] T1 未命中(%s)，T2 模板降级命中", loc.name, res.reason)
                return fres
            res.reason += f"；T2 降级亦失败: {fres.reason}"
        return res

    def _in_roi(self, c: Candidate, roi, shot: Screenshot) -> bool:
        h, w = shot.img.shape[:2]
        l, t, r, b = roi
        return l * w <= c.cx <= r * w and t * h <= c.cy <= b * h

    def _relation_ok(self, c: Candidate, ref: Candidate, rel) -> bool:
        if rel.max_distance is not None:
            if math.hypot(c.cx - ref.cx, c.cy - ref.cy) > rel.max_distance:
                return False
        row_tol = 0.7 * max(c.h, ref.h)        # 同行判定：垂直中心差不超过行高的 0.7 倍
        near_row = 1.5 * max(c.h, ref.h)       # 左/右侧判定放宽到 1.5 倍（视觉上同一行附近）
        d = rel.direction
        if d == Direction.SAME_ROW:
            return abs(c.cy - ref.cy) <= row_tol and c is not ref
        if d == Direction.RIGHT_OF:
            return c.box[0] >= ref.box[2] - 2 and abs(c.cy - ref.cy) <= near_row
        if d == Direction.LEFT_OF:
            return c.box[2] <= ref.box[0] + 2 and abs(c.cy - ref.cy) <= near_row
        if d == Direction.ABOVE:
            return c.box[3] <= ref.box[1] + 2
        if d == Direction.BELOW:
            return c.box[1] >= ref.box[3] - 2
        return False

    def _pick(self, cands, pick: Pick, ref, roi, shot: Screenshot) -> Candidate:
        if pick == Pick.FIRST:
            return cands[0]
        if pick == Pick.TOPMOST:
            return min(cands, key=lambda c: (c.box[1], c.box[0]))
        # NEAREST：距参照锚点最近；无参照则距 ROI 中心 / 屏幕中心最近
        h, w = shot.img.shape[:2]
        if ref is not None:
            ox, oy = ref.cx, ref.cy
        elif roi:
            ox, oy = (roi[0] + roi[2]) / 2 * w, (roi[1] + roi[3]) / 2 * h
        else:
            ox, oy = w / 2, h / 2
        return min(cands, key=lambda c: math.hypot(c.cx - ox, c.cy - oy))

    # ------------------------------------------------------------------
    # T2 多尺度模板匹配
    # ------------------------------------------------------------------

    def _load_template(self, loc: TemplateLocator) -> np.ndarray | None:
        path = loc.path
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        cached = self._tmpl_cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        img = cv2.imread(str(path))
        if img is not None:
            self._tmpl_cache[str(path)] = (mtime, img)
        return img

    def _locate_template(self, loc: TemplateLocator, shot: Screenshot) -> LocateResult:
        res = LocateResult(locator=loc, shot=shot)
        tmpl = self._load_template(loc)
        if tmpl is None:
            res.reason = f"模板图不存在或不可读: {loc.path}"
            return res
        gray = cv2.cvtColor(shot.img, cv2.COLOR_BGR2GRAY)
        tgray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        ox = oy = 0
        if loc.roi:
            l, t, r, b = loc.roi
            ox, oy = int(l * W), int(t * H)
            gray = gray[oy:int(b * H), ox:int(r * W)]

        best = None  # (score, x, y, tw, th, resmap)
        s0, s1, step = loc.scales
        for scale in np.arange(s0, s1 + 1e-6, step):
            tw = int(round(tgray.shape[1] * scale))
            th = int(round(tgray.shape[0] * scale))
            if tw < 8 or th < 8 or tw >= gray.shape[1] or th >= gray.shape[0]:
                continue
            rt = cv2.resize(tgray, (tw, th), interpolation=cv2.INTER_AREA)
            rmap = cv2.matchTemplate(gray, rt, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(rmap)
            if best is None or mx > best[0]:
                best = (float(mx), ml[0], ml[1], tw, th, rmap)
        if best is None:
            res.reason = "模板尺寸与屏幕不匹配（各尺度均无法匹配）"
            return res

        score, bx, by, tw, th, rmap = best
        # 收集最佳尺度下的全部峰值（诊断页高亮全部候选）
        peak_thr = min(loc.threshold, score)
        for x, y in self._peaks(rmap, peak_thr, tw, th):
            c = Candidate(box=(x + ox, y + oy, x + ox + tw, y + oy + th),
                          score=float(rmap[y, x]))
            res.matches.append(c)
            if c.score >= loc.threshold:
                res.survivors.append(c)

        if score < loc.threshold:
            res.reason = f"最高匹配度 {score:.2f} < 阈值 {loc.threshold}"
            return res
        chosen = Candidate(box=(bx + ox, by + oy, bx + ox + tw, by + oy + th), score=score)
        res.chosen = chosen
        res.click_point = click_point_of(loc, chosen)
        res.ok = True
        return res

    @staticmethod
    def _peaks(rmap: np.ndarray, thr: float, tw: int, th: int, cap: int = 8):
        ys, xs = np.where(rmap >= thr)
        if len(xs) == 0:
            return []
        order = np.argsort(rmap[ys, xs])[::-1][:200]
        kept: list[tuple[int, int]] = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            if all(abs(x - kx) > tw // 2 or abs(y - ky) > th // 2 for kx, ky in kept):
                kept.append((x, y))
                if len(kept) >= cap:
                    break
        return kept

    # ------------------------------------------------------------------
    # 存在性 / 等待
    # ------------------------------------------------------------------

    def exists(self, locator, shot: Screenshot) -> bool:
        """宽松存在性判定（页面状态特征用）：不要求唯一，命中即真。"""
        try:
            if isinstance(locator, TextLocator):
                cands = []
                for c in self.ocr_lines(shot, locator.roi):
                    if _text_matches(locator, c.text or ""):
                        cands.append(c)
                if locator.roi:
                    cands = [c for c in cands if self._in_roi(c, locator.roi, shot)]
                if cands:
                    return True
                if locator.fallback is not None and locator.fallback.path.exists():
                    return self._locate_template(locator.fallback, shot).ok
                return False
            if not locator.path.exists():
                return False
            return self._locate_template(locator, shot).ok
        except Exception:
            logger.debug("exists(%s) 异常", getattr(locator, "name", "?"), exc_info=True)
            return False

    def page_state(self, shot: Screenshot | None = None) -> str:
        """README §3 状态判定：'main' / 'login' / 'unknown'。先判主页面再判登录页。"""
        if shot is None:
            shot = self.capture()
        if any(self.exists(f, shot) for f in L.MAIN_PAGE_FEATURES):
            return "main"
        if any(self.exists(f, shot) for f in L.LOGIN_PAGE_FEATURES):
            return "login"
        return "unknown"

    def wait_for(self, locator, timeout: float | None = None,
                 interval: float = 0.6,
                 on_attempt: "Callable[[LocateResult], None] | None" = None) -> LocateResult:
        """等待元素出现（每步通用骨架的第一环）。超时返回 ok=False 的最后一次结果。

        on_attempt：每轮截屏定位后回调一次（含未命中轮），供「运行实况」实时镜像
        “搜索中”的每一帧；默认 None，不影响既有调用。
        """
        if timeout is None:
            timeout = self.settings.engine.step_timeout
        deadline = time.monotonic() + timeout
        while True:
            shot = self.capture()
            res = self.locate(locator, shot)
            if on_attempt is not None:
                try:
                    on_attempt(res)
                except Exception:
                    logger.debug("wait_for.on_attempt 回调异常", exc_info=True)
            if res.ok:
                return res
            if time.monotonic() >= deadline:
                res.reason = f"等待超时({timeout}s): {res.reason}"
                return res
            time.sleep(interval)
