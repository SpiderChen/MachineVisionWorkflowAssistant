"""算术验证码识别：图片型「1+1=?」类计算题（README §5）。

针对小字、浅色背景和散点干扰，OCR 前会自适应放大、留白、去噪并生成多个
二值化变体；同时容错常见的数字/算符误识别和 OCR 片段乱序。识别失败
返回 None，由引擎刷新验证码后重试。
"""
from __future__ import annotations

import logging
import re

import cv2

logger = logging.getLogger(__name__)

_OPS = {"＋": "+", "加": "+",
        "十": "+", "✢": "+", "†": "+",
        "－": "-", "−": "-", "—": "-", "–": "-", "_": "-",
        "~": "-", "一": "-", "减": "-",
        "×": "*", "x": "*", "X": "*", "乘": "*",
        "÷": "/", ":": "/", "：": "/", "除": "/"}
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")
_DIGIT_CONFUSIONS = str.maketrans({
    "O": "0", "o": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8",
})


def parse_arith(text: str) -> tuple[str, str] | None:
    """从 OCR 文本解析算式。成功返回 (答案, 规范化算式)。"""
    t = text.translate(_FULLWIDTH).translate(_DIGIT_CONFUSIONS)
    t = "".join(_OPS.get(ch, ch) for ch in t)
    t = re.sub(r"[\s=＝?？几等于于以]", "", t)
    m = re.search(r"(\d{1,3})([+\-*/])(\d{1,3})", t)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        v = a + b
    elif op == "-":
        v = a - b
    elif op == "*":
        v = a * b
    else:
        if b == 0 or a % b:
            return None
        v = a // b
    return str(v), f"{a}{op}{b}"


def _remove_speckles(binary: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """清理二值图中的孤立散点；输入约定黑字白底。"""
    foreground = 255 - binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8)
    min_area = max(2, int(round(binary.shape[0] * binary.shape[1] * 0.00035)))
    cleaned = foreground.copy()
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == label] = 0
    return 255 - cleaned


def _with_border(img: cv2.typing.MatLike, *, inverted: bool = False):
    """RapidOCR 对紧贴裁剪边缘的字符较敏感，增加少量留白。"""
    pad = max(8, int(round(min(img.shape[:2]) * 0.12)))
    color = 0 if inverted else 255
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(color, color, color))


def _ocr_variants(img_bgr):
    """按预期成功率排列预处理变体，让常见场景尽量一次 OCR 命中。"""
    h, _ = img_bgr.shape[:2]
    scale = min(6.0, max(3.0, 140.0 / max(h, 1)))
    big = cv2.resize(
        img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)

    _, otsu = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    otsu = _remove_speckles(otsu)

    block = min(51, max(15, (min(gray.shape[:2]) // 4) | 1))
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, 7)
    adaptive = _remove_speckles(adaptive)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(denoised)
    return (
        _with_border(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
        _with_border(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
        _with_border(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)),
        _with_border(cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR), inverted=True),
        _with_border(big),
    )


def _candidate_texts(cands, min_score: float):
    """同时尝试单个 OCR 结果和按横坐标重组的整行结果。"""
    eligible = [c for c in cands if c.score >= min_score and (c.text or "").strip()]
    if not eligible:
        return
    for candidate in sorted(eligible, key=lambda c: c.score, reverse=True):
        yield candidate.text or ""
    ordered = sorted(eligible, key=lambda c: (c.cx, c.cy))
    yield "".join(c.text or "" for c in ordered)
    yield " ".join(c.text or "" for c in ordered)


def solve_arith(vision, img_bgr) -> tuple[str, str] | None:
    """识别算术验证码图片。成功返回 (答案, 算式)，无法解析返回 None。"""
    if img_bgr is None or img_bgr.size == 0:
        return None
    # 去噪二值图优先；每个变体先使用高置信度结果，再放宽。
    for variant in _ocr_variants(img_bgr):
        try:
            cands = vision.ocr_image(variant)
        except Exception as e:
            logger.warning("验证码 OCR 失败: %s", e)
            continue
        for min_score in (0.75, 0.0):
            for text in _candidate_texts(cands, min_score):
                got = parse_arith(text)
                if got:
                    logger.info("验证码识别: %r → %s = %s",
                                text, got[1], got[0])
                    return got
    logger.warning("验证码无法解析为算术题")
    return None
