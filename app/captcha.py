"""图片验证码识别：算术题与英文字母/数字验证码（README §5）。

针对小字、浅色背景和散点干扰，OCR 前会自适应放大、留白、去噪并生成多个
二值化变体；同时容错 OCR 片段乱序。算术模式还会修正常见的数字/算符
误识别。识别失败返回 None，由引擎刷新验证码后重试。
"""
from __future__ import annotations

import logging
import re
import unicodedata

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
_ALNUM_MIN_LENGTH = 3
_ALNUM_MAX_LENGTH = 8


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


def parse_alnum(text: str, *, min_length: int = _ALNUM_MIN_LENGTH,
                max_length: int = _ALNUM_MAX_LENGTH) -> str | None:
    """提取英文字母/数字验证码，保留 OCR 识别出的大小写。

    全角字符先转为半角，再忽略空格与干扰符号。默认只接受 3~8 位，避免把
    裁剪区域边缘的标签或单个噪点误当成验证码。
    """
    if min_length < 1 or max_length < min_length:
        raise ValueError("验证码长度范围无效")
    normalized = unicodedata.normalize("NFKC", text or "")
    code = "".join(re.findall(r"[A-Za-z0-9]", normalized))
    return code if min_length <= len(code) <= max_length else None


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


def _candidate_texts(cands, min_score: float, *, combined_first: bool = False):
    """同时尝试单个 OCR 结果和按横坐标重组的整行结果。"""
    eligible = [c for c in cands if c.score >= min_score and (c.text or "").strip()]
    if not eligible:
        return
    ordered = sorted(eligible, key=lambda c: (c.cx, c.cy))
    combined = (
        "".join(c.text or "" for c in ordered),
        " ".join(c.text or "" for c in ordered),
    )
    if combined_first and len(eligible) > 1:
        yield from combined
    for candidate in sorted(eligible, key=lambda c: c.score, reverse=True):
        yield candidate.text or ""
    if not combined_first or len(eligible) <= 1:
        yield from combined


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


def solve_alnum(vision, img_bgr) -> tuple[str, str] | None:
    """识别 3~8 位英文字母/数字验证码。

    成功返回 ``(待输入字符, OCR 原文)``，无法可靠提取时返回 None。多个 OCR
    片段优先按屏幕横坐标拼接，避免只输入验证码的一部分。
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    for variant in _ocr_variants(img_bgr):
        try:
            cands = vision.ocr_image(variant)
        except Exception as e:
            logger.warning("验证码 OCR 失败: %s", e)
            continue
        # 字母数字没有“算式可计算”这种语义校验，低置信度结果宁可刷新也不误填。
        for min_score in (0.75, 0.50):
            for text in _candidate_texts(cands, min_score, combined_first=True):
                code = parse_alnum(text)
                if code is not None:
                    logger.info("字母数字验证码识别: %r → %s", text, code)
                    return code, (text or "").strip()
    logger.warning("验证码无法解析为 3~8 位英文字母/数字")
    return None
