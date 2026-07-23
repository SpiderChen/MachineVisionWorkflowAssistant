"""算术验证码识别：图片型「1+1=?」类计算题（README §5 验证码接入点的第一版实现）。

放大 3 倍后 OCR，原图/二值化/反色三个变体依次尝试；支持全角数字、
×÷ 及「加减乘除」汉字算符。识别失败返回 None（引擎按步骤 optional 配置
跳过或报错转人工）。更复杂的扭曲验证码后续可在此替换为打码接口。
"""
from __future__ import annotations

import logging
import re

import cv2

logger = logging.getLogger(__name__)

_OPS = {"＋": "+", "加": "+",
        "－": "-", "−": "-", "—": "-", "一": "-", "减": "-",
        "×": "*", "x": "*", "X": "*", "乘": "*",
        "÷": "/", "除": "/"}
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")


def parse_arith(text: str) -> tuple[str, str] | None:
    """从 OCR 文本解析算式。成功返回 (答案, 规范化算式)。"""
    t = text.translate(_FULLWIDTH)
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


def solve_arith(vision, img_bgr) -> tuple[str, str] | None:
    """识别算术验证码图片。成功返回 (答案, 算式)，无法解析返回 None。"""
    if img_bgr is None or img_bgr.size == 0:
        return None
    big = cv2.resize(img_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 二值化去噪的变体优先；每个变体先只用高置信度行，失败再放宽
    for variant in (cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(255 - bw, cv2.COLOR_GRAY2BGR), big):
        try:
            cands = vision.ocr_image(variant)
        except Exception as e:
            logger.warning("验证码 OCR 失败: %s", e)
            return None
        for min_score in (0.75, 0.0):
            text = " ".join(c.text or "" for c in cands if c.score >= min_score)
            got = parse_arith(text)
            if got:
                logger.info("验证码识别: %r → %s = %s", text, got[1], got[0])
                return got
    logger.warning("验证码无法解析为算术题")
    return None
