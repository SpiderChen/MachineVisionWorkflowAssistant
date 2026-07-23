"""日志 + 失败快照（README §6：失败任务附整屏截图，存 logs/snapshots/）。"""
from __future__ import annotations

import logging
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .settings import LOGS_DIR, SNAPSHOTS_DIR

LOG_FILE = LOGS_DIR / "app.log"
_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def init_logging(level: str = "INFO") -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        RotatingFileHandler(LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"),
    ]
    # 打包为无控制台窗口的 exe 时 sys.stderr 为 None，加了 StreamHandler 会在每条日志上报错
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format=_FMT, handlers=handlers
    )
    # 第三方库太吵
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def save_snapshot(img_bgr, tag: str = "fail") -> Path | None:
    """保存整屏失败快照，返回文件路径。这些截图同时是将来 T3(YOLO) 的训练素材。"""
    import cv2  # 延迟导入，避免无 opencv 的环境连日志都用不了

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_tag = re.sub(r"[^\w\-]+", "_", tag)[:40] or "fail"
    path = SNAPSHOTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_tag}.png"
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        logging.getLogger(__name__).warning("快照写入失败: %s", path)
        return None
    return path


def tail_log(max_lines: int = 300) -> str:
    """读取日志文件末尾若干行（配置页「日志」Tab 用）。"""
    if not LOG_FILE.exists():
        return "(暂无日志)"
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])
