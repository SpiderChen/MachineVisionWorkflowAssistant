"""运行事件总线（配置页「运行实况」标签的数据源）。

被动镜像设计：引擎每步定位后，把它**真正看到、真正判定**的 LocateResult 广播出来，
UI 直接画这张图（复用 vision.draw_result），不另开截屏器与引擎抢 OCR/CPU。

publish 在引擎工作线程调用；订阅者须自行把回调编组回自己的线程（UI 用 Qt 信号，
跨线程自动排队到主线程）。见 config_ui/window.py 的 _wire_live。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .vision import LocateResult

logger = logging.getLogger(__name__)


@dataclass
class RunEvent:
    """一次运行事件。result 携带 shot + 候选框，供 UI 画实时标注。"""

    kind: str = "step"          # flow_start | step | flow_end | state | idle | busy
    flow_title: str = ""
    step_index: int = 0
    step_total: int = 0
    step_label: str = ""
    phase: str = ""             # start | searching | found | acting | ok | skip | timeout | fail
    result: "LocateResult | None" = None
    message: str = ""
    ts: float = field(default_factory=time.time)


Subscriber = Callable[[RunEvent], None]


class RunMonitor:
    """线程安全的发布/订阅。引擎持有一个实例，UI 打开时订阅、关闭时退订。"""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._lock = threading.Lock()
        self.last: RunEvent | None = None

    def subscribe(self, cb: Subscriber) -> None:
        with self._lock:
            if cb not in self._subs:
                self._subs.append(cb)

    def unsubscribe(self, cb: Subscriber) -> None:
        with self._lock:
            if cb in self._subs:
                self._subs.remove(cb)

    def publish(self, ev: RunEvent) -> None:
        self.last = ev
        with self._lock:
            subs = list(self._subs)          # 复制后在锁外回调，避免订阅者反向增删导致死锁
        for cb in subs:
            try:
                cb(ev)
            except Exception:
                logger.debug("运行事件订阅者异常", exc_info=True)
