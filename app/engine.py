"""状态机执行引擎 + 任务队列（README §3）。

同一时间只执行一个打印任务，其余排队。每一步：等待元素出现(超时) → 计算点击坐标 →
移动鼠标 → 点击 → 校验下一状态；任何一步失败即中止本任务并告警，不盲目继续点。
连续失败达到阈值自动熔断暂停。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from . import locators as L
from .flows import FlowError, FlowStore, Step
from .input_ctl import InputController, InputError, activate_window
from .logger import save_snapshot
from .login_flow import LoginFlow
from .monitor import RunEvent, RunMonitor
from .reactive import Outcome, default_rules
from .vision import LocateResult, Screenshot, Vision

logger = logging.getLogger(__name__)


class StepError(Exception):
    """流程中某一步失败。可携带失败瞬间的截图用于快照。"""

    def __init__(self, message: str, shot: Screenshot | None = None):
        super().__init__(message)
        self.shot = shot


@dataclass
class Task:
    vehicle_no: str
    source: str = "manual"                 # db / api / manual / cli
    flow: str = "print_order"              # 执行的流程 key（流程编排页定义）
    external_id: str | None = None         # 外部单号（API 触发）
    callback_url: str | None = None
    db_id: int | None = None
    on_done: Optional[Callable[["TaskResult"], None]] = None


@dataclass
class TaskResult:
    vehicle_no: str
    source: str
    ok: bool
    message: str = ""
    external_id: str | None = None
    snapshot: str | None = None
    finished_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_payload(self) -> dict:
        return {
            "vehicle_no": self.vehicle_no,
            "task_id": self.external_id,
            "ok": self.ok,
            "message": self.message,
            "finished_at": self.finished_at,
        }


class Engine:
    def __init__(self, settings, vision: Vision, inputs: InputController,
                 notifier: Callable[[str, str], None] | None = None):
        self.s = settings
        self.vision = vision
        self.inputs = inputs
        self.notifier = notifier          # 托盘启动后指向 tray.notify
        self.monitor = RunMonitor()       # 运行事件总线（配置页「运行实况」订阅）
        self.queue: "queue.Queue[Task]" = queue.Queue()
        self._stop = threading.Event()
        self._paused = not settings.engine.enabled
        self._busy = False
        self._execution_lock = threading.Lock()  # 正式任务 / CLI / UI 动作测试互斥
        self._consecutive_failures = 0
        self.last_result: TaskResult | None = None
        self._thread: threading.Thread | None = None
        self._cur_flow = ""               # 当前流程标题 / 步号（供事件自带上下文）
        self._cur_idx = 0
        self._cur_total = 0
        self.flows = FlowStore.load()
        self.rules = default_rules()       # 反应式派发规则（截屏→命中谁→执行谁）
        self.login_flow = LoginFlow(settings, vision, inputs, self._notify,
                                    run_flow=self.run_flow)

    # ------------------------------------------------------------------
    # 对外控制面
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="engine", daemon=True)
        self._thread.start()
        logger.info("执行引擎已启动（%s）", "运行中" if not self._paused else "暂停")

    def stop(self) -> None:
        self._stop.set()

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True
        logger.info("执行引擎已暂停")

    def resume(self) -> None:
        self._paused = False
        self._consecutive_failures = 0
        logger.info("执行引擎已恢复运行")

    def toggle(self) -> None:
        (self.resume if self._paused else self.pause)()

    def set_running(self, running: bool) -> None:
        (self.resume if running else self.pause)()

    def submit(self, task: Task) -> int:
        self.queue.put(task)
        size = self.queue.qsize()
        logger.info("任务入队: %s (来源 %s)，当前队列 %d", task.vehicle_no, task.source, size)
        return size

    def status(self) -> dict:
        state = ("busy" if self._busy or self._execution_lock.locked()
                 else "paused" if self._paused else "idle")
        return {
            "state": state,
            "queue_size": self.queue.qsize(),
            "consecutive_failures": self._consecutive_failures,
            "last_result": self.last_result.to_payload() if self.last_result else None,
        }

    def execute_sync(self, task: Task) -> TaskResult:
        """同步执行一单（CLI --print，M1 验证核心可行性）。"""
        with self._execution_lock:
            return self._execute(task)

    def execute_steps_sync(self, flow_title: str, steps: list[Step],
                           ctx: dict | None = None) -> None:
        """同步执行 UI 选中的真实动作，与正式流程共用步骤执行器。

        使用非阻塞互斥：正式任务正在跑时，UI 测试直接报错，不与打印任务抢鼠标。
        """
        if not steps:
            raise StepError("没有可执行的步骤")
        if not self._execution_lock.acquire(blocking=False):
            raise StepError("引擎正在执行其他任务，请稍候再测试")
        if self._busy:
            self._execution_lock.release()
            raise StepError("正式任务已开始，请完成后再测试")
        prev = (self._cur_flow, self._cur_idx, self._cur_total)
        self._cur_flow, self._cur_total = flow_title, len(steps)
        self._emit("flow_start", message=f"开始动作测试「{flow_title}」（{len(steps)} 步）")
        try:
            if self.s.engine.window_title:
                activate_window(self.s.engine.window_title)
                time.sleep(0.8)
            for i, step in enumerate(steps, 1):
                self._cur_idx = i
                logger.info("动作测试[%s] 步骤 %d/%d: %s",
                            flow_title, i, len(steps), step.label())
                self._emit("step", phase="start", label=step.label())
                self._run_step(step, ctx or {})
                self._emit("step", phase="ok", label=step.label())
            self._emit("flow_end", phase="ok", message=f"✔ 动作测试完成：{flow_title}")
        except Exception as exc:
            self._emit("flow_end", phase="fail", message=f"✘ 动作测试失败：{exc}")
            raise
        finally:
            self._cur_flow, self._cur_idx, self._cur_total = prev
            self._execution_lock.release()

    # ------------------------------------------------------------------
    # 工作线程
    # ------------------------------------------------------------------

    def _notify(self, title: str, msg: str) -> None:
        logger.info("[通知] %s: %s", title, msg)
        if self.notifier:
            try:
                self.notifier(title, msg)
            except Exception:
                logger.debug("托盘通知失败", exc_info=True)

    def _emit(self, kind: str, phase: str = "", label: str = "",
              result: LocateResult | None = None, message: str = "") -> None:
        """向「运行实况」广播一个事件（无订阅者时是廉价空转）。"""
        self.monitor.publish(RunEvent(
            kind=kind, flow_title=self._cur_flow, step_index=self._cur_idx,
            step_total=self._cur_total, step_label=label, phase=phase,
            result=result, message=message))

    def _searching_cb(self, label: str) -> Callable[[LocateResult], None]:
        """给 vision.wait_for 的每轮回调：把“搜索中”的每一帧镜像到运行实况。"""
        return lambda res: self._emit("step", phase="searching", label=label, result=res)

    def _worker(self) -> None:
        while not self._stop.is_set():
            if self._paused:
                self._stop.wait(0.5)
                continue
            try:
                task = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._busy = True
            try:
                with self._execution_lock:
                    result = self._execute(task)
            finally:
                self._busy = False
            self._dispatch_result(task, result)
            self._emit("idle", message="空闲，等待任务…")

    def _dispatch_result(self, task: Task, result: TaskResult) -> None:
        self.last_result = result
        self._emit("flow_end", phase=("ok" if result.ok else "fail"),
                   message=f"{'✔ 完成' if result.ok else '✘ 失败'}：{result.vehicle_no} — {result.message}")
        if result.ok:
            self._consecutive_failures = 0
            logger.info("任务完成: %s", result.vehicle_no)
        else:
            self._consecutive_failures += 1
            logger.error("任务失败: %s — %s（快照: %s）",
                         result.vehicle_no, result.message, result.snapshot)
            self._notify("打印任务失败", f"{result.vehicle_no}: {result.message}")
            if self._consecutive_failures >= self.s.engine.max_consecutive_failures:
                self.pause()
                self._notify("引擎已熔断暂停",
                             f"连续失败 {self._consecutive_failures} 次，请检查后手动恢复运行")
        if task.on_done:
            try:
                task.on_done(result)
            except Exception:
                logger.exception("任务回写失败: %s", task.vehicle_no)
        if task.callback_url:
            threading.Thread(target=self._post_callback,
                             args=(task.callback_url, result.to_payload()),
                             daemon=True).start()

    @staticmethod
    def _post_callback(url: str, payload: dict) -> None:
        try:
            import requests

            requests.post(url, json=payload, timeout=8)
            logger.info("结果已回调 %s", url)
        except Exception as e:
            logger.warning("回调 %s 失败: %s", url, e)

    # ------------------------------------------------------------------
    # 单任务执行（含重试与快照）
    # ------------------------------------------------------------------

    def _execute(self, task: Task) -> TaskResult:
        attempts = self.s.engine.task_retries + 1
        err: StepError | None = None
        for i in range(1, attempts + 1):
            try:
                self._execute_once(task)
                return TaskResult(vehicle_no=task.vehicle_no, source=task.source,
                                  external_id=task.external_id, ok=True, message="完成")
            except (StepError, InputError, FlowError) as e:
                err = e if isinstance(e, StepError) else StepError(str(e))
                logger.warning("任务 %s 第 %d/%d 次尝试失败: %s",
                               task.vehicle_no, i, attempts, e)
            except Exception as e:
                err = StepError(f"未预期异常: {e!r}")
                logger.exception("任务 %s 执行异常", task.vehicle_no)
            if i < attempts:
                self._stop.wait(2.0)

        snap_path = None
        try:
            shot = err.shot if (err and err.shot is not None) else self.vision.capture()
            p = save_snapshot(shot.img, task.vehicle_no)
            snap_path = str(p) if p else None
        except Exception:
            logger.warning("保存失败快照失败", exc_info=True)
        return TaskResult(vehicle_no=task.vehicle_no, source=task.source,
                          external_id=task.external_id, ok=False,
                          message=str(err), snapshot=snap_path)

    def _execute_once(self, task: Task) -> None:
        cfg = self.s.engine
        # 0. 置顶浏览器窗口（README §8 唯一一处窗口级 API）
        if cfg.window_title:
            activate_window(cfg.window_title)
            time.sleep(0.8)
        # 反应式派发：每轮截一次屏，看当前屏幕命中哪条规则，就执行对应的操作。
        self._dispatch(task)

    # ------------------------------------------------------------------
    # 反应式派发循环（截屏 → 匹配规则 → 执行；替代原固定顺序的状态分支）
    # ------------------------------------------------------------------

    def _dispatch(self, task: Task) -> None:
        """每轮只截一次屏，按优先级匹配规则，命中第一条即执行，处理完再派发。

        - 多规则同屏命中：靠优先级确定性消歧（不猜）。
        - 无任何规则命中：F5 刷新一轮，仍无命中则失败并快照（宁可失败不乱点）。
        - 有进展（HANDLED）即重置 F5 配额与超时——人工登录等长等待不算超时。
        - 同一规则反复命中却无进展则中止，防死循环。
        """
        cfg = self.s.engine
        ctx = {"vehicle_no": task.vehicle_no}
        rules = sorted(self.rules, key=lambda r: r.priority)
        deadline = time.monotonic() + cfg.dispatch_timeout
        refreshed = False
        last_shot = None
        last_fired = None
        repeat = 0
        self._emit("state", message="派发中：看当前屏幕是什么，就做对应的事…")
        while True:
            if self._stop.is_set():
                raise StepError("引擎已停止")
            if time.monotonic() >= deadline:
                raise StepError(
                    f"派发超时（{cfg.dispatch_timeout:.0f}s 内无法完成任务）", last_shot)

            shot = self.vision.capture()
            last_shot = shot
            fired = None
            for rule in rules:
                try:
                    if rule.detect(self, shot):
                        fired = rule
                        break
                except Exception:
                    logger.debug("规则 %s 匹配异常", rule.name, exc_info=True)

            if fired is None:
                if not refreshed:
                    logger.info("未匹配到任何规则，F5 刷新后重试")
                    self._emit("state", message="未识别当前页面，F5 刷新后重试…")
                    self.inputs.key_tap("f5")
                    refreshed = True
                    self._stop.wait(cfg.refresh_wait)
                    continue
                raise StepError("无法识别当前页面（无任何规则命中，宁可失败不乱点）", shot)

            repeat = repeat + 1 if fired is last_fired else 0
            last_fired = fired
            if repeat >= 5:
                raise StepError(
                    f"规则「{fired.title}」反复命中却无进展，中止以防死循环", shot)

            logger.info("派发命中规则「%s」(%s)", fired.title, fired.name)
            self._emit("state", message=f"命中规则「{fired.title}」→ {fired.about}")
            if fired.act(self, task, ctx) is Outcome.DONE:
                return
            # 有进展：重置刷新配额与派发超时（人工登录等长等待不应被判超时）
            refreshed = False
            deadline = time.monotonic() + cfg.dispatch_timeout

    def _run_main_task(self, task: Task, ctx: dict) -> None:
        """主页面规则的处理器：执行任务指定的流程（有序），并做可选成功校验。"""
        cfg = self.s.engine
        self.run_flow(task.flow, ctx)
        # 流程⑥（可选）：打印成功特征校验（内置打印流程保留原开关语义）
        if task.flow == "print_order" and cfg.verify_success and L.PRINT_SUCCESS.path.exists():
            res = self.vision.wait_for(L.PRINT_SUCCESS, timeout=cfg.step_timeout)
            if not res.ok:
                raise StepError(f"未检测到打印成功特征: {res.reason}", res.shot)

    # ------------------------------------------------------------------
    # 流程解释器：逐步执行流程编排页定义的步骤
    # ------------------------------------------------------------------

    def run_flow(self, flow_key: str, ctx: dict | None = None) -> None:
        """执行一条流程；任何一步失败抛 StepError。登录恢复流也走这里。"""
        flow = self.flows.get(flow_key)
        if not flow.steps:
            raise StepError(f"流程「{flow.title}」没有任何步骤")
        ctx = ctx or {}
        prev = (self._cur_flow, self._cur_idx, self._cur_total)   # 支持嵌套流程（登录恢复流）
        self._cur_flow, self._cur_total = flow.title, len(flow.steps)
        self._emit("flow_start", message=f"开始流程「{flow.title}」（{len(flow.steps)} 步）")
        try:
            for i, step in enumerate(flow.steps, 1):
                self._cur_idx = i
                logger.info("流程[%s] 步骤 %d/%d: %s",
                            flow.title, i, len(flow.steps), step.label())
                self._emit("step", phase="start", label=step.label())
                self._run_step(step, ctx)
        finally:
            self._cur_flow, self._cur_idx, self._cur_total = prev

    def _run_step(self, step: Step, ctx: dict) -> None:
        a = step.action
        if a == "sleep":
            self._stop.wait(max(step.seconds, 0.0))
            return
        if a == "key":
            self.inputs.key_tap(step.key)
            time.sleep(0.2)
            return

        loc = step.build_locator()
        timeout = step.timeout or self.s.engine.step_timeout
        if a == "wait":
            res = self.vision.wait_for(loc, timeout=timeout,
                                       on_attempt=self._searching_cb(step.label()))
            if res.ok:
                self._emit("step", phase="found", label=step.label(), result=res)
            else:
                self._emit("step", phase=("skip" if step.optional else "timeout"),
                           label=step.label(), result=res, message=res.reason)
            if not res.ok and not step.optional:
                raise StepError(f"步骤「{step.label()}」超时未出现: {res.reason}", res.shot)
            if not res.ok:
                logger.info("可选步骤「%s」未出现，跳过", step.label())
            return
        if a in ("click", "double_click", "right_click"):
            res = self._wait_step(step, loc, timeout)
            if res is None:
                return
            self._emit("step", phase="acting", label=step.label(), result=res)
            x, y = self.vision.to_screen(res.shot, res.click_point)
            {"click": self.inputs.click,
             "double_click": self.inputs.double_click,
             "right_click": self.inputs.right_click}[a](x, y)
            time.sleep(0.5)
            return
        if a == "input":
            self._input_step(step, loc, timeout, ctx)
            return
        raise StepError(f"未知动作类型: {a}")

    def _wait_step(self, step: Step, loc, timeout: float) -> LocateResult | None:
        """等待步骤目标出现；可选步骤失败返回 None（跳过），否则抛错。"""
        res = self.vision.wait_for(loc, timeout=timeout,
                                   on_attempt=self._searching_cb(step.label()))
        if res.ok:
            self._emit("step", phase="found", label=step.label(), result=res)
            return res
        if step.optional:
            self._emit("step", phase="skip", label=step.label(), result=res, message=res.reason)
            logger.info("可选步骤「%s」未定位到目标，跳过: %s", step.label(), res.reason)
            return None
        self._emit("step", phase="timeout", label=step.label(), result=res, message=res.reason)
        raise StepError(f"步骤「{step.label()}」未定位到目标: {res.reason}", res.shot)

    def _resolve_value(self, step: Step, ctx: dict,
                       res: LocateResult | None = None) -> tuple[str, bool]:
        """输入步骤取值。返回 (文本, 是否敏感)——敏感值跳过回读校验、不记日志。"""
        src = step.value_source
        if src == "fixed":
            return step.text, False
        if src == "vehicle_no":
            v = ctx.get("vehicle_no")
            if not v:
                raise StepError("任务未携带车辆自编号，无法执行输入步骤")
            return v, False
        if src in ("username", "password"):
            try:
                creds = self.s.get_credentials()
            except RuntimeError as e:
                raise StepError(str(e))
            if not creds:
                raise StepError("未配置登录凭据，请在配置页「登录凭据」中设置")
            return (creds[0], False) if src == "username" else (creds[1], True)
        if src == "captcha":
            return self._solve_captcha(step, res), False
        raise StepError(f"未知输入值来源: {src}")

    def _solve_captcha(self, step: Step, res: LocateResult | None) -> str:
        """按标注比例裁剪验证码图片区域 → 算术题识别（app.captcha）。"""
        from .captcha import parse_arith, solve_arith

        if res is None or res.chosen is None or res.shot is None:
            raise StepError("验证码步骤需要框选定位目标（输入框），无法计算图片区域")
        c = res.chosen

        # 动态算式定位已经拿到当前 OCR 文本，直接计算；
        # 不再二次裁剪/OCR，避免蓝框轻微偏移导致「定位成功但无法解析」。
        loc = res.locator
        if (isinstance(loc, L.TextLocator) and loc.match == L.Match.ARITH
                and not res.used_fallback):
            direct = parse_arith(c.text or "")
            if direct is not None:
                answer, expr = direct
                logger.info("验证码动态锚点 %s → %s", expr, answer)
                return answer

        H, W = res.shot.img.shape[:2]

        if res.used_fallback and step.box and step.captcha_box:
            # T2 候选框是「目标输入框模板」，尺寸与 OCR 文字框不同。
            # 用标注时蓝框相对红框的几何关系重建，避免混用两种框的比例。
            bl, bt, br, bb = (float(v) for v in step.box)
            ql, qt, qr, qb = (float(v) for v in step.captcha_box)
            sx = c.w / max(br - bl, 1.0)
            sy = c.h / max(bb - bt, 1.0)
            l = max(0, int(c.box[0] + (ql - bl) * sx))
            t = max(0, int(c.box[1] + (qt - bt) * sy))
            r = min(W, int(c.box[0] + (qr - bl) * sx))
            b = min(H, int(c.box[1] + (qb - bt) * sy))
        else:
            rl, rt, rr, rb = step.captcha_ratio
            l = max(0, int(c.cx + rl * c.w)); t = max(0, int(c.cy + rt * c.h))
            r = min(W, int(c.cx + rr * c.w)); b = min(H, int(c.cy + rb * c.h))
        got = solve_arith(self.vision, res.shot.img[t:b, l:r]) if r > l and b > t else None
        if got is None:
            raise StepError("验证码识别失败（无法解析算术题），请人工登录",
                            self.vision.capture())
        answer, expr = got
        logger.info("验证码 %s → %s", expr, answer)
        return answer

    def _input_step(self, step: Step, loc, timeout: float, ctx: dict) -> None:
        """流程②③④：点击聚焦 → 清空 → 注入 → 回读校验（失败降级剪贴板粘贴）。

        取值在定位之后：验证码来源需要按命中框比例裁剪图片区域。
        """
        res = None
        if loc is not None:
            res = self._wait_step(step, loc, timeout)
            if res is None:
                return
        if step.value_source == "captcha" and not step.captcha_ratio:
            msg = f"步骤「{step.label()}」未框选验证码图片区域（流程编排页标注）"
            if step.optional:
                logger.warning("%s，按可选步骤跳过", msg)
                return
            raise StepError(msg)
        value, secret = self._resolve_value(step, ctx, res)
        if res is not None:
            self._emit("step", phase="acting", label=step.label(), result=res)
            self.inputs.click(*self.vision.to_screen(res.shot, res.click_point))
            time.sleep(0.3)
        if step.clear_first:
            self.inputs.select_all_clear()
        self.inputs.type_text(value)
        time.sleep(0.3)

        if secret or not step.verify or not self.s.engine.verify_input or res is None:
            return
        if self._verify_input(value, res.click_point, res.chosen):
            return
        logger.warning("步骤「%s」输入校验未通过，降级为剪贴板粘贴重试", step.label())
        res = self._wait_step(step, loc, timeout)
        if res is None:
            return
        self.inputs.click(*self.vision.to_screen(res.shot, res.click_point))
        time.sleep(0.25)
        if step.clear_first:
            self.inputs.select_all_clear()
        self.inputs.paste_text(value)
        time.sleep(0.3)
        if not self._verify_input(value, res.click_point, res.chosen):
            raise StepError(f"步骤「{step.label()}」输入校验失败（注入与粘贴均不匹配）",
                            self.vision.capture())

    def _verify_input(self, value: str, click_point, anchor) -> bool:
        """OCR 回读输入框区域，校验文本已正确填入（流程④）。"""
        import re

        shot = self.vision.capture()
        h = max(anchor.h, 18) if anchor else 24
        x, y = click_point
        region = (x - 30, y - h, x + 320, y + h)
        text = self.vision.read_region(shot, region)
        norm = re.sub(r"\s+", "", text).upper()
        ok = value.upper() in norm
        if not ok:
            logger.warning("输入回读: %r，未包含 %r", text, value)
        return ok

    # ------------------------------------------------------------------
    # 页面状态（README §3 左侧分支）
    # ------------------------------------------------------------------

    def _handle_login(self) -> None:
        if self.s.engine.auto_login:
            logger.info("检测到登录页，自动登录开关已开，执行自动登录")
            if not self.login_flow.login():
                raise StepError("自动登录失败", self.vision.capture())
        else:
            self._wait_manual_login()

    def _wait_manual_login(self) -> None:
        """自动登录关闭：挂起任务，气泡提醒，每 10s 重新检测（README §5）。"""
        self._notify("请先手动登录", "检测到登录页且自动登录未开启，登录完成后将自动继续")
        logger.info("等待人工登录 ...")
        while not self._stop.is_set():
            if self._paused:
                raise StepError("等待人工登录期间引擎被暂停，任务中止")
            self._stop.wait(10.0)
            if self._stop.is_set():
                break
            if self.vision.page_state() == "main":
                logger.info("检测到已登录，继续任务")
                return
            logger.info("仍在登录页，继续等待人工登录 ...")
        raise StepError("引擎已停止")
