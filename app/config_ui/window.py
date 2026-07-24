"""PySide6 配置窗口（README §6）：四个 Tab —— 基础 / 流程编排 / 运行实况 / 登录凭据。

「运行实况」被动镜像引擎的真实执行：左=实时标注截图（引擎真正看到的那一帧，黄=候选
橙=约束后 绿=选中 红点=点击点）+ 当前步骤横幅，右=实时日志；并内置手动「测试定位 /
重新截取模板」（装机验收：把定位歧义暴露在部署阶段，模板类元素可拉框覆盖 templates/）。
"""
from __future__ import annotations

import html
import logging
import os
import subprocess
import sys
import time

import cv2

from PySide6.QtCore import QObject, QRect, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .. import locators
from ..engine import Task
from ..flows import LOCATOR_ACTIONS
from ..locators import TemplateLocator, TextLocator
from ..logger import LOG_FILE, tail_log
from ..settings import SNAPSHOTS_DIR
from ..vision import draw_result
from .flow_tab import FlowTab

logger = logging.getLogger(__name__)

# 数据库类型 → (显示名, 连接串示例, 时间函数)。db_type 仅作填充助手，真实驱动看连接串。
DB_TYPES = {
    "mssql": ("SQL Server", "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server", "GETDATE()"),
    "mysql": ("MySQL", "mysql+pymysql://user:pass@host:3306/db", "NOW()"),
    "postgresql": ("PostgreSQL", "postgresql+psycopg://user:pass@host:5432/db", "NOW()"),
    "oracle": ("Oracle", "oracle+oracledb://user:pass@host:1521/?service_name=orcl", "SYSDATE"),
    "sqlite": ("SQLite", "sqlite:///print_task.db", "CURRENT_TIMESTAMP"),
}


def default_sql(db_type: str) -> dict:
    """按数据库类型生成 4 段默认 SQL（仅时间函数按方言不同）。"""
    now = DB_TYPES.get(db_type, DB_TYPES["mssql"])[2]
    return {
        "poll": "SELECT id, vehicle_no FROM print_task WHERE status = 0 ORDER BY id",
        "claim": f"UPDATE print_task SET status = 1, updated_at = {now} WHERE id = :id AND status = 0",
        "success": f"UPDATE print_task SET status = 2, updated_at = {now} WHERE id = :id",
        "fail": f"UPDATE print_task SET status = 3, err_msg = :err, updated_at = {now} WHERE id = :id",
    }


def open_folder(path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def np_to_pixmap(img_bgr) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _LiveBridge(QObject):
    """跨线程信号桥：引擎工作线程 emit，槽在主线程执行（Qt 自动排队到主线程）。"""

    event = Signal(object)        # RunEvent
    logline = Signal(str, int)    # (格式化文本, levelno)


class _QtLogHandler(logging.Handler):
    """把日志记录转成 Qt 信号，喂给「运行实况」右侧的实时日志流。"""

    def __init__(self, emit_fn):
        super().__init__()
        self._emit_fn = emit_fn

    def emit(self, record):
        try:
            self._emit_fn(self.format(record), record.levelno)
        except Exception:
            pass


class SnipOverlay(QWidget):
    """全屏框选截取器：在冻结的截图上拉框，Esc 取消。"""

    def __init__(self, shot, on_done):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._shot = shot
        self._on_done = on_done
        self._pix = np_to_pixmap(shot.img)
        self._origin = None
        self._current = None
        self.setCursor(Qt.CrossCursor)
        self.setGeometry(shot.left, shot.top,
                         shot.img.shape[1], shot.img.shape[0])
        self.showFullScreen()

    def paintEvent(self, event):
        p = QPainter(self)
        p.drawPixmap(self.rect(), self._pix)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)
        p.setOpacity(0.35)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._origin and self._current:
            sel = QRect(self._origin, self._current).normalized()
            p.setOpacity(1.0)
            src = self._map_to_pix(sel)
            p.drawPixmap(sel, self._pix, src)
            p.setPen(Qt.GlobalColor.red)
            p.drawRect(sel)
        p.end()

    def _map_to_pix(self, rect: QRect) -> QRect:
        sx = self._pix.width() / max(self.width(), 1)
        sy = self._pix.height() / max(self.height(), 1)
        return QRect(int(rect.x() * sx), int(rect.y() * sy),
                     int(rect.width() * sx), int(rect.height() * sy))

    def mousePressEvent(self, event):
        self._origin = event.position().toPoint()
        self._current = self._origin
        self.update()

    def mouseMoveEvent(self, event):
        if self._origin:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self._origin:
            return
        sel = self._map_to_pix(QRect(self._origin, self._current).normalized())
        self.close()
        if sel.width() >= 8 and sel.height() >= 8:
            crop = self._shot.img[sel.y():sel.y() + sel.height(),
                                  sel.x():sel.x() + sel.width()]
            self._on_done(crop)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


class ConfigWindow(QMainWindow):
    def __init__(self, settings, vision, engine):
        super().__init__()
        self.settings = settings
        self.vision = vision
        self.engine = engine
        self._overlay = None
        self.setWindowTitle("出货单自动打印助手 — 配置")
        self.resize(1000, 700)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_basic_tab(), "基础")
        self.tabs.addTab(self._build_db_tab(), "数据库触发")
        self.flow_tab = FlowTab(settings, vision, engine, self._capture_hidden)
        self.tabs.addTab(self.flow_tab, "流程编排")
        self.live_tab = self._build_live_tab()
        self.tabs.addTab(self.live_tab, "运行实况")
        self.tabs.addTab(self._build_cred_tab(), "登录凭据")
        self.setCentralWidget(self.tabs)
        # 切到「运行实况」时刷新手动步骤下拉，反映流程编排页的最新编辑
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._load_basic()
        self._load_db()
        self._wire_live()      # 订阅引擎事件总线 + 挂实时日志 handler + 空闲底图心跳

    # 托盘应用：关窗仅隐藏
    def closeEvent(self, event):
        event.ignore()
        self.hide()

    # ------------------------------------------------------------------
    # Tab 1 基础
    # ------------------------------------------------------------------

    def _build_basic_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.chk_run = QCheckBox("执行引擎运行中")
        self.chk_auto_login = QCheckBox("自动登录（默认关，README §5）")
        self.chk_api = QCheckBox("启用 HTTP API 触发（改动重启后生效）")
        self.le_api_host = QLineEdit()
        self.sb_api_port = QSpinBox(); self.sb_api_port.setRange(1, 65535)
        self.le_api_token = QLineEdit()
        self.sb_retries = QSpinBox(); self.sb_retries.setRange(0, 10)
        self.sb_fuse = QSpinBox(); self.sb_fuse.setRange(1, 100)
        self.sb_timeout = QDoubleSpinBox(); self.sb_timeout.setRange(1.0, 120.0); self.sb_timeout.setSuffix(" s")
        self.sb_monitor = QSpinBox(); self.sb_monitor.setRange(1, 8)
        self.le_window_title = QLineEdit()
        self.le_window_title.setPlaceholderText("浏览器窗口标题包含串；留空则不做窗口激活")

        form.addRow(self.chk_run)
        form.addRow(self.chk_auto_login)
        form.addRow(self.chk_api)
        form.addRow("API 监听地址", self.le_api_host)
        form.addRow("API 端口", self.sb_api_port)
        form.addRow("API Token", self.le_api_token)
        form.addRow("失败重试次数", self.sb_retries)
        form.addRow("连败熔断阈值", self.sb_fuse)
        form.addRow("单步等待超时", self.sb_timeout)
        form.addRow("目标显示器编号", self.sb_monitor)
        form.addRow("目标窗口标题", self.le_window_title)

        btn = QPushButton("保存")
        btn.clicked.connect(self._save_basic)
        form.addRow(btn)
        return w

    # ------------------------------------------------------------------
    # Tab 数据库触发（直接写 SQL 适配任意库表/视图/存储过程）
    # ------------------------------------------------------------------

    def _build_db_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.chk_db = QCheckBox("启用数据库轮询触发（改动需重启程序生效）")
        self.cmb_db_type = QComboBox()
        for key, (label, _url, _now) in DB_TYPES.items():
            self.cmb_db_type.addItem(label, key)
        self.le_db_url = QLineEdit()
        self.sb_poll = QDoubleSpinBox(); self.sb_poll.setRange(1.0, 300.0); self.sb_poll.setSuffix(" s")
        self.sb_db_batch = QSpinBox(); self.sb_db_batch.setRange(1, 500)
        self.cmb_db_flow = QComboBox()
        self.chk_db_writeback = QCheckBox(
            "回写对方数据库（默认关闭＝只读，仅在客户授权写权限时勾选）")
        self.chk_db_writeback.toggled.connect(self._on_writeback_toggled)
        self.txt_sql_poll = self._sql_edit()
        self.txt_sql_claim = self._sql_edit()
        self.txt_sql_success = self._sql_edit()
        self.txt_sql_fail = self._sql_edit()

        form.addRow(self.chk_db)
        type_row = QHBoxLayout()
        type_row.addWidget(self.cmb_db_type)
        btn_fill = QPushButton("按类型填充连接串/SQL 模板")
        btn_fill.clicked.connect(self._fill_db_defaults)
        type_row.addWidget(btn_fill); type_row.addStretch()
        tr = QWidget(); tr.setLayout(type_row)
        form.addRow("数据库类型", tr)
        form.addRow("连接串", self.le_db_url)
        form.addRow("轮询间隔", self.sb_poll)
        form.addRow("单批最多处理", self.sb_db_batch)
        form.addRow("触发任务走的流程", self.cmb_db_flow)
        form.addRow(self.chk_db_writeback)

        help_lbl = QLabel(
            "只读模式（推荐，默认）：每轮只跑「捞取」这一条 SELECT，客户只需给只读账号；"
            "「已处理」记在本机 db_seen.sqlite 按主键去重，完全不改动对方业务表。\n"
            "回写模式（需客户授权写权限）：捞取 →「抢占」(rowcount>0 才算抢到，防多实例重复) "
            "→ 入队执行 → 完成后按结果跑「成功/失败」回写（下方三条 SQL 仅此模式生效）。\n"
            "① 捞取 SQL 的前两列必须是「主键、车辆自编号」；② 占位符（参数化防注入）："
            ":id = 主键、:vehicle_no = 车号、:err = 失败原因。\n"
            "表名 / 状态值 / 联表 / 视图 / 存储过程都写在 SQL 里，程序不假设任何 schema。")
        help_lbl.setWordWrap(True)
        help_lbl.setTextFormat(Qt.PlainText)
        help_lbl.setStyleSheet("color:#666; padding:4px;")
        form.addRow(help_lbl)
        form.addRow("捞取待处理", self.txt_sql_poll)
        form.addRow("抢占 (:id)", self.txt_sql_claim)
        form.addRow("成功回写 (:id)", self.txt_sql_success)
        form.addRow("失败回写 (:id,:err)", self.txt_sql_fail)

        btns = QHBoxLayout()
        btn_test = QPushButton("测试连接 / 试跑捞取")
        btn_test.clicked.connect(self._test_db_conn)
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self._save_db)
        btns.addWidget(btn_test); btns.addWidget(btn_save); btns.addStretch()
        bw = QWidget(); bw.setLayout(btns)
        form.addRow(bw)
        return w

    def _on_writeback_toggled(self, on: bool) -> None:
        # 只读模式下回写三条 SQL 用不到，禁用以示区分（捞取 SQL 两种模式都要，不禁用）
        for e in (self.txt_sql_claim, self.txt_sql_success, self.txt_sql_fail):
            e.setEnabled(on)

    @staticmethod
    def _sql_edit() -> QPlainTextEdit:
        e = QPlainTextEdit()
        e.setMaximumHeight(60)
        e.setLineWrapMode(QPlainTextEdit.NoWrap)
        return e

    def _fill_db_defaults(self) -> None:
        key = self.cmb_db_type.currentData()
        nonempty = any(t.toPlainText().strip() for t in (
            self.txt_sql_poll, self.txt_sql_claim, self.txt_sql_success, self.txt_sql_fail))
        if self.le_db_url.text().strip() or nonempty:
            if QMessageBox.question(
                    self, "覆盖确认",
                    "将用所选类型的示例连接串与默认 SQL 覆盖当前内容，继续？") != QMessageBox.Yes:
                return
        self.le_db_url.setText(DB_TYPES[key][1])
        sql = default_sql(key)
        self.txt_sql_poll.setPlainText(sql["poll"])
        self.txt_sql_claim.setPlainText(sql["claim"])
        self.txt_sql_success.setPlainText(sql["success"])
        self.txt_sql_fail.setPlainText(sql["fail"])

    def _load_db(self) -> None:
        s = self.settings
        self.chk_db.setChecked(s.db.enabled)
        idx = self.cmb_db_type.findData(s.db.db_type)
        self.cmb_db_type.setCurrentIndex(idx if idx >= 0 else 0)
        self.le_db_url.setText(s.db.url)
        self.sb_poll.setValue(s.db.poll_interval)
        self.sb_db_batch.setValue(s.db.batch)
        self.cmb_db_flow.clear()
        keys = list(self.engine.flows.flows.keys())
        if s.db.flow and s.db.flow not in keys:
            keys.insert(0, s.db.flow)
        for k in keys:
            fl = self.engine.flows.flows.get(k)
            self.cmb_db_flow.addItem(fl.title if fl else k, k)
        fidx = self.cmb_db_flow.findData(s.db.flow)
        if fidx >= 0:
            self.cmb_db_flow.setCurrentIndex(fidx)
        self.chk_db_writeback.setChecked(s.db.writeback)
        self._on_writeback_toggled(s.db.writeback)
        self.txt_sql_poll.setPlainText(s.db.sql_poll)
        self.txt_sql_claim.setPlainText(s.db.sql_claim)
        self.txt_sql_success.setPlainText(s.db.sql_success)
        self.txt_sql_fail.setPlainText(s.db.sql_fail)

    def _collect_db(self) -> None:
        s = self.settings
        s.db.enabled = self.chk_db.isChecked()
        s.db.db_type = self.cmb_db_type.currentData() or "mssql"
        s.db.url = self.le_db_url.text().strip()
        s.db.poll_interval = self.sb_poll.value()
        s.db.batch = self.sb_db_batch.value()
        s.db.flow = self.cmb_db_flow.currentData() or "print_order"
        s.db.writeback = self.chk_db_writeback.isChecked()
        s.db.sql_poll = self.txt_sql_poll.toPlainText().strip()
        s.db.sql_claim = self.txt_sql_claim.toPlainText().strip()
        s.db.sql_success = self.txt_sql_success.toPlainText().strip()
        s.db.sql_fail = self.txt_sql_fail.toPlainText().strip()

    def _save_db(self) -> None:
        self._collect_db()
        self.settings.save()
        QMessageBox.information(
            self, "已保存", "数据库触发配置已保存。\n启用/连接串/SQL 改动需重启程序生效。")

    def _test_db_conn(self) -> None:
        """只读试跑「捞取」SQL：验证连接串与查询是否可用（不抢占、不改数据）。"""
        url = self.le_db_url.text().strip()
        sql = self.txt_sql_poll.toPlainText().strip()
        if not url or not sql:
            QMessageBox.warning(self, "提示", "请先填写连接串与「捞取待处理」SQL")
            return
        try:
            import sqlalchemy as sa

            eng = sa.create_engine(url, pool_pre_ping=True)
            with eng.connect() as conn:
                rows = conn.execute(sa.text(sql)).fetchmany(5)
            eng.dispose()
        except Exception as e:
            QMessageBox.warning(self, "连接/查询失败", str(e))
            return
        sample = "\n".join(f"  {r[0]} | {r[1]}" for r in rows) if rows else "  （当前无待处理行）"
        QMessageBox.information(
            self, "连接成功",
            f"「捞取」SQL 可执行，前几条（主键 | 车号）：\n{sample}")

    def _load_basic(self) -> None:
        s = self.settings
        self.chk_run.setChecked(not self.engine.paused)
        self.chk_auto_login.setChecked(s.engine.auto_login)
        self.chk_api.setChecked(s.api.enabled)
        self.le_api_host.setText(s.api.host)
        self.sb_api_port.setValue(s.api.port)
        self.le_api_token.setText(s.api.token)
        self.sb_retries.setValue(s.engine.task_retries)
        self.sb_fuse.setValue(s.engine.max_consecutive_failures)
        self.sb_timeout.setValue(s.engine.step_timeout)
        self.sb_monitor.setValue(s.engine.monitor)
        self.le_window_title.setText(s.engine.window_title)

    def _save_basic(self) -> None:
        s = self.settings
        s.engine.enabled = self.chk_run.isChecked()
        s.engine.auto_login = self.chk_auto_login.isChecked()
        s.api.enabled = self.chk_api.isChecked()
        s.api.host = self.le_api_host.text().strip() or "127.0.0.1"
        s.api.port = self.sb_api_port.value()
        s.api.token = self.le_api_token.text().strip()
        s.engine.task_retries = self.sb_retries.value()
        s.engine.max_consecutive_failures = self.sb_fuse.value()
        s.engine.step_timeout = self.sb_timeout.value()
        s.engine.monitor = self.sb_monitor.value()
        s.engine.window_title = self.le_window_title.text().strip()
        s.save()
        self.engine.set_running(s.engine.enabled)
        QMessageBox.information(
            self, "已保存",
            "配置已保存。\n运行/自动登录等开关立即生效；API 端口改动需重启程序。")

    # ------------------------------------------------------------------
    # Tab 3 运行实况（实时画面 + 步骤横幅 + 实时日志 + 内置手动定位诊断）
    # ------------------------------------------------------------------

    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)

        # 顶部：引擎状态 + 手动测试定位工具（吸收原「定位诊断」）
        top = QHBoxLayout()
        self.lbl_engine_state = QLabel("○ —")
        top.addWidget(self.lbl_engine_state)
        top.addStretch()
        top.addWidget(QLabel("手动步骤:"))
        self.cmb_locator = QComboBox()
        self.cmb_locator.setMinimumWidth(300)
        self._reload_step_choices()
        top.addWidget(self.cmb_locator)
        btn_one = QPushButton("测试选中")
        btn_all = QPushButton("测试全部")
        btn_recap = QPushButton("重新截取模板")
        btn_one.clicked.connect(self._test_selected)
        btn_all.clicked.connect(self._test_all)
        btn_recap.clicked.connect(self._recapture_template)
        for b in (btn_one, btn_all, btn_recap):
            top.addWidget(b)
        outer.addLayout(top)

        # 当前步骤横幅
        self.lbl_step = QLabel("空闲，等待任务…")
        self.lbl_step.setStyleSheet("font-weight:bold; padding:6px; font-size:14px;")
        outer.addWidget(self.lbl_step)

        # 左=实时标注截图，右=实时日志 + 定位报告
        splitter = QSplitter()
        self.preview = QLabel(
            "（运行时此处实时显示引擎看到的标注截图：\n黄=候选 橙=约束后 绿=选中 红点=点击点）")
        self.preview.setMinimumSize(520, 340)
        self.preview.setAlignment(Qt.AlignCenter)
        splitter.addWidget(self.preview)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("实时日志"))
        self.live_log = QPlainTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMaximumBlockCount(800)     # 限制内存，滚动窗口
        rv.addWidget(self.live_log, stretch=3)
        rv.addWidget(QLabel("定位报告（手动测试 / 最近一步）"))
        self.diag_text = QPlainTextEdit()
        self.diag_text.setReadOnly(True)
        rv.addWidget(self.diag_text, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([620, 380])
        outer.addWidget(splitter, stretch=1)

        # 底部：快照 / 日志目录
        bottom = QHBoxLayout()
        btn_snap = QPushButton("打开失败快照目录")
        btn_snap.clicked.connect(lambda: open_folder(SNAPSHOTS_DIR))
        btn_logdir = QPushButton("打开日志目录")
        btn_logdir.clicked.connect(lambda: open_folder(LOG_FILE.parent))
        bottom.addWidget(btn_snap)
        bottom.addWidget(btn_logdir)
        bottom.addStretch()
        bottom.addWidget(QLabel(f"日志文件: {LOG_FILE}"))
        outer.addLayout(bottom)
        return w

    # ---- 实时接线：事件总线 + 日志 handler + 空闲底图心跳 ----

    def _wire_live(self) -> None:
        self._bridge = _LiveBridge()
        self._bridge.event.connect(self._on_run_event)
        self._bridge.logline.connect(self._on_log_line)
        # 订阅引擎事件总线：回调在引擎线程，转 Qt 信号后自动排队到主线程
        self._mon_sub = lambda ev: self._bridge.event.emit(ev)
        self.engine.monitor.subscribe(self._mon_sub)
        # 实时日志：把根 logger 的每条记录推进右侧面板
        self._log_handler = _QtLogHandler(
            lambda msg, lvl: self._bridge.logline.emit(msg, lvl))
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(self._log_handler)
        self.live_log.setPlainText(tail_log(200))     # 预填最近历史
        self._scroll_log_bottom()
        self._update_engine_state()
        # 空闲时每 1.5s 纯截屏刷新底图（不跑 OCR），仅本标签可见时
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(1500)
        self._live_timer.timeout.connect(self._idle_refresh)
        self._live_timer.start()

    def _on_run_event(self, ev) -> None:
        """主线程渲染一个运行事件：更新步骤横幅 + 实时标注截图 + 定位报告。"""
        if ev.kind == "flow_start":
            self.lbl_step.setText(f"▶ {ev.flow_title}（共 {ev.step_total} 步）")
        elif ev.kind == "flow_end":
            self.lbl_step.setText(ev.message or "流程结束")
        elif ev.kind == "idle":
            self.lbl_step.setText("空闲，等待任务…")
        elif ev.kind == "state":
            self.lbl_step.setText(ev.message)
        elif ev.kind == "step":
            tag = {"start": "", "searching": "· 搜索中…", "found": "· ✔ 命中",
                   "acting": "· 点击 / 输入中…", "timeout": "· ✘ 超时",
                   "skip": "· 跳过（可选）"}.get(ev.phase, "")
            self.lbl_step.setText(
                f"步骤 {ev.step_index}/{ev.step_total} 「{ev.step_label}」 {tag}")
        res = ev.result
        if res is not None and getattr(res, "shot", None) is not None:
            annotated = res.shot.img.copy()
            draw_result(annotated, res)
            self._show_pixmap(annotated)
            try:
                self.diag_text.setPlainText(res.describe())
            except Exception:
                pass
        self._update_engine_state()

    def _on_log_line(self, msg: str, level: int) -> None:
        color = {logging.WARNING: "#c8860a", logging.ERROR: "#c0392b",
                 logging.CRITICAL: "#c0392b"}.get(level)
        if color:
            self.live_log.appendHtml(
                f'<span style="color:{color}">{html.escape(msg)}</span>')
        else:
            self.live_log.appendPlainText(msg)
        self._scroll_log_bottom()

    def _scroll_log_bottom(self) -> None:
        sb = self.live_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _update_engine_state(self) -> None:
        st = self.engine.status()
        label = {"paused": "⏸ 已暂停", "busy": "● 运行中", "idle": "○ 空闲"}.get(
            st["state"], st["state"])
        self.lbl_engine_state.setText(
            f"{label}   队列 {st['queue_size']}   连败 {st['consecutive_failures']}")

    def _idle_refresh(self) -> None:
        """空闲且本标签可见时，纯截屏刷新底图（不跑 OCR，几乎零成本）。"""
        if not self.isVisible() or self.tabs.currentWidget() is not self.live_tab:
            return
        if self.engine.status()["state"] != "idle":
            return
        try:
            shot = self.vision.capture()
        except Exception:
            return
        self._show_pixmap(shot.img)
        self._update_engine_state()

    def _show_pixmap(self, img_bgr) -> None:
        # 全黑帧检测：mss 在 WSL/WSLg（X11 抓屏拿到黑帧）或 macOS 未授权「屏幕录制」
        # 时会返回近纯黑的一帧。此时不画黑块，改显示原因提示 + 打一次 warning。
        if float(img_bgr.max()) < 8:
            self._show_black_hint()
            return
        self._black_warned = False
        pix = np_to_pixmap(img_bgr)
        self.preview.setPixmap(pix.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_black_hint(self) -> None:
        self.preview.setText(
            "截屏为全黑帧。\n\n"
            "· WSL/WSLg 里跑 GUI：X11 抓屏拿不到真实画面，属开发机现象，\n"
            "  打包成 Windows exe 在真机上正常。\n"
            "· macOS：系统设置 → 隐私与安全性 → 屏幕录制 勾选本程序后重启程序。")
        if not getattr(self, "_black_warned", False):
            logging.getLogger(__name__).warning(
                "截屏为全黑帧（WSL 抓屏 / macOS 未授权屏幕录制？），已跳过预览渲染")
            self._black_warned = True

    # ---- 手动步骤下拉：枚举「流程编排」里配置的可定位步骤 ----

    @staticmethod
    def _step_has_target(step) -> bool:
        """该步骤是否有可在屏幕上定位的目标（否则不进手动测试下拉）。

        排除等待/按键/纯延时，以及「打给当前焦点」的无定位输入步骤。
        """
        if step.action not in LOCATOR_ACTIONS:
            return False
        return step.locator is not None or bool(step.ref)

    @staticmethod
    def _step_kind(step) -> str:
        if step.locator is not None:
            return "T2模板" if step.locator.kind == "template" else "T1文字"
        loc = locators.ALL.get(step.ref)
        return "T2模板" if isinstance(loc, TemplateLocator) else "T1文字"

    def _reload_step_choices(self) -> None:
        """按 流程 › 步骤 重建手动下拉，尽量保留当前选中项。"""
        prev = self.cmb_locator.currentData()
        self.cmb_locator.blockSignals(True)
        self.cmb_locator.clear()
        for flow in self.engine.flows.flows.values():
            for step in flow.steps:
                if not self._step_has_target(step):
                    continue
                text = f"{flow.title} › {step.label()}  [{self._step_kind(step)}]"
                self.cmb_locator.addItem(text, (flow.key, step.id))
        if prev is not None:
            idx = self.cmb_locator.findData(prev)
            if idx >= 0:
                self.cmb_locator.setCurrentIndex(idx)
        self.cmb_locator.blockSignals(False)

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.live_tab:
            self._reload_step_choices()

    def _resolve_step(self, data):
        """(flow_key, step_id) → (flow, step)；找不到返回 (None, None)。"""
        if not data:
            return None, None
        flow_key, step_id = data
        flow = self.engine.flows.flows.get(flow_key)
        if flow is None:
            return None, None
        step = next((s for s in flow.steps if s.id == step_id), None)
        return flow, step

    def _capture_hidden(self):
        """隐藏配置窗口后截屏，避免自己挡住目标页面。"""
        self.hide()
        QGuiApplication.processEvents()
        time.sleep(0.5)
        try:
            return self.vision.capture()
        finally:
            self.show()

    def _test_selected(self) -> None:
        flow, step = self._resolve_step(self.cmb_locator.currentData())
        if step is None:
            QMessageBox.information(self, "提示", "请先选择一个步骤")
            return
        self._run_test([step])

    def _test_all(self) -> None:
        steps = [s for flow in self.engine.flows.flows.values()
                 for s in flow.steps if self._step_has_target(s)]
        if not steps:
            QMessageBox.information(self, "提示", "当前没有可定位的步骤")
            return
        self._run_test(steps)

    def _run_test(self, steps: list) -> None:
        try:
            shot = self._capture_hidden()
        except Exception as e:
            QMessageBox.warning(self, "截屏失败", str(e))
            return
        annotated = shot.img.copy()
        reports = []
        for step in steps:
            label = step.label()
            try:
                loc = step.build_locator()
            except Exception as e:
                reports.append(f"[{label}] ✘ 定位配置无效: {e}")
                continue
            if loc is None:
                continue
            try:
                res = self.vision.locate(loc, shot)
            except Exception as e:
                reports.append(f"[{label}] ✘ 异常: {e}")
                continue
            draw_result(annotated, res)
            reports.append(f"[{label}]\n{res.describe()}")
        pix = np_to_pixmap(annotated)
        self.preview.setPixmap(pix.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.diag_text.setPlainText("\n\n".join(reports) or "（无可定位步骤）")

    def _recapture_template(self) -> None:
        flow, step = self._resolve_step(self.cmb_locator.currentData())
        if step is None:
            QMessageBox.information(self, "提示", "请先选择一个步骤")
            return
        label = step.label()
        try:
            loc = step.build_locator()
        except Exception as e:
            QMessageBox.warning(self, "定位配置无效", str(e))
            return
        if loc is None:
            QMessageBox.information(self, "提示", f"步骤「{label}」无可定位目标")
            return
        if isinstance(loc, TextLocator):
            if loc.fallback is None:
                QMessageBox.warning(
                    self, "提示", f"步骤「{label}」是文字锚点定位且无模板 fallback，无需截取模板")
                return
            loc = loc.fallback
        try:
            shot = self._capture_hidden()
        except Exception as e:
            QMessageBox.warning(self, "截屏失败", str(e))
            return
        self.hide()

        def _save(crop) -> None:
            cv2.imwrite(str(loc.path), crop)
            self.show()
            QMessageBox.information(self, "已保存", f"模板已更新: {loc.path}")

        # 关闭 overlay 后若未框选也要恢复窗口
        self._overlay = SnipOverlay(shot, _save)
        self._overlay.destroyed.connect(lambda: self.show())

    # ------------------------------------------------------------------
    # Tab 4 登录凭据
    # ------------------------------------------------------------------

    def _build_cred_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.le_user = QLineEdit()
        self.le_pwd = QLineEdit()
        self.le_pwd.setEchoMode(QLineEdit.Password)
        try:
            creds = self.settings.get_credentials()
            if creds:
                self.le_user.setText(creds[0])
        except Exception:
            pass
        form.addRow(QLabel("凭据经 keyring 存入 Windows 凭据管理器（DPAPI 加密），不落明文。"))
        form.addRow("用户名", self.le_user)
        form.addRow("密码", self.le_pwd)
        btn = QPushButton("保存凭据")
        btn.clicked.connect(self._save_creds)
        form.addRow(btn)
        return w

    def _save_creds(self) -> None:
        user = self.le_user.text().strip()
        pwd = self.le_pwd.text()
        if not user or not pwd:
            QMessageBox.warning(self, "提示", "用户名与密码均不能为空")
            return
        try:
            self.settings.set_credentials(user, pwd)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.le_pwd.clear()
        QMessageBox.information(self, "已保存", "凭据已写入系统凭据管理器")

    # ------------------------------------------------------------------
    # 托盘「立即测试打印」
    # ------------------------------------------------------------------

    def prompt_test_print(self) -> None:
        text, ok = QInputDialog.getText(self, "立即测试打印", "车辆自编号：")
        if ok and text.strip():
            size = self.engine.submit(Task(vehicle_no=text.strip(), source="manual"))
            QMessageBox.information(self, "已入队", f"任务已入队，当前队列 {size} 个")
