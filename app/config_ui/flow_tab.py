"""「流程编排」Tab：截图 → 框选 → 选动作，可视化定义/编辑工作流。

用户操作只有三件事：截屏（或载入图片）、在图上拖框选中目标、下拉选动作类型；
锚点词/ROI/模板降级/点击点偏移全部由 flows.derive_locator 自动推导并展示推导说明。
每个步骤保存标注底图（flows_assets/）——既是活文档，也是重新编辑与回归验证的素材。

画布注意（踩坑记录）：缩放必须在 paintEvent 里现算并同步给鼠标坐标映射，
不得在 resizeEvent 里对子控件 setFixedSize（布局反馈环会导致窗口乱跳、点击错位）。
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ..qt_compat import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QPlainTextEdit, QPoint, QPushButton, QPixmap, QRect,
    QImage, QPainter, QPen, QColor, QSizePolicy, QSplitter, QToolButton, Qt,
    QVBoxLayout, QWidget, Signal, event_pos, qt_exec,
)

from ..flows import (
    ACTION_LABELS, KEY_CHOICES, LOCATOR_ACTIONS, SHOTS_DIR, VALUE_SOURCES,
    FlowError, Step, derive_locator,
)
from ..settings import TEMPLATES_DIR
from ..vision import Screenshot, draw_result
from .background import BackgroundTask

logger = logging.getLogger(__name__)


def _np_to_pixmap(img_bgr) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class AnnotCanvas(QWidget):
    """标注画布：自适应缩放显示截图；左键拖框选目标，右键指定点击点。"""

    boxSelected = Signal(int, int, int, int)     # (l, t, r, b) 图像坐标
    pointPicked = Signal(int, int)               # 右键指定点击点，图像坐标

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.setCursor(Qt.CrossCursor)
        self._pix: QPixmap | None = None
        self._box: tuple | None = None           # 图像坐标 (l,t,r,b)
        self._click: tuple | None = None         # 图像坐标 (x,y)
        self._extra_box: tuple | None = None     # 附加区域（验证码图片，蓝色）
        self._drag_start: QPoint | None = None
        self._drag_cur: QPoint | None = None
        self._hint = "尚未截图。点击上方「重新截屏」或「载入图片」开始标注"

    # ---- 外部接口 ----

    def set_image(self, img_bgr: np.ndarray | None, hint: str = "") -> None:
        self._pix = _np_to_pixmap(img_bgr) if img_bgr is not None else None
        if hint:
            self._hint = hint
        self._box = None
        self._click = None
        self._drag_start = None
        self.update()

    def set_annotation(self, box, click, extra_box=None) -> None:
        self._box = tuple(box) if box else None
        self._click = tuple(click) if click else None
        self._extra_box = tuple(extra_box) if extra_box else None
        self.update()

    # ---- 坐标映射（缩放参数在用到时现算，与绘制保持同源）----

    def _geom(self):
        if self._pix is None:
            return 1.0, 0.0, 0.0
        pw, ph = self._pix.width(), self._pix.height()
        scale = min(self.width() / pw, self.height() / ph)
        ox = (self.width() - pw * scale) / 2
        oy = (self.height() - ph * scale) / 2
        return scale, ox, oy

    def _to_img(self, pos: QPoint) -> tuple[int, int] | None:
        if self._pix is None:
            return None
        scale, ox, oy = self._geom()
        x = (pos.x() - ox) / scale
        y = (pos.y() - oy) / scale
        return (int(max(0, min(self._pix.width() - 1, x))),
                int(max(0, min(self._pix.height() - 1, y))))

    # ---- 绘制 ----

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(32, 32, 36))
        if self._pix is None:
            p.setPen(QColor(170, 170, 170))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._hint)
            p.end()
            return
        scale, ox, oy = self._geom()
        target = QRect(int(ox), int(oy), int(self._pix.width() * scale),
                       int(self._pix.height() * scale))
        p.drawPixmap(target, self._pix)

        def img_rect(box) -> QRect:
            return QRect(int(ox + box[0] * scale), int(oy + box[1] * scale),
                         int((box[2] - box[0]) * scale), int((box[3] - box[1]) * scale))

        if self._box:
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.drawRect(img_rect(self._box))
        if self._extra_box:
            p.setPen(QPen(QColor(60, 140, 255), 2))
            p.drawRect(img_rect(self._extra_box))
        if self._click:
            cx = int(ox + self._click[0] * scale)
            cy = int(oy + self._click[1] * scale)
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.drawLine(cx - 8, cy, cx + 8, cy)
            p.drawLine(cx, cy - 8, cx, cy + 8)
            p.setBrush(QColor(255, 60, 60))
            p.drawEllipse(QPoint(cx, cy), 3, 3)
        if self._drag_start and self._drag_cur:
            p.setPen(QPen(QColor(80, 180, 255), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(self._drag_start, self._drag_cur).normalized())
        p.end()

    # ---- 鼠标 ----

    def mousePressEvent(self, event):
        if self._pix is None:
            return
        if event.button() == Qt.LeftButton:
            self._drag_start = event_pos(event)
            self._drag_cur = self._drag_start
            self.update()
        elif event.button() == Qt.RightButton:
            pt = self._to_img(event_pos(event))
            if pt:
                self.pointPicked.emit(*pt)

    def mouseMoveEvent(self, event):
        if self._drag_start:
            self._drag_cur = event_pos(event)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self._drag_start:
            return
        p0 = self._to_img(self._drag_start)
        p1 = self._to_img(event_pos(event))
        self._drag_start = self._drag_cur = None
        self.update()
        if not p0 or not p1:
            return
        l, r = sorted((p0[0], p1[0]))
        t, b = sorted((p0[1], p1[1]))
        if r - l >= 6 and b - t >= 6:
            self.boxSelected.emit(l, t, r, b)


# 动作下拉的展示顺序
_ACTION_ORDER = ("click", "double_click", "right_click", "input", "wait", "key", "sleep")
_SOURCE_ORDER = ("vehicle_no", "username", "password", "captcha", "fixed")


class FlowTab(QWidget):
    """流程编排页。store 与执行引擎共享同一 FlowStore 实例，保存即生效。"""

    def __init__(self, settings, vision, engine, capture_hidden):
        super().__init__()
        self.settings = settings
        self.vision = vision
        self.store = engine.flows
        self._capture_hidden = capture_hidden    # ConfigWindow._capture_hidden
        self.flow = None
        self.step: Step | None = None
        self._shot: Screenshot | None = None     # 画布当前底图（可标注）
        self._canvas_preview = False             # True=画布是测试结果图，不可标注
        self._pending_add = False                # 「框选添加步骤」等待拖框
        self._pending_captcha = False            # 等待框选验证码图片区域
        self._reports: dict[str, str] = {}       # step.id → 最近一次推导说明
        self._loading = False
        self._task = BackgroundTask(self)
        self._build_ui()
        self._reload_flows()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter()
        root = QHBoxLayout(self)
        root.addWidget(splitter)

        # ---- 左：流程与步骤列表 ----
        left = QWidget()
        lv = QVBoxLayout(left)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("流程"))
        self.cmb_flow = QComboBox()
        self.cmb_flow.currentIndexChanged.connect(self._on_flow_changed)
        frow.addWidget(self.cmb_flow, stretch=1)
        lv.addLayout(frow)
        brow = QHBoxLayout()
        for text, fn in (("新建", self._new_flow), ("重命名", self._rename_flow),
                         ("删除", self._delete_flow)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            brow.addWidget(b)
        lv.addLayout(brow)

        self.list_steps = QListWidget()
        self.list_steps.setDragDropMode(QListWidget.InternalMove)
        self.list_steps.currentRowChanged.connect(self._on_step_selected)
        self.list_steps.model().rowsMoved.connect(self._on_rows_moved)
        self.list_steps.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_steps.customContextMenuRequested.connect(self._on_step_menu)
        lv.addWidget(QLabel("步骤（拖拽排序；右键上移 / 下移 / 删除）"))
        lv.addWidget(self.list_steps, stretch=1)

        # 一个按钮搞定所有添加：点击=框选添加，下拉里选按键/等待
        self.btn_add = QToolButton()
        self.btn_add.setText("➕ 添加步骤")
        self.btn_add.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_add.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_add.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_add.clicked.connect(self._start_add)
        add_menu = QMenu(self.btn_add)
        add_menu.addAction("框选添加（点击 / 输入等）", self._start_add)
        add_menu.addAction("按键步骤（Enter/F5 等）", lambda: self._add_plain("key"))
        add_menu.addAction("等待秒数步骤", lambda: self._add_plain("sleep"))
        self.btn_add.setMenu(add_menu)
        lv.addWidget(self.btn_add)
        btn_save = QPushButton("💾 保存流程")
        btn_save.clicked.connect(self._save)
        lv.addWidget(btn_save)

        # ---- 右：画布 + 步骤设置 ----
        right = QWidget()
        rv = QVBoxLayout(right)
        trow = QHBoxLayout()
        btn_shot = QPushButton("📷 重新截屏")
        btn_shot.clicked.connect(self._capture)
        btn_open = QPushButton("载入图片...")
        btn_open.clicked.connect(self._load_image)
        self.lbl_hint = QLabel("左键拖框选中目标元素；右键可微调点击点")
        self.lbl_hint.setStyleSheet("color: #888;")
        trow.addWidget(btn_shot)
        trow.addWidget(btn_open)
        trow.addWidget(self.lbl_hint, stretch=1)
        rv.addLayout(trow)

        self.canvas = AnnotCanvas()
        self.canvas.boxSelected.connect(self._on_box)
        self.canvas.pointPicked.connect(self._on_point)
        rv.addWidget(self.canvas, stretch=1)

        box = QGroupBox("步骤设置")
        form = QFormLayout(box)
        self.form = form
        self.cmb_action = QComboBox()
        for a in _ACTION_ORDER:
            self.cmb_action.addItem(ACTION_LABELS[a], a)
        self.cmb_action.currentIndexChanged.connect(self._on_prop_changed)
        form.addRow("动作", self.cmb_action)

        self.cmb_source = QComboBox()
        for s in _SOURCE_ORDER:
            self.cmb_source.addItem(VALUE_SOURCES[s], s)
        self.cmb_source.currentIndexChanged.connect(self._on_prop_changed)
        self.le_fixed = QLineEdit()
        self.le_fixed.setPlaceholderText("固定文本内容")
        self.le_fixed.editingFinished.connect(self._on_prop_changed)
        self.chk_clear = QCheckBox("输入前清空原内容（Ctrl+A + Delete）")
        self.chk_clear.toggled.connect(self._on_prop_changed)
        self.chk_verify = QCheckBox("输入后 OCR 回读校验（密码自动跳过）")
        self.chk_verify.toggled.connect(self._on_prop_changed)
        self.btn_captcha = QPushButton("🖼 框选验证码图片区域")
        self.btn_captcha.clicked.connect(self._start_captcha_box)
        form.addRow("输入内容", self.cmb_source)
        form.addRow("", self.le_fixed)
        form.addRow("", self.chk_clear)
        form.addRow("", self.chk_verify)
        form.addRow("", self.btn_captcha)

        self.cmb_key = QComboBox()
        for k in KEY_CHOICES:
            self.cmb_key.addItem(k.upper(), k)
        self.cmb_key.currentIndexChanged.connect(self._on_prop_changed)
        form.addRow("按键", self.cmb_key)

        self.sb_seconds = QDoubleSpinBox()
        self.sb_seconds.setRange(0.1, 600.0)
        self.sb_seconds.setSuffix(" 秒")
        self.sb_seconds.valueChanged.connect(self._on_prop_changed)
        form.addRow("等待时长", self.sb_seconds)

        self.chk_optional = QCheckBox("目标未出现时跳过此步（不判任务失败）")
        self.chk_optional.toggled.connect(self._on_prop_changed)
        self.sb_timeout = QDoubleSpinBox()
        self.sb_timeout.setRange(0.0, 120.0)
        self.sb_timeout.setSuffix(" 秒")
        self.sb_timeout.setSpecialValueText("默认")
        self.sb_timeout.valueChanged.connect(self._on_prop_changed)
        form.addRow("", self.chk_optional)
        form.addRow("单步超时", self.sb_timeout)

        self.txt_report = QPlainTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setMaximumHeight(96)
        self.txt_report.setPlaceholderText("框选后此处显示系统自动推导的定位方式与自检结果")
        form.addRow(self.txt_report)
        self.btn_test = QPushButton("🎯 测试此步定位（当前屏幕）")
        self.btn_test.clicked.connect(self._test_step)
        form.addRow(self.btn_test)
        rv.addWidget(box)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([320, 720])

    # ------------------------------------------------------------------
    # 流程 / 步骤列表
    # ------------------------------------------------------------------

    def _reload_flows(self, select: str = "") -> None:
        self._loading = True
        self.cmb_flow.clear()
        for f in self.store.flows.values():
            title = f.title + ("（内置）" if f.builtin else "")
            self.cmb_flow.addItem(title, f.key)
        self._loading = False
        idx = self.cmb_flow.findData(select) if select else 0
        self.cmb_flow.setCurrentIndex(max(idx, 0))
        self._on_flow_changed()

    def _on_flow_changed(self) -> None:
        if self._loading:
            return
        key = self.cmb_flow.currentData()
        self.flow = self.store.flows.get(key) if key else None
        self._refresh_steps()

    def _refresh_steps(self, keep_row: int = -1) -> None:
        self._loading = True
        self.list_steps.clear()
        if self.flow:
            for i, s in enumerate(self.flow.steps, 1):
                item = QListWidgetItem(f"{i}. {s.label()}")
                item.setData(Qt.UserRole, s.id)      # 拖拽排序后据此回写步骤顺序
                self.list_steps.addItem(item)
        self._loading = False
        if self.flow and self.flow.steps:
            row = keep_row if 0 <= keep_row < len(self.flow.steps) else 0
            self.list_steps.setCurrentRow(row)
        else:
            self.step = None
            self.canvas.set_image(None, "此流程还没有步骤：点击左侧「框选添加步骤」开始")
            self.txt_report.clear()

    def _on_step_selected(self, row: int) -> None:
        if self._loading or not self.flow or not (0 <= row < len(self.flow.steps)):
            return
        self.step = self.flow.steps[row]
        self._pending_add = False
        self._load_step_panel()
        self._load_step_canvas()

    def _load_step_panel(self) -> None:
        s = self.step
        self._loading = True
        self.cmb_action.setCurrentIndex(max(self.cmb_action.findData(s.action), 0))
        self.cmb_source.setCurrentIndex(max(self.cmb_source.findData(s.value_source), 0))
        self.le_fixed.setText(s.text)
        self.chk_clear.setChecked(s.clear_first)
        self.chk_verify.setChecked(s.verify)
        self.cmb_key.setCurrentIndex(max(self.cmb_key.findData(s.key), 0))
        self.sb_seconds.setValue(max(s.seconds, 0.1))
        self.chk_optional.setChecked(s.optional)
        self.sb_timeout.setValue(s.timeout)
        self._loading = False
        self._update_prop_visibility()
        self.txt_report.setPlainText(self._reports.get(s.id) or self._locator_summary(s))

    def _update_prop_visibility(self) -> None:
        a = self.cmb_action.currentData()
        is_input = a == "input"
        for w in (self.cmb_source, self.le_fixed, self.chk_clear, self.chk_verify):
            w.setVisible(is_input)
        self.le_fixed.setEnabled(self.cmb_source.currentData() == "fixed")
        self.btn_captcha.setVisible(
            is_input and self.cmb_source.currentData() == "captcha")
        self.cmb_key.setVisible(a == "key")
        self.sb_seconds.setVisible(a == "sleep")
        has_target = a in LOCATOR_ACTIONS
        self.chk_optional.setVisible(has_target)
        self.sb_timeout.setVisible(has_target)
        # QFormLayout 的行标签不随字段自动隐藏，需手动同步
        for w in (self.cmb_source, self.cmb_key, self.sb_seconds, self.sb_timeout):
            lbl = self.form.labelForField(w)
            if lbl is not None:
                lbl.setVisible(w.isVisibleTo(self))

    @staticmethod
    def _locator_summary(s: Step) -> str:
        if s.action not in LOCATOR_ACTIONS:
            return ""
        if s.locator is not None:
            sl = s.locator
            if sl.kind == "text":
                lines = [f"文字锚点「{sl.anchor}」定位（T1）"]
                if sl.roi:
                    lines.append("已限定搜索区域消歧")
                if sl.template:
                    lines.append("失败自动降级图像模板")
                return "；".join(lines)
            return "图像模板匹配定位（T2）"
        if s.ref:
            return f"使用内置注册元素 {s.ref} 定位；在此页重新框选可覆盖为本机标注"
        return "⚠ 尚未框选目标：请截屏后在图上拖框"

    def _load_step_canvas(self) -> None:
        s = self.step
        self._canvas_preview = False
        if s.shot:
            path = SHOTS_DIR / s.shot
            img = cv2.imread(str(path))
            if img is not None:
                self._shot = Screenshot(img=img)
                self.canvas.set_image(img)
                self.canvas.set_annotation(s.box, s.click, s.captcha_box)
                self.lbl_hint.setText("重新拖框可修改此步目标；右键微调点击点")
                return
        self._shot = None
        if s.action in LOCATOR_ACTIONS:
            self.canvas.set_image(
                None, "此步骤暂无标注截图。点「重新截屏」后拖框即可为它标注"
                if s.ref else "请点「重新截屏」后在图上拖框选中目标")
        else:
            self.canvas.set_image(None, "此步骤（按键/等待）无需框选目标")

    # ------------------------------------------------------------------
    # 截屏 / 载图 / 标注
    # ------------------------------------------------------------------

    def _capture(self) -> None:
        try:
            self._shot = self._capture_hidden()
        except Exception as e:
            QMessageBox.warning(self, "截屏失败", str(e))
            return
        self._canvas_preview = False
        self.canvas.set_image(self._shot.img)
        if self.step and self.step.box:
            self.canvas.set_annotation(None, None)
        self.lbl_hint.setText("左键拖框选中目标元素；右键可微调点击点")

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入页面截图", "",
                                              "图片 (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        data = np.fromfile(path, dtype=np.uint8)      # 兼容中文路径
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "载入失败", "无法读取该图片文件")
            return
        self._shot = Screenshot(img=img)
        self._canvas_preview = False
        self.canvas.set_image(img)
        self.lbl_hint.setText("左键拖框选中目标元素；右键可微调点击点")

    def _start_add(self) -> None:
        if not self.flow:
            return
        self._pending_add = True
        if self._shot is None or self._canvas_preview:
            self._capture()
        if self._shot is not None:
            self.lbl_hint.setText("⬅ 正在添加新步骤：请在截图上拖框选中目标元素")

    def _add_plain(self, action: str) -> None:
        if not self.flow:
            return
        step = Step(action=action)
        self.flow.steps.append(step)
        self._refresh_steps(keep_row=len(self.flow.steps) - 1)

    def _start_captcha_box(self) -> None:
        """值来源=验证码：目标输入框之外再框一个验证码图片区域（蓝框）。"""
        if self.step is None or self._shot is None or self._canvas_preview:
            QMessageBox.information(
                self, "提示", "请先截屏并框选目标输入框，再框选验证码图片区域")
            return
        self._pending_captcha = True
        self.lbl_hint.setText("⬅ 请在截图上拖框选中【验证码图片】区域（蓝框）")

    def _on_box(self, l: int, t: int, r: int, b: int) -> None:
        if self._canvas_preview:
            self.lbl_hint.setText("当前显示的是定位测试结果，请先「重新截屏」再标注")
            return
        if self._shot is None or not self.flow:
            return
        if self._pending_captcha:
            self._pending_captcha = False
            self._annotate_captcha((l, t, r, b))
            return
        if self._pending_add:
            step = Step(action="click")
            self.flow.steps.append(step)
            self.step = step
            self._pending_add = False
        elif self.step is not None and self.step.action in LOCATOR_ACTIONS:
            step = self.step
        else:
            self.lbl_hint.setText("请先点「框选添加步骤」，或在左侧选中要重新框选的步骤")
            return
        self._annotate(step, (l, t, r, b), ((l + r) // 2, (t + b) // 2))

    def _on_point(self, x: int, y: int) -> None:
        """右键微调点击点：沿用已框选区域，按新点击点重新推导偏移。"""
        s = self.step
        if (self._canvas_preview or s is None or self._shot is None
                or not s.box or s.action not in LOCATOR_ACTIONS):
            return
        self._annotate(s, tuple(s.box), (x, y))

    def _annotate(self, step: Step, box, click) -> None:
        shot = self._shot
        shot_name = f"{step.id}.png"

        def _work():
            sl, report = derive_locator(self.vision, shot, box, click, step.id)
            if not cv2.imwrite(str(SHOTS_DIR / shot_name), shot.img):
                raise FlowError(f"无法保存标注截图: {SHOTS_DIR / shot_name}")
            return sl, report

        def _done(result):
            sl, report = result
            step.locator = sl
            step.ref = ""
            step.box = list(box)
            step.click = list(click)
            step.shot = shot_name
            self._reports[step.id] = "\n".join(report)
            if self.flow is None or step not in self.flow.steps:
                return
            row = self.flow.steps.index(step)
            self._refresh_steps(keep_row=row)      # 触发 _on_step_selected 刷新面板
            self.lbl_hint.setText(
                "已生成定位（见下方说明）。可继续拖框修正或右键微调点击点")

        self._run_vision_task("正在初始化 OCR 并生成定位…", _work, _done, "标注失败")

    def _annotate_captcha(self, box) -> None:
        """验证码图片区域：换算为相对目标命中框的比例存储，并当场试识别一次。"""
        from ..captcha import solve_arith

        s = self.step
        shot = self._shot

        def _work():
            loc = s.build_locator()
            res = self.vision.locate(loc, shot) if loc is not None else None
            if res is None or not res.ok or res.chosen is None:
                raise FlowError(
                    "目标输入框在当前截图上未定位成功，请先框选/修正目标输入框")
            c = res.chosen
            l, t, r, b = box
            ratio = [round((l - c.cx) / max(c.w, 1), 3),
                     round((t - c.cy) / max(c.h, 1), 3),
                     round((r - c.cx) / max(c.w, 1), 3),
                     round((b - c.cy) / max(c.h, 1), 3)]
            got = solve_arith(self.vision, shot.img[t:b, l:r])
            return ratio, got

        def _done(result):
            ratio, got = result
            s.captcha_box = [int(v) for v in box]
            s.captcha_ratio = ratio
            self.canvas.set_annotation(s.box, s.click, s.captcha_box)
            extra = (f"试识别：{got[1]} = {got[0]} ✔" if got
                     else "⚠ 试识别失败：当前图上未解析出算术题（真实页面上再测）")
            report = (self._reports.get(s.id) or self._locator_summary(s)).rstrip()
            self._reports[s.id] = (
                f"{report}\n验证码图片区域已记录（相对目标框比例）\n{extra}")
            self.txt_report.setPlainText(self._reports[s.id])
            self.lbl_hint.setText("验证码区域已记录（蓝框）。可重新框选覆盖")

        self._run_vision_task(
            "正在定位输入框并识别验证码…", _work, _done, "无法标注验证码区域")

    # ------------------------------------------------------------------
    # 步骤属性 / 增删移
    # ------------------------------------------------------------------

    def _on_prop_changed(self, *_):
        if self._loading or self.step is None:
            return
        s = self.step
        s.action = self.cmb_action.currentData() or s.action
        s.value_source = self.cmb_source.currentData() or s.value_source
        s.text = self.le_fixed.text()
        s.clear_first = self.chk_clear.isChecked()
        s.verify = self.chk_verify.isChecked()
        s.key = self.cmb_key.currentData() or s.key
        s.seconds = self.sb_seconds.value()
        s.optional = self.chk_optional.isChecked()
        s.timeout = self.sb_timeout.value()
        if not s.ref:
            s.name = ""                            # 非内置步骤：名称随属性自动更新
        self._update_prop_visibility()
        row = self.list_steps.currentRow()
        if self.flow and 0 <= row < len(self.flow.steps):
            self.list_steps.item(row).setText(f"{row + 1}. {s.label()}")

    def _on_step_menu(self, pos) -> None:
        item = self.list_steps.itemAt(pos)
        if item is None:
            return
        self.list_steps.setCurrentItem(item)
        menu = QMenu(self.list_steps)
        menu.addAction("⬆ 上移", lambda: self._move_step(-1))
        menu.addAction("⬇ 下移", lambda: self._move_step(1))
        menu.addSeparator()
        menu.addAction("删除步骤", self._delete_step)
        qt_exec(menu, self.list_steps.mapToGlobal(pos))

    def _on_rows_moved(self, *_) -> None:
        """拖拽结束后：按列表当前顺序回写 flow.steps，并重排序号。"""
        if self._loading or not self.flow:
            return
        by_id = {s.id: s for s in self.flow.steps}
        order = [self.list_steps.item(i).data(Qt.UserRole)
                 for i in range(self.list_steps.count())]
        self.flow.steps = [by_id[i] for i in order if i in by_id]
        for i in range(self.list_steps.count()):
            s = by_id[self.list_steps.item(i).data(Qt.UserRole)]
            self.list_steps.item(i).setText(f"{i + 1}. {s.label()}")

    def _move_step(self, delta: int) -> None:
        row = self.list_steps.currentRow()
        if not self.flow or row < 0:
            return
        new = row + delta
        if not (0 <= new < len(self.flow.steps)):
            return
        steps = self.flow.steps
        steps[row], steps[new] = steps[new], steps[row]
        self._refresh_steps(keep_row=new)

    def _delete_step(self) -> None:
        row = self.list_steps.currentRow()
        if not self.flow or row < 0:
            return
        step = self.flow.steps.pop(row)
        for p in ((SHOTS_DIR / step.shot) if step.shot else None,
                  TEMPLATES_DIR / f"step_{step.id}.png"):
            try:
                if p and p.exists():
                    p.unlink()
            except OSError:
                pass
        self._refresh_steps(keep_row=min(row, len(self.flow.steps) - 1))

    # ------------------------------------------------------------------
    # 流程管理 / 保存 / 测试
    # ------------------------------------------------------------------

    def _new_flow(self) -> None:
        title, ok = QInputDialog.getText(self, "新建流程", "流程名称：")
        if ok and title.strip():
            f = self.store.create(title.strip())
            self._reload_flows(select=f.key)

    def _rename_flow(self) -> None:
        if not self.flow:
            return
        title, ok = QInputDialog.getText(self, "重命名流程", "流程名称：",
                                         text=self.flow.title)
        if ok and title.strip():
            self.flow.title = title.strip()
            self._reload_flows(select=self.flow.key)

    def _delete_flow(self) -> None:
        if not self.flow:
            return
        if QMessageBox.question(self, "删除流程",
                                f"确定删除流程「{self.flow.title}」？") != QMessageBox.Yes:
            return
        try:
            self.store.delete(self.flow.key)
        except FlowError as e:
            QMessageBox.warning(self, "无法删除", str(e))
            return
        self._reload_flows()

    def _save(self) -> None:
        try:
            self.store.save()
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        QMessageBox.information(self, "已保存", "流程已保存并即刻生效（无需重启）")

    def _run_vision_task(self, status: str, fn, on_done, error_title: str) -> None:
        """Run slow OCR/locating work without blocking the Qt event loop."""
        if self._task.busy:
            QMessageBox.information(self, "正在处理", "上一个 OCR/定位任务尚未完成，请稍候。")
            return
        self.setEnabled(False)
        self.lbl_hint.setText(status + "（窗口不会卡住）")
        logger.info(status.rstrip("…"))

        def _finished(result, error):
            self.setEnabled(True)
            if error is not None:
                QMessageBox.warning(self, error_title, str(error))
                self.lbl_hint.setText(f"{error_title}：{error}")
                return
            on_done(result)

        self._task.start(status.rstrip("…"), fn, _finished)

    def _test_step(self) -> None:
        if self.step is None:
            QMessageBox.information(self, "提示", "请先选择一个步骤")
            return
        if self.step.action not in LOCATOR_ACTIONS:
            QMessageBox.information(self, "提示", "按键/等待步骤没有定位目标，无需测试")
            return
        try:
            loc = self.step.build_locator()
        except FlowError as e:
            QMessageBox.warning(self, "无法测试", str(e))
            return
        if loc is None:
            QMessageBox.information(self, "提示", "此步骤作用于当前焦点，无定位可测")
            return
        try:
            shot = self._capture_hidden()
        except Exception as e:
            QMessageBox.warning(self, "截屏失败", str(e))
            return

        def _work():
            t0 = time.time()
            res = self.vision.locate(loc, shot)
            annotated = shot.img.copy()
            draw_result(annotated, res)
            return res, annotated, time.time() - t0

        def _done(result):
            res, annotated, elapsed = result
            self._canvas_preview = True
            self.canvas.set_image(annotated)
            self.lbl_hint.setText(
                "正在显示定位测试结果；重新选择步骤或重新截屏可返回标注")
            self.txt_report.setPlainText(f"{res.describe()}\n（耗时 {elapsed:.2f}s）")

        self._run_vision_task("正在初始化 OCR 并测试此步定位…", _work, _done, "测试失败")
