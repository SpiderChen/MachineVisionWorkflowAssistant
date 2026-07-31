# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包规格：机器视觉工作流助手（跨平台）。

在 GitHub Actions 上构建：
  - Windows（windows-latest）      → 单文件 MachineVisionWorkflowAssistant.exe
  - macOS（macos-13 / macos-14）   → MachineVisionWorkflowAssistant.app（onedir 包）

本机（Linux）无法实测产物运行时行为，故对易漏依赖一律从宽收集，宁多勿缺：
- rapidocr_onnxruntime / onnxruntime：随包的 .onnx 模型与运行库必须一起打包，否则 OCR 起不来
- uvicorn：运行期动态导入 loops / protocols，静态分析抓不全
- keyring（collect_all 已含各平台后端）+ win32ctypes：Windows 凭据后端经入口点发现
- pystray：按平台动态选择托盘后端
构建：pyinstaller MachineVisionWorkflowAssistant.spec --noconfirm --clean
"""
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

datas = []
binaries = []
hiddenimports = []

# 带模型 / 数据文件 / 动态库的包：整包收集（datas + binaries + 子模块）
for pkg in ("rapidocr_onnxruntime", "onnxruntime", "keyring"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 运行期动态 import 子模块的包：把全部子模块补进隐藏导入
for pkg in ("uvicorn", "pystray"):
    hiddenimports += collect_submodules(pkg)

hiddenimports += ["pyodbc"]  # 仅 Windows 有该驱动，其它平台缺失只告警不致命

if IS_WIN:
    hiddenimports += [
        "mss.windows",
        "win32ctypes.core",
        "win32ctypes.pywin32.win32cred",
    ]
elif IS_MAC:
    hiddenimports += ["mss.darwin"]

# 本包内在函数体里延迟导入的模块，静态分析偶有遗漏，显式列出保底
hiddenimports += [
    "app.main", "app.engine", "app.vision", "app.input_ctl", "app.settings",
    "app.locators", "app.flows", "app.captcha", "app.monitor", "app.logger",
    "app.login_flow", "app.tray", "app.triggers.api_server", "app.triggers.db_poller",
    "app.config_ui", "app.config_ui.window", "app.config_ui.flow_tab",
    "app.config_ui.background",
]

block_cipher = None
excludes = [
    "tkinter", "matplotlib", "pytest",
    "PyQt5.QtWebEngine", "PyQt5.QtWebEngineCore", "PySide6.QtWebEngineCore",
]

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if IS_MAC:
    # macOS：onedir + .app 包（GUI 双击即开，比 onefile 在 mac 上更稳）
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="MachineVisionWorkflowAssistant",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="MachineVisionWorkflowAssistant",
    )
    app = BUNDLE(
        coll,
        name="MachineVisionWorkflowAssistant.app",
        icon=None,
        bundle_identifier="com.spiderchen.mvwa",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "0.1.3",
            # 截屏 / 输入会在运行时触发 macOS TCC 授权（屏幕录制 / 辅助功能），首次运行
            # 需在「系统设置 → 隐私与安全性」手动放行；不设 LSUIElement，保留 Dock 图标。
        },
    )
else:
    # Windows：单文件 exe，崩溃由 run.py 兜底记录并弹窗
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="MachineVisionWorkflowAssistant",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
