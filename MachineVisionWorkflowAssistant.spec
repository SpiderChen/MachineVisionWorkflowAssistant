# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包规格：机器视觉工作流助手（单文件 GUI exe）。

在 CI（GitHub Actions / windows-latest）上跑，本机无法在 Windows 验证，故对易漏的
依赖一律从宽收集，宁多勿缺：
- rapidocr_onnxruntime / onnxruntime：随包的 .onnx 模型与运行库 DLL 必须一起打包，否则 OCR 起不来
- uvicorn：运行期动态导入 loops / protocols，静态分析抓不全
- keyring + win32ctypes：Windows 凭据后端经入口点发现，需显式带上
- pystray：按平台动态选择托盘后端
- mss：按平台动态选择截屏后端
构建命令：pyinstaller MachineVisionWorkflowAssistant.spec --noconfirm --clean
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# 带模型 / 数据文件 / DLL 的包：整包收集（datas + binaries + hiddenimports）
for pkg in ("rapidocr_onnxruntime", "onnxruntime", "keyring"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 运行期动态 import 子模块的包：把全部子模块补进隐藏导入
for pkg in ("uvicorn", "pystray"):
    hiddenimports += collect_submodules(pkg)

hiddenimports += [
    "mss.windows",
    "win32ctypes.core",
    "win32ctypes.pywin32.win32cred",
    "keyring.backends.Windows",
    "keyring.backends.chainer",
    "keyring.backends.fail",
    "pyodbc",
    # 本包内在函数体里延迟导入的模块，静态分析偶有遗漏，显式列出保底
    "app.main",
    "app.engine",
    "app.vision",
    "app.input_ctl",
    "app.settings",
    "app.locators",
    "app.flows",
    "app.captcha",
    "app.monitor",
    "app.logger",
    "app.login_flow",
    "app.tray",
    "app.triggers.api_server",
    "app.triggers.db_poller",
    "app.config_ui",
    "app.config_ui.window",
    "app.config_ui.flow_tab",
]

block_cipher = None

a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest", "PySide6.QtWebEngineCore"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
    console=False,          # 托盘常驻 GUI，不弹控制台窗口；崩溃由 run.py 兜底记录并弹窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
