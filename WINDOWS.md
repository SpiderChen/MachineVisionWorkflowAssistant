# Windows 便携版

## 下载选择

- Windows 10/11、Windows Server 2016/2019/2022/2025（64 位）：下载
  `MachineVisionWorkflowAssistant-windows-x64.zip`，这是推荐版本。
- 32 位 Windows：下载 `MachineVisionWorkflowAssistant-windows-x86.zip`。
- Windows on ARM：优先尝试 x86 包，通过 Windows 自带的 x86 兼容层运行。

ZIP 中已包含 EXE、默认配置、示例和说明，不需要安装 Python。请先完整解压到普通可写
目录，再运行 `MachineVisionWorkflowAssistant.exe`；不要直接在压缩包内双击运行，也不要
把 EXE 单独移出目录（同目录的 `_internal` 原生运行库必须保留）。

## 支持边界

构建使用 GitHub 官方 Windows Server 2022 Runner，并在 Runner 上实际启动成品 EXE，
完成一次 ONNX OCR 初始化与推理后才上传。目标支持范围是 Windows Server 2016 及以上、
Windows 10/11。Windows 7/8/8.1、Vista 和 XP 已不受当前 Python、Qt 与 ONNX Runtime
依赖支持，因此不存在能可靠承诺“所有 Windows 都可用”的单一现代构建。

## 首次运行

1. 解压 ZIP，保留 EXE 与 `config.yaml` 在同一目录。
2. 双击 EXE；首次 OCR 初始化可能需要几十秒。
3. 打开配置页完成目标窗口、数据库/API 和页面元素配置。
4. 若系统报告缺少 VC++ DLL，请从微软官网下载并安装 Microsoft Visual C++
   2015-2022 Redistributable（x64 包对应 x64，x86 包对应 x86）。
5. 日志位于 EXE 同目录的 `logs/app.log`；启动前崩溃会写入 `startup_error.log`。

## 部署注意

- 该程序需要可交互的桌面会话；锁屏、注销或纯 Session 0 Windows 服务环境不能执行
  截屏和鼠标键盘自动化。
- 建议使用专用 Windows 用户，关闭锁屏、息屏和屏保，并保持目标程序/浏览器可见。
- 数据库驱动（例如 SQL Server ODBC Driver 17/18）仍需按现场数据库类型安装。
