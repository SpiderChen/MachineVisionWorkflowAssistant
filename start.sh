#!/usr/bin/env bash
# 一键启动（WSL / Linux 开发用）：建/复用 venv → 装依赖 → 完整常驻启动。
#
# 默认完整常驻 = 托盘图标 + 配置窗口 + DB 轮询 + HTTP API（等价 `python run.py`）。
# 需要图形环境：Windows 11 的 WSLg 自动提供；否则装 X server，或改用 --headless。
#
# 用法：
#   ./start.sh                 # 完整常驻（默认）
#   ./start.sh --headless      # 透传给 run.py：无 UI（仅触发源 + 引擎）
#   ./start.sh --print A0231   # 透传：命令行执行一单
#   ./start.sh --locate all    # 透传：定位诊断，标注图存 logs/
#   REINSTALL=1 ./start.sh     # 强制重装依赖（不重建 venv）
#   RECREATE=1  ./start.sh     # 删除并重建 venv
set -euo pipefail

# 无论从哪个目录调用，都切到脚本所在目录（= 仓库根）
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_DIR="${VENV_DIR:-.venv}"
PY="${PYTHON:-python3}"
REQ="requirements.txt"
STAMP="$VENV_DIR/.req.sha256"

log()  { printf '\033[36m[start]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[start]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[start]\033[0m %s\n' "$*" >&2; exit 1; }

command -v "$PY" >/dev/null 2>&1 || die "找不到 $PY，请先安装 Python 3.10+"

# RECREATE=1 → 删掉旧环境重来
if [ "${RECREATE:-0}" = "1" ] && [ -d "$VENV_DIR" ]; then
  log "RECREATE=1 → 删除旧虚拟环境 $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

# 1) 虚拟环境：没有就建
FRESH=0
if [ ! -d "$VENV_DIR" ]; then
  log "创建虚拟环境 $VENV_DIR（$("$PY" --version 2>&1)）"
  "$PY" -m venv "$VENV_DIR" || die "创建 venv 失败，缺 venv 模块请装：sudo apt install python3-venv"
  FRESH=1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2) 依赖：首次创建 / requirements 变更 / REINSTALL=1 才装，否则秒起
need_install=0
if [ "$FRESH" = "1" ] || [ "${REINSTALL:-0}" = "1" ]; then
  need_install=1
elif [ ! -f "$STAMP" ] || ! sha256sum -c --status "$STAMP" 2>/dev/null; then
  need_install=1
fi

if [ "$need_install" = "1" ]; then
  log "安装/更新依赖（$REQ）…"
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r "$REQ"
  sha256sum "$REQ" > "$STAMP"
else
  log "依赖已是最新（$REQ 未变），跳过安装"
fi

# 3) 完整常驻需要图形环境；无显示且未走无头模式时给出提示（不阻断）
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  case " $* " in
    *" --headless "*|*" --print "*|*" --locate "*) : ;;  # 这些模式不需要 GUI
    *)
      warn "未检测到 \$DISPLAY / \$WAYLAND_DISPLAY —— 完整常驻需要图形环境。"
      warn "Windows 11 的 WSLg 会自动提供；否则请装 X server，或改用：./start.sh --headless"
      ;;
  esac
fi

# 4) 启动（透传全部参数给 run.py）。exec 让信号直达、进程替换当前 shell
log "启动：python run.py $*"
exec python run.py "$@"
