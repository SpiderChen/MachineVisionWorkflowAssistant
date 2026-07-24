#!/usr/bin/env python3
"""把 arm64 与 x86_64 两份 PyInstaller .app 合并成 universal2 胖二进制 .app。

用法：make_universal.py <arm64.app> <x86_64.app> <out.app>

以 arm64 那份为骨架整树复制（保留符号链接），再逐个把 Mach-O 二进制用 `lipo -create`
与 x86_64 对应文件合体；非 Mach-O 文件（Python 脚本 / .onnx 模型 / 资源）两架构完全一致，
直接沿用 arm64 的即可。合并后 .app 内每个可执行 / 动态库都同时含两种架构，Intel 与
Apple Silicon 通吃。合并会使原有代码签名失效，调用方需在其后重新 ad-hoc 签名。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Mach-O 魔数：32/64 位、大小端，以及已是 fat 的（capabilities/ba be）
_MACHO_MAGIC = {
    b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",   # 64/32 位小端
    b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce",   # 64/32 位大端
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",   # fat（已含多架构）
}


def is_macho(p: Path) -> bool:
    if p.is_symlink() or not p.is_file():
        return False
    try:
        with p.open("rb") as f:
            return f.read(4) in _MACHO_MAGIC
    except OSError:
        return False


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    arm, x86, out = (Path(a) for a in sys.argv[1:4])
    for src in (arm, x86):
        if not src.is_dir():
            print(f"✗ 找不到输入 .app：{src}")
            return 1

    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(arm, out, symlinks=True)

    merged = arm_only = 0
    for cur in out.rglob("*"):
        if not is_macho(cur):
            continue
        rel = cur.relative_to(out)
        x86_file = x86 / rel
        if not x86_file.is_file():
            arm_only += 1
            print(f"  ⚠ 仅 arm64 有此二进制、x86_64 缺失，保持单架构：{rel}")
            continue
        subprocess.run(
            ["lipo", "-create", str(cur), str(x86_file), "-output", str(cur)],
            check=True,
        )
        merged += 1

    print(f"✓ universal2 合并完成：{merged} 个二进制已含双架构，{arm_only} 个仅 arm64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
