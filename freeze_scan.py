"""Dump hiddenimports that runtime torch/vision/easyocr need from the freeze.

Torch is excluded from PyInstaller analysis, so those imports never get packed.
Run this WHILE torch is still installed (CI: after pip install, before uninstall).

Writes one module name per line. Also prints a top-level summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Live in AppData vendor dir — do not require these inside the exe.
VENDOR_PREFIXES = (
    "torch", "torchvision", "torchaudio", "nvidia",
    "sympy", "mpmath", "networkx", "jinja2", "markupsafe",
    "fsspec", "filelock",
)


def _is_vendor(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in VENDOR_PREFIXES)


def scan(vendor: Path | None) -> list[str]:
    if vendor:
        sys.path.insert(0, str(vendor))
    before = set(sys.modules)
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import easyocr  # noqa: F401
    names = []
    for name in sorted(set(sys.modules) - before):
        if not name or name.startswith("_frozen_importlib"):
            continue
        if _is_vendor(name):
            continue
        names.append(name)
    return names


def _toplevel(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for n in names:
        top = n.split(".", 1)[0]
        if top not in seen:
            seen.add(top)
            out.append(top)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vendor", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    names = scan(args.vendor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(names) + "\n", encoding="utf-8")
    tops = _toplevel(names)
    print(f"freeze_scan: {len(names)} modules, {len(tops)} top-level -> {args.out}")
    print("top-level:", ", ".join(tops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
