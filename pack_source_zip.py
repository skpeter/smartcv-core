"""Assemble source.zip for a SmartCV sibling.

Launch scripts live in this core repo. The packer copies them to the
archive root so source.zip users can run them next to routines.py.

Siblings invoke from repo root:

    python core/pack_source_zip.py --out dist/source.zip --name MyGame-main
"""
from __future__ import annotations

import argparse
import stat
import subprocess
import zipfile
from pathlib import Path

LAUNCHERS = ("smartcv.bat", "smartcv.sh", "smartcv.command")
EXTRA_FILES = (Path("core") / "build_info.py",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--out", type=Path, default=Path("dist/source.zip"))
    p.add_argument("--name", default=None, help="Folder name inside the zip")
    return p.parse_args()


def tracked_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--recurse-submodules", "-z"],
        cwd=root,
    )
    rels = [Path(p) for p in raw.decode("utf-8").split("\0") if p]
    seen = set(rels)
    for extra in EXTRA_FILES:
        if extra not in seen and (root / extra).is_file():
            rels.append(extra)
    return [rel for rel in rels if (root / rel).is_file()]


def add_file(zf: zipfile.ZipFile, abs_path: Path, arc: str) -> None:
    info = zipfile.ZipInfo(arc)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if Path(arc).suffix in {".sh", ".command"} else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    zf.writestr(info, abs_path.read_bytes())


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    name = args.name or root.name
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in tracked_files(root):
            add_file(zf, root / rel, f"{name}/{rel.as_posix()}")
        for launcher in LAUNCHERS:
            src = root / "core" / launcher
            if src.is_file():
                add_file(zf, src, f"{name}/{launcher}")


if __name__ == "__main__":
    main()
