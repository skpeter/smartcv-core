"""Replay a VOD at SmartCV's poll interval and print detector state changes.

Run from a game sibling (the repo that contains routines.py):

    python core/validate_vod.py path/to.mp4
    python core/validate_vod.py path/to.mp4 --start 50 --end 380 --step 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _prepare_root() -> None:
    if not (ROOT / "routines.py").is_file():
        raise SystemExit(
            "Run from a SmartCV game repo: routines.py must sit next to core/.\n"
            "Example: python core/validate_vod.py match.mp4"
        )
    os.chdir(ROOT)
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if not (ROOT / "config.ini").is_file():
        example = HERE / "config.ini.example"
        if not example.is_file():
            raise SystemExit("missing core/config.ini.example")
        shutil.copy(example, ROOT / "config.ini")


_prepare_root()

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

import routines  # noqa: E402
import core.core as core  # noqa: E402


def snapshot(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def fmt(payload: dict) -> str:
    bits = [f"state={payload.get('state')}"]
    if "round" in payload:
        bits.append(f"round={payload['round']}")
    players = payload.get("players") or []
    if players:
        rounds = "-".join(str(p.get("rounds", "-")) for p in players)
        games = "-".join(str(p.get("games", "-")) for p in players)
        chars = " vs ".join(str(p.get("character") or "-") for p in players)
        bits.append(f"rounds={rounds}")
        if any("games" in p for p in players):
            bits.append(f"games={games}")
        bits.append(chars)
    return " ".join(bits)


def run(path: str, start: float, end: float | None, step: float, ocr: bool) -> None:
    if hasattr(routines, "ocr_enabled"):
        routines.ocr_enabled = ocr
    elif ocr:
        print("routines has no ocr_enabled; --ocr ignored")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    if end is None:
        duration_ms = cap.get(cv2.CAP_PROP_DURATION)
        if duration_ms <= 0:
            cap.release()
            raise SystemExit("could not read duration; pass --end")
        end = duration_ms / 1000.0

    base_w = getattr(core, "base_width", 1920)
    base_h = getattr(core, "base_height", 1080)
    t = start
    prev = None
    print(f"scan {path}")
    print(f"range {start}s .. {end:.1f}s step {step}s ocr={ocr}")
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        scale_x = w / base_w
        scale_y = h / base_h
        if hasattr(routines, "_now"):
            routines._now = lambda ts=t: ts
        funcs = routines.states_to_functions.get(routines.payload.get("state"), [])
        for func in funcs:
            func(routines.payload, img, scale_x, scale_y)
        cur = snapshot(routines.payload)
        if cur != prev:
            print(f"  t={t:7.1f}  {fmt(routines.payload)}")
            prev = cur
        t += step
    cap.release()
    print("done", fmt(routines.payload))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replay a VOD through the sibling's pixel detectors."
    )
    p.add_argument("vod", help="path to a video file")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds (default: video duration)")
    p.add_argument("--step", type=float, default=0.5, help="poll interval in seconds")
    p.add_argument("--ocr", action="store_true", help="enable OCR if the sibling supports it")
    args = p.parse_args()
    run(args.vod, args.start, args.end, args.step, args.ocr)


if __name__ == "__main__":
    main()
