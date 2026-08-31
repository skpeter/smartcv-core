"""Replay a VOD at SmartCV's poll interval and print detector state changes.

Run from a game sibling (the repo that contains routines.py):

    python core/validate_vod.py path/to.mp4
    python core/validate_vod.py path/to.mp4 --start 50 --end 380 --step 0.5
    python core/validate_vod.py path/to.mp4 --jsonl logs/run.jsonl --metrics logs/run.json
    python core/validate_vod.py path/to.mp4 --offset 0.25
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

try:
    import psutil
except ImportError:
    psutil = None


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
    if payload.get("stage"):
        bits.append(f"stage={payload['stage']}")
    if "round" in payload:
        bits.append(f"round={payload['round']}")
    players = payload.get("players") or []
    if players:
        rounds = "-".join(str(p.get("rounds", "-")) for p in players)
        games = "-".join(str(p.get("games", "-")) for p in players)
        chars = " vs ".join(str(p.get("character") or "-") for p in players)
        stocks = "-".join(
            str(p.get("stocks") if p.get("stocks") is not None else "-")
            for p in players
        )
        bits.append(f"rounds={rounds}")
        if any("games" in p for p in players):
            bits.append(f"games={games}")
        bits.append(chars)
        bits.append(f"stocks={stocks}")
    return " ".join(bits)


def _video_end_s(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps > 1e-3 and n > 0:
        return n / fps
    duration_ms = cap.get(cv2.CAP_PROP_DURATION) or 0.0
    if duration_ms > 0:
        return duration_ms / 1000.0 if duration_ms > 1000 else duration_ms
    raise SystemExit("could not read duration; pass --end")


def _pctile(samples: list[float], p: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _nvidia_smi() -> dict | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    line = out.strip().splitlines()[0] if out.strip().splitlines() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        return {
            "gpu_util": float(parts[0]),
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
        }
    except ValueError:
        return None


def _sample_resources(proc) -> dict:
    sample = {"ts": time.time()}
    if proc is not None:
        sample["cpu_percent"] = proc.cpu_percent(interval=None)
        sample["rss_mb"] = proc.memory_info().rss / (1024 * 1024)
    gpu = _nvidia_smi()
    if gpu:
        sample.update(gpu)
    return sample


def _summarize_samples(resource_samples: list[dict]) -> dict:
    def _col(key: str) -> list[float]:
        return [s[key] for s in resource_samples if key in s]

    out = {}
    for key in ("cpu_percent", "rss_mb", "gpu_util", "vram_used_mb"):
        vals = _col(key)
        if not vals:
            continue
        out[key] = {
            "avg": round(statistics.fmean(vals), 2),
            "peak": round(max(vals), 2),
        }
    return out


def _resolve_offset(offset: float | None) -> float:
    if offset is None:
        offset = random.uniform(0.0, 0.5)
    if not 0.0 <= offset <= 0.5:
        raise SystemExit("--offset must be between 0 and 0.5")
    return offset


def run(
    path: str,
    start: float,
    end: float | None,
    step: float,
    ocr: bool,
    jsonl_path: str | None,
    metrics_path: str | None,
    sample_every: int,
    offset: float | None,
) -> None:
    offset = _resolve_offset(offset)
    if hasattr(routines, "ocr_enabled"):
        routines.ocr_enabled = ocr
    elif ocr:
        print("routines has no ocr_enabled; --ocr ignored")

    if hasattr(core, "reset_ocr_stats"):
        core.reset_ocr_stats()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    if end is None:
        end = _video_end_s(cap)

    jsonl_fp = None
    if jsonl_path:
        jsonl_file = Path(jsonl_path)
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = jsonl_file.open("w", encoding="utf-8")

    proc = psutil.Process() if psutil else None
    if proc is not None:
        proc.cpu_percent(interval=None)

    base_w = getattr(core, "base_width", 1920)
    base_h = getattr(core, "base_height", 1080)
    t = start + offset
    prev = None
    frames = 0
    frame_ms: list[float] = []
    resource_samples: list[dict] = []
    wall0 = time.perf_counter()
    print(f"scan {path}")
    print(f"range {t:.3f}s .. {end:.1f}s step {step}s offset={offset:.3f} ocr={ocr}")
    try:
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
            t_frame = time.perf_counter()
            for func in funcs:
                if func is None:
                    continue
                func(routines.payload, img, scale_x, scale_y)
            elapsed_ms = (time.perf_counter() - t_frame) * 1000.0
            frames += 1
            if len(frame_ms) < 20000:
                frame_ms.append(elapsed_ms)
            if sample_every > 0 and frames % sample_every == 0:
                resource_samples.append(_sample_resources(proc))
            cur = snapshot(routines.payload)
            if cur != prev:
                print(f"  t={t:7.1f}  {fmt(routines.payload)}")
                if jsonl_fp:
                    jsonl_fp.write(json.dumps({"t": t, "payload": routines.payload}, default=str) + "\n")
                    jsonl_fp.flush()
                resource_samples.append(_sample_resources(proc))
                prev = cur
            t += step
    finally:
        cap.release()
        if jsonl_fp:
            jsonl_fp.close()

    wall_s = time.perf_counter() - wall0
    print("done", fmt(routines.payload))

    ocr_stats = getattr(core, "ocr_stats", {}) or {}
    ocr_samples = list(ocr_stats.get("ms_samples") or [])
    metrics = {
        "vod": str(path),
        "start": start,
        "end": end,
        "step": step,
        "offset": round(offset, 4),
        "frames": frames,
        "wall_s": round(wall_s, 3),
        "frame_ms": {
            "mean": round(statistics.fmean(frame_ms), 2) if frame_ms else None,
            "p95": _pctile(frame_ms, 95),
        },
        "ocr": {
            "calls": ocr_stats.get("calls", 0),
            "ms_total": round(float(ocr_stats.get("ms_total") or 0.0), 2),
            "mean": round(statistics.fmean(ocr_samples), 2) if ocr_samples else None,
            "p95": _pctile(ocr_samples, 95),
        },
        "resources": _summarize_samples(resource_samples),
        "budget_ms": step * 1000.0,
    }
    if metrics["frame_ms"]["p95"] is not None:
        metrics["frame_ms"]["p95"] = round(metrics["frame_ms"]["p95"], 2)
    if metrics["ocr"]["p95"] is not None:
        metrics["ocr"]["p95"] = round(metrics["ocr"]["p95"], 2)
    print("metrics", json.dumps(metrics, indent=2))
    if metrics_path:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replay a VOD through the sibling's pixel detectors."
    )
    p.add_argument("vod", help="path to a video file")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds (default: video duration)")
    p.add_argument("--step", type=float, default=0.5, help="poll interval in seconds")
    p.add_argument(
        "--offset",
        type=float,
        default=None,
        help="phase offset in [0, 0.5] seconds added to --start. "
        "If omitted, a random value in that range is chosen.",
    )
    p.add_argument("--ocr", action="store_true", help="enable OCR if the sibling supports it")
    p.add_argument("--jsonl", default=None, help="write payload changes as JSONL")
    p.add_argument("--metrics", default=None, help="write resource/timing metrics JSON")
    p.add_argument("--sample-every", type=int, default=20, help="resource sample every N frames")
    args = p.parse_args()
    run(
        args.vod, args.start, args.end, args.step, args.ocr,
        args.jsonl, args.metrics, args.sample_every, args.offset,
    )


if __name__ == "__main__":
    main()
